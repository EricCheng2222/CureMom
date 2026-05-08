"""Drug-name detection in queries + structured lookup with PubChem fallback.

Pipeline:
  1. Detect drug-like terms in the user's query (HF biomedical NER + a
     normalized-name regex).
  2. For each candidate, look it up in `fda_drugs` (exact match on
     generic_name / brand_names, plus trigram fuzzy match for misspellings).
  3. If openFDA didn't have it (older or non-US drug), fall back to PubChem
     REST + PUG-View. Cache hits in `fda_drugs` so the next request is fast.

Output is a list of "drug cards" — short, structured text blocks that the
API layer prepends to the LLM context so the model has authoritative
clinical information to cite alongside the retrieved literature passages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

PUBCHEM_NAME_TO_CID = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
PUBCHEM_VIEW        = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"

# Wikipedia REST API — page summary + section content. Free, no key, fast.
WIKIPEDIA_SEARCH  = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_EXTRACT = "https://en.wikipedia.org/w/api.php"

# Heuristic: drug-name candidates are 4+ letter words, alphabetic, not common English.
# We trust the upstream NER pipeline to flag CHEMICAL / MEDICATION entities;
# this regex is a safety net for drug names embedded in conversational queries.
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]{3,}\b")

# Words that the regex safety net should NEVER treat as a candidate drug name.
# Includes common English, generic medical concepts, body parts, conditions —
# anything where a Wikipedia "lookup" produces irrelevant noise.
_STOP = {
    # Conversational filler
    "what","when","where","which","whose","who","whom","this","that","they","them","their","there",
    "tell","show","find","list","know","want","need","need","wonder","help","helps","helping",
    "best","good","great","most","more","less","kind","type","sort","example","examples",
    "available","modern","common","approved","useful","effective",
    # Question shells / quantifiers
    "does","do","is","are","was","were","be","been","have","had","has","take","using","used","uses",
    "about","with","without","from","over","under","into","through","between",
    # Medical concept words (NOT drug names — keep these for the reverse lookup)
    "drug","drugs","medication","medications","medicine","medicines","compound","compounds",
    "treatment","treatments","therapy","therapies","therapies","cure","cures",
    "agent","agents","class","classes","supplement","supplements",
    "side","effect","effects","reaction","reactions","interaction","interactions",
    # Generic medical / body-part / condition nouns
    "muscle","muscles","muscular","skeletal","cardiac","kidney","kidneys","renal","liver","brain",
    "growth","relaxation","relaxant","spasm","spasms","pain","ache","aches","fatigue",
    "fever","headache","nausea","cough","rash","inflammation","disease","syndrome","condition",
    "symptoms","symptom","cancer","diabetes","hypertension","cholesterol","blood","pressure",
    "lupus","autoimmune","arthritis","rheumatoid","nephritis",
    "anxiety","depression","insomnia","epilepsy","stroke","attack","disorder",
    # Time / scope
    "year","years","old","male","female","adult","child","children","patient","patients",
    "fast","slow","quick","quickly",
}


@dataclass
class DrugCard:
    """A structured drug-info block to include in the LLM context."""
    name: str
    source: str                 # 'fda' | 'pubchem'
    indications: str | None = None
    mechanism: str | None = None
    pharmacology: str | None = None
    contraindications: str | None = None
    warnings: str | None = None
    dosage: str | None = None
    brand_names: list[str] | None = None

    def to_text(self, max_field_chars: int = 1500) -> str:
        """Format as a compact passage for the LLM context window.

        Default 1500 chars per field — enough to fit the leading paragraph
        of an FDA Clinical Pharmacology section (where mechanism details
        usually live for older drugs that lack a dedicated MOA section).
        """
        def truncate(s: str | None) -> str | None:
            if not s:
                return None
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > max_field_chars:
                # Snap to a sentence boundary if one is nearby
                cut = s[:max_field_chars]
                last_period = cut.rfind(". ")
                if last_period > max_field_chars - 200:
                    return cut[:last_period + 1]
                return cut.rsplit(" ", 1)[0] + "…"
            return s

        lines = [f"DRUG REFERENCE — {self.name.upper()} ({self.source.upper()})"]
        if self.brand_names:
            lines.append(f"Brand names: {', '.join(self.brand_names[:5])}")
        for label, val in [
            ("Indications",       truncate(self.indications)),
            ("Mechanism",         truncate(self.mechanism)),
            ("Pharmacology",      truncate(self.pharmacology)),
            ("Dosing",            truncate(self.dosage)),
            ("Contraindications", truncate(self.contraindications)),
            ("Warnings",          truncate(self.warnings)),
        ]:
            if val:
                lines.append(f"{label}: {val}")
        return "\n".join(lines)


# ─── Detection ────────────────────────────────────────────────────────────

def candidate_drug_names(query: str) -> list[str]:
    """Pull drug-name candidates out of a query.

    Combines HF biomedical NER (CHEMICAL / MEDICATION entities) with a
    keyword regex fallback. Lowercased, deduped.
    """
    candidates: set[str] = set()

    try:
        from ..embeddings.ner_pipeline import _NERRunner
        # Reuse the global query-time runner if HippoRAG already loaded one
        from ..search.hipporag import _query_ner_runner as cached
        runner = cached if cached else _NERRunner()
        for ent in runner.extract(query, paper_id=0, chunk_id=0):
            if ent.entity_type == "CHEMICAL":
                candidates.add(ent.entity_text.lower())
    except Exception as exc:
        logger.debug("NER unavailable for drug detection (%s); using regex.", exc)

    # Regex safety net — only fires for tokens that LOOK like drug names.
    # Drug-shape suffixes are highly specific (penicilLIN, omepraZOLE, atorvaSTATIN…).
    # Generic long English words are filtered out via _STOP.
    drug_suffixes = (
        "cillin","mycin","sporin","statin","prazole","sartan","olol","pril",
        "tinib","ciclib","zumab","ximab","imab","ridine","oxetine","tropine",
        "oxalone","carbamol","cyclobenzaprine","barbital","azepam","azolam",
        "phylline","triptan","glitazone","gliflozin","semide","floxacin",
    )
    for w in _WORD_RE.findall(query):
        wl = w.lower()
        if wl in _STOP:
            continue
        if wl.endswith(drug_suffixes):
            candidates.add(wl)
        # Long unusual tokens that aren't English stopwords also pass — but
        # only if they're length >=8 (filters out general-purpose words like
        # "muscles", "kidneys", "relaxation"). Real drug names are typically
        # 8+ chars when not abbreviations.
        elif len(wl) >= 8 and any(ch in wl for ch in "xyz") or len(wl) >= 10:
            candidates.add(wl)

    return sorted(candidates)


# ─── Database lookup ──────────────────────────────────────────────────────

def lookup_in_db(conn: psycopg.Connection, name: str) -> DrugCard | None:
    """Find a drug in fda_drugs by exact-match (generic or brand) or trigram fuzzy."""
    with conn.cursor(row_factory=dict_row) as cur:
        # Exact case-insensitive match on generic_name OR brand_names
        cur.execute(
            """
            SELECT * FROM fda_drugs
            WHERE LOWER(generic_name) = LOWER(%s)
               OR EXISTS (
                   SELECT 1 FROM unnest(brand_names) b WHERE LOWER(b) = LOWER(%s)
               )
            LIMIT 1
            """,
            (name, name),
        )
        row = cur.fetchone()
        if not row:
            # Trigram fuzzy match — only for clearly non-trivial similarity
            cur.execute(
                """
                SELECT *, similarity(LOWER(generic_name), LOWER(%s)) AS sim
                FROM fda_drugs
                WHERE LOWER(generic_name) %% LOWER(%s)
                ORDER BY sim DESC
                LIMIT 1
                """,
                (name, name),
            )
            row = cur.fetchone()
            if row and float(row.get("sim") or 0) < 0.5:
                row = None

    if not row:
        return None
    # Recover the original source: openFDA-fetched rows have raw_json with
    # the full FDA label and no top-level "source" key; cached fallbacks
    # store {"source": "wikipedia"} (etc.) in raw_json.
    raw = row.get("raw_json") or {}
    source = raw.get("source") if isinstance(raw, dict) else None
    if source not in ("wikipedia", "pubchem"):
        source = "fda"
    return DrugCard(
        name=row["generic_name"],
        source=source,
        indications=row.get("indications_and_usage"),
        mechanism=row.get("mechanism_of_action"),
        pharmacology=row.get("pharmacology"),
        contraindications=row.get("contraindications"),
        warnings=row.get("warnings"),
        dosage=row.get("dosage_and_administration"),
        brand_names=row.get("brand_names") or [],
    )


# ─── PubChem fallback ─────────────────────────────────────────────────────

# Substring-match these headings → bucket. Order matters: more-specific first.
_PUBCHEM_HEADING_RULES = [
    ("indications",       ["drug indication", "therapeutic use", "indication and usage"]),
    ("mechanism",         ["mechanism of action", "moa"]),
    ("pharmacology",      ["pharmacology", "pharmacodynamics", "pharmacokinetics",
                           "absorption", "metabolism"]),
    ("dosage",            ["dosage and administration", "dosage", "fda approved products"]),
    ("contraindications", ["contraindication"]),
    ("warnings",          ["drug warning", "boxed warning", "drug-drug interaction",
                           "drug interactions"]),
]


def _bucket_for_heading(heading: str) -> str | None:
    h = heading.lower()
    for bucket, keys in _PUBCHEM_HEADING_RULES:
        if any(k in h for k in keys):
            return bucket
    return None


def _walk_pubchem_sections(node: dict, out: dict[str, list[str]]) -> None:
    """Depth-first walk of PUG-View JSON; collect text under recognized headings."""
    heading = node.get("TOCHeading") or ""
    bucket = _bucket_for_heading(heading) if heading else None
    if bucket:
        for info in node.get("Information", []) or []:
            for v in info.get("Value", {}).get("StringWithMarkup", []) or []:
                s = v.get("String")
                if s and len(s) > 15:
                    out.setdefault(bucket, []).append(s)
            # PubChem also embeds plain text in the 'Number' / 'Unit' fields
            # for some sections, ignore those.
    for child in node.get("Section", []) or []:
        _walk_pubchem_sections(child, out)


def lookup_wikipedia(name: str) -> DrugCard | None:
    """Fetch a drug card from Wikipedia. Used when openFDA doesn't have the
    drug (older / discontinued / non-US drugs). Wikipedia drug articles
    typically include sections like "Medical uses", "Mechanism of action",
    "Pharmacology", "Side effects" — exactly the buckets we want.
    """
    headers = {
        # Wikipedia requires a real User-Agent per their API policy
        "User-Agent": "CureMom/0.1 (https://github.com/EricCheng2222/CureMom; medical-literature-rag)",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=15, follow_redirects=True, headers=headers) as client:
            # Step 1: search for the article ID via the page-search API
            search = client.get(WIKIPEDIA_SEARCH, params={
                "action": "query", "format": "json",
                "list": "search", "srsearch": name,
                "srlimit": 1, "srprop": "",
            })
            search.raise_for_status()
            hits = search.json().get("query", {}).get("search", [])
            if not hits:
                return None
            title = hits[0]["title"]

            # Sanity check: is this article actually about the drug we asked for?
            # (Wikipedia search can match weird unrelated pages for short queries.)
            if name.lower() not in title.lower() and title.lower() not in name.lower():
                # Could still be valid (e.g., brand-name → generic redirect),
                # but require some token overlap.
                if not any(tok in title.lower() for tok in name.lower().split()):
                    return None

            # Step 2: pull the full page extract (plain text, no markup)
            extract = client.get(WIKIPEDIA_EXTRACT, params={
                "action": "query", "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
            })
            extract.raise_for_status()
            pages = extract.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            text = page.get("extract") or ""
            if not text or len(text) < 200:
                return None
    except Exception as exc:
        logger.warning("Wikipedia lookup failed for %r (%s)", name, exc)
        return None

    # Wikipedia plain-text extract is divided by section headings like
    # "== Medical uses ==", "== Mechanism of action ==", etc. (in the
    # explaintext mode they show up as bare titles followed by content).
    sections = _split_wiki_sections(text)
    summary = sections.get("summary")  # the lead paragraph(s) before any heading

    indications = (
        sections.get("medical uses")
        or sections.get("uses")
        or sections.get("indications")
        or summary  # stub articles often have no "Medical uses" section; use lead
    )
    mechanism = (
        sections.get("mechanism of action")
        or sections.get("pharmacodynamics")
        # If the summary mentions "inhibits"/"acts on" etc. and we don't have
        # a dedicated mechanism section, the summary often doubles as both.
        or (summary if indications != summary else None)
    )

    return DrugCard(
        name=name.title(),
        source="wikipedia",
        indications=indications,
        mechanism=mechanism,
        pharmacology=sections.get("pharmacology") or sections.get("pharmacokinetics"),
        contraindications=sections.get("contraindications"),
        warnings=sections.get("side effects") or sections.get("adverse effects"),
        dosage=sections.get("dosage") or sections.get("dose"),
    )


def _split_wiki_sections(text: str) -> dict[str, str]:
    """Split a Wikipedia explaintext extract into a {lower_heading: content} map."""
    out: dict[str, str] = {}
    # Wikipedia explaintext separates headings with a leading newline + 1+ word
    # followed by another newline. Heuristic split: lines that are short and
    # title-cased, followed by a blank line.
    lines = text.split("\n")
    current = "summary"
    body: list[str] = []
    for line in lines:
        s = line.strip()
        # Heuristic: heading lines are short, no trailing punctuation, mostly title case
        if (
            s and len(s) < 60
            and not s.endswith((".", ":", "?", "!"))
            and not s[0].islower()
            and len(s.split()) <= 6
            and any(c.isalpha() for c in s)
        ):
            # commit previous
            if body:
                out[current.lower()] = " ".join(body).strip()
            current = s
            body = []
        else:
            if s:
                body.append(s)
    if body:
        out[current.lower()] = " ".join(body).strip()
    return out


def cache_external_in_db(conn: psycopg.Connection, card: DrugCard) -> None:
    """Persist a non-FDA-sourced card into fda_drugs so the next request is fast."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fda_drugs
                (generic_name, indications_and_usage, mechanism_of_action,
                 pharmacology, contraindications, warnings,
                 dosage_and_administration, marketing_status, raw_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (generic_name) DO NOTHING
            """,
            (
                card.name, card.indications, card.mechanism, card.pharmacology,
                card.contraindications, card.warnings, card.dosage,
                card.source,
                Jsonb({"source": card.source}),
            ),
        )
    conn.commit()


# ─── Top-level entry point ───────────────────────────────────────────────

# Stopwords for the reverse-lookup query parser. Drops generic English filler
# AND drug-question shell words ("drug", "medication", "treatment", etc.) so
# the actual disease/effect terms drive FTS rank.
_REVERSE_STOPWORDS = {
    "what", "which", "when", "where", "how", "does", "do", "is", "are", "the",
    "for", "with", "and", "or", "of", "to", "an", "from", "by", "in",
    "tell", "me", "show", "list", "find", "good", "best", "most",
    "drug", "drugs", "medication", "medications", "medicine", "medicines",
    "treatment", "treatments", "therapy", "therapies", "compound", "compounds",
    "agent", "agents", "approved", "fda", "rx",
    "help", "helps", "treat", "treats", "used", "use", "useful",
    "available", "common", "modern",
}


def _query_content_terms(query: str) -> list[str]:
    """Extract content terms (>=3 chars, not stopwords) from a free-text query."""
    return [
        w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower())
        if w not in _REVERSE_STOPWORDS
    ]


def lookup_drugs_by_indication(
    conn: psycopg.Connection,
    query: str,
    limit: int = 3,
    min_rank: float = 0.01,
) -> list[DrugCard]:
    """Find FDA drugs whose indication/mechanism text matches the query.

    Ranking layers (combined into a single score):
      • OR over content terms — recall layer
      • ILIKE phrase match on the bigram (e.g. 'muscle relaxation') —
        boosts drugs that mention the FULL phrase, not just one word
      • Indications field weighted higher than pharmacology
    """
    terms = _query_content_terms(query)
    if not terms:
        return []

    or_tsq = " | ".join(terms)
    # Build phrase candidates: consecutive bigrams of content terms, plus
    # common medical morpheme variants ("muscle relax" → muscle relaxation,
    # muscle relaxant). Use ILIKE rather than tsvector for phrase boost so
    # exact-phrase prose hits score high regardless of how Postgres lemmatizes.
    bigrams = [
        f"%{terms[i]}%{terms[i+1]}%"
        for i in range(len(terms) - 1)
    ]
    # Also try truncated-stem variants for common pharmacology phrasing
    stems = []
    for i in range(len(terms) - 1):
        a, b = terms[i], terms[i+1]
        # "relaxation" / "relaxant" both share "relax" — match it
        if len(b) > 5:
            stems.append(f"%{a}%{b[:5]}%")
    phrase_patterns = bigrams + stems

    with conn.cursor(row_factory=dict_row) as cur:
        try:
            cur.execute(
                """
                WITH scored AS (
                  SELECT generic_name, brand_names, indications_and_usage,
                         mechanism_of_action, pharmacology, contraindications,
                         warnings, dosage_and_administration, raw_json,
                         ts_rank_cd(
                           setweight(to_tsvector('english', coalesce(indications_and_usage,'')), 'A') ||
                           setweight(to_tsvector('english', coalesce(mechanism_of_action,'')), 'B')   ||
                           setweight(to_tsvector('english', coalesce(pharmacology,'')), 'C'),
                           to_tsquery('english', %s)
                         ) AS fts_rank,
                         CASE WHEN %s::text[] IS NOT NULL AND EXISTS (
                                SELECT 1 FROM unnest(%s::text[]) p
                                WHERE LOWER(coalesce(indications_and_usage,'') || ' ' ||
                                            coalesce(mechanism_of_action,'')) LIKE p
                              )
                              THEN 1.0 ELSE 0.0
                         END AS phrase_hit
                  FROM fda_drugs
                  WHERE to_tsvector('english',
                          coalesce(indications_and_usage,'') || ' ' ||
                          coalesce(mechanism_of_action,'')   || ' ' ||
                          coalesce(pharmacology,'')
                        ) @@ to_tsquery('english', %s)
                )
                SELECT *, (fts_rank + phrase_hit * 5.0) AS rank
                FROM scored
                ORDER BY rank DESC
                LIMIT %s
                """,
                (or_tsq, phrase_patterns, phrase_patterns, or_tsq, limit * 2),
            )
            rows = cur.fetchall()
        except psycopg.errors.SyntaxError:
            return []

    cards: list[DrugCard] = []
    for row in rows:
        rank = float(row.get("rank") or 0)
        if rank < min_rank:
            continue
        raw = row.get("raw_json") or {}
        source = raw.get("source") if isinstance(raw, dict) else None
        if source not in ("wikipedia", "pubchem"):
            source = "fda"
        cards.append(DrugCard(
            name=row["generic_name"],
            source=source,
            indications=row.get("indications_and_usage"),
            mechanism=row.get("mechanism_of_action"),
            pharmacology=row.get("pharmacology"),
            contraindications=row.get("contraindications"),
            warnings=row.get("warnings"),
            dosage=row.get("dosage_and_administration"),
            brand_names=row.get("brand_names") or [],
        ))
        if len(cards) >= limit:
            break
    return cards


def lookup_drugs_for_query(
    conn: psycopg.Connection,
    query: str,
    max_drugs: int = 3,
    use_external_fallback: bool = True,
    use_reverse_lookup: bool = True,
    use_llm_analysis: bool = True,
) -> list[DrugCard]:
    """Detect drug names in the query and return matching drug cards.

    Pipeline (when LLM analysis is enabled — default):
      1. Ask the LLM to classify the query intent + extract clean drug names
         and canonical indication terms.
      2. If intent="about_specific_drug" → forward lookup on each drug name.
         If intent="find_drugs_by_effect" → reverse FTS on indication terms.
         If intent="general" → no drug cards (literature retrieval is enough).
      3. Wikipedia fallback only fires for forward-lookup misses.

    When LLM analysis is disabled, falls back to the regex-based detection.
    """
    cards: list[DrugCard] = []
    seen: set[str] = set()

    drug_names: list[str] = []
    indication_terms: list[str] = []
    intent = "general"

    if use_llm_analysis:
        try:
            from .query_analyzer import analyze_query
            analysis = analyze_query(query)
            intent = analysis.intent
            drug_names = analysis.drug_names
            indication_terms = analysis.indication_terms
            logger.info(
                "Query analyzed: intent=%s drug_names=%s indication_terms=%s",
                intent, drug_names, indication_terms,
            )
        except Exception as exc:
            logger.warning("LLM analysis unavailable (%s); using regex.", exc)

    # If LLM analysis didn't produce candidates, fall back to regex extraction
    if not drug_names and intent != "find_drugs_by_effect":
        drug_names = candidate_drug_names(query)

    # Forward lookup (drug names)
    if intent != "find_drugs_by_effect":
        for name in drug_names:
            if len(cards) >= max_drugs:
                break
            if name in seen:
                continue
            seen.add(name)
            card = lookup_in_db(conn, name)
            if card is None and use_external_fallback:
                card = lookup_wikipedia(name)
                if card is not None:
                    try:
                        cache_external_in_db(conn, card)
                    except Exception as exc:
                        logger.debug("Failed to cache wiki hit %r (%s)", name, exc)
            if card is not None:
                cards.append(card)

    # Reverse lookup (indication terms)
    if not cards and use_reverse_lookup and intent != "about_specific_drug":
        if indication_terms:
            # Use the LLM's canonical indication terms — much higher recall
            expanded = " ".join(indication_terms)
            cards = lookup_drugs_by_indication(conn, expanded, limit=max_drugs)
        else:
            # No LLM hint — fall back to using the raw query
            cards = lookup_drugs_by_indication(conn, query, limit=max_drugs)

    return cards
