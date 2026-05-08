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
_STOP = {
    "what","when","where","which","does","this","that","they","them","their",
    "from","with","about","have","take","take","help","helps","need","best",
    "lupus","muscle","growth","kidney","fatigue","pain","fever","headache",
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

    def to_text(self, max_field_chars: int = 600) -> str:
        """Format as a compact passage for the LLM context window."""
        def truncate(s: str | None) -> str | None:
            if not s:
                return None
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > max_field_chars:
                return s[:max_field_chars].rsplit(" ", 1)[0] + "…"
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

    # Regex safety net — single-word drug-name shapes
    for w in _WORD_RE.findall(query):
        wl = w.lower()
        if wl in _STOP:
            continue
        # Drug names tend to have specific endings or be long.
        if len(wl) >= 6 or wl.endswith(("zole","cillin","mycin","statin","prazole","oxalone")):
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
    return DrugCard(
        name=row["generic_name"],
        source="fda",
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

def lookup_drugs_for_query(
    conn: psycopg.Connection,
    query: str,
    max_drugs: int = 3,
    use_external_fallback: bool = True,
) -> list[DrugCard]:
    """Detect drug names in the query and return matching drug cards.

    Returns at most `max_drugs` cards. Lookup order:
      1. local fda_drugs (openFDA labels — ~1.7K modern Rx drugs)
      2. Wikipedia REST (older / discontinued / non-US drugs)

    Wikipedia hits are persisted into fda_drugs so subsequent calls are local.
    """
    cards: list[DrugCard] = []
    seen: set[str] = set()

    for name in candidate_drug_names(query):
        if len(cards) >= max_drugs:
            break
        if name in seen:
            continue
        seen.add(name)

        # Try the local DB first
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

    return cards
