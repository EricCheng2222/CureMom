#!/usr/bin/env python3
"""Backfill mechanism_of_action + dosage fields on the fda_drugs table.

Problem
-------
`fetch_fda_drugs.py` keeps the most-recent SPL per generic_name. For many
common drugs (acetaminophen, fexofenadine, ibuprofen, omeprazole, etc.)
the latest SPL is an OTC generic or a re-labelled bulk product that
doesn't carry a mechanism_of_action section — those clinical-prose
fields only live on the original brand-name innovator label.

Result today (2026-05-12): 1,230 / 1,760 fda_drugs rows have NULL
mechanism_of_action. Coverage of the actual drug names is fine; the
problem is field completeness.

Fix
---
For every row with NULL mechanism_of_action OR NULL dosage_and_administration,
query openFDA for ALL labels matching that generic_name. Pick the
sibling label that has the missing field filled (prefer original brand-
name / prescription-only labels). UPDATE the row in place, merging
field-by-field — never overwrite a field that's already populated.

Source of truth: openFDA's drug/label.json — same dataset the original
fetcher uses, just queried with a different sort/filter so we find the
clinical-detail-carrying SPLs that were skipped on the first pass.

Usage:
    PYTHONPATH=. python scripts/backfill_fda_mechanism.py
    PYTHONPATH=. python scripts/backfill_fda_mechanism.py --limit 50
    PYTHONPATH=. python scripts/backfill_fda_mechanism.py --names atorvastatin metformin
    PYTHONPATH=. python scripts/backfill_fda_mechanism.py --dry-run

Rate limit: openFDA allows 240 req/min unauthenticated; this fetches one
page of up to 10 labels per missing-field generic_name. With 1,230
candidates, the run takes ~5 min at 4 req/s.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import psycopg
from psycopg.rows import dict_row
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# Fields we attempt to backfill. Order matters for the "first non-null
# wins" merge — we walk siblings most-likely-to-have-content first.
BACKFILLABLE_FIELDS = (
    "mechanism_of_action",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "pharmacology",            # mapped from clinical_pharmacology
    "pharmacokinetics",
    "contraindications",
    "warnings",                # mapped from warnings_and_cautions / warnings
    "adverse_reactions",
    "drug_interactions",
    "indications_and_usage",
)

# Map our DB column → list of openFDA JSON keys to try (first non-empty wins).
_FDA_KEYS = {
    "mechanism_of_action":        ["mechanism_of_action"],
    "dosage_and_administration":  ["dosage_and_administration"],
    "dosage_forms_and_strengths": ["dosage_forms_and_strengths"],
    "pharmacology":               ["clinical_pharmacology", "pharmacology"],
    "pharmacokinetics":           ["pharmacokinetics"],
    "contraindications":          ["contraindications"],
    "warnings":                   ["warnings_and_cautions", "warnings"],
    "adverse_reactions":          ["adverse_reactions"],
    "drug_interactions":          ["drug_interactions"],
    "indications_and_usage":      ["indications_and_usage"],
}


def _first_text(field: Any) -> str | None:
    if not field:
        return None
    if isinstance(field, list):
        return field[0] if field else None
    return field


def _extract(label: dict, db_col: str) -> str | None:
    """Pull a clinical-text field from an openFDA label, trying each
    candidate JSON key. Returns None if nothing matches."""
    for fda_key in _FDA_KEYS.get(db_col, ()):
        val = _first_text(label.get(fda_key))
        if val and val.strip():
            return val.strip()
    return None


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _fetch_siblings(client: httpx.Client, generic_name: str, limit: int = 10) -> list[dict]:
    """Find all SPLs for this generic_name. We sort by
    effective_time desc so the freshest labels come first; brand-name
    originals usually appear near the top, but generic OTCs may dominate.
    The caller iterates until it finds one with the missing field."""
    # openFDA's search syntax: exact-match on a quoted phrase, ANDed with
    # the field-existence constraint. The escaped quotes are required.
    search = f'openfda.generic_name.exact:"{generic_name}"'
    params = {
        "search": search,
        "sort":   "effective_time:desc",
        "limit":  limit,
    }
    r = client.get(OPENFDA_LABEL_URL, params=params, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json().get("results", [])


def _missing_fields(row: dict) -> list[str]:
    """Which backfillable fields on this DB row are NULL or empty?"""
    return [
        col for col in BACKFILLABLE_FIELDS
        if not (row.get(col) or "").strip()
    ]


def _select_candidates(
    conn: psycopg.Connection,
    *, names: list[str] | None, limit: int | None,
) -> list[dict]:
    """Pick rows that need backfilling. Priority: rows missing
    mechanism_of_action first (the largest gap), then rows missing
    dosage_and_administration. Optionally filter by a name list.

    Important: filter to REAL FDA drugs (application_number OR spl_id
    set). The fda_drugs table also caches runtime drug-lookup misses
    (Wikipedia / PubChem fallback path in drug_lookup.py) for words the
    NER incorrectly flagged as drug-like — "Filaggrin", "Phosphorylation",
    "Malfunction", etc. Those have no FDA identifiers and will never
    return a sibling SPL; querying openFDA for them just burns API quota.
    """
    # Real-FDA marker: application_number OR spl_id present.
    base_where = (
        "(application_number IS NOT NULL OR spl_id IS NOT NULL)"
    )
    missing_clause = (
        "((mechanism_of_action IS NULL OR mechanism_of_action = '') "
        "OR (dosage_and_administration IS NULL OR dosage_and_administration = ''))"
    )
    where = f"{base_where} AND {missing_clause}"
    params: list[Any] = []
    if names:
        # Match each requested name as a case-insensitive substring of
        # generic_name. Common drugs ship as "ATORVASTATIN CALCIUM" etc.,
        # so substring is the right call, not exact.
        like_terms = [f"%{n.lower()}%" for n in names]
        where = f"LOWER(generic_name) ILIKE ANY(%s) AND ({where})"
        params.append(like_terms)
    sql = f"""
        SELECT id, generic_name,
               mechanism_of_action, dosage_and_administration,
               dosage_forms_and_strengths, pharmacology, pharmacokinetics,
               contraindications, warnings, adverse_reactions,
               drug_interactions, indications_and_usage
          FROM fda_drugs
         WHERE {where}
         ORDER BY
           (mechanism_of_action IS NULL OR mechanism_of_action = '') DESC,
           (dosage_and_administration IS NULL OR dosage_and_administration = '') DESC,
           id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _update_row(conn: psycopg.Connection, row_id: int, updates: dict[str, str]) -> None:
    if not updates:
        return
    cols = list(updates.keys())
    set_clause = ", ".join(f"{c} = %s" for c in cols)
    values = [updates[c] for c in cols] + [row_id]
    with conn.cursor() as cur:
        cur.execute(f"UPDATE fda_drugs SET {set_clause} WHERE id = %s", values)
    conn.commit()


def _wikipedia_updates(row: dict, missing: list[str]) -> dict[str, str]:
    """Try Wikipedia for the fields openFDA couldn't fill. Reuses the
    fetcher in src.api.drug_lookup which already handles the search →
    extract → section-split pipeline + name-relevance sanity check.

    Returns a dict of {db_col: text} for whichever missing fields
    Wikipedia carried. Empty if Wikipedia didn't find the drug.
    """
    from src.api.drug_lookup import lookup_wikipedia

    # Pure monograph names work best on Wikipedia (e.g. "ATORVASTATIN
    # CALCIUM" → article on atorvastatin). Try the generic_name first;
    # for combination products (NAME1, NAME2, NAME3) Wikipedia will
    # usually punt — that's fine, we just don't fill those.
    name = row["generic_name"]
    # Strip dose-form noise like "ACETAMINOPHEN 325 MG" → "ACETAMINOPHEN"
    cleaned = name.split(",")[0]
    cleaned = " ".join(w for w in cleaned.split() if not w.replace(".", "").isdigit() and "MG" not in w.upper())
    cleaned = cleaned.strip()
    if len(cleaned) < 3:
        return {}

    card = lookup_wikipedia(cleaned)
    if card is None:
        return {}

    field_map = {
        "mechanism_of_action":       card.mechanism,
        "indications_and_usage":     card.indications,
        "pharmacology":              card.pharmacology,
        "contraindications":         card.contraindications,
        "warnings":                  card.warnings,
        "dosage_and_administration": getattr(card, "dosage", None),
    }
    return {
        col: val.strip()
        for col, val in field_map.items()
        if col in missing and val and val.strip()
    }


def backfill(
    *, names: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    rate_per_sec: float = 4.0,
    wikipedia: bool = False,
) -> dict[str, int]:
    """Walk every row missing one or more clinical fields and merge in
    whatever can be salvaged from sibling openFDA labels.

    When `wikipedia=True`, runs a second pass for whatever's still missing
    after the openFDA pass — fetches the matching article via
    src.api.drug_lookup.lookup_wikipedia and pulls mechanism / indications /
    pharmacology / contraindications / warnings / dosage from the section
    map. Pure monographs (aspirin, ibuprofen, warfarin, ...) almost always
    resolve; combination OTC products usually don't, which is the right
    answer — their "mechanism" is the sum of components and isn't a
    single Wikipedia section.

    Returns counters: {rows_touched, fields_filled, no_sibling_help,
    wiki_touched, wiki_fields_filled}.
    """
    counters = {
        "rows_touched": 0, "fields_filled": 0, "no_sibling_help": 0,
        "wiki_touched": 0, "wiki_fields_filled": 0,
    }
    interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0

    with psycopg.connect(DB_DSN) as conn:
        candidates = _select_candidates(conn, names=names, limit=limit)
        log.info("Found %d rows to backfill (wikipedia=%s)", len(candidates), wikipedia)
        with httpx.Client(timeout=30) as client:
            for i, row in enumerate(candidates):
                missing = _missing_fields(row)
                if not missing:
                    continue
                t0 = time.monotonic()
                try:
                    siblings = _fetch_siblings(client, row["generic_name"])
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%d/%d] %s — fetch failed: %s",
                                i + 1, len(candidates), row["generic_name"], exc)
                    continue

                # Pass 1: walk siblings; for each missing field take the
                # first sibling that has a non-empty value.
                updates: dict[str, str] = {}
                for sib in siblings:
                    for col in list(missing):
                        if col in updates:
                            continue
                        val = _extract(sib, col)
                        if val:
                            updates[col] = val
                    if all(c in updates for c in missing):
                        break

                if not updates:
                    counters["no_sibling_help"] += 1
                else:
                    counters["rows_touched"] += 1
                    counters["fields_filled"] += len(updates)
                    log.info("[%d/%d] %s — fda: filled %s",
                             i + 1, len(candidates),
                             row["generic_name"], ", ".join(sorted(updates.keys())))

                # Pass 2: Wikipedia for whatever's still missing.
                still_missing = [c for c in missing if c not in updates]
                wiki_updates: dict[str, str] = {}
                if wikipedia and still_missing:
                    try:
                        wiki_updates = _wikipedia_updates(row, still_missing)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("[%d/%d] %s — wiki failed: %s",
                                    i + 1, len(candidates), row["generic_name"], exc)
                if wiki_updates:
                    counters["wiki_touched"] += 1
                    counters["wiki_fields_filled"] += len(wiki_updates)
                    log.info("[%d/%d] %s — wiki: filled %s",
                             i + 1, len(candidates),
                             row["generic_name"], ", ".join(sorted(wiki_updates.keys())))
                    updates.update(wiki_updates)

                if updates and not dry_run:
                    _update_row(conn, row["id"], updates)

                # Respect rate limit (openFDA: 240/min unauth = 4/s).
                # Wikipedia is also pretty generous (no published cap on
                # the public REST API), no need to throttle separately.
                elapsed = time.monotonic() - t0
                sleep = max(0.0, interval - elapsed)
                if sleep:
                    time.sleep(sleep)

    return counters


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="*", default=None,
                   help="Only backfill rows whose generic_name contains any of these substrings (case-insensitive).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N rows.")
    p.add_argument("--dry-run", action="store_true",
                   help="Log what would be filled but don't UPDATE.")
    p.add_argument("--rate", type=float, default=4.0,
                   help="Max openFDA requests per second (default 4 — well under the 240/min cap).")
    p.add_argument("--wikipedia", action="store_true",
                   help="Second pass: for rows still missing fields after the openFDA "
                        "sibling search, fetch the drug's Wikipedia article and pull "
                        "mechanism / indications / pharmacology / contraindications / "
                        "warnings / dosage from its section map.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log.info("Backfilling fda_drugs (dry_run=%s names=%s limit=%s wikipedia=%s)",
             args.dry_run, args.names, args.limit, args.wikipedia)
    counters = backfill(
        names=args.names, limit=args.limit,
        dry_run=args.dry_run, rate_per_sec=args.rate,
        wikipedia=args.wikipedia,
    )
    log.info(
        "Done. fda: rows_touched=%d fields_filled=%d no_sibling_help=%d  |  "
        "wiki: rows_touched=%d fields_filled=%d",
        counters["rows_touched"], counters["fields_filled"], counters["no_sibling_help"],
        counters["wiki_touched"], counters["wiki_fields_filled"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
