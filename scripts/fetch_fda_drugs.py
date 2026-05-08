#!/usr/bin/env python3
"""Pull ~1-3K commonly-prescribed drugs from openFDA's Drug Label API into
the `fda_drugs` table.

openFDA returns Structured Product Labels (SPLs). Many drugs have multiple
labels (different formulations / reformulations across years), so we
deduplicate by generic_name and keep the most recently-updated label.

The label ships richer prose than ChEMBL: indications_and_usage, mechanism_
of_action, dosage_and_administration, contraindications, etc. — exactly
what an LLM needs to answer "what is mephenoxalone used for?" with
specifics.

Free, no API key needed (rate-limited 240 req/min). With limit=100/page
and ~150K total labels, fetching the first 5K gives ~1.5K-3K unique
drugs after dedup.

Usage:
    PYTHONPATH=. python scripts/fetch_fda_drugs.py
    PYTHONPATH=. python scripts/fetch_fda_drugs.py --max-pages 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import psycopg
from psycopg.types.json import Jsonb
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
PAGE_SIZE = 100


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _fetch_page(client: httpx.Client, skip: int) -> list[dict]:
    """Fetch one page of labels, sorted by most-recently-updated."""
    params = {
        "search": "_exists_:openfda.generic_name AND _exists_:indications_and_usage",
        "sort":   "effective_time:desc",
        "limit":  PAGE_SIZE,
        "skip":   skip,
    }
    r = client.get(OPENFDA_LABEL_URL, params=params, timeout=30)
    if r.status_code == 404:
        return []  # past the end of results
    r.raise_for_status()
    return r.json().get("results", [])


def _first_text(field: list | str | None) -> str | None:
    """openFDA returns most prose fields as `[str]` lists. Pick the first."""
    if not field:
        return None
    if isinstance(field, list):
        return field[0] if field else None
    return field


def _parse_label(label: dict) -> dict | None:
    """Map an openFDA label record to fda_drugs row fields."""
    openfda = label.get("openfda", {}) or {}
    generic = openfda.get("generic_name") or []
    if not generic:
        return None
    return {
        "generic_name":              generic[0].strip(),
        "brand_names":               list({n.strip() for n in (openfda.get("brand_name") or []) if n}),
        "application_number":        (openfda.get("application_number") or [None])[0],
        "sponsor":                   (openfda.get("manufacturer_name") or [None])[0],
        "route":                     openfda.get("route") or [],
        "dosage_form":               openfda.get("dosage_form") or [],
        "marketing_status":          (openfda.get("product_type") or [None])[0],
        "indications_and_usage":     _first_text(label.get("indications_and_usage")),
        "mechanism_of_action":       _first_text(label.get("mechanism_of_action")),
        "pharmacology":              _first_text(label.get("clinical_pharmacology")),
        "pharmacokinetics":          _first_text(label.get("pharmacokinetics")),
        "contraindications":         _first_text(label.get("contraindications")),
        "warnings":                  _first_text(label.get("warnings_and_cautions") or label.get("warnings")),
        "adverse_reactions":         _first_text(label.get("adverse_reactions")),
        "drug_interactions":         _first_text(label.get("drug_interactions")),
        "dosage_and_administration": _first_text(label.get("dosage_and_administration")),
        "dosage_forms_and_strengths": _first_text(label.get("dosage_forms_and_strengths")),
        "rxcui":                     openfda.get("rxcui") or [],
        "unii":                      openfda.get("unii") or [],
        "spl_id":                    (openfda.get("spl_id") or [None])[0],
        "last_label_update":         label.get("effective_time"),
    }


def _upsert(conn: psycopg.Connection, row: dict, raw: dict) -> int:
    """Upsert by generic_name (case-insensitive)."""
    last_update = row.pop("last_label_update", None)
    # openFDA effective_time is YYYYMMDD string
    parsed_date = None
    if last_update and len(last_update) == 8 and last_update.isdigit():
        parsed_date = f"{last_update[:4]}-{last_update[4:6]}-{last_update[6:8]}"

    cols = list(row.keys()) + ["last_label_update", "raw_json"]
    vals = list(row.values()) + [parsed_date, Jsonb(raw)]
    placeholders = ",".join(["%s"] * len(cols))
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "generic_name")

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO fda_drugs ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT (generic_name) DO UPDATE SET {update_set}
            RETURNING id
            """,
            vals,
        )
        return cur.fetchone()[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-pages", type=int, default=50,
                   help="Max pages of 100 = 5,000 raw labels, ~1.5-3K unique drugs (default)")
    args = p.parse_args()

    conn = psycopg.connect(DB_DSN, autocommit=False)
    seen_generic: set[str] = set()
    inserted = updated = skipped = 0

    # Pre-load existing generic names so subsequent runs can skip dedup work
    with conn.cursor() as cur:
        cur.execute("SELECT LOWER(generic_name) FROM fda_drugs")
        seen_generic.update(r[0] for r in cur.fetchall())
    log.info("Existing fda_drugs rows: %d", len(seen_generic))

    with httpx.Client(follow_redirects=True) as client:
        for page in range(args.max_pages):
            skip = page * PAGE_SIZE
            try:
                labels = _fetch_page(client, skip=skip)
            except httpx.HTTPStatusError as exc:
                log.warning("Page skip=%d failed (%s); stopping.", skip, exc)
                break
            if not labels:
                log.info("No more results at skip=%d.", skip)
                break

            for label in labels:
                row = _parse_label(label)
                if row is None:
                    skipped += 1
                    continue
                key = row["generic_name"].lower()
                if key in seen_generic:
                    # Already have this generic; skip subsequent labels.
                    # (Sorted by effective_time desc, so first label wins.)
                    skipped += 1
                    continue
                seen_generic.add(key)
                _upsert(conn, row, label)
                inserted += 1

            conn.commit()
            log.info(
                "page %d/%d (skip=%d): page hits=%d | unique drugs so far=%d | dups skipped=%d",
                page + 1, args.max_pages, skip, len(labels), inserted, skipped,
            )

    log.info("Done. Unique drugs ingested: %d (skipped %d duplicate/no-generic-name labels)",
             inserted, skipped)
    conn.close()


if __name__ == "__main__":
    main()
