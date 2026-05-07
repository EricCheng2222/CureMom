#!/usr/bin/env python3
"""Fetch full-text JATS XML from PMC for papers with a PMCID.

Pulls the JATS XML through NCBI's efetch API (`db=pmc`), parses it with
`src/ingestion/jats_parser.py`, and populates `paper_sections`. After running
this, run:

    PYTHONPATH=. python scripts/embed.py --chunk-fulltext

to generate section-aware chunks (results-per-paragraph, sliding-window for
intro/methods/discussion) and embed them with PubMedBERT.

Examples:
    PYTHONPATH=. python scripts/fetch_pmc.py
    PYTHONPATH=. python scripts/fetch_pmc.py --limit 100
    PYTHONPATH=. python scripts/fetch_pmc.py --pmcids PMC1234567 PMC2345678
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import psycopg
from psycopg.rows import dict_row
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.jats_parser import parse_jats_xml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
NCBI_EMAIL   = os.environ.get("NCBI_EMAIL", "curemom@example.com")
TOOL_NAME    = "CureMom"

# 3 req/s without a key, 10 req/s with one
MIN_INTERVAL_S = 0.10 if NCBI_API_KEY else 0.34


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _fetch_pmc_xml(client: httpx.Client, pmcid: str) -> bytes | None:
    """Fetch JATS XML for one PMC paper. Returns None if paper is not OA."""
    pmc_id_numeric = pmcid.removeprefix("PMC")
    params = {
        "db": "pmc",
        "id": pmc_id_numeric,
        "rettype": "xml",
        "retmode": "xml",
        "tool": TOOL_NAME,
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    r = client.get(EFETCH_URL, params=params, timeout=30)
    r.raise_for_status()
    body = r.content

    # Non-OA papers come back as a tiny XML "not available" message instead of
    # a real article body. Detect by checking for <article> presence.
    if b"<article" not in body:
        return None
    return body


def _store_sections(conn: psycopg.Connection, paper_db_id: int, pmcid: str, sections) -> int:
    """Insert parsed sections; mark paper.has_full_text = true. Returns count."""
    with conn.cursor() as cur:
        # Wipe any prior sections (idempotent re-runs)
        cur.execute("DELETE FROM paper_sections WHERE paper_id = %s", (paper_db_id,))
        for s in sections:
            if not s.content or len(s.content.strip()) < 30:
                continue
            cur.execute(
                """
                INSERT INTO paper_sections
                    (paper_id, section_type, section_order, title, content)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (paper_db_id, s.section_type, s.section_order, s.title, s.content),
            )
        cur.execute(
            "UPDATE papers SET has_full_text = TRUE, last_updated = NOW() WHERE id = %s",
            (paper_db_id,),
        )
    conn.commit()
    return len(sections)


def fetch_pmc_full_text(
    conn: psycopg.Connection,
    pmcids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Fetch JATS XML for all eligible papers. Returns (ok, not_oa, errors)."""
    where = "WHERE p.pmcid IS NOT NULL AND NOT p.has_full_text"
    params: list = []
    if pmcids:
        where += " AND p.pmcid = ANY(%s)"
        params.append(pmcids)
    sql = f"SELECT p.id, p.pmid, p.pmcid FROM papers p {where} ORDER BY p.id"
    if limit:
        sql += f" LIMIT {int(limit)}"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        log.info("No eligible papers (pmcid IS NOT NULL AND NOT has_full_text).")
        return 0, 0, 0

    log.info(
        "Fetching full text for %d papers (rate=%s req/s)",
        len(rows), int(1 / MIN_INTERVAL_S),
    )

    ok = not_oa = errors = 0
    last_request = 0.0
    with httpx.Client(follow_redirects=True) as client:
        for i, row in enumerate(rows, start=1):
            wait = MIN_INTERVAL_S - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
            last_request = time.monotonic()

            pmcid = row["pmcid"]
            try:
                xml = _fetch_pmc_xml(client, pmcid)
                if xml is None:
                    not_oa += 1
                    continue
                parsed = parse_jats_xml(xml)
                if parsed is None or not parsed.sections:
                    not_oa += 1
                    continue
                _store_sections(conn, row["id"], pmcid, parsed.sections)
                ok += 1
            except Exception as exc:
                log.warning("PMCID %s failed: %s", pmcid, exc)
                errors += 1

            if i % 100 == 0 or i == len(rows):
                log.info("  processed %d / %d (ok=%d, not_oa=%d, err=%d)",
                         i, len(rows), ok, not_oa, errors)

    return ok, not_oa, errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pmcids", nargs="*", default=None,
                   help="Limit to specific PMCIDs (e.g. PMC1234567)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N papers")
    args = p.parse_args()

    conn = psycopg.connect(DB_DSN)
    try:
        ok, not_oa, errors = fetch_pmc_full_text(
            conn, pmcids=args.pmcids, limit=args.limit,
        )
        log.info("Done. Full-text ingested: %d | non-OA: %d | errors: %d",
                 ok, not_oa, errors)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
