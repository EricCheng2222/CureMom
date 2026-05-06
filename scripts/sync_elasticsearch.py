"""Sync PostgreSQL papers → Elasticsearch + create abstract chunks in paper_chunks.

Run once (or re-run safely — idempotent):
    PYTHONPATH=. python scripts/sync_elasticsearch.py

What it does:
  1. Creates the Elasticsearch 'papers' index (skips if already exists).
  2. Reads papers from PostgreSQL in batches and bulk-indexes to ES.
  3. Inserts one abstract chunk per paper into paper_chunks (skips existing).
"""

from __future__ import annotations

import logging
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.elasticsearch_client import bulk_index_papers, ensure_index, get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)
ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")
BATCH = 500


def _fetch_batch(conn: psycopg.Connection, offset: int, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.id,
                p.pmid,
                p.pmcid,
                p.doi,
                p.title,
                p.abstract,
                p.pub_year,
                p.publication_types,
                p.language,
                p.has_full_text,
                p.grant_agencies,
                j.title AS journal_title,
                ARRAY_AGG(DISTINCT mt.descriptor_name) FILTER (WHERE mt.descriptor_name IS NOT NULL) AS mesh_terms,
                ARRAY_AGG(DISTINCT mt.descriptor_name) FILTER (WHERE pm.is_major_topic AND mt.descriptor_name IS NOT NULL) AS mesh_major_terms,
                ARRAY_AGG(a.last_name || COALESCE(' ' || a.initials, '') ORDER BY pa.position) FILTER (WHERE a.last_name IS NOT NULL) AS author_names
            FROM papers p
            LEFT JOIN journals j ON p.journal_id = j.id
            LEFT JOIN paper_mesh pm ON p.id = pm.paper_id
            LEFT JOIN mesh_terms mt ON pm.mesh_id = mt.id
            LEFT JOIN paper_authors pa ON p.id = pa.paper_id
            LEFT JOIN authors a ON pa.author_id = a.id
            GROUP BY p.id, p.pmid, p.pmcid, p.doi, p.title, p.abstract,
                     p.pub_year, p.publication_types, p.language, p.has_full_text,
                     p.grant_agencies, j.title
            ORDER BY p.id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        return cur.fetchall()


def _build_es_doc(row: dict) -> dict:
    authors = row["author_names"] or []
    authors_text = "; ".join(authors[:10])  # cap at 10 for ES
    return {
        "pmid":             row["pmid"],
        "pmcid":            row["pmcid"],
        "doi":              row["doi"],
        "title":            row["title"] or "",
        "abstract":         row["abstract"] or "",
        "pub_year":         row["pub_year"],
        "journal_title":    row["journal_title"],
        "publication_types": row["publication_types"] or [],
        "mesh_terms":       row["mesh_terms"] or [],
        "mesh_major_terms": row["mesh_major_terms"] or [],
        "language":         row["language"],
        "has_full_text":    row["has_full_text"],
        "grant_agencies":   row["grant_agencies"] or [],
        "authors":          authors_text,
    }


def sync_elasticsearch(conn: psycopg.Connection, es) -> int:
    """Bulk-index all papers into ES. Returns total indexed."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM papers")
        total = cur.fetchone()["n"]
    log.info("Syncing %d papers to Elasticsearch...", total)

    indexed = 0
    offset = 0
    while True:
        rows = _fetch_batch(conn, offset, BATCH)
        if not rows:
            break
        docs = [_build_es_doc(r) for r in rows]
        ok, err = bulk_index_papers(es, docs)
        indexed += ok
        if err:
            log.warning("Batch offset=%d: %d errors", offset, err)
        offset += BATCH
        log.info("  ES indexed %d / %d", indexed, total)
    return indexed


def sync_chunks(conn: psycopg.Connection) -> int:
    """Insert one abstract chunk per paper (skip if already exists)."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM paper_chunks")
        existing = cur.fetchone()["n"]
    if existing > 0:
        log.info("paper_chunks already has %d rows — skipping chunk creation.", existing)
        return existing

    log.info("Creating abstract chunks in paper_chunks...")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO paper_chunks
                (paper_id, section_id, chunk_index, chunk_text, source_type,
                 start_char, end_char, paragraph_index, token_count)
            SELECT
                p.id,
                NULL,
                0,
                COALESCE(p.abstract, p.title, ''),
                'abstract',
                0,
                LENGTH(COALESCE(p.abstract, p.title, '')),
                0,
                -- rough token estimate: chars / 4
                LENGTH(COALESCE(p.abstract, p.title, '')) / 4
            FROM papers p
            WHERE p.abstract IS NOT NULL OR p.title IS NOT NULL
            ON CONFLICT DO NOTHING
            """
        )
        inserted = cur.rowcount
    conn.commit()
    log.info("Inserted %d abstract chunks.", inserted)
    return inserted


def main() -> None:
    t0 = time.monotonic()
    es = get_client(ES_HOST)
    ensure_index(es)

    conn = psycopg.connect(DB_DSN, row_factory=dict_row)
    try:
        es_count = sync_elasticsearch(conn, es)
        chunk_count = sync_chunks(conn)
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    log.info(
        "Done in %.1fs — ES: %d papers indexed, paper_chunks: %d rows",
        elapsed, es_count, chunk_count,
    )


if __name__ == "__main__":
    main()
