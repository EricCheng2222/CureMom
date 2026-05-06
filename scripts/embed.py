#!/usr/bin/env python3
"""Generate PubMedBERT embeddings for chunks lacking them.

Examples:
    # Embed everything pending
    PYTHONPATH=. python scripts/embed.py

    # Embed only specific papers
    PYTHONPATH=. python scripts/embed.py --paper-ids 1 2 3

    # Smaller batch size on CPU
    PYTHONPATH=. python scripts/embed.py --batch-size 8

    # After bulk load, build the HNSW index for fast vector search
    PYTHONPATH=. python scripts/embed.py --index-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from src.embeddings.chunk_pipeline import (
    chunk_paper,
    embed_pending_chunks,
    ensure_hnsw_index,
    ensure_pgvector_extension,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate chunk embeddings")
    p.add_argument("--paper-ids", nargs="*", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Embed at most N chunks (useful for smoke tests)")
    p.add_argument("--chunk-fulltext", action="store_true",
                   help="Re-chunk paper sections (introduction/methods/etc) before embedding")
    p.add_argument("--index-only", action="store_true",
                   help="Only build/refresh the HNSW index; don't embed")
    args = p.parse_args()

    conn = psycopg.connect(DB_DSN)
    try:
        ensure_pgvector_extension(conn)

        if args.index_only:
            log.info("Building HNSW index on paper_chunks.embedding…")
            ensure_hnsw_index(conn)
            log.info("Done.")
            return

        if args.chunk_fulltext:
            paper_ids = args.paper_ids
            if not paper_ids:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT paper_id FROM paper_sections ORDER BY paper_id")
                    paper_ids = [r[0] for r in cur.fetchall()]
            log.info("Re-chunking %d papers from paper_sections…", len(paper_ids))
            for pid in paper_ids:
                inserted = chunk_paper(conn, pid)
                if inserted:
                    log.info("  paper %d: +%d chunks", pid, inserted)
            conn.commit()

        embedded = embed_pending_chunks(
            conn, batch_size=args.batch_size,
            paper_ids=args.paper_ids, limit=args.limit,
        )
        log.info("Embedded %d chunks total.", embedded)

        if embedded:
            log.info("Building HNSW index…")
            ensure_hnsw_index(conn)
            log.info("HNSW index ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
