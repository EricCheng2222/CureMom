#!/usr/bin/env python3
"""Encode chunks with SPLADE and write sparse vectors into Elasticsearch.

    PYTHONPATH=. python scripts/encode_splade.py
    PYTHONPATH=. python scripts/encode_splade.py --paper-ids 1 2 --limit 100
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from src.embeddings.splade_pipeline import encode_and_index_chunks
from src.search.elasticsearch_client import get_client as get_es_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)
ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-ids", nargs="*", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    es = get_es_client(ES_HOST)
    conn = psycopg.connect(DB_DSN)
    try:
        n = encode_and_index_chunks(
            conn, es,
            paper_ids=args.paper_ids,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        logging.info("SPLADE-encoded %d papers.", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
