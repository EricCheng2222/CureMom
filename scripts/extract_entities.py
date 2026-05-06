#!/usr/bin/env python3
"""Run scispaCy NER over chunks and populate `paper_entities`.

Examples:
    PYTHONPATH=. python scripts/extract_entities.py
    PYTHONPATH=. python scripts/extract_entities.py --paper-ids 1 2 --with-linker
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from src.embeddings.ner_pipeline import extract_entities_for_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-ids", nargs="*", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--with-linker", action="store_true",
                   help="Resolve entities to UMLS CUIs via the scispaCy linker (slower)")
    args = p.parse_args()

    conn = psycopg.connect(DB_DSN)
    try:
        n = extract_entities_for_chunks(
            conn,
            paper_ids=args.paper_ids,
            limit=args.limit,
            with_linker=args.with_linker,
        )
        logging.info("Extracted %d entities total.", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
