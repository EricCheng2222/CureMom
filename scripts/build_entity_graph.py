#!/usr/bin/env python3
"""Build the entity co-occurrence graph from `paper_entities`.

Run this after `extract_entities.py`. Re-run whenever paper_entities grows
materially (more than ~5%).

    PYTHONPATH=. python scripts/build_entity_graph.py
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from src.search.hipporag import build_entity_graph

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
    conn = psycopg.connect(DB_DSN)
    try:
        n = build_entity_graph(conn)
        log.info("Entity graph rebuilt: %d edges.", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
