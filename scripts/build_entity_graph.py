#!/usr/bin/env python3
"""Build the entity co-occurrence graph for HippoRAG.

Two source modes:

  --source mesh    (default, fast)
      Build from MeSH descriptor co-occurrence on shared papers. No NER
      required — uses paper_mesh / mesh_terms which were populated during
      ingestion. ~minutes on 33K papers.

  --source ner     (high-resolution, slow)
      Build from paper_entities populated by scispaCy. Run
      extract_entities.py first. Captures non-MeSH entities (specific
      proteins, chemicals, cell lines).

  --source both    (recommended once NER is done)
      Build from MeSH first, then merge in NER entities.

    PYTHONPATH=. python scripts/build_entity_graph.py
    PYTHONPATH=. python scripts/build_entity_graph.py --source ner
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from src.search.hipporag import build_entity_graph, build_mesh_graph

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
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["mesh", "ner", "both"], default="mesh")
    args = p.parse_args()

    conn = psycopg.connect(DB_DSN)
    try:
        if args.source == "mesh":
            n = build_mesh_graph(conn)
            log.info("MeSH graph: %d edges.", n)
        elif args.source == "ner":
            n = build_entity_graph(conn)
            log.info("NER graph: %d edges.", n)
        else:
            n_mesh = build_mesh_graph(conn)
            log.info("MeSH graph: %d edges.", n_mesh)
            log.info("NER graph would overwrite — skipping. Run --source ner separately to add NER entities.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
