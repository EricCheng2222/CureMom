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

from src.search.hipporag import build_entity_graph, build_mesh_graph, merge_ner_into_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)


def _refresh_hipporag_pickle() -> None:
    """Final step: rebuild the in-memory NetworkX graph from the freshly-
    written entity_graph table and save it to disk as a pickle.

    This must run AFTER all upserts in this process have committed,
    inside the same script — that way the read sees only committed
    rows and no other writer can race to add more during the read.
    Eliminates the read-write drift we saw when running the rebuild as
    a separate process.
    """
    from pathlib import Path
    from src.search.hipporag import HippoRAGRetriever

    log.info("Refreshing HippoRAG pickle from the freshly-built entity_graph…")
    # Drop any stale cache so HippoRAGRetriever takes the cold path
    # (rebuild from Postgres + write fresh pickle + matching meta).
    for name in ("hipporag_graph.pkl", "hipporag_graph.meta.json"):
        f = Path("data") / name
        if f.exists():
            f.unlink()
    r = HippoRAGRetriever(DB_DSN)
    r._ensure_loaded()
    log.info(
        "HippoRAG pickle ready: %d edges → data/hipporag_graph.pkl",
        r._graph.number_of_edges() if r._graph is not None else 0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["mesh", "ner", "both"], default="mesh")
    p.add_argument("--no-pickle", action="store_true",
                   help="Skip the post-build HippoRAG pickle refresh (useful when chaining multiple builds).")
    args = p.parse_args()

    conn = psycopg.connect(DB_DSN)
    try:
        if args.source == "mesh":
            n = build_mesh_graph(conn)
            log.info("MeSH graph: %d edges.", n)
        elif args.source == "ner":
            n = build_entity_graph(conn)
            log.info("NER graph: %d edges.", n)
        else:  # both
            n_mesh = build_mesh_graph(conn)
            log.info("MeSH graph: %d edges.", n_mesh)
            n_ner = merge_ner_into_graph(conn)
            log.info("Merged %d NER entity-pair upserts on top.", n_ner)
    finally:
        conn.close()

    if not args.no_pickle:
        _refresh_hipporag_pickle()


if __name__ == "__main__":
    main()
