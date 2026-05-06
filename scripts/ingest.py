#!/usr/bin/env python3
"""CLI for running the PubMed ingestion pipeline.

Usage:
  python scripts/ingest.py --topics sle_core lupus_nephritis --dry-run
  python scripts/ingest.py --topics sle_core --date-from 2020/01/01
  python scripts/ingest.py --all-topics
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingestion.pipeline import PipelineConfig, run_ingestion
from src.ingestion.topics import TOPICS, get_topic_by_name, get_topics_by_priority
from src.search.elasticsearch_client import get_client as get_es_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CureMom ingestion pipeline")
    parser.add_argument(
        "--topics", nargs="+", metavar="TOPIC",
        help="Topic names to ingest (e.g. sle_core lupus_nephritis)",
    )
    parser.add_argument(
        "--all-topics", action="store_true",
        help="Ingest all defined topics",
    )
    parser.add_argument(
        "--priority", type=int, default=1,
        help="Ingest topics up to this priority level (default: 1 = SLE only)",
    )
    parser.add_argument("--date-from", metavar="YYYY/MM/DD")
    parser.add_argument("--date-to", metavar="YYYY/MM/DD")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and queue PMIDs only, don't fetch full records")
    parser.add_argument("--db-dsn", default=None,
                        help="PostgreSQL DSN (default: from env vars)")
    parser.add_argument("--es-host", default=None,
                        help="Elasticsearch URL (default: from ES_HOST env var or http://localhost:9200)")
    args = parser.parse_args()

    db_dsn = args.db_dsn or (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'curemom')}"
    )

    config = PipelineConfig(
        db_dsn=db_dsn,
        ncbi_api_key=os.environ.get("NCBI_API_KEY"),
        ncbi_email=os.environ.get("NCBI_EMAIL", "curemom@example.com"),
    )

    if args.all_topics:
        topics = TOPICS
    elif args.topics:
        topics = []
        for name in args.topics:
            t = get_topic_by_name(name)
            if t is None:
                print(f"Unknown topic: {name!r}")
                print("Available:", [t.name for t in TOPICS])
                sys.exit(1)
            topics.append(t)
    else:
        topics = get_topics_by_priority(args.priority)

    print(f"Running ingestion for {len(topics)} topic(s):")
    for t in topics:
        print(f"  [{t.priority}] {t.name}: {t.description}")

    es_host = args.es_host or os.environ.get("ES_HOST", "http://localhost:9200")
    es = get_es_client(es_host)

    run_ingestion(
        config=config,
        topics=topics,
        date_from=args.date_from,
        date_to=args.date_to,
        dry_run=args.dry_run,
        es=es,
    )


if __name__ == "__main__":
    main()
