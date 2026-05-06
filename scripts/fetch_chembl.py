#!/usr/bin/env python3
"""Fetch compound-target data from ChEMBL and queue PubMed papers for ingestion.

Workflow:
  1. For each molecular target in targets.py, look up its ChEMBL target ID
  2. Fetch all compounds active against that target (above confidence threshold)
  3. Store compounds and compound-target relationships in the DB
  4. Queue PubMed papers for each compound via ESearch on compound names
  5. Ingestion pipeline (ingest.py) then fetches the full papers

No API key required — ChEMBL is fully open access.

Usage:
  python scripts/fetch_chembl.py
  python scripts/fetch_chembl.py --genes JAK1 JAK2 MTOR
  python scripts/fetch_chembl.py --biology sle
  python scripts/fetch_chembl.py --dry-run
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import psycopg
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ingestion.chembl_client import ChEMBLClient, ChEMBLCompound, get_pubmed_search_terms
from src.ingestion.targets import TARGETS, MolecularTarget, get_targets_by_biology

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    wait=wait_exponential(min=2, max=30),
    stop=stop_after_attempt(3),
)
def _esearch_pmids(http: httpx.Client, query: str, email: str) -> list[str]:
    """PubMed ESearch for a compound name — returns matching PMIDs."""
    r = http.get(NCBI_ESEARCH, params={
        "db": "pubmed",
        "term": f'"{query}"[Title/Abstract] AND ("mechanism"[Title/Abstract] OR "pharmacology"[Title/Abstract] OR "signaling"[Title/Abstract])',
        "retmax": "200",
        "retmode": "json",
        "tool": "curemom",
        "email": email,
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _upsert_compound(conn: psycopg.Connection, compound: ChEMBLCompound) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO compounds (chembl_id, name, synonyms, max_phase, molecule_type,
                                   molecular_formula, molecular_weight, inchi_key)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chembl_id) DO UPDATE SET
                name             = EXCLUDED.name,
                synonyms         = EXCLUDED.synonyms,
                max_phase        = EXCLUDED.max_phase,
                molecule_type    = EXCLUDED.molecule_type,
                molecular_formula = EXCLUDED.molecular_formula,
                molecular_weight = EXCLUDED.molecular_weight,
                inchi_key        = EXCLUDED.inchi_key
            RETURNING id
            """,
            (
                compound.chembl_id,
                compound.name,
                compound.synonyms or [],
                compound.max_phase,
                compound.molecule_type,
                compound.molecular_formula,
                compound.molecular_weight,
                compound.inchi_key,
            ),
        )
        return cur.fetchone()[0]  # type: ignore[index]


def _upsert_target(conn: psycopg.Connection, target: MolecularTarget) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO molecular_targets (chembl_target_id, gene_name, pref_name,
                                           target_type, biology)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (chembl_target_id) DO UPDATE SET
                gene_name  = EXCLUDED.gene_name,
                biology    = EXCLUDED.biology
            RETURNING id
            """,
            (
                target.chembl_target_id,
                target.gene_name,
                target.gene_name,
                "SINGLE PROTEIN",
                target.biology,
            ),
        )
        return cur.fetchone()[0]  # type: ignore[index]


def _link_compound_target(
    conn: psycopg.Connection,
    compound_db_id: int,
    target_db_id: int,
    action_type: str | None,
    activity_type: str | None,
    activity_value: float | None,
    assay_type: str | None,
    confidence_score: int | None,
    document_year: int | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO compound_targets (compound_id, target_id, action_type,
                activity_type, activity_value, assay_type, confidence_score, document_year)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (compound_id, target_id, activity_type) DO NOTHING
            """,
            (compound_db_id, target_db_id, action_type, activity_type,
             activity_value, assay_type, confidence_score, document_year),
        )


def _queue_pmids(conn: psycopg.Connection, pmids: list[str], compound_db_id: int) -> int:
    """Add PMIDs to ingestion_log and link them to the compound."""
    if not pmids:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO ingestion_log (pmid, status) VALUES (%s, 'queued') ON CONFLICT DO NOTHING",
            [(pmid,) for pmid in pmids],
        )
        cur.executemany(
            "INSERT INTO compound_papers (compound_id, pmid) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(compound_db_id, pmid) for pmid in pmids],
        )
    conn.commit()
    return len(pmids)


def run(
    targets: list[MolecularTarget],
    dry_run: bool = False,
    email: str = "curemom@example.com",
) -> None:
    conn = psycopg.connect(DB_DSN, autocommit=False)
    ncbi_http = httpx.Client(follow_redirects=True)
    total_compounds = 0
    total_pmids = 0

    with ChEMBLClient() as chembl:
        for target in targets:
            logger.info("Processing target: %s", target.gene_name)

            # Step 1: Resolve ChEMBL target ID
            chembl_targets = chembl.find_target_by_gene(target.gene_name)
            if not chembl_targets:
                logger.warning("No ChEMBL target found for gene: %s", target.gene_name)
                continue

            # Use first match (best match for human SINGLE PROTEIN)
            ct = chembl_targets[0]
            target.chembl_target_id = ct.chembl_id
            logger.info("  → ChEMBL target: %s (%s)", ct.chembl_id, ct.pref_name)

            if dry_run:
                continue

            target_db_id = _upsert_target(conn, target)
            conn.commit()

            # Step 2: Fetch all active compounds for this target
            activities = chembl.get_activities_for_target(ct.chembl_id)
            logger.info("  → %d activities found", len(activities))

            # Deduplicate compounds
            seen_chembl_ids: set[str] = set()
            for act in activities:
                cid = act.compound_chembl_id
                if not cid or cid in seen_chembl_ids:
                    continue
                seen_chembl_ids.add(cid)

                # Step 3: Fetch compound details
                compound = chembl.get_compound(cid)
                if compound is None:
                    continue

                compound_db_id = _upsert_compound(conn, compound)
                _link_compound_target(
                    conn, compound_db_id, target_db_id,
                    act.action_type, act.activity_type, act.activity_value,
                    act.assay_type, act.confidence_score, act.document_year,
                )
                conn.commit()
                total_compounds += 1

                # Step 4: Queue PubMed papers for this compound
                search_terms = get_pubmed_search_terms(compound)
                all_pmids: set[str] = set()
                for term in search_terms:
                    try:
                        pmids = _esearch_pmids(ncbi_http, term, email)
                        all_pmids.update(pmids)
                        time.sleep(0.34)  # ~3 req/s without API key
                    except Exception as e:
                        logger.warning("ESearch failed for '%s': %s", term, e)

                queued = _queue_pmids(conn, list(all_pmids), compound_db_id)
                total_pmids += queued
                if queued:
                    logger.info("    Compound %s (%s): %d PMIDs queued",
                                cid, compound.name or "unnamed", queued)

    ncbi_http.close()
    conn.close()
    logger.info("Done. Compounds stored: %d | PMIDs queued: %d", total_compounds, total_pmids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ChEMBL compound-target data")
    parser.add_argument("--genes", nargs="+", metavar="GENE",
                        help="Specific gene names (e.g. JAK1 MTOR BTK)")
    parser.add_argument("--biology", metavar="DOMAIN",
                        help="Filter targets by biology domain (e.g. sle muscle)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve ChEMBL IDs only, don't fetch compounds or queue papers")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", "curemom@example.com"))
    args = parser.parse_args()

    if args.genes:
        targets = [t for t in TARGETS if t.gene_name in args.genes]
    elif args.biology:
        targets = get_targets_by_biology(args.biology)
    else:
        targets = TARGETS

    logger.info("Processing %d targets: %s", len(targets), [t.gene_name for t in targets])
    run(targets, dry_run=args.dry_run, email=args.email)


if __name__ == "__main__":
    main()
