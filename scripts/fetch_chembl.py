#!/usr/bin/env python3
from __future__ import annotations

"""Fetch compound-target data from ChEMBL and queue PubMed papers for ingestion.

Workflow:
  1. Resolve molecular targets (hand-picked seed list OR full clinical druggable proteome)
  2. Fetch all compounds active against each target (above confidence threshold)
  3. Store compounds and compound-target relationships in the DB
  4. Queue PubMed papers for each compound via ESearch on compound names
  5. Ingestion pipeline (ingest.py) then fetches the full papers

No API key required — ChEMBL is fully open access.

Usage:
  # Full druggable proteome — all human single-protein targets with clinical compounds
  python scripts/fetch_chembl.py --all-targets

  # Same, but only approved drugs (phase 4)
  python scripts/fetch_chembl.py --all-targets --min-phase 4

  # Seed list only (30 hand-curated targets from targets.py)
  python scripts/fetch_chembl.py --seed-only

  # Specific genes from seed list
  python scripts/fetch_chembl.py --genes JAK1 JAK2 MTOR

  # Filter seed list by biology domain
  python scripts/fetch_chembl.py --biology sle

  # Dry-run: resolve target IDs only, no DB writes
  python scripts/fetch_chembl.py --all-targets --dry-run
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

from src.ingestion.chembl_client import ChEMBLClient, ChEMBLCompound, ChEMBLTarget, get_pubmed_search_terms
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


def _upsert_target_from_chembl(conn: psycopg.Connection, target: ChEMBLTarget) -> int:
    """Upsert a ChEMBLTarget record into molecular_targets."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO molecular_targets (chembl_target_id, gene_name, pref_name,
                                           target_type, biology)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (chembl_target_id) DO UPDATE SET
                gene_name  = EXCLUDED.gene_name,
                pref_name  = EXCLUDED.pref_name
            RETURNING id
            """,
            (
                target.chembl_id,
                target.gene_name or target.pref_name,
                target.pref_name,
                "SINGLE PROTEIN",
                [],   # biology field is documentation-only; not available for dynamic targets
            ),
        )
        return cur.fetchone()[0]  # type: ignore[index]


def _upsert_target_from_molecular(conn: psycopg.Connection, target: MolecularTarget) -> int:
    """Upsert a MolecularTarget (seed list) record into molecular_targets."""
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


def _is_target_processed(conn: psycopg.Connection, chembl_target_id: str) -> bool:
    """Return True if this target already has compound-target rows in the DB (resumability)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM molecular_targets mt
                JOIN compound_targets ct ON ct.target_id = mt.id
                WHERE mt.chembl_target_id = %s
                LIMIT 1
            )
            """,
            (chembl_target_id,),
        )
        return bool(cur.fetchone()[0])


def _process_target(
    chembl: ChEMBLClient,
    conn: psycopg.Connection,
    ncbi_http: httpx.Client,
    target_db_id: int,
    chembl_target_id: str,
    email: str,
) -> tuple[int, int]:
    """Fetch activities, store compounds, queue PubMed PMIDs for one target.

    Returns (compounds_stored, pmids_queued).
    """
    activities = chembl.get_activities_for_target(chembl_target_id)
    if not activities:
        return 0, 0

    total_compounds = 0
    total_pmids = 0
    seen_chembl_ids: set[str] = set()

    for act in activities:
        cid = act.compound_chembl_id
        if not cid or cid in seen_chembl_ids:
            continue
        seen_chembl_ids.add(cid)

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

    return total_compounds, total_pmids


def run_all_targets(
    min_phase: int = 1,
    dry_run: bool = False,
    email: str = "curemom@example.com",
) -> None:
    """Run the full ChEMBL druggable proteome pipeline.

    Discovers all human single-protein targets with ≥1 clinical compound,
    then processes each one (with resumability — skips already-processed targets).
    """
    conn = psycopg.connect(DB_DSN, autocommit=False)
    ncbi_http = httpx.Client(follow_redirects=True)
    total_compounds = 0
    total_pmids = 0

    with ChEMBLClient() as chembl:
        targets = chembl.get_all_clinical_targets(min_phase=min_phase)
        logger.info("Starting pipeline for %d clinical targets (min_phase=%d)",
                    len(targets), min_phase)

        if dry_run:
            logger.info("[dry-run] Would process %d targets:", len(targets))
            for t in targets[:20]:
                logger.info("  %s — %s (%s)", t.chembl_id, t.gene_name or "?", t.pref_name)
            if len(targets) > 20:
                logger.info("  ... and %d more", len(targets) - 20)
            ncbi_http.close()
            conn.close()
            return

        for i, target in enumerate(targets):
            chembl_id = target.chembl_id

            # Resumability: skip targets already fully processed
            if _is_target_processed(conn, chembl_id):
                logger.debug("Skipping already-processed target: %s (%s)",
                             chembl_id, target.gene_name or target.pref_name)
                continue

            logger.info("[%d/%d] Processing %s — %s",
                        i + 1, len(targets), chembl_id, target.gene_name or target.pref_name)

            target_db_id = _upsert_target_from_chembl(conn, target)
            conn.commit()

            n_compounds, n_pmids = _process_target(
                chembl, conn, ncbi_http, target_db_id, chembl_id, email
            )
            total_compounds += n_compounds
            total_pmids += n_pmids
            logger.info("  → %d compounds, %d PMIDs", n_compounds, n_pmids)

    ncbi_http.close()
    conn.close()
    logger.info("Done. Compounds stored: %d | PMIDs queued: %d", total_compounds, total_pmids)


def run_seed_targets(
    targets: list[MolecularTarget],
    dry_run: bool = False,
    email: str = "curemom@example.com",
) -> None:
    """Run the pipeline for the hand-curated seed target list (targets.py)."""
    conn = psycopg.connect(DB_DSN, autocommit=False)
    ncbi_http = httpx.Client(follow_redirects=True)
    total_compounds = 0
    total_pmids = 0

    with ChEMBLClient() as chembl:
        for target in targets:
            logger.info("Processing seed target: %s", target.gene_name)

            # Resolve ChEMBL target ID if not already set
            if not target.chembl_target_id:
                chembl_targets = chembl.find_target_by_gene(target.gene_name)
                if not chembl_targets:
                    logger.warning("No ChEMBL target found for gene: %s", target.gene_name)
                    continue
                ct = chembl_targets[0]
                target.chembl_target_id = ct.chembl_id
                logger.info("  → ChEMBL target: %s (%s)", ct.chembl_id, ct.pref_name)

            if dry_run:
                continue

            # Resumability
            if _is_target_processed(conn, target.chembl_target_id):
                logger.info("  Skipping already-processed: %s", target.chembl_target_id)
                continue

            target_db_id = _upsert_target_from_molecular(conn, target)
            conn.commit()

            n_compounds, n_pmids = _process_target(
                chembl, conn, ncbi_http, target_db_id, target.chembl_target_id, email
            )
            total_compounds += n_compounds
            total_pmids += n_pmids
            logger.info("  → %d compounds, %d PMIDs", n_compounds, n_pmids)

    ncbi_http.close()
    conn.close()
    logger.info("Done. Compounds stored: %d | PMIDs queued: %d", total_compounds, total_pmids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ChEMBL compound-target data")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-targets", action="store_true",
                      help="Full druggable proteome: all ChEMBL human single-protein targets "
                           "with ≥1 clinical compound (default when no other filter given)")
    mode.add_argument("--seed-only", action="store_true",
                      help="Use only the 30 hand-curated targets from targets.py")
    mode.add_argument("--genes", nargs="+", metavar="GENE",
                      help="Specific gene names from the seed list (e.g. JAK1 MTOR BTK)")
    mode.add_argument("--biology", metavar="DOMAIN",
                      help="Filter seed list by biology domain (e.g. sle muscle)")

    parser.add_argument("--min-phase", type=int, default=1, metavar="N",
                        help="Minimum clinical phase for --all-targets mode (default: 1; "
                             "use 4 for approved drugs only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve targets only — no DB writes, no PubMed queuing")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", "curemom@example.com"))
    args = parser.parse_args()

    # Default to --all-targets when no mode flag is given
    use_all = args.all_targets or (not args.seed_only and not args.genes and not args.biology)

    if use_all:
        logger.info("Mode: full druggable proteome (min_phase=%d)", args.min_phase)
        run_all_targets(min_phase=args.min_phase, dry_run=args.dry_run, email=args.email)
    else:
        if args.genes:
            seed_targets = [t for t in TARGETS if t.gene_name in args.genes]
        elif args.biology:
            seed_targets = get_targets_by_biology(args.biology)
        else:
            seed_targets = TARGETS
        logger.info("Mode: seed list (%d targets): %s",
                    len(seed_targets), [t.gene_name for t in seed_targets])
        run_seed_targets(seed_targets, dry_run=args.dry_run, email=args.email)


if __name__ == "__main__":
    main()
