"""ChEMBL REST API client — no API key required.

Queries compounds by molecular target (gene name), not by assumed disease
relevance. The system discovers compound→target→pathway connections from data.

ChEMBL API docs: https://www.ebi.ac.uk/chembl/api/data/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
MIN_CONFIDENCE = 6      # ChEMBL confidence score 0-9; 6+ = "expert curated direct assay"
MAX_ACTIVITY_VALUE = 10_000  # nM — ignore very weak binders (>10 µM)
REQUEST_INTERVAL = 0.2  # seconds between requests (polite crawling)


@dataclass
class ChEMBLTarget:
    chembl_id: str
    pref_name: str
    target_type: str    # SINGLE PROTEIN, PROTEIN COMPLEX, etc.
    gene_name: str | None
    organism: str | None


@dataclass
class ChEMBLCompound:
    chembl_id: str
    name: str | None            # preferred name
    synonyms: list[str]         # trade names, INN, etc.
    max_phase: int | None       # 0-4 (4 = approved drug)
    molecule_type: str | None   # Small molecule, Protein, Antibody, etc.
    molecular_formula: str | None
    molecular_weight: float | None
    inchi_key: str | None


@dataclass
class TargetActivity:
    compound_chembl_id: str
    target_chembl_id: str
    target_gene: str | None
    action_type: str | None     # INHIBITOR, AGONIST, ANTAGONIST, etc.
    activity_type: str | None   # IC50, Ki, Kd, EC50, etc.
    activity_value: float | None  # in nM
    assay_type: str | None      # B=binding, F=functional, A=ADME
    confidence_score: int | None
    reference_pmid: str | None
    document_year: int | None


class ChEMBLClient:
    def __init__(self) -> None:
        self._client = httpx.Client(
            base_url=CHEMBL_BASE,
            headers={"Accept": "application/json"},
            timeout=30,
            follow_redirects=True,
        )
        self._last_request: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        self._throttle()
        r = self._client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    def _paginate(self, path: str, params: dict[str, Any]) -> Generator[dict, None, None]:
        """Yield all result items across paginated ChEMBL responses."""
        params = {**params, "limit": PAGE_SIZE, "offset": 0, "format": "json"}
        while True:
            data = self._get(path, params)
            items = data.get("targets") or data.get("activities") or data.get("molecules") or data.get("mechanisms") or []
            yield from items
            page_meta = data.get("page_meta", {})
            if not page_meta.get("next"):
                break
            params["offset"] += PAGE_SIZE

    def find_target_by_gene(self, gene_name: str) -> list[ChEMBLTarget]:
        """Look up ChEMBL targets matching a gene name (human proteins only)."""
        data = self._get("/target", params={
            "target_synonym__icontains": gene_name,
            "organism": "Homo sapiens",
            "target_type": "SINGLE PROTEIN",
            "format": "json",
            "limit": 10,
        })
        targets: list[ChEMBLTarget] = []
        for t in data.get("targets", []):
            # Filter: must have the gene name in synonyms/components
            gene = None
            for comp in t.get("target_components", []):
                for syn in comp.get("target_component_synonyms", []):
                    if syn.get("syn_type") == "GENE_SYMBOL" and syn.get("component_synonym", "").upper() == gene_name.upper():
                        gene = syn["component_synonym"]
            targets.append(ChEMBLTarget(
                chembl_id=t["target_chembl_id"],
                pref_name=t.get("pref_name", ""),
                target_type=t.get("target_type", ""),
                gene_name=gene,
                organism=t.get("organism"),
            ))
        return targets

    def get_activities_for_target(
        self,
        target_chembl_id: str,
        min_confidence: int = MIN_CONFIDENCE,
        max_activity_nm: float = MAX_ACTIVITY_VALUE,
    ) -> list[TargetActivity]:
        """Return all compound-target activities above confidence threshold."""
        activities: list[TargetActivity] = []
        for item in self._paginate("/activity", {
            "target_chembl_id": target_chembl_id,
            "assay_type__in": "B,F",   # binding or functional assays
            "standard_units": "nM",
            "confidence_score__gte": min_confidence,
        }):
            value = item.get("standard_value")
            try:
                value_f = float(value) if value is not None else None
            except (TypeError, ValueError):
                value_f = None

            if value_f is not None and value_f > max_activity_nm:
                continue  # too weak

            activities.append(TargetActivity(
                compound_chembl_id=item.get("molecule_chembl_id", ""),
                target_chembl_id=target_chembl_id,
                target_gene=item.get("target_pref_name"),
                action_type=None,  # filled from mechanism endpoint
                activity_type=item.get("standard_type"),
                activity_value=value_f,
                assay_type=item.get("assay_type"),
                confidence_score=item.get("confidence_score"),
                reference_pmid=item.get("document_chembl_id"),  # note: ChEMBL doc ID, not PMID
                document_year=item.get("document_year"),
            ))
        return activities

    def get_compound(self, chembl_id: str) -> ChEMBLCompound | None:
        """Fetch full compound record."""
        try:
            data = self._get(f"/molecule/{chembl_id}", params={"format": "json"})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

        props = data.get("molecule_properties") or {}
        synonyms = [
            s["molecule_synonym"]
            for s in data.get("molecule_synonyms", [])
            if s.get("molecule_synonym")
        ]

        return ChEMBLCompound(
            chembl_id=chembl_id,
            name=data.get("pref_name"),
            synonyms=synonyms,
            max_phase=data.get("max_phase"),
            molecule_type=data.get("molecule_type"),
            molecular_formula=props.get("full_molformula"),
            molecular_weight=float(props["full_mwt"]) if props.get("full_mwt") else None,
            inchi_key=data.get("molecule_structures", {}).get("standard_inchi_key"),
        )

    def get_mechanisms_for_target(self, target_chembl_id: str) -> list[dict]:
        """Get mechanism-of-action records for compounds against a target."""
        data = self._get("/mechanism", params={
            "target_chembl_id": target_chembl_id,
            "format": "json",
            "limit": PAGE_SIZE,
        })
        return data.get("mechanisms", [])

    def get_all_clinical_targets(self, min_phase: int = 1) -> list[ChEMBLTarget]:
        """Return all human single-protein targets with ≥1 compound at clinical phase ≥ min_phase.

        Scans all ChEMBL mechanism-of-action records to identify targets in the
        clinically-validated druggable space (~3,000 targets for min_phase=1).
        No hand-picking — covers the complete druggable proteome.

        Steps:
          1. Paginate /mechanism → collect unique target ChEMBL IDs and their linked molecules
          2. If min_phase > 1, batch-query molecule max_phase and filter accordingly
          3. Batch-fetch /target details, keeping only Homo sapiens SINGLE PROTEIN targets
        """
        logger.info("Scanning ChEMBL mechanism records for all clinical targets...")

        # Step 1: Collect unique target IDs and their linked molecule IDs from all mechanism records
        target_ids: set[str] = set()
        mol_ids_by_target: dict[str, set[str]] = {}
        record_count = 0

        for item in self._paginate("/mechanism", {}):
            tid = item.get("target_chembl_id")
            mid = item.get("molecule_chembl_id")
            if tid and mid:
                target_ids.add(tid)
                mol_ids_by_target.setdefault(tid, set()).add(mid)
            record_count += 1
            if record_count % 10_000 == 0:
                logger.info("  ... scanned %d mechanism records, %d unique targets so far",
                            record_count, len(target_ids))

        logger.info("Scanned %d mechanism records → %d unique target IDs",
                    record_count, len(target_ids))

        # Step 2: Filter targets by molecule clinical phase if min_phase > 1
        if min_phase > 1:
            all_mol_ids = sorted({mid for mids in mol_ids_by_target.values() for mid in mids})
            clinical_mol_ids: set[str] = set()
            batch_size = 100
            logger.info("Filtering %d molecules by max_phase >= %d...", len(all_mol_ids), min_phase)

            for i in range(0, len(all_mol_ids), batch_size):
                batch = all_mol_ids[i:i + batch_size]
                data = self._get("/molecule", params={
                    "molecule_chembl_id__in": ",".join(batch),
                    "max_phase__gte": str(min_phase),
                    "format": "json",
                    "limit": batch_size,
                })
                for mol in data.get("molecules", []):
                    mid = mol.get("molecule_chembl_id")
                    if mid:
                        clinical_mol_ids.add(mid)

            target_ids = {
                tid for tid, mids in mol_ids_by_target.items()
                if mids & clinical_mol_ids
            }
            logger.info("After phase >= %d filter: %d targets remain", min_phase, len(target_ids))

        # Step 3: Batch-fetch target details, filter for Homo sapiens SINGLE PROTEIN
        targets: list[ChEMBLTarget] = []
        target_id_list = sorted(target_ids)
        batch_size = 50

        logger.info("Fetching details for %d candidate targets...", len(target_id_list))
        for i in range(0, len(target_id_list), batch_size):
            batch = target_id_list[i:i + batch_size]
            data = self._get("/target", params={
                "target_chembl_id__in": ",".join(batch),
                "organism": "Homo sapiens",
                "target_type": "SINGLE PROTEIN",
                "format": "json",
                "limit": batch_size,
            })
            for t in data.get("targets", []):
                gene = None
                for comp in t.get("target_components", []):
                    for syn in comp.get("target_component_synonyms", []):
                        if syn.get("syn_type") == "GENE_SYMBOL":
                            gene = syn.get("component_synonym")
                            break
                    if gene:
                        break
                targets.append(ChEMBLTarget(
                    chembl_id=t["target_chembl_id"],
                    pref_name=t.get("pref_name", ""),
                    target_type=t.get("target_type", ""),
                    gene_name=gene,
                    organism=t.get("organism"),
                ))

            if i > 0 and (i // batch_size) % 20 == 0:
                logger.info("  ... fetched %d/%d", i, len(target_id_list))

        logger.info("Total human single-protein clinical targets: %d", len(targets))
        return targets

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ChEMBLClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def get_pubmed_search_terms(compound: ChEMBLCompound) -> list[str]:
    """Return a list of search terms for PubMed from a compound record.

    Prioritises INN/preferred name, then short synonyms.
    Avoids very generic or long strings that would match too broadly.
    """
    terms: list[str] = []
    if compound.name:
        terms.append(compound.name)
    for syn in compound.synonyms:
        # skip very long strings (likely IUPAC names) and duplicates
        if len(syn) <= 60 and syn.lower() not in {t.lower() for t in terms}:
            terms.append(syn)
    return terms[:5]  # top 5 names max
