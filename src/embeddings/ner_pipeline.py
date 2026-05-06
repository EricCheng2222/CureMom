"""Biomedical NER over chunks — populates `paper_entities`.

Uses scispaCy:
  • `en_ner_bc5cdr_md` — DISEASE / CHEMICAL
  • `en_core_sci_lg`   — general biomedical entities (genes, cell types, etc.)

scispaCy is heavy (3–5 GB of model files) and adds noise to a fresh venv;
keep it as an opt-in dep installed only on machines that run NER.

Typical usage:
    PYTHONPATH=. python scripts/extract_entities.py
    PYTHONPATH=. python scripts/extract_entities.py --paper-ids 1 2 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# Mapping scispaCy labels → our normalized types in paper_entities.entity_type
LABEL_MAP = {
    "DISEASE":              "DISEASE",
    "CHEMICAL":             "CHEMICAL",
    "GENE_OR_GENE_PRODUCT": "GENE_OR_GENE_PRODUCT",
    "CELL_TYPE":            "CELL_TYPE",
    "ORGANISM":             "ORGANISM",
    "PROTEIN":              "GENE_OR_GENE_PRODUCT",
    "GENE":                 "GENE_OR_GENE_PRODUCT",
    "CELL":                 "CELL_TYPE",
}


@dataclass
class ExtractedEntity:
    paper_id: int
    chunk_id: int
    entity_text: str
    entity_type: str
    start_char: int
    end_char: int
    kb_id: str | None = None


class _NERRunner:
    """Lazy wrapper — defers heavy imports until needed."""

    def __init__(self, with_linker: bool = True):
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "scispaCy not installed. Run:\n"
                "  pip install scispacy\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            ) from exc

        logger.info("Loading scispaCy models…")
        self._nlp_general = spacy.load("en_core_sci_lg")
        self._nlp_bc5cdr  = spacy.load("en_ner_bc5cdr_md")

        if with_linker:
            try:
                from scispacy.linking import EntityLinker  # noqa: F401
                self._nlp_general.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
                logger.info("UMLS linker attached.")
            except Exception as exc:
                logger.warning("scispaCy linker unavailable (%s) — entities will lack kb_id.", exc)

    def extract(self, text: str, paper_id: int, chunk_id: int) -> list[ExtractedEntity]:
        results: list[ExtractedEntity] = []
        seen: set[tuple[str, str, int]] = set()

        # Pass 1 — disease/chemical via BC5CDR
        for ent in self._nlp_bc5cdr(text).ents:
            label = LABEL_MAP.get(ent.label_)
            if not label:
                continue
            key = (ent.text.lower(), label, ent.start_char)
            if key in seen:
                continue
            seen.add(key)
            results.append(ExtractedEntity(
                paper_id=paper_id, chunk_id=chunk_id,
                entity_text=ent.text, entity_type=label,
                start_char=ent.start_char, end_char=ent.end_char,
            ))

        # Pass 2 — general biomedical (genes, cells, proteins)
        doc = self._nlp_general(text)
        for ent in doc.ents:
            label = LABEL_MAP.get(ent.label_, "OTHER")
            if label == "OTHER":
                continue
            key = (ent.text.lower(), label, ent.start_char)
            if key in seen:
                continue
            seen.add(key)

            kb_id: str | None = None
            if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                # Highest-confidence UMLS CUI
                kb_id = ent._.kb_ents[0][0]

            results.append(ExtractedEntity(
                paper_id=paper_id, chunk_id=chunk_id,
                entity_text=ent.text, entity_type=label,
                start_char=ent.start_char, end_char=ent.end_char,
                kb_id=kb_id,
            ))

        return results


def extract_entities_for_chunks(
    conn: psycopg.Connection,
    paper_ids: list[int] | None = None,
    limit: int | None = None,
    with_linker: bool = False,
) -> int:
    """Extract entities for chunks lacking them. Returns total entities inserted."""
    where = (
        "WHERE NOT EXISTS (SELECT 1 FROM paper_entities pe WHERE pe.chunk_id = pc.id)"
    )
    params: list = []
    if paper_ids:
        where += " AND pc.paper_id = ANY(%s)"
        params.append(paper_ids)
    sql = (
        "SELECT pc.id AS chunk_id, pc.paper_id, pc.chunk_text "
        f"FROM paper_chunks pc {where} ORDER BY pc.id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"

    runner = _NERRunner(with_linker=with_linker)
    total = 0

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        logger.info("No chunks pending NER.")
        return 0
    logger.info("Running NER over %d chunks…", len(rows))

    for i, row in enumerate(rows, start=1):
        ents = runner.extract(row["chunk_text"] or "", row["paper_id"], row["chunk_id"])
        if ents:
            with conn.cursor() as cur:
                for e in ents:
                    cur.execute(
                        """
                        INSERT INTO paper_entities
                            (paper_id, chunk_id, entity_text, entity_type,
                             start_char, end_char, kb_id)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (e.paper_id, e.chunk_id, e.entity_text, e.entity_type,
                         e.start_char, e.end_char, e.kb_id),
                    )
            conn.commit()
            total += len(ents)

        if i % 100 == 0:
            logger.info("  processed %d / %d chunks (%d entities so far)", i, len(rows), total)

    return total
