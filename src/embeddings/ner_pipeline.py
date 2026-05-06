"""Biomedical NER over chunks — populates `paper_entities`.

Uses a HuggingFace transformers BERT-based biomedical NER model. We pivoted
from scispaCy because its current versions pin Python 3.10+, while the rest
of the project still runs on 3.9.

Default model: `d4data/biomedical-ner-all` — covers diseases, chemicals,
medications, anatomy, dosage, history, lab values, etc. (~110 MB).

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

# Map raw HF NER labels to our normalized entity types.
# d4data/biomedical-ner-all label set is broad — we collapse to the schema
# used in paper_entities.entity_type.
LABEL_MAP = {
    # d4data/biomedical-ner-all label set
    "DISEASE_DISORDER":      "DISEASE",
    "SIGN_SYMPTOM":          "SYMPTOM",
    "MEDICATION":            "CHEMICAL",
    "BIOLOGICAL_STRUCTURE":  "ANATOMY",          # tissues, organs, cell lines
    "DIAGNOSTIC_PROCEDURE":  "PROCEDURE",
    "THERAPEUTIC_PROCEDURE": "PROCEDURE",
    # Everything below = drop (too generic / not useful for graph)
    "DETAILED_DESCRIPTION":  None,
    "BIOLOGICAL_ATTRIBUTE":  None,
    "CLINICAL_EVENT":        None,
    "LAB_VALUE":             None,
    "FAMILY_HISTORY":        None,
    "HISTORY":               None,
    "AGE":                   None, "SEX": None,
    "DURATION":              None, "FREQUENCY": None,
    "DATE":                  None, "TIME": None,
    "AREA":                  None, "VOLUME": None, "MASS": None, "DISTANCE": None,
    "DOSAGE":                None,
    "OUTCOME":               None,
    "ADMINISTRATION":        None,
    "PERSONAL_BACKGROUND":   None,
    "ACTIVITY":              None, "SUBJECT": None, "SEVERITY": None, "COREFERENCE": None,
    # Fallbacks for older / alternate models
    "DISEASE":               "DISEASE",
    "CHEMICAL":              "CHEMICAL",
    "GENE":                  "GENE_OR_GENE_PRODUCT",
    "PROTEIN":               "GENE_OR_GENE_PRODUCT",
    "CELL":                  "CELL_TYPE",
    "ORGANISM":              "ORGANISM",
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

    def __init__(
        self,
        model_name: str = "d4data/biomedical-ner-all",
        device: str | None = None,
    ):
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        logger.info("Loading NER model %s on %s", model_name, device)

        tok   = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model.to(device).eval()

        # device argument: pipeline accepts -1 for CPU, integer index for CUDA, "mps" string
        device_arg: int | str
        if device == "cpu":
            device_arg = -1
        elif device == "mps":
            device_arg = device
        else:
            device_arg = 0  # cuda:0

        self._pipe = pipeline(
            task="ner",
            model=model,
            tokenizer=tok,
            # 'first' uses the label of the first sub-token in each word and
            # produces cleaner spans than 'simple'; 'max' picks the highest
            # score across sub-tokens.
            aggregation_strategy="first",
            device=device_arg,
        )

    def extract(self, text: str, paper_id: int, chunk_id: int) -> list[ExtractedEntity]:
        if not text:
            return []
        try:
            spans = self._pipe(text)
        except Exception as exc:
            logger.warning("NER pipeline failed on chunk %d (%s); skipping.", chunk_id, exc)
            return []

        results: list[ExtractedEntity] = []
        seen: set[tuple[str, str, int]] = set()

        for span in spans:
            raw_label = span.get("entity_group") or span.get("entity") or ""
            label = LABEL_MAP.get(raw_label.upper())
            if label is None:
                continue

            # Re-extract text from the original string using the offsets so we
            # avoid WordPiece-leftover '##' markers and get correct token spans.
            start = int(span.get("start", 0))
            end = int(span.get("end", start))
            if end <= start:
                continue
            txt = text[start:end].strip()

            # Filter junk:
            #   - sub-word fragments and very short strings
            #   - tokens that are pure stopwords or single non-letter chars
            if len(txt) < 3 or txt.startswith("##"):
                continue
            if not any(ch.isalpha() for ch in txt):
                continue
            # Drop spans with score below a confidence threshold
            score = float(span.get("score", 1.0))
            if score < 0.5:
                continue

            key = (txt.lower(), label, start)
            if key in seen:
                continue
            seen.add(key)
            results.append(ExtractedEntity(
                paper_id=paper_id, chunk_id=chunk_id,
                entity_text=txt, entity_type=label,
                start_char=start, end_char=end,
            ))
        return results


def extract_entities_for_chunks(
    conn: psycopg.Connection,
    paper_ids: list[int] | None = None,
    limit: int | None = None,
    model_name: str = "d4data/biomedical-ner-all",
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

    runner = _NERRunner(model_name=model_name)
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
                cur.executemany(
                    """
                    INSERT INTO paper_entities
                        (paper_id, chunk_id, entity_text, entity_type,
                         start_char, end_char, kb_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [(e.paper_id, e.chunk_id, e.entity_text, e.entity_type,
                      e.start_char, e.end_char, e.kb_id) for e in ents],
                )
            conn.commit()
            total += len(ents)

        if i % 100 == 0 or i == len(rows):
            logger.info("  processed %d / %d chunks (%d entities so far)", i, len(rows), total)

    return total
