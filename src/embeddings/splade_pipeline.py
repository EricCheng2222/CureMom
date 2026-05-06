"""SPLADE — learned sparse vectors stored as Elasticsearch sparse_vector fields.

SPLADE (Sparse Lexical AnD Expansion) produces sparse vectors over the BERT
vocabulary where each non-zero dimension represents a learned-important
token weight. Unlike BM25, it can match "antimalarial" ↔ "hydroxychloroquine"
because the encoder learns vocabulary expansion. Unlike dense embeddings,
it stays interpretable and works with Elasticsearch's standard inverted index
infrastructure.

Models worth trying for biomedicine:
  • naver/splade-v3                   — general SPLADE, public weights
  • naver/efficient-splade-VI-BT-large-doc / -query  — separate doc/query encoders
  • prithivida/Splade_PP_en_v1        — smaller, distilled variant

We default to `naver/splade-v3` which works for both queries and docs.
"""

from __future__ import annotations

import logging
from typing import Iterable

import psycopg
from psycopg.rows import dict_row

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)

SPLADE_MODEL = "naver/splade-v3"
SPLADE_INDEX_FIELD = "splade_vector"


class _SpladeEncoder:
    """Lazy encoder; defers torch/transformers imports."""

    def __init__(self, model_name: str = SPLADE_MODEL, device: str | None = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        logger.info("Loading SPLADE %s on %s", model_name, device)

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self._model.eval()
        self._device = device

    def encode(self, texts: list[str], batch_size: int = 8, max_length: int = 256) -> list[dict[str, float]]:
        """Returns one dict[token: weight] per input — non-zero entries only."""
        torch = self._torch
        tok = self._tok
        out_dicts: list[dict[str, float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self._device)
                logits = self._model(**enc).logits          # (B, T, V)
                # SPLADE pooling: log(1+ReLU(logits)) max-pooled over tokens
                weights = torch.log1p(torch.relu(logits))    # (B, T, V)
                attn = enc["attention_mask"].unsqueeze(-1)
                weights = weights * attn
                pooled = weights.max(dim=1).values           # (B, V)

                vocab = tok.get_vocab()
                inv_vocab = {idx: tok_str for tok_str, idx in vocab.items()}

                for vec in pooled.cpu():
                    nz = (vec > 0).nonzero(as_tuple=True)[0]
                    out_dicts.append({inv_vocab[int(idx)]: float(vec[idx]) for idx in nz})
        return out_dicts


def ensure_splade_field(es: Elasticsearch, index: str = "papers") -> None:
    """Add a sparse_vector field to the existing index mapping (idempotent)."""
    try:
        es.indices.put_mapping(
            index=index,
            properties={SPLADE_INDEX_FIELD: {"type": "sparse_vector"}},
        )
        logger.info("Mapping for %s.%s ensured.", index, SPLADE_INDEX_FIELD)
    except Exception as exc:
        logger.warning("Could not add SPLADE field mapping (%s).", exc)


def encode_and_index_chunks(
    conn: psycopg.Connection,
    es: Elasticsearch,
    paper_ids: list[int] | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    index: str = "papers",
) -> int:
    """Encode chunks with SPLADE and update each paper's ES doc with the merged
    sparse vector (max-pooled across the paper's chunks).

    Returns number of papers updated.
    """
    ensure_splade_field(es, index=index)
    encoder = _SpladeEncoder()

    where = "WHERE 1=1"
    params: list = []
    if paper_ids:
        where += " AND p.id = ANY(%s)"
        params.append(paper_ids)
    sql = (
        "SELECT p.id AS paper_id, p.pmid, "
        " ARRAY_AGG(pc.chunk_text ORDER BY pc.chunk_index) AS chunk_texts "
        f"FROM papers p JOIN paper_chunks pc ON pc.paper_id = p.id {where} "
        "GROUP BY p.id, p.pmid"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        papers = cur.fetchall()
    if not papers:
        logger.info("No papers to encode.")
        return 0

    logger.info("SPLADE-encoding %d papers…", len(papers))

    actions = []
    updated = 0
    for paper in papers:
        chunk_texts = paper["chunk_texts"] or []
        if not chunk_texts:
            continue
        vectors = encoder.encode(chunk_texts, batch_size=batch_size)
        # Merge per-chunk vectors with element-wise max
        merged: dict[str, float] = {}
        for vec in vectors:
            for tok, w in vec.items():
                if w > merged.get(tok, 0.0):
                    merged[tok] = w
        actions.append({
            "_op_type": "update",
            "_index": index,
            "_id": paper["pmid"],
            "doc": {SPLADE_INDEX_FIELD: merged},
            "doc_as_upsert": False,
        })
        updated += 1
        if len(actions) >= 50:
            bulk(es, actions, raise_on_error=False)
            actions.clear()
            logger.info("  ES updated: %d / %d", updated, len(papers))
    if actions:
        bulk(es, actions, raise_on_error=False)
    return updated


def search_splade(
    es: Elasticsearch,
    query: str,
    top_k: int = 20,
    index: str = "papers",
    encoder: _SpladeEncoder | None = None,
) -> list[dict]:
    """Run a sparse_vector query against the SPLADE field."""
    if encoder is None:
        encoder = _SpladeEncoder()
    qvec = encoder.encode([query], batch_size=1)[0]
    if not qvec:
        return []

    body = {
        "size": top_k,
        "query": {
            "sparse_vector": {
                "field": SPLADE_INDEX_FIELD,
                "query_vector": qvec,
            }
        },
        "_source": ["pmid", "title", "pub_year", "journal_title", "publication_types"],
    }
    res = es.search(index=index, body=body)
    return [
        {
            "pmid": h["_source"]["pmid"],
            "score": h["_score"],
            "title": h["_source"].get("title", ""),
            "pub_year": h["_source"].get("pub_year"),
            "journal_title": h["_source"].get("journal_title"),
            "publication_types": h["_source"].get("publication_types", []),
        }
        for h in res["hits"]["hits"]
    ]
