"""Section-aware chunking + PubMedBERT embedding generation.

This module owns two pipelines:

  1. Chunking — splits a paper's text into context-coherent chunks
     (abstract = 1 chunk; intro/methods/discussion = 512-token sliding window
     with 128-token overlap; results = one chunk per paragraph).

  2. Embedding — runs PubMedBERT inference over chunks lacking an embedding,
     storing 768-dim vectors in `paper_chunks.embedding`.

The two pipelines are decoupled so chunks can be created at ingestion time
(cheap) and embeddings added in a separate batch (GPU/CPU intensive).

Typical usage:
    python scripts/embed.py                  # embeds all chunks missing embeddings
    python scripts/embed.py --paper-ids 1 2  # embeds chunks for specific papers
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
EMBEDDING_DIM = 768
TARGET_TOKENS = 512
OVERLAP_TOKENS = 128


# ─── Chunking ──────────────────────────────────────────────────────────────

@dataclass
class ProposedChunk:
    """A chunk to be inserted (no embedding yet)."""
    paper_id: int
    section_id: int | None
    chunk_index: int
    chunk_text: str
    source_type: str          # 'abstract', 'introduction', 'methods', 'results', 'discussion'
    start_char: int
    end_char: int
    paragraph_index: int | None
    token_count: int


def _approx_token_count(text: str) -> int:
    """Cheap proxy: ~4 chars per token. Avoids loading a tokenizer just for stats."""
    return max(1, len(text) // 4)


def _sliding_window(text: str, target_tokens: int, overlap_tokens: int) -> list[tuple[str, int, int]]:
    """Split `text` into overlapping windows. Returns [(chunk_text, start_char, end_char)]."""
    if not text:
        return []

    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4
    step = max(target_chars - overlap_chars, 1)

    windows: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + target_chars, n)
        # Snap to nearest sentence/whitespace boundary going backward, but don't go too far
        if end < n:
            for boundary in (". ", "? ", "! ", "\n"):
                idx = text.rfind(boundary, start + step // 2, end)
                if idx > start:
                    end = idx + len(boundary)
                    break
        chunk = text[start:end].strip()
        if chunk:
            windows.append((chunk, start, end))
        if end >= n:
            break
        start = end - overlap_chars
    return windows


def chunk_section(
    paper_id: int,
    section_id: int | None,
    section_type: str,
    section_text: str,
    starting_chunk_index: int = 0,
) -> list[ProposedChunk]:
    """Apply section-type-specific chunking rules to a single section."""
    text = (section_text or "").strip()
    if not text:
        return []

    section_type_lower = section_type.lower()
    chunks: list[ProposedChunk] = []
    idx = starting_chunk_index

    if section_type_lower == "abstract":
        # Whole abstract = 1 chunk
        chunks.append(ProposedChunk(
            paper_id=paper_id, section_id=section_id, chunk_index=idx,
            chunk_text=text, source_type="abstract",
            start_char=0, end_char=len(text), paragraph_index=0,
            token_count=_approx_token_count(text),
        ))
        return chunks

    if section_type_lower == "results":
        # One chunk per paragraph (results sections are dense numerical content)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        offset = 0
        for p_idx, para in enumerate(paragraphs):
            start = text.find(para, offset)
            end = start + len(para)
            offset = end
            if len(para) < 30:  # skip stub fragments
                continue
            chunks.append(ProposedChunk(
                paper_id=paper_id, section_id=section_id, chunk_index=idx,
                chunk_text=para, source_type="results",
                start_char=start, end_char=end, paragraph_index=p_idx,
                token_count=_approx_token_count(para),
            ))
            idx += 1
        return chunks

    # Default for introduction/methods/discussion: sliding window
    windows = _sliding_window(text, TARGET_TOKENS, OVERLAP_TOKENS)
    for window_text, start, end in windows:
        chunks.append(ProposedChunk(
            paper_id=paper_id, section_id=section_id, chunk_index=idx,
            chunk_text=window_text, source_type=section_type_lower,
            start_char=start, end_char=end, paragraph_index=None,
            token_count=_approx_token_count(window_text),
        ))
        idx += 1
    return chunks


def chunk_paper(conn: psycopg.Connection, paper_id: int) -> int:
    """Generate and insert chunks for a single paper from its sections.

    Skips abstract chunks if they already exist (created during ingestion).
    Returns the number of new chunks inserted.
    """
    inserted = 0
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, section_type, section_order, content FROM paper_sections "
            "WHERE paper_id = %s ORDER BY section_order",
            (paper_id,),
        )
        sections = cur.fetchall()

        # Determine starting chunk_index — continue past existing chunks
        cur.execute("SELECT COALESCE(MAX(chunk_index), -1) AS max_idx FROM paper_chunks WHERE paper_id = %s", (paper_id,))
        starting_idx = (cur.fetchone()["max_idx"] or -1) + 1

        for section in sections:
            stype = (section["section_type"] or "").lower()
            if stype == "abstract":
                # Already inserted at ingestion time
                continue
            new_chunks = chunk_section(
                paper_id=paper_id,
                section_id=section["id"],
                section_type=stype,
                section_text=section["content"] or "",
                starting_chunk_index=starting_idx,
            )
            for c in new_chunks:
                cur.execute(
                    """
                    INSERT INTO paper_chunks
                        (paper_id, section_id, chunk_index, chunk_text, source_type,
                         start_char, end_char, paragraph_index, token_count)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                    """,
                    (c.paper_id, c.section_id, c.chunk_index, c.chunk_text, c.source_type,
                     c.start_char, c.end_char, c.paragraph_index, c.token_count),
                )
                inserted += 1
            starting_idx += len(new_chunks)
    return inserted


# ─── Embedding ─────────────────────────────────────────────────────────────

class _Embedder:
    """Lazy wrapper around the PubMedBERT model so we only import torch on demand."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        logger.info("Loading %s on %s", model_name, device)

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()
        self._device = device

    def encode(self, texts: list[str], batch_size: int = 16, max_length: int = 512) -> list[list[float]]:
        """Mean-pool the last hidden state to get a 768-dim vector per text."""
        torch = self._torch
        all_vecs: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self._tok(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(self._device)
                out = self._model(**enc)
                hidden = out.last_hidden_state                  # (B, T, 768)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-6)
                pooled = summed / counts                        # mean pooling
                # Normalize for cosine similarity
                pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
                all_vecs.extend(pooled.cpu().tolist())
        return all_vecs


@contextmanager
def get_embedder(model_name: str = EMBEDDING_MODEL, device: str | None = None) -> Iterator[_Embedder]:
    """Context manager so callers can release GPU memory deterministically."""
    emb = _Embedder(model_name=model_name, device=device)
    try:
        yield emb
    finally:
        # Best-effort GPU cleanup
        del emb._model
        try:
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            pass


def embed_pending_chunks(
    conn: psycopg.Connection,
    batch_size: int = 32,
    paper_ids: list[int] | None = None,
    limit: int | None = None,
) -> int:
    """Embed all chunks where embedding IS NULL. Returns count embedded."""
    where = "WHERE embedding IS NULL"
    params: list = []
    if paper_ids:
        where += " AND paper_id = ANY(%s)"
        params.append(paper_ids)
    sql = f"SELECT id, chunk_text FROM paper_chunks {where} ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"

    with get_embedder() as emb:
        total = 0
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        if not rows:
            logger.info("No chunks pending embedding.")
            return 0
        logger.info("Embedding %d chunks (batch_size=%d)…", len(rows), batch_size)

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [r["chunk_text"] for r in batch]
            vecs = emb.encode(texts, batch_size=batch_size)

            with conn.cursor() as cur:
                for row, vec in zip(batch, vecs):
                    cur.execute(
                        "UPDATE paper_chunks SET embedding = %s::vector WHERE id = %s",
                        (vec, row["id"]),
                    )
            conn.commit()
            total += len(batch)
            if total % (batch_size * 10) == 0 or total == len(rows):
                logger.info("  embedded %d / %d", total, len(rows))
    return total


def ensure_pgvector_extension(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()


def ensure_hnsw_index(conn: psycopg.Connection) -> None:
    """Create HNSW index after the bulk load. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON paper_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    conn.commit()


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> list[float]:
    """One-off helper to embed a single query string at request time."""
    with get_embedder(model_name=model_name) as emb:
        return emb.encode([query])[0]
