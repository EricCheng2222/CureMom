"""Verify that LLM-generated [N] citations point at real chunks.

Catches two failure modes common in cited-RAG:
  1. Hallucinated indices — the LLM emits [7] when only 5 chunks were given.
  2. Out-of-context claims — text immediately preceding a [N] doesn't actually
     appear (semantically) in chunk N. Detected with cheap word-overlap; an
     NLI-based check is a future upgrade.

Returns a list of warnings (empty = clean).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..search.hybrid_retriever import RetrievedChunk

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class CitationWarning:
    citation_index: int
    severity: str   # 'invalid' | 'weak'
    message: str


def _word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text or "")}


def verify_citations(
    response_text: str,
    chunks: list[RetrievedChunk],
    weak_overlap_threshold: float = 0.05,
) -> list[CitationWarning]:
    warnings: list[CitationWarning] = []
    n_chunks = len(chunks)

    # Build a context window around each [N] marker (the sentence containing it)
    sentences = re.split(r"(?<=[.!?])\s+", response_text)

    for sent in sentences:
        for m in CITATION_RE.finditer(sent):
            idx = int(m.group(1))
            # 1) Out-of-range index
            if idx < 1 or idx > n_chunks:
                warnings.append(CitationWarning(
                    citation_index=idx,
                    severity="invalid",
                    message=f"[{idx}] references a passage that wasn't provided "
                            f"(only {n_chunks} chunks were retrieved).",
                ))
                continue

            # 2) Weak support — claim words don't overlap with chunk text
            chunk = chunks[idx - 1]
            claim_words = _word_set(sent.replace(m.group(0), ""))
            chunk_words = _word_set(chunk.chunk_text)
            if not claim_words or not chunk_words:
                continue
            overlap = len(claim_words & chunk_words) / len(claim_words)
            if overlap < weak_overlap_threshold:
                warnings.append(CitationWarning(
                    citation_index=idx,
                    severity="weak",
                    message=f"[{idx}] cites chunk for claim with low lexical "
                            f"overlap ({overlap:.0%}); manually verify.",
                ))

    return warnings


def warnings_to_dicts(warnings: list[CitationWarning]) -> list[dict]:
    return [
        {"citation_index": w.citation_index, "severity": w.severity, "message": w.message}
        for w in warnings
    ]
