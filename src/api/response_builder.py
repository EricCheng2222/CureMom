"""Build structured query responses with full citation provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..search.hybrid_retriever import RetrievedChunk


@dataclass
class CitationChunk:
    section: str
    paragraph_index: int | None
    start_char: int | None
    end_char: int | None
    text: str


@dataclass
class Citation:
    citation_index: int         # 1-indexed, matches [N] in response text
    chunk_id: int
    pmid: str
    pmcid: str | None
    doi: str | None
    title: str
    authors: str
    journal: str | None
    year: int | None
    publication_types: list[str]
    pubmed_url: str
    pmc_url: str | None
    chunk: CitationChunk
    relevance_score: float


@dataclass
class QueryResponse:
    query: str
    response_text: str
    citations: list[Citation]
    retrieval_strategy: str
    model_used: str
    total_chunks_retrieved: int
    metadata: dict[str, Any] = field(default_factory=dict)


def build_citation(index: int, chunk: RetrievedChunk) -> Citation:
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{chunk.pmid}/"
    pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{chunk.pmcid}/" if chunk.pmcid else None

    return Citation(
        citation_index=index,
        chunk_id=chunk.chunk_id,
        pmid=chunk.pmid,
        pmcid=chunk.pmcid,
        doi=chunk.doi,
        title=chunk.title,
        authors=chunk.authors_short,
        journal=chunk.journal,
        year=chunk.pub_year,
        publication_types=chunk.publication_types,
        pubmed_url=pubmed_url,
        pmc_url=pmc_url,
        chunk=CitationChunk(
            section=chunk.section_type,
            paragraph_index=chunk.paragraph_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            text=chunk.chunk_text,
        ),
        relevance_score=chunk.relevance_score,
    )


def build_response(
    query: str,
    chunks: list[RetrievedChunk],
    response_text: str,
    cited_chunk_ids: list[int],
    model_used: str,
    retrieval_strategy: str,
) -> QueryResponse:
    """Assemble the full structured response with citations.

    Only chunks that were actually cited (by chunk_id) are included in citations.
    If cited_chunk_ids is empty (extractive mode with all chunks as sources),
    all chunks are included.
    """
    # Build index by chunk_id
    chunk_by_id = {c.chunk_id: c for c in chunks}

    citations: list[Citation] = []
    if cited_chunk_ids:
        seen: set[int] = set()
        idx = 1
        for chunk_id in cited_chunk_ids:
            if chunk_id in chunk_by_id and chunk_id not in seen:
                citations.append(build_citation(idx, chunk_by_id[chunk_id]))
                seen.add(chunk_id)
                idx += 1
    else:
        # Extractive mode: include all retrieved chunks as citations
        for idx, chunk in enumerate(chunks, start=1):
            citations.append(build_citation(idx, chunk))

    return QueryResponse(
        query=query,
        response_text=response_text,
        citations=citations,
        retrieval_strategy=retrieval_strategy,
        model_used=model_used,
        total_chunks_retrieved=len(chunks),
    )


def response_to_dict(r: QueryResponse) -> dict[str, Any]:
    """Serialize QueryResponse to a JSON-serializable dict."""
    return {
        "query": r.query,
        "response": r.response_text,
        "citations": [
            {
                "citation_index": c.citation_index,
                "chunk_id": c.chunk_id,
                "pmid": c.pmid,
                "pmcid": c.pmcid,
                "doi": c.doi,
                "title": c.title,
                "authors": c.authors,
                "journal": c.journal,
                "year": c.year,
                "publication_types": c.publication_types,
                "pubmed_url": c.pubmed_url,
                "pmc_url": c.pmc_url,
                "chunk": {
                    "section": c.chunk.section,
                    "paragraph_index": c.chunk.paragraph_index,
                    "start_char": c.chunk.start_char,
                    "end_char": c.chunk.end_char,
                    "text": c.chunk.text,
                },
                "relevance_score": c.relevance_score,
            }
            for c in r.citations
        ],
        "metadata": {
            "retrieval_strategy": r.retrieval_strategy,
            "model_used": r.model_used,
            "total_chunks_retrieved": r.total_chunks_retrieved,
            **r.metadata,
        },
    }
