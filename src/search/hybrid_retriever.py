"""Hybrid retrieval: BM25 (Elasticsearch) + dense vector (pgvector) via RRF fusion.

Phase 1: BM25 only (dense_weight=0)
Phase 2: Hybrid BM25 + dense with Reciprocal Rank Fusion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg
from elasticsearch import Elasticsearch
from psycopg.rows import dict_row

from .elasticsearch_client import search_bm25
from .mesh_expander import MeSHExpander

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant


@dataclass
class RetrievedChunk:
    chunk_id: int
    paper_id: int
    pmid: str
    pmcid: str | None
    doi: str | None
    title: str
    authors_short: str      # "Smith J, et al."
    journal: str | None
    pub_year: int | None
    publication_types: list[str]
    section_type: str
    chunk_text: str
    paragraph_index: int | None
    start_char: int | None
    end_char: int | None
    relevance_score: float  # final RRF or BM25 score (normalized 0-1)
    bm25_rank: int | None
    dense_rank: int | None


def _rrf_score(ranks: list[int | None]) -> float:
    """Compute RRF score for a document given its ranks in multiple lists."""
    total = 0.0
    for rank in ranks:
        if rank is not None:
            total += 1.0 / (RRF_K + rank)
    return total


class HybridRetriever:
    def __init__(
        self,
        db_dsn: str,
        es: Elasticsearch,
        mesh_expander: MeSHExpander | None = None,
        hipporag=None,  # Optional HippoRAGRetriever for entity-graph re-ranking
    ) -> None:
        self._db_dsn = db_dsn
        self._es = es
        self._mesh_expander = mesh_expander
        self._hipporag = hipporag
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._db_dsn, row_factory=dict_row)
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def _expand_query(self, query: str, filters: dict[str, Any]) -> dict[str, Any]:
        """Optionally expand filters with MeSH descendants."""
        if not self._mesh_expander or not filters.get("use_mesh_expansion"):
            return filters

        expanded = dict(filters)
        if "mesh_terms" in filters:
            all_uis: list[str] = []
            for ui in filters["mesh_terms"]:
                all_uis.append(ui)
                descendants = self._mesh_expander.get_descendants(ui)
                all_uis.extend(d["descriptor_ui"] for d in descendants)
            expanded["mesh_terms"] = list(set(all_uis))

        return expanded

    def _fetch_chunks_by_pmids(
        self, pmids: list[str], top_k_per_paper: int = 3
    ) -> dict[str, list[dict]]:
        """Fetch the most relevant chunks for each PMID from PostgreSQL."""
        if not pmids:
            return {}

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    pc.id AS chunk_id,
                    pc.paper_id,
                    p.pmid,
                    p.pmcid,
                    p.doi,
                    p.title,
                    p.pub_year,
                    p.publication_types,
                    j.title AS journal_title,
                    pc.source_type,
                    pc.chunk_text,
                    pc.paragraph_index,
                    pc.start_char,
                    pc.end_char,
                    ARRAY_AGG(a.last_name || COALESCE(' ' || a.initials, '') ORDER BY pa.position) AS author_names
                FROM paper_chunks pc
                JOIN papers p ON pc.paper_id = p.id
                LEFT JOIN journals j ON p.journal_id = j.id
                LEFT JOIN paper_authors pa ON p.id = pa.paper_id
                LEFT JOIN authors a ON pa.author_id = a.id
                WHERE p.pmid = ANY(%s)
                GROUP BY pc.id, pc.paper_id, p.pmid, p.pmcid, p.doi, p.title,
                         p.pub_year, p.publication_types, j.title,
                         pc.source_type, pc.chunk_text, pc.paragraph_index,
                         pc.start_char, pc.end_char
                ORDER BY p.pmid, pc.chunk_index
                """,
                (pmids,),
            )
            rows = cur.fetchall()

        # Group by PMID, keep top_k_per_paper chunks per paper
        by_pmid: dict[str, list[dict]] = {}
        for row in rows:
            pmid = row["pmid"]
            if pmid not in by_pmid:
                by_pmid[pmid] = []
            if len(by_pmid[pmid]) < top_k_per_paper:
                by_pmid[pmid].append(dict(row))

        return by_pmid

    def _dense_search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Vector similarity search via pgvector. Returns [(pmid, score)]."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.pmid, 1 - (pc.embedding <=> %s::vector) AS cosine_similarity
                FROM paper_chunks pc
                JOIN papers p ON pc.paper_id = p.id
                WHERE pc.embedding IS NOT NULL
                ORDER BY pc.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            return [(row["pmid"], row["cosine_similarity"]) for row in cur.fetchall()]

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        dense_weight: float = 0.0,  # 0.0 = BM25 only; 0.5 = hybrid
        query_embedding: list[float] | None = None,
        use_hipporag: bool = False,
        hipporag_weight: float = 0.3,
    ) -> list[RetrievedChunk]:
        """Main retrieval method.

        Phase 1: dense_weight=0 (BM25 only, no embedding needed).
        Phase 2: dense_weight>0 with query_embedding provided.
        """
        effective_filters = self._expand_query(query, filters or {})
        # Strip internal flags before passing to ES
        es_filters = {k: v for k, v in effective_filters.items() if k != "use_mesh_expansion"}

        # BM25 retrieval
        bm25_results = search_bm25(self._es, query, es_filters, top_k=top_k * 2)
        bm25_ranks: dict[str, int] = {r["pmid"]: i + 1 for i, r in enumerate(bm25_results)}

        # Dense retrieval (Phase 2+)
        dense_ranks: dict[str, int] = {}
        if dense_weight > 0 and query_embedding:
            dense_results = self._dense_search(query_embedding, top_k=top_k * 2)
            dense_ranks = {pmid: i + 1 for i, (pmid, _) in enumerate(dense_results)}

        # Combine via RRF
        all_pmids = set(bm25_ranks) | set(dense_ranks)
        scored: list[tuple[str, float]] = []
        for pmid in all_pmids:
            score = _rrf_score([bm25_ranks.get(pmid), dense_ranks.get(pmid)])
            scored.append((pmid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Pull more candidates than top_k so HippoRAG re-ranking has room to work
        candidate_pool_size = top_k * 3 if (use_hipporag and self._hipporag) else top_k
        top_pmids = [pmid for pmid, _ in scored[:candidate_pool_size]]
        score_map = dict(scored)

        # Fetch chunks from DB
        chunks_by_pmid = self._fetch_chunks_by_pmids(top_pmids)

        # HippoRAG entity-graph re-ranking — no try/except: if the user
        # asked for hipporag/full and it errors, surface the error.
        if use_hipporag and self._hipporag is not None:
            from .hipporag import extract_query_entities
            query_entities = extract_query_entities(query)
            all_chunk_ids = [c["chunk_id"] for cs in chunks_by_pmid.values() for c in cs]
            hipporag_boosts = self._hipporag.rerank_chunks(all_chunk_ids, query_entities)
            pmid_boost: dict[str, float] = {}
            for pmid, cs in chunks_by_pmid.items():
                pmid_boost[pmid] = max(
                    (hipporag_boosts.get(c["chunk_id"], 0.0) for c in cs),
                    default=0.0,
                )
            combined = sorted(
                top_pmids,
                key=lambda p: score_map.get(p, 0.0) + hipporag_weight * pmid_boost.get(p, 0.0),
                reverse=True,
            )
            top_pmids = combined[:top_k]
        else:
            top_pmids = top_pmids[:top_k]

        results: list[RetrievedChunk] = []
        for pmid in top_pmids:
            chunks = chunks_by_pmid.get(pmid, [])
            if not chunks:
                continue

            for chunk_row in chunks:
                author_names: list[str] = chunk_row.get("author_names") or []
                if len(author_names) > 2:
                    authors_short = f"{author_names[0]}, et al."
                elif author_names:
                    authors_short = ", ".join(author_names)
                else:
                    authors_short = "Unknown"

                # Normalize score to [0, 1]
                raw_score = score_map.get(pmid, 0.0)
                max_possible = _rrf_score([1, 1])  # best possible RRF
                normalized = min(raw_score / max_possible, 1.0) if max_possible > 0 else 0.0

                results.append(RetrievedChunk(
                    chunk_id=chunk_row["chunk_id"],
                    paper_id=chunk_row["paper_id"],
                    pmid=pmid,
                    pmcid=chunk_row.get("pmcid"),
                    doi=chunk_row.get("doi"),
                    title=chunk_row["title"],
                    authors_short=authors_short,
                    journal=chunk_row.get("journal_title"),
                    pub_year=chunk_row.get("pub_year"),
                    publication_types=chunk_row.get("publication_types") or [],
                    section_type=chunk_row["source_type"],
                    chunk_text=chunk_row["chunk_text"],
                    paragraph_index=chunk_row.get("paragraph_index"),
                    start_char=chunk_row.get("start_char"),
                    end_char=chunk_row.get("end_char"),
                    relevance_score=round(normalized, 4),
                    bm25_rank=bm25_ranks.get(pmid),
                    dense_rank=dense_ranks.get(pmid),
                ))

        # Sort by relevance score descending
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results
