"""CureMom FastAPI application."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

# Load .env BEFORE anything reads os.environ — providers, DSN, etc.
# override=True so editing .env and triggering uvicorn --reload picks up
# the new values (default override=False keeps stale env vars).
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import psycopg
from elasticsearch import Elasticsearch
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from ..search.elasticsearch_client import get_client as get_es_client, ensure_index
from ..search.hybrid_retriever import HybridRetriever
from ..search.mesh_expander import MeSHExpander
from .classifier import classify_query
from .citation_verifier import verify_citations, warnings_to_dicts
from .drug_lookup import lookup_drugs_for_query
from .graph_extractor import extract_graph
from .llm_providers import get_provider
from .response_builder import build_response, response_to_dict

logger = logging.getLogger(__name__)

DB_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'curemom')}"
)
ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")

# ─── Application state ───────────────────────────────────────────────────────

_es: Elasticsearch | None = None
_retriever: HybridRetriever | None = None
_mesh_expander: MeSHExpander | None = None
_embedder: Any = None      # PubMedBERT — loaded lazily on first hybrid query
_hipporag: Any = None      # HippoRAG retriever — loaded lazily on first request that needs it


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _es, _retriever, _mesh_expander
    _es = get_es_client(ES_HOST)
    ensure_index(_es)
    _mesh_expander = MeSHExpander(DB_DSN)
    _retriever = HybridRetriever(DB_DSN, _es, _mesh_expander)
    logger.info("CureMom started. ES: %s | DB: %s", ES_HOST, DB_DSN.split("@")[-1])
    yield
    if _retriever:
        _retriever.close()
    if _mesh_expander:
        _mesh_expander.close()
    if _es:
        _es.close()


app = FastAPI(
    title="CureMom Medical Literature API",
    description="Query thousands of peer-reviewed medical papers with citation-grounded responses.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# "/" and static files are served via app.mount at module bottom.


# ─── Dependency helpers ───────────────────────────────────────────────────────

def get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return _retriever


def get_db() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


# ─── Request/Response models ──────────────────────────────────────────────────

class QueryFilters(BaseModel):
    pub_year_from: int | None = None
    pub_year_to: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    language: str | None = "eng"
    use_mesh_expansion: bool = True


class QueryOptions(BaseModel):
    top_k: int = Field(default=10, ge=1, le=50)
    retrieval_strategy: str = Field(default="full", pattern="^(bm25|hybrid|hipporag|full)$")
    include_full_passages: bool = True
    llm_provider: str | None = None   # override LLM_PROVIDER env var
    plain_language: bool = False      # patient-mode tone + follow-up questions


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=8000)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    query_type: str | None = Field(default="factual", pattern="^(factual|exploratory|comparative)$")
    filters: QueryFilters = Field(default_factory=QueryFilters)
    options: QueryOptions = Field(default_factory=QueryOptions)
    # Prior turns of this conversation (oldest first, most recent last).
    # The current `query` is NOT included — only past turns. Each assistant
    # entry should be just the answer text (no [N] markers, no source list);
    # the frontend strips those before sending.
    history: list[ChatMessage] = Field(default_factory=list)


class GraphChunkRef(BaseModel):
    id: int
    text: str = Field(..., max_length=20000)


class GraphExtractRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=20000)
    chunks: list[GraphChunkRef] = Field(default_factory=list, max_length=20)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/v1/query")
async def query(
    req: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> dict[str, Any]:
    """Main Q&A endpoint: retrieve relevant passages and return a cited response."""
    start = time.monotonic()

    filter_dict = req.filters.model_dump(exclude_none=False)
    strategy = req.options.retrieval_strategy
    dense_weight = 0.5 if strategy in ("hybrid", "full") else 0.0
    use_hipporag = strategy in ("hipporag", "full")
    query_embedding: list[float] | None = None

    # Multi-turn awareness: fuse prior user turns into the current query for
    # retrieval and drug lookup, so pronouns ("its side effects") and
    # follow-ups resolve to entities mentioned earlier. The LLM still gets
    # the full turn-by-turn history separately for generation.
    effective_query = req.query
    if req.history:
        prior_user_terms = " ".join(
            m.content for m in req.history[-6:] if m.role == "user"
        )
        if prior_user_terms:
            effective_query = f"{prior_user_terms} {req.query}"

    if dense_weight > 0:
        global _embedder
        if _embedder is None:
            from ..embeddings.chunk_pipeline import _Embedder
            logger.info("Loading PubMedBERT for hybrid retrieval (one-time)…")
            try:
                _embedder = _Embedder()
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Strategy '{strategy}' requires PubMedBERT embeddings, "
                        f"but the model failed to load: {type(exc).__name__}: {exc}. "
                        "Use retrieval_strategy='bm25' or 'hipporag', or install "
                        "the embedding model: `pip install transformers torch`."
                    ),
                )
        query_embedding = _embedder.encode([effective_query])[0]

    if use_hipporag:
        global _hipporag
        if _hipporag is None:
            from ..search.hipporag import HippoRAGRetriever
            logger.info("Loading HippoRAG entity graph (one-time)…")
            try:
                _hipporag = HippoRAGRetriever(DB_DSN)
                _hipporag._ensure_loaded()
                if _hipporag._graph is None or _hipporag._graph.number_of_edges() == 0:
                    raise RuntimeError(
                        "entity_graph table is empty — run "
                        "`python scripts/build_entity_graph.py` first."
                    )
            except Exception as exc:
                _hipporag = None  # don't cache the failure
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Strategy '{strategy}' requires the HippoRAG entity graph, "
                        f"which failed to load: {type(exc).__name__}: {exc}"
                    ),
                )
        retriever._hipporag = _hipporag

    try:
        chunks = retriever.retrieve(
            query=effective_query,
            filters=filter_dict,
            top_k=req.options.top_k,
            dense_weight=dense_weight,
            query_embedding=query_embedding,
            use_hipporag=use_hipporag,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=502,
            detail=f"Retrieval failed: {type(exc).__name__}: {exc}",
        )

    if not chunks:
        return {
            "query": req.query,
            "response": "No relevant papers were found for this query.",
            "citations": [],
            "metadata": {"retrieval_strategy": req.options.retrieval_strategy, "model_used": "none"},
        }

    # Drug lookup — if the query (or recent turns) mentions any known drug,
    # fetch its FDA card (with Wikipedia fallback) and inject as authoritative
    # context.
    drug_cards: list[str] = []
    drug_card_names: list[str] = []
    try:
        with psycopg.connect(DB_DSN) as drug_conn:
            cards = lookup_drugs_for_query(drug_conn, effective_query, max_drugs=3)
            for c in cards:
                drug_cards.append(c.to_text())
                drug_card_names.append(f"{c.name} ({c.source})")
        if drug_cards:
            logger.info("Injected %d drug card(s): %s",
                        len(drug_cards), ", ".join(drug_card_names))
    except Exception as exc:
        logger.warning("Drug lookup failed (%s); continuing without drug cards.", exc)

    try:
        provider = get_provider(req.options.llm_provider)
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
        synthesis = provider.synthesize(
            req.query, chunks,
            plain_language=req.options.plain_language,
            drug_cards=drug_cards or None,
            history=history_dicts or None,
        )
    except Exception as exc:
        logger.exception("LLM provider %r failed", req.options.llm_provider)
        raise HTTPException(
            status_code=502,
            detail=f"Provider '{req.options.llm_provider}' failed: {type(exc).__name__}: {exc}",
        )

    result = build_response(
        query=req.query,
        chunks=chunks,
        response_text=synthesis.response_text,
        cited_chunk_ids=synthesis.cited_chunk_ids,
        model_used=synthesis.model_used,
        retrieval_strategy=req.options.retrieval_strategy,
    )
    output = response_to_dict(result)
    output["metadata"]["latency_ms"] = int((time.monotonic() - start) * 1000)

    # Classify query and surface citation warnings (informational; never blocks)
    classification = classify_query(req.query)
    output["metadata"]["query_type"] = classification.query_type
    citation_warnings = verify_citations(synthesis.response_text, chunks)
    if citation_warnings:
        output["metadata"]["citation_warnings"] = warnings_to_dicts(citation_warnings)
    if drug_card_names:
        output["metadata"]["drug_cards"] = drug_card_names
    return output


@app.post("/api/v1/graph_extract")
async def graph_extract(req: GraphExtractRequest) -> dict[str, Any]:
    """Per-turn knowledge-graph payload for the chat panel.

    Called by the frontend AFTER the answer renders. Runs biomedical NER
    over the question + answer + cited chunk texts, then asks the LLM to
    emit JSON triples between those entities. Hallucinated triples
    (entities not in the NER set, or with no evidence chunk_ids) are
    dropped server-side. The frontend merges the returned nodes/edges
    into its session-local graph state.

    This is a separate endpoint from /api/v1/query so the chat answer
    can render at its current speed; the graph spinner waits on the
    second LLM call without blocking the user.
    """
    chunk_dicts = [{"id": c.id, "text": c.text} for c in req.chunks]
    try:
        payload = extract_graph(req.query, req.answer, chunk_dicts)
    except Exception as exc:
        logger.exception("graph_extract failed")
        raise HTTPException(
            status_code=502,
            detail=f"graph_extract failed: {type(exc).__name__}: {exc}",
        )
    return payload.to_dict()


@app.get("/api/v1/papers/{pmid}")
async def get_paper(pmid: str, db: Annotated[psycopg.Connection, Depends(get_db)]) -> dict:
    """Return full paper metadata for a given PMID."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT p.*, j.title AS journal_title, j.abbreviation AS journal_abbrev
            FROM papers p
            LEFT JOIN journals j ON p.journal_id = j.id
            WHERE p.pmid = %s
            """,
            (pmid,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Paper {pmid} not found")
    return dict(row)


@app.get("/api/v1/papers/{pmid}/similar")
async def get_similar_papers(
    pmid: str,
    top_k: Annotated[int, Query(ge=1, le=20)] = 10,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)] = None,
    db: Annotated[psycopg.Connection, Depends(get_db)] = None,
) -> dict:
    """Find semantically similar papers (requires Phase 2 embeddings)."""
    with db.cursor() as cur:
        cur.execute("SELECT title, abstract FROM papers WHERE pmid = %s", (pmid,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Paper {pmid} not found")

    query_text = f"{row['title']} {row['abstract'] or ''}".strip()
    chunks = retriever.retrieve(query=query_text, top_k=top_k)
    seen_pmids: set[str] = {pmid}
    results = []
    for c in chunks:
        if c.pmid not in seen_pmids:
            seen_pmids.add(c.pmid)
            results.append({"pmid": c.pmid, "title": c.title, "score": c.relevance_score})
    return {"source_pmid": pmid, "similar": results}


@app.get("/api/v1/papers/search")
async def search_papers(
    q: str = Query(..., min_length=2),
    pub_year_from: int | None = None,
    pub_year_to: int | None = None,
    publication_type: list[str] = Query(default=[]),
    mesh_term: list[str] = Query(default=[]),
    language: str = "eng",
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    retriever: Annotated[HybridRetriever, Depends(get_retriever)] = None,
) -> dict:
    """Full-text search with filters and facets."""
    filters: dict[str, Any] = {"language": language}
    if pub_year_from:
        filters["pub_year_from"] = pub_year_from
    if pub_year_to:
        filters["pub_year_to"] = pub_year_to
    if publication_type:
        filters["publication_types"] = publication_type
    if mesh_term:
        filters["mesh_terms"] = mesh_term

    chunks = retriever.retrieve(query=q, filters=filters, top_k=limit + offset)
    # Deduplicate by PMID
    seen: set[str] = set()
    papers = []
    for c in chunks:
        if c.pmid not in seen:
            seen.add(c.pmid)
            papers.append({
                "pmid": c.pmid,
                "title": c.title,
                "authors": c.authors_short,
                "journal": c.journal,
                "year": c.pub_year,
                "publication_types": c.publication_types,
                "score": c.relevance_score,
            })

    return {
        "results": papers[offset: offset + limit],
        "total": len(papers),
    }


@app.get("/api/v1/papers/{pmid}/citations")
async def get_citations(pmid: str, db: Annotated[psycopg.Connection, Depends(get_db)]) -> dict:
    """Papers cited by this paper (reference list)."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT c.cited_pmid_raw, p.title, p.pub_year
            FROM citations c
            JOIN papers p2 ON c.citing_paper_id = p2.id
            LEFT JOIN papers p ON c.cited_paper_id = p.id
            WHERE p2.pmid = %s
            ORDER BY c.cited_pmid_raw
            """,
            (pmid,),
        )
        rows = cur.fetchall()
    return {"pmid": pmid, "citations": [dict(r) for r in rows]}


@app.get("/api/v1/papers/{pmid}/cited_by")
async def get_cited_by(pmid: str, db: Annotated[psycopg.Connection, Depends(get_db)]) -> dict:
    """Papers that cite this paper."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT p.pmid, p.title, p.pub_year
            FROM citations c
            JOIN papers p ON c.citing_paper_id = p.id
            JOIN papers cited ON c.cited_paper_id = cited.id
            WHERE cited.pmid = %s
            ORDER BY p.pub_year DESC
            """,
            (pmid,),
        )
        rows = cur.fetchall()
    return {"pmid": pmid, "cited_by": [dict(r) for r in rows]}


@app.get("/api/v1/mesh/suggest")
async def mesh_suggest(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    """MeSH term autocomplete."""
    if _mesh_expander is None:
        raise HTTPException(status_code=503, detail="MeSH expander not initialized")
    suggestions = _mesh_expander.suggest_mesh(q, limit=limit)
    return {"suggestions": [dict(s) for s in suggestions]}


@app.get("/api/v1/ingestion/status")
async def ingestion_status(db: Annotated[psycopg.Connection, Depends(get_db)]) -> dict:
    """Summary of ingestion pipeline state."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM ingestion_log
            GROUP BY status
            ORDER BY status
            """
        )
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) AS total FROM papers")
        total = cur.fetchone()["total"]
    return {
        "total_papers": total,
        "ingestion_log": {r["status"]: r["count"] for r in rows},
    }


@app.get("/api/v1/stats")
async def stats(db: Annotated[psycopg.Connection, Depends(get_db)]) -> dict:
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS papers FROM papers")
        papers = cur.fetchone()["papers"]
        cur.execute("SELECT COUNT(*) AS chunks FROM paper_chunks")
        chunks = cur.fetchone()["chunks"]
        cur.execute("SELECT COUNT(*) AS with_full_text FROM papers WHERE has_full_text")
        oa = cur.fetchone()["with_full_text"]
    return {"total_papers": papers, "total_chunks": chunks, "papers_with_full_text": oa}


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/hipporag/reload", include_in_schema=True)
async def reload_hipporag() -> dict:
    """Reload the HippoRAG entity graph from the DB.

    Call this after `scripts/build_entity_graph.py` rebuilds the graph so
    the running server picks up the new edges without a full restart.
    """
    global _hipporag
    if _hipporag is None:
        from ..search.hipporag import HippoRAGRetriever
        _hipporag = HippoRAGRetriever(DB_DSN)
    _hipporag.reload()
    # Touch _ensure_loaded() to immediately rebuild
    _hipporag._ensure_loaded()
    return {
        "status": "reloaded",
        "nodes": _hipporag._graph.number_of_nodes() if _hipporag._graph else 0,
        "edges": _hipporag._graph.number_of_edges() if _hipporag._graph else 0,
    }


@app.get("/api/v1/llm/status")
async def llm_status() -> dict:
    """Report which providers are configured and reachable."""
    import httpx

    out: dict[str, Any] = {
        "configured_provider": os.environ.get("LLM_PROVIDER", "extractive"),
        "providers": {},
    }

    out["providers"]["extractive"] = {"available": True, "model": "extractive"}

    # Ollama: ping /api/tags and report current default model
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "biomistral")
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{ollama_url}/api/tags")
            installed = [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
        out["providers"]["ollama"] = {
            "available": ollama_model in installed or f"{ollama_model}:latest" in installed,
            "model": ollama_model,
            "installed_models": installed,
            "endpoint": ollama_url,
        }
    except Exception as exc:
        out["providers"]["ollama"] = {
            "available": False, "model": ollama_model, "error": str(exc),
            "endpoint": ollama_url,
        }

    out["providers"]["claude"] = {
        "available": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()
                          and os.environ.get("ANTHROPIC_API_KEY") != "your_anthropic_api_key_here"),
        "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
    }
    out["providers"]["openai"] = {
        "available": bool(os.environ.get("OPENAI_API_KEY", "").strip()
                          and os.environ.get("OPENAI_API_KEY") != "your_openai_api_key_here"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    }
    return out


@app.get("/api/v1/query/classify")
async def query_classify(q: str = Query(..., min_length=2)) -> dict:
    """Classify a query without retrieving — useful for UI hints."""
    c = classify_query(q)
    return {
        "query": q,
        "query_type": c.query_type,
        "suggested_top_k": c.suggested_top_k,
        "suggested_provider": c.suggested_provider,
        "reason": c.reason,
    }


# ── Static files (frontend SPA) ───────────────────────────────────────────────
import pathlib as _pathlib
_frontend_dir = _pathlib.Path(__file__).parent.parent.parent / "frontend"

# Explicit routes for the three known frontend files — more reliable than
# relying solely on a catch-all StaticFiles mount at "/".
@app.get("/", include_in_schema=False)
async def _serve_index():
    return FileResponse(str(_frontend_dir / "index.html"))

@app.get("/style.css", include_in_schema=False)
async def _serve_css():
    return FileResponse(str(_frontend_dir / "style.css"), media_type="text/css")

@app.get("/app.js", include_in_schema=False)
async def _serve_js():
    return FileResponse(str(_frontend_dir / "app.js"), media_type="application/javascript")

@app.get("/graph.js", include_in_schema=False)
async def _serve_graph_js():
    return FileResponse(str(_frontend_dir / "graph.js"), media_type="application/javascript")

# StaticFiles handles any other assets (fonts, images, etc.)
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir)), name="frontend")
