"""CureMom FastAPI application."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from ..search.elasticsearch_client import get_client as get_es_client, ensure_index
from ..search.hybrid_retriever import HybridRetriever
from ..search.mesh_expander import MeSHExpander
from .auth import (
    KeyRecord, bootstrap_admin_key, generate_child_key, init_keys_table,
    require_api_key,
)
from .classifier import classify_query
from .citation_verifier import verify_citations, warnings_to_dicts
from .drug_lookup import lookup_drugs_for_query
from .graph_extractor import extract_graph
from .graph_merger import dedup_entities
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
    # Upper cap on returned chunks. Retriever pulls a 100-candidate pool
    # and returns 10% of it (floored at 5, capped at this top_k).
    top_k: int = Field(default=20, ge=1, le=50)
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
    # Same dropdown choice the user picked for QA. When given as
    # "ollama/<model>", graph extraction uses that model so the answer
    # and the graph come from the same brain. Anything else falls back
    # to OLLAMA_GRAPH_MODEL / OLLAMA_MODEL env defaults.
    llm_provider: str | None = None


class GraphDedupRequest(BaseModel):
    labels: list[str] = Field(default_factory=list, max_length=200)
    llm_provider: str | None = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup_init_auth() -> None:
    """Bootstrap the api_keys table + admin key on first run."""
    try:
        with psycopg.connect(DB_DSN) as conn:
            init_keys_table(conn)
            bootstrap_admin_key(conn)   # logs the key on first run
    except Exception as exc:
        logger.error("Failed to initialise api_keys table: %s", exc)


@app.post("/api/v1/query")
def query(
    req: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, Any]:
    """Main Q&A endpoint: retrieve relevant passages and return a cited response.

    Plain `def` (not async) so FastAPI runs this in its threadpool. The
    LLM provider clients (Anthropic SDK, OpenAI SDK, NIM) are synchronous
    and would otherwise block the event loop for the entire LLM call,
    freezing every other endpoint until it finishes.
    """
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


def _with_heartbeat(source, interval_s: float = 3.0):
    """Wrap a sync generator so the stream emits a real SSE data event
    every `interval_s` seconds during gaps.

    The keepalive is a real `data: {"stage":"keepalive"}` event — NOT a
    comment line. Serveo's edge proxy ignores SSE comments for its
    upstream-idle timer (we measured ~10 s before it 502s a long Sonnet
    call even with 3 s `: keepalive` comments). A real data event counts
    as activity everywhere.

    Clients should filter `stage === "keepalive"` events out (the
    /query/stream parser ignores them via _STAGE_LABELS lookup; the
    graph SSE parser explicitly skips them).

    The source generator runs on a daemon thread; the main loop here
    pumps real items as they arrive and falls back to a heartbeat when
    the queue is idle.
    """
    import queue
    import threading

    q: "queue.Queue" = queue.Queue()
    DONE = object()

    def _producer() -> None:
        try:
            for chunk in source:
                q.put(("item", chunk))
        except BaseException as exc:   # noqa: BLE001
            q.put(("exc", exc))
        finally:
            q.put((DONE, None))

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        try:
            kind, payload = q.get(timeout=interval_s)
        except queue.Empty:
            yield 'data: {"stage":"keepalive"}\n\n'
            continue
        if kind is DONE:
            return
        if kind == "exc":
            raise payload
        yield payload


def _run_query_pipeline(
    req: "QueryRequest",
    retriever: "HybridRetriever",
    update: Callable[..., None],
) -> dict[str, Any]:
    """Synchronous QA pipeline. Calls `update(stage=..., **extra)` at each
    stage boundary so the polling endpoint can surface progress to the client.
    Returns the final response dict (same shape /query used to return).
    """
    start = time.monotonic()
    update(stage="analyzing")

    filter_dict = req.filters.model_dump(exclude_none=False)
    strategy = req.options.retrieval_strategy
    dense_weight = 0.5 if strategy in ("hybrid", "full") else 0.0
    use_hipporag = strategy in ("hipporag", "full")
    query_embedding: list[float] | None = None

    effective_query = req.query
    if req.history:
        prior_user_terms = " ".join(
            m.content for m in req.history[-6:] if m.role == "user"
        )
        if prior_user_terms:
            effective_query = f"{prior_user_terms} {req.query}"

    if dense_weight > 0:
        update(stage="embedding")
        global _embedder
        if _embedder is None:
            from ..embeddings.chunk_pipeline import _Embedder
            logger.info("Loading PubMedBERT for hybrid retrieval (one-time)…")
            _embedder = _Embedder()
        query_embedding = _embedder.encode([effective_query])[0]

    if use_hipporag:
        global _hipporag
        if _hipporag is None:
            update(stage="loading_graph")
            from ..search.hipporag import HippoRAGRetriever
            logger.info("Loading HippoRAG entity graph (one-time)…")
            _hipporag = HippoRAGRetriever(DB_DSN)
            _hipporag._ensure_loaded()
            if _hipporag._graph is None or _hipporag._graph.number_of_edges() == 0:
                raise RuntimeError(
                    "entity_graph table is empty — run scripts/build_entity_graph.py"
                )
        retriever._hipporag = _hipporag

    update(stage="retrieving")
    chunks = retriever.retrieve(
        query=effective_query,
        filters=filter_dict,
        top_k=req.options.top_k,
        dense_weight=dense_weight,
        query_embedding=query_embedding,
        use_hipporag=use_hipporag,
    )

    if not chunks:
        return {
            "query": req.query,
            "response": "No relevant papers were found for this query.",
            "citations": [],
            "metadata": {"retrieval_strategy": strategy, "model_used": "none"},
        }

    update(stage="drug_lookup")
    drug_cards: list[str] = []
    drug_card_names: list[str] = []
    try:
        with psycopg.connect(DB_DSN) as drug_conn:
            cards = lookup_drugs_for_query(drug_conn, effective_query, max_drugs=3)
            for c in cards:
                drug_cards.append(c.to_text())
                drug_card_names.append(f"{c.name} ({c.source})")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Drug lookup failed (%s); continuing.", exc)

    provider = get_provider(req.options.llm_provider)
    update(stage="synthesizing", model=provider.name)
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    synthesis = provider.synthesize(
        req.query, chunks,
        plain_language=req.options.plain_language,
        drug_cards=drug_cards or None,
        history=history_dicts or None,
    )

    update(stage="verifying")
    result = build_response(
        query=req.query, chunks=chunks,
        response_text=synthesis.response_text,
        cited_chunk_ids=synthesis.cited_chunk_ids,
        model_used=synthesis.model_used,
        retrieval_strategy=strategy,
    )
    output = response_to_dict(result)
    output["metadata"]["latency_ms"] = int((time.monotonic() - start) * 1000)
    classification = classify_query(req.query)
    output["metadata"]["query_type"] = classification.query_type
    citation_warnings = verify_citations(synthesis.response_text, chunks)
    if citation_warnings:
        output["metadata"]["citation_warnings"] = warnings_to_dicts(citation_warnings)
    if drug_card_names:
        output["metadata"]["drug_cards"] = drug_card_names
    return output


@app.post("/api/v1/query/async")
def query_async(
    req: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, str]:
    """Start a background QA job. Returns {"job_id": "..."} immediately.

    Poll GET /api/v1/query/job/{job_id} until status is "done" or "error".
    The poll response also includes the current `stage` (analyzing /
    embedding / retrieving / synthesizing / verifying) for live progress.
    """
    def work(update: Callable[..., None]) -> dict[str, Any]:
        return _run_query_pipeline(req, retriever, update)

    job_id = _start_job(work, initial={"stage": "analyzing"}, pass_update=True)
    return {"job_id": job_id}


@app.get("/api/v1/query/job/{job_id}")
def query_job(
    job_id: str,
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, Any]:
    """Poll a QA job. Returns {status, stage?, model?, payload?, error?}."""
    return _job_response(job_id)


# Legacy SSE endpoint kept as a thin wrapper that runs the pipeline synchronously
# and streams the result. Used by clients that haven't migrated to polling.
@app.post("/api/v1/query/stream")
def query_stream(
    req: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> StreamingResponse:
    """Legacy SSE pipeline kept for backward compatibility.

    The frontend now uses /query/async + polling; this endpoint runs the
    same pipeline but streams stage events as SSE. Will break through
    tunnels with wall-clock caps (serveo) on long LLM calls — that's
    the reason the polling path exists.
    """
    def evt(stage: str, **extra: Any) -> str:
        return f"data: {json.dumps({'stage': stage, **extra})}\n\n"

    def gen():
        try:
            # Adapter: forward each pipeline stage as an SSE event.
            def update(**kwargs: Any) -> None:
                stage = kwargs.pop("stage", None)
                if stage:
                    # Note: yield from here doesn't work since update is a plain fn.
                    # We use a queue-style hack via the pipeline's own yield path
                    # below to keep this endpoint working.
                    raise NotImplementedError
            # Simpler approach: just call the pipeline and emit a single complete event.
            # Stage granularity is lost on this legacy path; clients wanting it
            # should use /query/async.
            yield evt("analyzing")
            output = _run_query_pipeline(req, retriever, lambda **_: None)
            yield evt("complete", result=output)
        except Exception as exc:  # noqa: BLE001
            logger.exception("query/stream pipeline failed")
            yield evt("error", detail=f"{type(exc).__name__}: {exc}", status=502)

    return StreamingResponse(
        _with_heartbeat(gen(), interval_s=3),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable proxy buffering (Cloudflare/nginx)
            "Connection": "keep-alive",
        },
    )


# ─── Async job store for long compute-bound operations ────────────────────────
# Streaming + heartbeats don't survive serveo's ~10 s tunnel cap. Use the
# standard pattern instead: POST starts a background job and returns a job_id
# immediately; client polls GET /job/<id> until done. Every HTTP round-trip
# stays well under any proxy timeout. Same pattern as OpenAI runs, HF endpoints,
# Vertex long-running operations, etc.
#
# Generic to all compute-bound endpoints: QA (/query/async), graph extraction
# (/graph_extract), graph dedup (/graph_dedup). The work() function may
# optionally accept an `update` callback to publish progress (current stage,
# model name, etc.) that the poll endpoint surfaces back to the client.

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_S = 1200  # 20 min — bigger than the LLM + polling deadlines (both
                   # 600 s) so a result that completes near the polling
                   # deadline doesn't get GC'd before the last poll picks it up.


def _gc_jobs() -> None:
    now = time.monotonic()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items()
                   if now - j.get("created", 0) > _JOB_TTL_S]
        for jid in expired:
            del _jobs[jid]


def _start_job(
    work: Callable[..., dict[str, Any]],
    *,
    initial: dict[str, Any] | None = None,
    pass_update: bool = False,
) -> str:
    """Run `work` on a daemon thread; return a job_id.

    work() returns the payload dict on success or raises on error.
    If pass_update=True, work is called as work(update_fn) where update_fn
    takes kwargs to merge into the live job state (for stage progress).
    """
    _gc_jobs()
    job_id = uuid.uuid4().hex[:16]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "created": time.monotonic(),
            **(initial or {}),
        }

    def _update(**kwargs: Any) -> None:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(kwargs)

    def _runner() -> None:
        try:
            payload = work(_update) if pass_update else work()
            _update(status="done", payload=payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("job %s failed", job_id)
            _update(status="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_runner, daemon=True).start()
    return job_id


# Poll endpoints must NEVER be cached. Without Cache-Control, intermediaries
# (serveo's HTTP/2 edge, Cloudflare, browser caches) are free to cache GET
# responses indefinitely — and they will. The first poll returns "pending",
# gets cached, every subsequent poll re-serves the cached "pending" even
# after the server flipped to "done". This is the actual cause of the
# "spinner disappears, no graph" bug users hit through the tunnel.
_NO_CACHE_JOB_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _job_response(job_id: str) -> JSONResponse:
    """Return the current job state with anti-cache headers.

    Both the 200 (pending/done/error) and the 404 (unknown id) paths carry
    the same headers — the 404 is defensive: if a proxy cached it, a
    fresh job with the same id (vanishingly rare with uuid4 but possible)
    would inherit the stale 404.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        snapshot = dict(job) if job else None
    if not snapshot:
        return JSONResponse(
            status_code=404,
            content={"detail": "job not found or expired"},
            headers=_NO_CACHE_JOB_HEADERS,
        )
    snapshot.pop("created", None)
    return JSONResponse(content=snapshot, headers=_NO_CACHE_JOB_HEADERS)


# Kept for the regression tests that call this helper directly.
def _get_job_snapshot(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        snapshot = dict(job) if job else None
    if not snapshot:
        raise HTTPException(status_code=404, detail="job not found or expired")
    snapshot.pop("created", None)
    return snapshot


@app.post("/api/v1/graph_extract")
def graph_extract(
    req: GraphExtractRequest,
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, str]:
    """Start a background graph-extract job. Poll /graph_extract/job/{id}."""
    def work() -> dict[str, Any]:
        chunk_dicts = [{"id": c.id, "text": c.text} for c in req.chunks]
        payload = extract_graph(req.query, req.answer, chunk_dicts,
                                provider_spec=req.llm_provider)
        return payload.to_dict()
    return {"job_id": _start_job(work)}


@app.get("/api/v1/graph_extract/job/{job_id}")
def graph_extract_job(
    job_id: str,
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, Any]:
    return _job_response(job_id)


@app.post("/api/v1/graph_dedup")
def graph_dedup(
    req: GraphDedupRequest,
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, str]:
    """Start a background graph-dedup job. Poll /graph_dedup/job/{id}."""
    def work() -> dict[str, Any]:
        groups = dedup_entities(req.labels, provider_spec=req.llm_provider)
        return {"groups": [g.to_dict() for g in groups]}
    return {"job_id": _start_job(work)}


@app.get("/api/v1/graph_dedup/job/{job_id}")
def graph_dedup_job(
    job_id: str,
    _key: Annotated[KeyRecord, Depends(require_api_key)] = None,
) -> dict[str, Any]:
    return _job_response(job_id)


# ─── Key management ───────────────────────────────────────────────────────────


@app.get("/api/v1/keys/me")
async def keys_me(
    key: Annotated[KeyRecord, Depends(require_api_key)],
    db: Annotated[psycopg.Connection, Depends(get_db)],
) -> dict:
    """Tell the caller about their key (admin? how many children?)."""
    from .auth import child_count
    return {
        "is_admin": key.is_admin,
        "children_minted": child_count(db, key.id),
        "can_mint_more": key.is_admin or child_count(db, key.id) < 1,
        "note": key.note,
    }


class KeyMintRequest(BaseModel):
    note: str | None = Field(default=None, max_length=200)


@app.post("/api/v1/keys/generate")
async def keys_generate(
    req: KeyMintRequest,
    key: Annotated[KeyRecord, Depends(require_api_key)],
    db: Annotated[psycopg.Connection, Depends(get_db)],
) -> dict:
    """Mint a child key. Admin = unlimited; non-admin = exactly one."""
    new_key = generate_child_key(db, key, note=req.note)
    return {"key": new_key, "note": req.note}


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


@app.get("/api/v1/version")
async def app_version() -> dict:
    """Returns the mtime of the served app.js so already-open tabs can
    detect a redeploy and prompt the user to refresh. The frontend polls
    this every ~30s; if the value differs from the one captured on first
    load, a "new version available" banner appears."""
    try:
        return {"app_js_mtime": int((_frontend_dir / "app.js").stat().st_mtime)}
    except OSError:
        return {"app_js_mtime": 0}


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
    out: dict[str, Any] = {
        "configured_provider": os.environ.get("LLM_PROVIDER", "extractive"),
        "providers": {},
    }

    out["providers"]["extractive"] = {"available": True, "model": "extractive"}

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
    out["providers"]["nim"] = {
        "available": bool(os.environ.get("NVIDIA_API_KEY", "").strip()
                          and os.environ.get("NVIDIA_API_KEY") != "your_nvidia_api_key_here"),
        "model": os.environ.get("NIM_MODEL", "minimaxai/minimax-m2.7"),
        "endpoint": os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
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

# index.html must never be cached: it points at versioned asset URLs
# (app.js?v=N, style.css?v=N) and a stale copy locks the user into
# whatever version those query strings referenced when it was cached.
# Two tabs opened at different times would otherwise run different code.
# The asset files themselves (.js / .css) keep aggressive caching because
# their URL changes on every release — the cache-bust IS the versioning.
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/", include_in_schema=False)
async def _serve_index():
    return FileResponse(str(_frontend_dir / "index.html"), headers=_NO_CACHE_HEADERS)

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
