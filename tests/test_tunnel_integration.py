"""Integration tests that hit the real tunnel URL (serveo / Cloudflare / ngrok).

These exercise the parts of the stack that the in-process unit tests can't
reach: serveo's HTTP/2 edge, the SSH-tunnelled forwarding, proxy caching
behaviour, and end-to-end LLM round-trips.

Skipped by default so CI and local pytest runs stay fast. Enable by setting
both env vars before running pytest:

    CUREMOM_TUNNEL_URL=https://abc123-118-160-141-29.serveousercontent.com \
    CUREMOM_API_KEY=...your-admin-or-child-key... \
    pytest tests/test_tunnel_integration.py -v

What these tests catch that unit tests don't:
  * Cache-Control headers actually surviving the serveo proxy
  * Polling architecture surviving the wall-clock tunnel cap (~10 s)
  * LLM round-trips landing real nodes/edges and stage progress
  * 404 on unknown jobs flowing back through the proxy correctly
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request

import pytest

TUNNEL = os.environ.get("CUREMOM_TUNNEL_URL", "").rstrip("/")
KEY = os.environ.get("CUREMOM_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not (TUNNEL and KEY),
    reason="set CUREMOM_TUNNEL_URL and CUREMOM_API_KEY to run tunnel integration tests",
)

# macOS Python 3.9 ships without a usable cert store. We're testing tunnel
# behaviour, not TLS — skip cert validation. Set CUREMOM_TUNNEL_VERIFY_TLS=1
# to opt back in if you want strict cert checks.
_VERIFY_TLS = os.environ.get("CUREMOM_TUNNEL_VERIFY_TLS") == "1"
_SSL_CTX = ssl.create_default_context()
if not _VERIFY_TLS:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE


def _post(path: str, body: dict, timeout: int = 30) -> tuple[int, dict, dict]:
    req = urllib.request.Request(
        TUNNEL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            return r.status, dict(r.headers), json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.load(e) if e.fp else {}


def _get(path: str, timeout: int = 30) -> tuple[int, dict, dict]:
    req = urllib.request.Request(TUNNEL + path, headers={"X-API-Key": KEY})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            return r.status, dict(r.headers), json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), {}


def _poll(job_path: str, max_s: int = 240, interval: float = 1.5) -> dict:
    """Drive the standard polling loop a real client would use. Returns the
    final job snapshot (status=done or status=error). Raises on timeout."""
    deadline = time.monotonic() + max_s
    while time.monotonic() < deadline:
        time.sleep(interval)
        code, _, body = _get(job_path)
        assert code == 200, f"poll returned HTTP {code}"
        if body.get("status") in ("done", "error"):
            return body
    raise TimeoutError(f"job at {job_path} did not finish in {max_s}s")


# ─── Sanity ──────────────────────────────────────────────────────────────────

def test_tunnel_health_reachable():
    code, _, body = _get("/api/v1/health")
    assert code == 200
    assert body.get("status") in ("ok", "healthy") or "ok" in body


def test_tunnel_version_reachable():
    # /api/v1/version is public — no auth — but we still hit it through the tunnel.
    # The current implementation returns {"app_js_mtime": <unix-time>}; the test
    # just asserts the route is reachable and returns a JSON object.
    req = urllib.request.Request(TUNNEL + "/api/v1/version")
    with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
        body = json.load(r)
    assert isinstance(body, dict) and body, "expected non-empty JSON object from /version"


# ─── The cache-header bug, but through the actual proxy ──────────────────────

@pytest.mark.parametrize("endpoint", [
    "/api/v1/query/job",
    "/api/v1/graph_extract/job",
    "/api/v1/graph_dedup/job",
])
def test_tunnel_poll_unknown_job_has_no_cache_header(endpoint):
    """A 404 from /job/{unknown-id} must still carry no-store/no-cache so the
    proxy doesn't cache the 404 either — otherwise a new job's freshly-issued
    id could collide with a cached 404 and the user sees 'job expired'.
    """
    code, headers, _ = _get(f"{endpoint}/non-existent-id")
    assert code == 404
    cc = headers.get("cache-control", headers.get("Cache-Control", "")).lower()
    # serveo passes through what uvicorn sends; we expect the same no-cache
    # tokens we apply server-side.
    assert "no-store" in cc or "no-cache" in cc, (
        f"poll 404 missing no-cache headers (got: {cc!r}) — proxies may cache it"
    )


# ─── End-to-end QA + graph_extract through the tunnel ────────────────────────

def test_tunnel_query_async_end_to_end():
    """User flow: POST /query/async, poll, verify the answer + citations.

    The query is a real one mom would ask — "explain the mechanism of SLE"
    — to exercise the full hybrid retrieval + drug lookup + LLM synthesis
    path through the tunnel, not just a toy round-trip.
    """
    code, _, body = _post("/api/v1/query/async", {
        "query": "explain the mechanism of SLE",
        "options": {
            "top_k": 8,
            "retrieval_strategy": "hybrid",
            "llm_provider": "claude/claude-haiku-4-5-20251001",
            "plain_language": True,
        },
        "history": [],
    })
    assert code == 200, f"start failed: HTTP {code}"
    assert "job_id" in body
    job_id = body["job_id"]

    final = _poll(f"/api/v1/query/job/{job_id}")
    assert final["status"] == "done", f"job ended in {final['status']}: {final.get('error')}"

    payload = final["payload"]
    response_text = payload.get("response", "")
    assert response_text, "no response text in payload"
    assert isinstance(payload.get("citations"), list), "citations missing or not a list"
    # The question is broad enough that the corpus should surface multiple
    # citations; if this fails, retrieval or LLM is dropping evidence.
    assert payload["citations"], "no citations returned for an SLE mechanism question"


def test_tunnel_graph_extract_end_to_end():
    """User flow: POST /graph_extract with a real answer, poll, verify nodes."""
    answer = (
        "C1q binds apoptotic cells [1] and activates the classical complement "
        "pathway [2]. Phagocytes then clear the tagged cells [3]."
    )
    code, _, body = _post("/api/v1/graph_extract", {
        "query": "How does C1q clear apoptotic cells?",
        "answer": answer,
        "chunks": [
            {"id": 1, "text": "c1q apoptotic"},
            {"id": 2, "text": "complement"},
            {"id": 3, "text": "phagocyte"},
        ],
        "llm_provider": "claude/claude-haiku-4-5-20251001",
    })
    assert code == 200, f"start failed: HTTP {code}"
    job_id = body["job_id"]

    final = _poll(f"/api/v1/graph_extract/job/{job_id}", max_s=180)
    assert final["status"] == "done", f"job ended in {final['status']}: {final.get('error')}"

    payload = final["payload"]
    assert isinstance(payload.get("nodes"), list), "nodes missing"
    assert isinstance(payload.get("edges"), list), "edges missing"
    # Sanity: a 3-sentence answer about C1q should produce at least one node
    # and one edge unless the LLM punted entirely. If this fails, look at
    # payload.get('error') for the LLM's reasoning.
    assert payload["nodes"], f"no nodes returned: error={payload.get('error')!r}"


def test_tunnel_polling_response_carries_no_cache():
    """A successful poll (status=pending or done) must have no-store headers.

    This is the actual regression that hit production: proxies were caching
    the first 'pending' response and re-serving it for the entire polling
    window. The unit test catches it at the FastAPI layer; this test
    catches it again at the real proxy layer (serveo).
    """
    code, _, body = _post("/api/v1/graph_extract", {
        "query": "test query",
        "answer": "Y is a thing [1].",
        "chunks": [{"id": 1, "text": "z"}],
        "llm_provider": "claude/claude-haiku-4-5-20251001",
    })
    assert code == 200, f"start failed: HTTP {code} body={body}"
    job_id = body["job_id"]

    # Catch the response while it's still pending (or just-done) and inspect headers.
    _, headers, _ = _get(f"/api/v1/graph_extract/job/{job_id}")
    cc = headers.get("cache-control", headers.get("Cache-Control", "")).lower()
    assert "no-store" in cc and "no-cache" in cc, (
        f"poll missing no-cache headers (got: {cc!r}) — proxy will cache pending forever"
    )
