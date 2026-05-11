"""Regression test for the "graph never shows" tunnel bug.

Background
----------
The async job-poll endpoints (/query/job, /graph_extract/job, /graph_dedup/job)
must return ``Cache-Control: no-store`` so intermediaries (serveo's HTTP/2 edge,
browser cache, anything else in the path) don't shadow the first ``pending``
response for the entire polling window.

Without these headers, the first poll's ``{"status":"pending"}`` is cached and
re-served indefinitely; the client polls for 180 s seeing only "pending", then
gives up. Spinner hides, no graph. We hit this in production through serveo —
the fix was wrapping the poll responses in JSONResponse with explicit
no-cache headers (commit 21404aa).

This test exercises each poll endpoint by injecting a synthetic job directly
into the in-memory store (no LLM call required) and asserting that the
response headers contain a non-cacheable directive.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with the API-key dependency bypassed.

    require_api_key normally hits the keys DB; we replace it with a no-op so
    tests don't need a live Postgres or a real admin key.
    """
    from src.api import main as api_main
    from src.api import auth

    api_main.app.dependency_overrides[auth.require_api_key] = lambda: auth.KeyRecord(
        id=0, key="test", parent_id=None, is_admin=True, note=None,
    )
    try:
        yield TestClient(api_main.app)
    finally:
        api_main.app.dependency_overrides.clear()


def _inject_job(job_id: str, **state) -> None:
    """Drop a fake job into the in-memory store so we can poll its state
    without running the LLM pipeline."""
    from src.api import main as api_main
    import time as _time
    with api_main._jobs_lock:
        api_main._jobs[job_id] = {
            "status": "pending",
            "created": _time.monotonic(),
            **state,
        }


def _assert_no_cache(headers: dict) -> None:
    """The Cache-Control directive must explicitly disable caching.

    We accept any value that contains both ``no-store`` and ``no-cache``;
    different frameworks word it slightly differently but those two tokens
    cover every well-behaved proxy.
    """
    cc = headers.get("cache-control", "").lower()
    assert cc, "Cache-Control header missing entirely — proxies will cache the poll"
    assert "no-store" in cc, f"Cache-Control missing no-store: {cc!r}"
    assert "no-cache" in cc, f"Cache-Control missing no-cache: {cc!r}"


@pytest.mark.parametrize("endpoint", [
    "/api/v1/query/job",
    "/api/v1/graph_extract/job",
    "/api/v1/graph_dedup/job",
])
def test_pending_poll_returns_no_cache_headers(client, endpoint):
    """First-poll-while-pending: headers must say no-store/no-cache."""
    job_id = f"test-pending-{endpoint.replace('/', '_')}"
    _inject_job(job_id, status="pending", stage="analyzing")

    r = client.get(f"{endpoint}/{job_id}")
    assert r.status_code == 200, r.text
    _assert_no_cache(r.headers)

    body = r.json()
    assert body["status"] == "pending"
    assert body["stage"] == "analyzing"


@pytest.mark.parametrize("endpoint", [
    "/api/v1/query/job",
    "/api/v1/graph_extract/job",
    "/api/v1/graph_dedup/job",
])
def test_done_poll_returns_no_cache_headers(client, endpoint):
    """After the work completes, the same response shape with payload must
    also carry the no-cache headers (otherwise a stale 'pending' would
    persist in a proxy that already cached the first poll)."""
    job_id = f"test-done-{endpoint.replace('/', '_')}"
    _inject_job(job_id, status="done", payload={"nodes": [], "edges": []})

    r = client.get(f"{endpoint}/{job_id}")
    assert r.status_code == 200, r.text
    _assert_no_cache(r.headers)

    body = r.json()
    assert body["status"] == "done"
    assert body["payload"] == {"nodes": [], "edges": []}


def test_unknown_job_returns_404(client):
    r = client.get("/api/v1/graph_extract/job/does-not-exist")
    assert r.status_code == 404
