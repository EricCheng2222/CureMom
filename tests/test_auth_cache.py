"""Regression test for the require_api_key TTL cache.

Without this cache, every request (including every poll on /job/{id}) opens
a fresh psycopg connection — at 1.5 s polling intervals a single 30 s job
churns 20 DB connections just to validate the same X-API-Key header.

The cache MUST:
  * hit on the second call within the TTL (no DB lookup)
  * miss when the cache is invalidated
  * miss when the cached entry is older than the TTL
  * NOT cache 401 results (a freshly minted key shouldn't be blocked by a
    prior 401 for up to a minute)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with an empty cache so the state doesn't leak between
    cases."""
    from src.api import auth
    auth._KEY_CACHE.clear()
    yield
    auth._KEY_CACHE.clear()


def test_second_call_within_ttl_skips_db(monkeypatch):
    """The whole point of the cache: the second call must NOT hit lookup_key."""
    from src.api import auth

    calls = {"n": 0}

    def fake_lookup(_conn, raw_key):
        calls["n"] += 1
        return auth.KeyRecord(id=42, key=raw_key, parent_id=None, is_admin=False, note=None)

    # Replace the DB-touching path so we count exactly how many real lookups
    # happen. The connection helper is irrelevant — it'll be discarded.
    monkeypatch.setattr(auth, "lookup_key", fake_lookup)
    monkeypatch.setattr(auth, "_open_conn",
                        lambda: _NullCtx())

    r1 = auth._cached_lookup("k-1")
    r2 = auth._cached_lookup("k-1")
    r3 = auth._cached_lookup("k-1")

    assert r1 is not None and r1.id == 42
    assert r2 is r1, "cache should return the same KeyRecord instance on hit"
    assert r3 is r1
    assert calls["n"] == 1, f"expected 1 DB call, got {calls['n']}"


def test_invalidate_clears_entry(monkeypatch):
    """Revoking a key should propagate immediately when the caller invalidates."""
    from src.api import auth

    calls = {"n": 0}

    def fake_lookup(_conn, raw_key):
        calls["n"] += 1
        return auth.KeyRecord(id=7, key=raw_key, parent_id=None, is_admin=False, note=None)

    monkeypatch.setattr(auth, "lookup_key", fake_lookup)
    monkeypatch.setattr(auth, "_open_conn", lambda: _NullCtx())

    auth._cached_lookup("k-2")
    auth._cached_lookup("k-2")  # cached, no DB
    auth.invalidate_key_cache("k-2")
    auth._cached_lookup("k-2")   # cache cleared, must hit DB

    assert calls["n"] == 2


def test_expired_entry_triggers_refresh(monkeypatch):
    """Once the TTL elapses, the next call goes back to the DB."""
    from src.api import auth

    calls = {"n": 0}

    def fake_lookup(_conn, raw_key):
        calls["n"] += 1
        return auth.KeyRecord(id=9, key=raw_key, parent_id=None, is_admin=False, note=None)

    monkeypatch.setattr(auth, "lookup_key", fake_lookup)
    monkeypatch.setattr(auth, "_open_conn", lambda: _NullCtx())

    # Anchor "now" so we can advance it past the TTL without sleeping.
    base = [1000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: base[0])
    auth._cached_lookup("k-3")
    base[0] += auth._KEY_CACHE_TTL_S + 1  # blast past expiry
    auth._cached_lookup("k-3")

    assert calls["n"] == 2, "stale cache entry should be re-fetched after TTL"


def test_negative_lookups_are_not_cached(monkeypatch):
    """A 401 must NOT poison the cache — the user might fix the typo next."""
    from src.api import auth

    calls = {"n": 0}

    def fake_lookup(_conn, raw_key):
        calls["n"] += 1
        return None  # always-fail lookup

    monkeypatch.setattr(auth, "lookup_key", fake_lookup)
    monkeypatch.setattr(auth, "_open_conn", lambda: _NullCtx())

    auth._cached_lookup("bad-key")
    auth._cached_lookup("bad-key")
    auth._cached_lookup("bad-key")

    assert calls["n"] == 3, "negative results must not be cached"


# ─── Helpers ─────────────────────────────────────────────────────────────────


class _NullCtx:
    """Minimal context manager standing in for a psycopg connection."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
