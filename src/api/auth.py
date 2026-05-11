"""API-key authentication for the public endpoint.

Two-tier key model:
  * Admin keys can mint unlimited child keys.
  * Each non-admin key can mint exactly ONE child key.

The admin key is bootstrapped at startup from `INITIAL_ADMIN_KEY` env if
set, otherwise a random 43-char token is generated and logged to stdout
on the first run with no admin key in the table.

Schema:
  api_keys(
    id            SERIAL PK,
    key           TEXT UNIQUE NOT NULL,
    parent_id     INT NULL REFERENCES api_keys(id),
    is_admin      BOOL NOT NULL DEFAULT FALSE,
    is_revoked    BOOL NOT NULL DEFAULT FALSE,
    note          TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
  )
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Annotated

import psycopg
from fastapi import Depends, Header, HTTPException, status
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


@dataclass
class KeyRecord:
    id: int
    key: str
    parent_id: int | None
    is_admin: bool
    note: str | None


def _new_token() -> str:
    """43-char URL-safe random token (32 random bytes, base64url-encoded)."""
    return secrets.token_urlsafe(32)


def init_keys_table(conn: psycopg.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS api_keys + indexes. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id          SERIAL PRIMARY KEY,
                key         TEXT UNIQUE NOT NULL,
                parent_id   INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
                is_admin    BOOLEAN NOT NULL DEFAULT FALSE,
                is_revoked  BOOLEAN NOT NULL DEFAULT FALSE,
                note        TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS api_keys_key_idx ON api_keys(key);")
        cur.execute("CREATE INDEX IF NOT EXISTS api_keys_parent_idx ON api_keys(parent_id);")
        conn.commit()


def bootstrap_admin_key(conn: psycopg.Connection) -> str | None:
    """If no admin key exists, create one and return it. Otherwise return None.

    Reads INITIAL_ADMIN_KEY from env if set (so the user can pin it across
    restarts); otherwise generates a fresh random token and logs it ONCE.
    The returned value should be printed to stdout by the caller so the
    operator can copy it.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM api_keys WHERE is_admin = TRUE AND is_revoked = FALSE LIMIT 1;")
        if cur.fetchone():
            return None

        admin_key = os.environ.get("INITIAL_ADMIN_KEY", "").strip() or _new_token()
        cur.execute(
            "INSERT INTO api_keys (key, is_admin, note) VALUES (%s, TRUE, %s) RETURNING id;",
            (admin_key, "bootstrap admin"),
        )
        conn.commit()
    logger.warning("=" * 72)
    logger.warning("BOOTSTRAP: created admin API key. Save it now — it won't be shown again:")
    logger.warning("  %s", admin_key)
    logger.warning("=" * 72)
    return admin_key


def lookup_key(conn: psycopg.Connection, raw_key: str) -> KeyRecord | None:
    """Look up a key string. Returns None if missing or revoked."""
    if not raw_key:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, key, parent_id, is_admin, is_revoked, note "
            "FROM api_keys WHERE key = %s;",
            (raw_key,),
        )
        row = cur.fetchone()
    if not row or row["is_revoked"]:
        return None
    return KeyRecord(
        id=row["id"], key=row["key"], parent_id=row["parent_id"],
        is_admin=row["is_admin"], note=row["note"],
    )


def child_count(conn: psycopg.Connection, parent_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM api_keys WHERE parent_id = %s AND is_revoked = FALSE;",
            (parent_id,),
        )
        return int(cur.fetchone()["n"])


def generate_child_key(conn: psycopg.Connection, parent: KeyRecord, note: str | None = None) -> str:
    """Mint a new key. Admin can mint unlimited; others get exactly one."""
    if not parent.is_admin:
        if child_count(conn, parent.id) >= 1:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Non-admin keys can mint exactly one child key. You've already used yours.",
            )
    new_key = _new_token()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (key, parent_id, is_admin, note) "
            "VALUES (%s, %s, FALSE, %s) RETURNING id;",
            (new_key, parent.id, note),
        )
        conn.commit()
    return new_key


# ─── FastAPI dependency ──────────────────────────────────────────────────────


def _open_conn() -> psycopg.Connection:
    """Open a fresh DB connection for the auth check.

    The auth dep runs before the regular get_db() dep can route the
    request, so we don't share that connection. Cheap — psycopg pools at
    the OS-socket level with most setups, and the lookup is one indexed
    query.
    """
    dsn = (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'curemom')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'curemom')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'curemom')}"
    )
    return psycopg.connect(dsn, row_factory=dict_row)


# In-process LRU-with-TTL cache for the key→KeyRecord lookup. Without this
# every request (including every poll on /job/{id}) opens a fresh psycopg
# connection — at 1.5 s polling intervals a single 30 s job churns 20 DB
# connections just to validate the same header. 60 s TTL is short enough
# that key revocations propagate within a minute. Negative results are NOT
# cached (a freshly minted key shouldn't be blocked for up to a minute by
# a prior 401 lookup).
_KEY_CACHE: dict[str, tuple[float, KeyRecord]] = {}
_KEY_CACHE_LOCK = threading.Lock()
_KEY_CACHE_TTL_S = 60.0
_KEY_CACHE_MAX = 1024  # bound memory; trim oldest on overflow


def _cached_lookup(raw_key: str) -> KeyRecord | None:
    now = time.monotonic()
    with _KEY_CACHE_LOCK:
        entry = _KEY_CACHE.get(raw_key)
        if entry and entry[0] > now:
            return entry[1]
        # Stale entries are dropped lazily on next hit; here we just fall through.

    # Cache miss — hit the DB.
    with _open_conn() as conn:
        rec = lookup_key(conn, raw_key)

    if rec is not None:
        with _KEY_CACHE_LOCK:
            if len(_KEY_CACHE) >= _KEY_CACHE_MAX:
                # Drop the oldest expiring entry — bounded eviction, no LRU.
                oldest = min(_KEY_CACHE.items(), key=lambda kv: kv[1][0])[0]
                _KEY_CACHE.pop(oldest, None)
            _KEY_CACHE[raw_key] = (now + _KEY_CACHE_TTL_S, rec)
    else:
        # Drop any stale positive entry — the key was just revoked or rotated.
        with _KEY_CACHE_LOCK:
            _KEY_CACHE.pop(raw_key, None)
    return rec


def invalidate_key_cache(raw_key: str | None = None) -> None:
    """Drop a single key (or the whole cache) — call this from any code path
    that revokes a key so the next request re-reads the DB instead of using
    a still-valid cache entry."""
    with _KEY_CACHE_LOCK:
        if raw_key is None:
            _KEY_CACHE.clear()
        else:
            _KEY_CACHE.pop(raw_key, None)


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> KeyRecord:
    """FastAPI dependency: extract X-API-Key header, validate against DB.
    Returns the KeyRecord on success, raises 401 otherwise. Validations are
    cached in-process for 60 s; revocations propagate within that window.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header. Get a key from the admin.",
        )
    rec = _cached_lookup(x_api_key.strip())
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )
    return rec
