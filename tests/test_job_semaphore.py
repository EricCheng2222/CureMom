"""Regression test for the per-key concurrent-job semaphore.

A single misbehaving client (or a bug in the frontend) shouldn't be able to
spawn unbounded background jobs — each one costs an LLM round-trip and a
Python thread stack. _acquire_key_slot enforces a per-key cap; the fourth
concurrent job must be refused (429 from the route handler) instead of
spawning.

These tests poke at the primitive directly rather than spinning up live
worker threads — the goal is the lock semantics, not the rest of the
pipeline.
"""

from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_semaphores():
    from src.api import main as api_main
    with api_main._KEY_SEMAPHORES_LOCK:
        api_main._KEY_SEMAPHORES.clear()
    yield
    with api_main._KEY_SEMAPHORES_LOCK:
        api_main._KEY_SEMAPHORES.clear()


def test_acquire_up_to_limit_succeeds():
    """A key can take exactly _KEY_JOB_LIMIT slots before being refused."""
    from src.api import main as api_main

    for _ in range(api_main._KEY_JOB_LIMIT):
        assert api_main._acquire_key_slot(1) is True


def test_acquire_beyond_limit_returns_false():
    """The (limit + 1)-th acquire must fail without blocking."""
    from src.api import main as api_main

    for _ in range(api_main._KEY_JOB_LIMIT):
        api_main._acquire_key_slot(2)

    assert api_main._acquire_key_slot(2) is False, (
        "(limit+1)-th acquire must return False to trigger the 429 in the handler"
    )


def test_release_makes_slot_available_again():
    """A released slot can be re-acquired — slots are not single-use."""
    from src.api import main as api_main

    for _ in range(api_main._KEY_JOB_LIMIT):
        api_main._acquire_key_slot(3)
    assert api_main._acquire_key_slot(3) is False

    api_main._release_key_slot(3)
    assert api_main._acquire_key_slot(3) is True


def test_separate_keys_have_independent_pools():
    """Key A saturating its slots must not affect key B."""
    from src.api import main as api_main

    for _ in range(api_main._KEY_JOB_LIMIT):
        api_main._acquire_key_slot(10)
    assert api_main._acquire_key_slot(10) is False

    # Different key — fresh budget.
    for _ in range(api_main._KEY_JOB_LIMIT):
        assert api_main._acquire_key_slot(11) is True


def test_release_unknown_key_is_safe():
    """A release on a key that never acquired anything must not crash —
    over-release would raise threading.Semaphore's ValueError without the
    guard. Defensive cleanup paths rely on this."""
    from src.api import main as api_main
    # Nothing acquired for key 99 — release should be a no-op.
    api_main._release_key_slot(99)


def test_concurrent_acquires_respect_limit():
    """Even under thread contention, no more than _KEY_JOB_LIMIT acquires
    can succeed concurrently. Spin up many threads, count successes."""
    from src.api import main as api_main

    granted: list[bool] = []
    lock = threading.Lock()

    def worker():
        ok = api_main._acquire_key_slot(20)
        with lock:
            granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert granted.count(True) == api_main._KEY_JOB_LIMIT, (
        f"expected exactly {api_main._KEY_JOB_LIMIT} successes, got "
        f"{granted.count(True)} ({granted})"
    )
