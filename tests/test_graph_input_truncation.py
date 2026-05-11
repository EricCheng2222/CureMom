"""Regression test for the silent answer-truncation surface.

`_build_user_message` caps the answer at 5000 chars before sending it to
the graph-extract LLM. Without this signal the user gets a graph built
from less than half their answer with no warning.

`extract_graph` now sets a `truncated input:` prefix on the GraphPayload
error string when the input exceeded the cap; the frontend uses that
prefix to surface "Graph partial — …" in the chrome.
"""

from __future__ import annotations

import pytest

from src.api import graph_extractor


def _fake_llm(monkeypatch, relations):
    """Replace the LLM call with a stub returning a fixed relations list.
    Bypasses the network so the test exercises only the surrounding logic.
    """
    monkeypatch.setattr(
        graph_extractor,
        "_llm_graph",
        lambda **_: {"relations": relations, "types": {}, "_raw": ""},
    )


def test_long_answer_sets_input_truncated_error(monkeypatch):
    """A >5000-char answer triggers the input-truncation surface even on a
    successful LLM call. The error string must START with 'truncated' so
    the frontend's regex catches it."""
    # An answer well past the 5000-char cap, mentioning concepts the fake
    # LLM will use as relation endpoints so we get a non-empty graph.
    answer = "C1q binds apoptotic cells. " + ("filler text " * 600)
    assert len(answer) > graph_extractor._ANSWER_CHAR_CAP

    _fake_llm(monkeypatch, [
        {"subject": "C1q", "predicate": "binds", "object": "apoptotic cells", "citations": [1]},
    ])

    payload = graph_extractor.extract_graph(
        query="How does C1q clear apoptotic cells?",
        answer=answer,
        chunks=[{"id": 1, "text": "..."}],
    )

    assert payload.nodes, "expected at least one node from the fake LLM"
    assert payload.error, "long-answer case must set an error message"
    assert payload.error.startswith("truncated"), (
        f"frontend regex /^truncated/i won't match: {payload.error!r}"
    )
    assert "input" in payload.error, (
        f"error must distinguish input vs output truncation: {payload.error!r}"
    )


def test_short_answer_does_not_trigger_truncation(monkeypatch):
    """A normal-length answer must NOT carry a truncation error — otherwise
    every successful graph would show 'partial' in the chrome."""
    answer = "C1q binds apoptotic cells [1] and activates complement [2]."
    assert len(answer) < graph_extractor._ANSWER_CHAR_CAP

    _fake_llm(monkeypatch, [
        {"subject": "C1q", "predicate": "binds", "object": "apoptotic cells", "citations": [1]},
    ])

    payload = graph_extractor.extract_graph(
        query="How does C1q clear apoptotic cells?",
        answer=answer,
        chunks=[{"id": 1, "text": "..."}],
    )

    assert payload.nodes, "expected at least one node"
    # error may be None or contain other diagnostics — but it must NOT
    # claim truncation.
    if payload.error:
        assert "truncated" not in payload.error.lower(), (
            f"short answer should not be flagged as truncated: {payload.error!r}"
        )
