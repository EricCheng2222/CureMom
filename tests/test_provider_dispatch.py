"""Regression test for the LLM provider resolver.

Background
----------
The frontend dropdown sends `<provider>/<model>` specs like
`claude/claude-haiku-4-5-20251001` or `nim/meta/llama-3.1-70b-instruct`.

A bug in `_resolve_provider` only matched bare provider names — anything
with a slash silently fell through to the NIM default. That meant picking
"Claude Haiku 4.5 (fast)" in the UI silently routed graph_extract +
graph_dedup to NIM, which:
  - made Haiku-selected calls 20× slower than they should be (NIM's free
    tier vs Anthropic's paid endpoint),
  - returned worse results (NIM's MiniMax was a reasoning model that
    timed out on dedup),
  - hid the bug from anyone testing with bare `claude` / `openai` specs.

Fix: resolver now treats anything before the first `/` as the provider
name, with a separate helper to extract the model override.

These tests pin the resolver behavior on both modules (extractor + merger)
so a future "simplify the spec parsing" PR can't accidentally re-introduce
the silent NIM fallback.
"""

from __future__ import annotations

import pytest

from src.api import graph_extractor, graph_merger


@pytest.mark.parametrize("spec, expected", [
    # Bare provider names.
    ("claude",  "claude"),
    ("openai",  "openai"),
    ("nim",     "nim"),

    # The case that USED TO BREAK — a model after the slash.
    ("claude/claude-haiku-4-5-20251001",  "claude"),
    ("claude/claude-sonnet-4-6",          "claude"),
    ("openai/gpt-4o",                     "openai"),
    ("nim/meta/llama-3.1-70b-instruct",   "nim"),
    ("nim/minimaxai/minimax-m2.7",        "nim"),

    # Defaults.
    (None,    "nim"),
    ("",      "nim"),
    ("garbage", "nim"),    # unrecognized provider → NIM fallback (free tier)
])
def test_extractor_resolver(spec, expected):
    assert graph_extractor._resolve_provider(spec) == expected, (
        f"extractor resolver dispatched {spec!r} to wrong provider"
    )


@pytest.mark.parametrize("spec, expected", [
    ("claude",  "claude"),
    ("openai",  "openai"),
    ("nim",     "nim"),
    ("claude/claude-haiku-4-5-20251001",  "claude"),
    ("openai/gpt-4o",                     "openai"),
    ("nim/meta/llama-3.1-70b-instruct",   "nim"),
    (None,    "nim"),
])
def test_merger_resolver_matches_extractor(spec, expected):
    """The two resolvers must agree — otherwise picking the same dropdown
    value would route graph_extract one way and merge another."""
    assert graph_merger._resolve_provider(spec) == expected


@pytest.mark.parametrize("spec, expected_model", [
    # Bare provider — no override, falls back to env default in the helper.
    ("claude",   None),
    ("openai",   None),
    ("nim",      None),
    (None,       None),

    # The frontend's actual values.
    ("claude/claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    ("claude/claude-sonnet-4-6",         "claude-sonnet-4-6"),
    ("openai/gpt-4o",                    "gpt-4o"),
    # NIM model names contain a slash themselves — the SECOND slash starts
    # the model. The split must be greedy on the first slash only.
    ("nim/meta/llama-3.1-70b-instruct",  "meta/llama-3.1-70b-instruct"),
    ("nim/minimaxai/minimax-m2.7",       "minimaxai/minimax-m2.7"),
])
def test_extractor_model_override(spec, expected_model):
    assert graph_extractor._model_override(spec) == expected_model, (
        f"extractor model override misparsed {spec!r}"
    )


@pytest.mark.parametrize("spec, expected_model", [
    ("claude/claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    ("nim/meta/llama-3.1-70b-instruct",  "meta/llama-3.1-70b-instruct"),
    ("nim",                               None),
    (None,                                None),
])
def test_merger_model_override_matches(spec, expected_model):
    assert graph_merger._model_override(spec) == expected_model
