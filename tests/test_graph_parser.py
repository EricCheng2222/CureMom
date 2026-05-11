"""Unit tests for graph_extractor._parse_graph.

The parser is the only thing standing between an LLM that emitted partial
or weird JSON and a server-side crash. These tests pin down the contract:
"never raise — degrade gracefully to an empty graph". When max_tokens cuts
a Sonnet response mid-relation, the parser must still return a usable shape.
"""

from __future__ import annotations

from src.api.graph_extractor import _parse_graph


def test_parses_clean_json():
    raw = '{"relations":[{"subject":"C1q","predicate":"binds","object":"PS","citations":[1]}],"types":{"c1q":"GENE","ps":"OTHER"}}'
    out = _parse_graph(raw)
    assert len(out["relations"]) == 1
    assert out["types"] == {"c1q": "GENE", "ps": "OTHER"}


def test_parses_json_inside_markdown_fence():
    """Sonnet/NIM sometimes wrap JSON in ```json ... ``` blocks. The regex
    picks up the first {...} and that's enough."""
    raw = '```json\n{"relations":[],"types":{}}\n```'
    out = _parse_graph(raw)
    assert out["relations"] == []
    assert out["types"] == {}


def test_truncated_json_mid_relation_returns_empty():
    """If max_tokens hits mid-output, JSON is invalid (no closing brace).
    Parser must NOT raise; must return the empty fallback shape so the
    pipeline reports an empty graph instead of crashing the job."""
    raw = '{"relations":[{"subject":"C1q","predicate":"binds","object":"PS","citation'
    out = _parse_graph(raw)
    assert out["relations"] == [], "truncated JSON should fall back to empty relations"
    assert out["types"] == {}
    assert out["_raw"] == raw, "raw text preserved for debugging"


def test_truncated_json_mid_types_returns_empty():
    """Truncation inside the types map: still a parse error → empty fallback."""
    raw = '{"relations":[],"types":{"c1q":"GE'
    out = _parse_graph(raw)
    assert out["relations"] == []
    assert out["types"] == {}


def test_no_json_block_at_all():
    """LLM emitted only prose — no `{...}` anywhere."""
    out = _parse_graph("I can't answer that question.")
    assert out["relations"] == []
    assert out["types"] == {}


def test_empty_string():
    out = _parse_graph("")
    assert out["relations"] == []
    assert out["types"] == {}


def test_invalid_type_value_falls_back_to_other():
    """LLM sometimes invents type labels (e.g. 'METABOLITE'). The parser
    clamps anything outside the allowed set to OTHER."""
    raw = '{"relations":[],"types":{"glucose":"METABOLITE","c1q":"GENE"}}'
    out = _parse_graph(raw)
    assert out["types"] == {"glucose": "OTHER", "c1q": "GENE"}


def test_relations_field_wrong_shape():
    """If `relations` is a dict instead of a list, drop it (don't crash)."""
    raw = '{"relations":{"oops":1},"types":{}}'
    out = _parse_graph(raw)
    assert out["relations"] == []


def test_types_field_wrong_shape():
    raw = '{"relations":[],"types":[]}'
    out = _parse_graph(raw)
    assert out["types"] == {}
