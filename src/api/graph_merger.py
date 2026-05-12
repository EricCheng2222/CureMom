"""On-demand entity dedup for the knowledge-graph panel.

The graph accumulates duplicate nodes for the same concept across turns
(e.g. "B-cell" / "B cell" / "B cells", "Myostatin" / "myostatin",
"IGF-1" / "IGF1"). We deliberately do NOT hardcode normalization rules
— there's no end to the variants once you start (case, hyphens,
abbreviations, plurals, slash-separated synonyms, …).

Instead, when the user clicks the "Merge" button, this module asks the
LLM to group equivalent labels. The frontend then collapses each group
into a single canonical node and redirects all incident edges.

Mirrors the call pattern in src/api/graph_extractor.py: same Ollama
JSON-mode pattern, same env knobs, same dropdown-driven model
resolution.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .graph_extractor import emit_heartbeat, _suffix_for_model, _route_nim_model_for_structured

logger = logging.getLogger(__name__)


@dataclass
class MergeGroup:
    canonical: str          # the chosen short canonical label
    members: list[str]      # all input labels (including canonical) that fold into this group

    def to_dict(self) -> dict[str, Any]:
        return {"canonical": self.canonical, "members": self.members}


_DEDUP_PROMPT = """You receive a list of biomedical entity labels collected across multiple research-paper Q&A turns. Some refer to the same concept under different surface forms (case, hyphens, abbreviations, plurals, synonyms).

Your job: group equivalent labels.

Rules:
  - Group ONLY labels that refer to the SAME biological concept.
    "B-cell" ≡ "B cell" ≡ "B cells"   (variant spelling — group)
    "Myostatin" ≡ "myostatin"          (case only — group)
    "IGF-1" ≡ "IGF1"                   (variant spelling — group)
    "B cell" vs "T cell"               (DIFFERENT cells — do NOT group)
    "muscle" vs "skeletal muscle"      (HYPONYM — do NOT group; they are different specificities)
    "muscle growth" vs "muscle hypertrophy"  (synonyms — group, canonical = "muscle hypertrophy" or "muscle growth", pick one)
  - Each group must contain ≥2 members. Singletons should be omitted entirely.
  - Pick a canonical label for each group:
      * Prefer the most commonly used scientific spelling
      * Prefer hyphenated forms for compound names ("B cell" over "B cells")
      * Lowercase unless it's an acronym (IGF-1, mTOR, NF-kB)
  - Use ONLY labels from the input list verbatim. Do NOT invent new spellings.

Output ONLY this JSON (no prose, no markdown fences):

{
  "groups": [
    {"canonical": "<chosen label from input>", "members": ["<label1>", "<label2>", ...]}
  ]
}

If no groups exist, return {"groups": []}.

EXAMPLE:
Input: ["B-cell", "B cell", "B cells", "T cell", "Myostatin", "myostatin", "IGF-1", "IGF1", "liver", "skeletal muscle", "muscle"]
Output:
{"groups":[
  {"canonical":"B cell","members":["B-cell","B cell","B cells"]},
  {"canonical":"myostatin","members":["Myostatin","myostatin"]},
  {"canonical":"IGF-1","members":["IGF-1","IGF1"]}
]}"""


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def dedup_entities(
    labels: list[str],
    provider_spec: str | None = None,
    timeout_s: float | None = None,
    update: Callable[..., None] | None = None,
) -> list[MergeGroup]:
    """Ask the LLM to group equivalent biomedical entity labels.

    Routes to the same provider the user picked in the dropdown:
      claude         → Anthropic Messages API
      openai         → OpenAI chat completions (json_object mode)
      nim / nim/<m>  → NVIDIA NIM (OpenAI-compatible)
      else           → defaults to NIM (free tier)

    Returns a list of MergeGroup (each with ≥2 members). Groups whose
    members aren't all in the input list are dropped (no invented
    labels). Singletons are never returned.
    """
    if len(labels) < 2:
        return []
    if timeout_s is None:
        timeout_s = float(os.environ.get("GRAPH_TIMEOUT_S", "120"))

    user_msg = f"Labels ({len(labels)}):\n" + "\n".join(f"- {lbl}" for lbl in labels)
    target = _resolve_provider(provider_spec)
    model_override = _model_override(provider_spec)

    # Heartbeat: tick elapsed_s on the job every 500 ms so the client can
    # show "Merging… Ns" instead of a frozen button while the LLM runs.
    with emit_heartbeat(update, stage="dedup_entities"):
        if target == "claude":
            raw = _claude_dedup(user_msg, timeout_s, model=model_override)
        elif target == "openai":
            raw = _openai_dedup(user_msg, timeout_s, model=model_override)
        else:
            raw = _nim_dedup(user_msg, model_override, timeout_s)

    return _parse_groups(raw, allowed=set(labels))


def _resolve_provider(provider_spec: str | None) -> str:
    """See graph_extractor._resolve_provider — same logic. Bare `claude` /
    `openai` / `nim` and `<provider>/<model>` forms both route correctly.
    Anything unrecognized falls back to nim.
    """
    if not provider_spec:
        return "nim"
    head = provider_spec.split("/", 1)[0].strip().lower()
    if head in ("claude", "openai", "nim"):
        return head
    return "nim"


def _model_override(provider_spec: str | None) -> str | None:
    if not provider_spec or "/" not in provider_spec:
        return None
    return provider_spec.split("/", 1)[1].strip() or None


def _claude_dedup(user_msg: str, timeout_s: float, model: str | None = None) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    model = model or os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    logger.info("graph_dedup: calling Claude (model=%s, timeout=%.0fs)", model, timeout_s)
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    msg = client.messages.create(
        model=model,
        max_tokens=32768,
        system=_DEDUP_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text if msg.content else ""


def _openai_dedup(user_msg: str, timeout_s: float, model: str | None = None) -> str:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError("OPENAI_API_KEY not set in env")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    logger.info("graph_dedup: calling OpenAI (model=%s, timeout=%.0fs)", model, timeout_s)
    client = openai.OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DEDUP_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _nim_dedup(user_msg: str, model: str | None, timeout_s: float) -> str:
    """Call NIM (OpenAI-compatible) for dedup.

    The NIM_MODEL default is `meta/llama-4-maverick-17b-128e-instruct` — a non-reasoning
    sibling on the same provider. Reasoning models like minimaxai/minimax-m2.7
    emit ~3 K thinking tokens before any JSON (100+ s wall clock) and NIM
    doesn't honor any of the standard "disable thinking" knobs. Users who
    explicitly want a reasoning model can pick `nim/minimaxai/minimax-m2.7`
    in the dropdown.

    `model` is the per-request override extracted from `nim/<model>` specs.
    Falls back to NIM_MODEL env when None.
    """
    import openai

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key == "your_nvidia_api_key_here":
        raise RuntimeError("NVIDIA_API_KEY not set in env")
    requested = model or os.environ.get("NIM_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
    # Same reasoning-model reroute as _nim_graph — MiniMax dedup is 100+ s,
    # llama dedup is <10 s. Users who pick MiniMax for QA still get
    # MiniMax for QA; this only affects the structured-output Merge call.
    model = _route_nim_model_for_structured(requested)
    base_url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    system_prompt = _DEDUP_PROMPT + _suffix_for_model(model)
    if model != requested:
        logger.info("graph_dedup: rerouted reasoning model %s → %s "
                    "(non-reasoning; QA still uses %s)", requested, model, requested)
    logger.info("graph_dedup: calling NIM (model=%s, timeout=%.0fs, no_think=%s)",
                model, timeout_s, bool(_suffix_for_model(model)))
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=32768,
    )
    return resp.choices[0].message.content or ""


def _parse_groups(raw: str, allowed: set[str]) -> list[MergeGroup]:
    """Parse the LLM JSON output. Drop groups that:
      - have <2 members
      - reference a label not in the input list (no invented labels)
      - have a canonical not in the members list
    Members and canonical are compared case-insensitively against the
    input set, then the original spelling from the input list is kept.
    """
    m = _JSON_RE.search(raw)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    raw_groups = obj.get("groups") if isinstance(obj.get("groups"), list) else []

    # Build a case-insensitive lookup back to the original spellings.
    lookup: dict[str, str] = {lbl.lower(): lbl for lbl in allowed}

    out: list[MergeGroup] = []
    for g in raw_groups:
        if not isinstance(g, dict):
            continue
        canonical_in = (g.get("canonical") or "").strip()
        members_in = g.get("members") or []
        if not isinstance(members_in, list):
            continue

        members: list[str] = []
        seen: set[str] = set()
        for m_raw in members_in:
            if not isinstance(m_raw, str):
                continue
            ml = m_raw.strip().lower()
            orig = lookup.get(ml)
            if orig is None or orig in seen:
                continue
            members.append(orig)
            seen.add(orig)

        if len(members) < 2:
            continue

        # Resolve canonical to one of the surviving members, preferring the
        # LLM's choice if it's actually in the input.
        canonical = lookup.get(canonical_in.lower())
        if canonical is None or canonical not in members:
            # Fall back to the first member.
            canonical = members[0]

        out.append(MergeGroup(canonical=canonical, members=members))

    logger.info("graph_dedup: %d groups returned (out of %d raw)", len(out), len(raw_groups))
    return out
