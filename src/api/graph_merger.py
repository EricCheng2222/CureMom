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
from dataclasses import dataclass
from typing import Any

import httpx

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
        timeout_s = float(os.environ.get("GRAPH_TIMEOUT_S", "600"))

    user_msg = f"Labels ({len(labels)}):\n" + "\n".join(f"- {lbl}" for lbl in labels)
    target = _resolve_provider(provider_spec)

    if target == "claude":
        raw = _claude_dedup(user_msg, timeout_s)
    elif target == "openai":
        raw = _openai_dedup(user_msg, timeout_s)
    else:
        raw = _nim_dedup(user_msg, provider_spec, timeout_s)

    return _parse_groups(raw, allowed=set(labels))


def _resolve_provider(provider_spec: str | None) -> str:
    if not provider_spec:
        return "nim"
    if provider_spec == "claude":
        return "claude"
    if provider_spec == "openai":
        return "openai"
    if provider_spec.startswith("nim/") or provider_spec == "nim":
        return "nim"
    return "nim"


def _claude_dedup(user_msg: str, timeout_s: float) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    logger.info("graph_dedup: calling Claude (model=%s, timeout=%.0fs)", model, timeout_s)
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_DEDUP_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text if msg.content else ""


def _openai_dedup(user_msg: str, timeout_s: float) -> str:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError("OPENAI_API_KEY not set in env")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

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


def _nim_dedup(user_msg: str, provider_spec: str | None, timeout_s: float) -> str:
    import openai

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key == "your_nvidia_api_key_here":
        raise RuntimeError("NVIDIA_API_KEY not set in env")
    if provider_spec and provider_spec.startswith("nim/"):
        model = provider_spec.split("/", 1)[1]
    else:
        model = os.environ.get("NIM_MODEL", "minimaxai/minimax-m2.7")
    base_url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    logger.info("graph_dedup: calling NIM (model=%s, timeout=%.0fs)", model, timeout_s)
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DEDUP_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=2048,
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
