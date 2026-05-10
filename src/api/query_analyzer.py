"""LLM-based query intent analysis for drug-aware retrieval.

Idea: medgemma already knows that "muscle relaxation" maps to
"skeletal muscle relaxant", "antispasmodic", "musculoskeletal pain", etc.
Rather than building a static synonym dictionary or relying on lexical
FTS alone, we let the LLM expand each query into:

  { "intent": "about_specific_drug" | "find_drugs_by_effect" | "general",
    "drug_names": [...],          # for forward lookup
    "indication_terms": [...]     # canonical clinical terms for reverse FTS
  }

This is one extra ~1-2s LLM call per query but dramatically improves
reverse drug lookup quality without needing a vector index over labels.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from .llm_providers import get_provider
from ..search.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    intent: Literal["about_specific_drug", "find_drugs_by_effect", "general"]
    drug_names: list[str] = field(default_factory=list)
    indication_terms: list[str] = field(default_factory=list)
    raw: str | None = None


_ANALYZER_PROMPT = """You analyze medical queries before retrieval. Output ONLY a JSON object — no prose, no markdown fences. Fields:

  intent: one of
    - "about_specific_drug"      — query names a specific drug, regardless of what aspect (uses, mechanism, dose, side effects, interactions, contraindications, pharmacology)
    - "find_drugs_by_effect"     — query describes an effect/condition WITHOUT naming a drug, and is asking what drugs treat it
    - "general"                  — neither: pure biology, anatomy, physiology, or no medication angle
  drug_names:        array of drug names actually MENTIONED in the query (lowercase, generic preferred). REQUIRED whenever a drug name appears, even for mechanism/biology questions about that drug.
  indication_terms:  array of canonical CLINICAL terms (max 6) for the effect or condition. Use FDA-label language. Only populate when intent="find_drugs_by_effect".

Examples:

Q: "What is hydroxychloroquine used for?"
{"intent":"about_specific_drug","drug_names":["hydroxychloroquine"],"indication_terms":[]}

Q: "What is the mechanism of cyclobenzaprine?"
{"intent":"about_specific_drug","drug_names":["cyclobenzaprine"],"indication_terms":[]}

Q: "How does atorvastatin lower cholesterol?"
{"intent":"about_specific_drug","drug_names":["atorvastatin"],"indication_terms":[]}

Q: "Side effects of metformin?"
{"intent":"about_specific_drug","drug_names":["metformin"],"indication_terms":[]}

Q: "what drugs help muscle relaxation"
{"intent":"find_drugs_by_effect","drug_names":[],"indication_terms":["skeletal muscle relaxant","muscle spasm","musculoskeletal pain","spasticity"]}

Q: "drugs for high blood pressure"
{"intent":"find_drugs_by_effect","drug_names":[],"indication_terms":["hypertension","essential hypertension","blood pressure reduction"]}

Q: "How do muscles grow?"
{"intent":"general","drug_names":[],"indication_terms":[]}

Q: "What causes lupus fatigue?"
{"intent":"general","drug_names":[],"indication_terms":[]}
"""


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def analyze_query(query: str, provider_spec: str | None = None) -> QueryAnalysis:
    """Run a one-shot LLM call to extract structured intent + terms.

    Always uses Claude Haiku (fast + cheap + reliable JSON). Independent
    of whichever provider the user picked for QA — query analysis is a
    meta-task and we want consistency.

    Falls back to QueryAnalysis(intent='general') if the LLM is
    unreachable or produces unparseable output, so the user's query
    still gets answered without drug-aware expansion.
    """
    try:
        return _analyze_via_claude(query)
    except Exception as exc:
        logger.warning("LLM query analysis failed (%s); using passthrough.", exc)
        return QueryAnalysis(intent="general", raw=str(exc))


def _analyze_via_claude(query: str) -> QueryAnalysis:
    """Call Anthropic Messages API directly for structured analysis.

    Cheap with Haiku (~$0.0001/call) and fast (1-2s). The system prompt
    instructs JSON output; _parse extracts it via regex.
    """
    import os
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=_ANALYZER_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    raw = msg.content[0].text if msg.content else ""
    return _parse(raw)


def _parse(raw: str) -> QueryAnalysis:
    """Parse the LLM's JSON output. Tolerant to extra whitespace / fences."""
    m = _JSON_RE.search(raw)
    if not m:
        return QueryAnalysis(intent="general", raw=raw)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return QueryAnalysis(intent="general", raw=raw)

    intent = obj.get("intent", "general")
    if intent not in {"about_specific_drug", "find_drugs_by_effect", "general"}:
        intent = "general"
    return QueryAnalysis(
        intent=intent,
        drug_names=[s.strip().lower() for s in (obj.get("drug_names") or []) if isinstance(s, str)],
        indication_terms=[s.strip() for s in (obj.get("indication_terms") or []) if isinstance(s, str)],
        raw=raw,
    )
