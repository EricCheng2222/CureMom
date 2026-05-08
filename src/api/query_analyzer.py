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

    Falls back to a simple QueryAnalysis(intent='general') if the LLM is
    unreachable or produces unparseable output. (This is the one place we
    *do* fall back gracefully — the user's query still gets answered, it
    just won't benefit from drug-aware expansion.)
    """
    try:
        provider = get_provider(provider_spec)
        # Build a tiny synthetic chunk list — providers expect chunks, but
        # this is a meta-task and we only want the model's text out.
        # We bypass synthesize() and call the LLM directly via the provider's
        # internals where possible. Simpler: use httpx directly to ollama
        # for medgemma, since that's the most common case.
        return _analyze_via_ollama(query, provider_spec)
    except Exception as exc:
        logger.warning("LLM query analysis failed (%s); using passthrough.", exc)
        return QueryAnalysis(intent="general", raw=str(exc))


def _analyze_via_ollama(query: str, provider_spec: str | None) -> QueryAnalysis:
    """Call Ollama directly for a structured analysis. Ignores non-Ollama provider
    specs and uses the configured default Ollama model — query analysis is
    cheap and we want consistency, not the user's choice of OpenAI/Claude
    just for this step."""
    import os, httpx
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "medgemma:4b")
    if provider_spec and provider_spec.startswith("ollama/"):
        model = provider_spec.split("/", 1)[1]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _ANALYZER_PROMPT},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "format": "json",                   # Ollama: enforce JSON output
        "options": {"num_ctx": 4096, "temperature": 0.0},
    }
    with httpx.Client(timeout=30) as client:
        r = client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        raw = r.json()["message"]["content"]

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
