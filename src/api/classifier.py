"""Query complexity classifier — selects retrieval/synthesis strategy.

Three categories:
  • factual      — single fact, named entity, narrow scope
                   ("What is the half-life of belimumab?")
  • exploratory  — broad/open-ended, may have multiple valid answers
                   ("How does lupus affect the kidneys?")
  • comparative  — explicit or implicit comparison between options
                   ("Is hydroxychloroquine better than methotrexate for SLE?")

The classification informs:
  • Default top_k (factual: 5, exploratory: 12, comparative: 10)
  • Provider routing (factual → extractive ok; exploratory/comparative → LLM)
  • Prompt template (comparative gets structured pro/con scaffolding)

V1 is rule-based — empirically good enough for this corpus. Upgrade to a
sklearn classifier once we have labeled training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

QueryType = Literal["factual", "exploratory", "comparative"]

_COMPARATIVE_PATTERNS = [
    r"\bvs\.?\b", r"\bversus\b",
    r"\bcompared?\s+(?:to|with|against)\b",
    r"\bbetter\s+than\b", r"\bworse\s+than\b",
    r"\bmore\s+effective\b", r"\bless\s+effective\b",
    r"\bdifference\s+between\b", r"\bsuperior\s+to\b",
    r"\bover\b.*\bor\b",
]
_COMPARATIVE_RE = re.compile("|".join(_COMPARATIVE_PATTERNS), re.IGNORECASE)

_FACTUAL_OPENERS = (
    "what is", "what are", "what's the",
    "define", "definition of",
    "how much", "how many",
    "when was", "where is", "which", "is the",
)

_EXPLORATORY_OPENERS = (
    "how does", "how do", "why does", "why do",
    "explain", "describe", "tell me about",
    "what role", "what factors",
)


@dataclass
class Classification:
    query_type: QueryType
    suggested_top_k: int
    suggested_provider: str  # 'extractive' | 'llm'
    reason: str


def classify_query(query: str) -> Classification:
    q = query.strip().lower()

    if _COMPARATIVE_RE.search(q):
        return Classification(
            query_type="comparative",
            suggested_top_k=10,
            suggested_provider="llm",
            reason="Comparative phrasing detected (vs / better than / compared to / etc.)",
        )

    if any(q.startswith(opener) for opener in _EXPLORATORY_OPENERS):
        return Classification(
            query_type="exploratory",
            suggested_top_k=12,
            suggested_provider="llm",
            reason="Exploratory phrasing — open-ended explanation requested.",
        )

    if any(q.startswith(opener) for opener in _FACTUAL_OPENERS):
        return Classification(
            query_type="factual",
            suggested_top_k=5,
            suggested_provider="extractive",
            reason="Factual phrasing — extractive likely sufficient.",
        )

    # Default: medium scope, extractive ok
    return Classification(
        query_type="exploratory",
        suggested_top_k=10,
        suggested_provider="llm",
        reason="No strong signal — defaulting to exploratory.",
    )
