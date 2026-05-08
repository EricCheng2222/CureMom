"""Swappable LLM provider layer.

Select provider via LLM_PROVIDER env var:
  extractive  — no LLM; returns top-ranked sentences with inline citations (default)
  ollama      — local LLM via Ollama HTTP API (biomistral, meditron:7b, etc.)
  claude      — Anthropic Claude API
  openai      — OpenAI API

All providers return a SynthesisResult with response_text and citation_indices
mapping back to the input chunks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..search.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

PATIENT_SYSTEM_PROMPT = """You are a patient-friendly medical assistant. You help non-medical readers understand what peer-reviewed research says about their condition, symptoms, or curiosity.

RULES:
1. Answer using ONLY the provided context passages — every factual claim must be supported by the passages.
2. Cite every claim with [N] where N is the passage number (1-indexed).
3. Use plain language. Avoid jargon. When a technical term is unavoidable, briefly explain it in parentheses (e.g. "anti-dsDNA antibody (a marker the immune system makes against the body's own DNA)").
4. If the passages do not contain enough information, say exactly: "The research I have access to doesn't have enough information to answer that confidently."
5. Be warm and direct. Short paragraphs. No medical-textbook tone.
6. Do NOT use knowledge beyond the provided passages.

Structure your answer:
  • Start with a 1-2 sentence direct answer to what they asked.
  • A short plain-language explanation with citations.
  • If the question implies clinical advice (treatment, dose, "should I take..."), end the main answer with: "This is for information only — please talk to a qualified healthcare professional before making any decisions."

Then, on a new section, ALWAYS include a follow-up section in this exact format:

**You might also want to know:**
- <one natural follow-up question the reader would likely ask next>
- <another follow-up question, focused on a related angle>
- <a third follow-up question, e.g. about treatments / mechanism / lifestyle / related conditions>

The follow-up questions should:
  • Be drawn from topics that the retrieved passages actually cover (don't suggest a follow-up if the literature can't answer it).
  • Be phrased the way a curious reader would phrase them, not the way a doctor would.
  • Be 3 questions, no more, no fewer.
"""


GROUNDING_SYSTEM_PROMPT = """You are a biomedical research assistant with access to retrieved passages from peer-reviewed medical literature.

RULES:
1. Answer ONLY using the provided context passages — every factual claim must be supported by the passages.
2. Cite every claim with [N] where N is the passage number (1-indexed). Multiple citations like [1][3] are encouraged.
3. If the passages do not contain sufficient information, say exactly: "The retrieved literature does not provide sufficient evidence to answer this question."
4. Do NOT use knowledge beyond the provided passages.

WHEN THE QUESTION ASKS ABOUT A CONDITION, MECHANISM, OR TREATMENT, structure the answer to cover (only the parts the passages actually support):
  • Brief description of the condition / phenomenon
  • Underlying mechanism — proteins, pathways, immune cells, cytokines, genes involved
  • Clinical evidence — symptoms, biomarkers, diagnostic findings
  • Interventions reported in the literature — drugs, dosages, supplements, lifestyle factors, with effect sizes when given

Be specific: prefer named molecules ("anti-dsDNA antibody", "complement C3") over vague terms ("autoantibodies", "immune factors"). Prefer quantitative data ("78% of patients", "OR 2.4") when available. Do not invent quantities.

End with a one-sentence summary of what the evidence collectively supports.
"""


@dataclass
class SynthesisResult:
    response_text: str
    citation_indices: list[int]         # 1-indexed passage numbers cited
    cited_chunk_ids: list[int]          # resolved chunk IDs from the DB
    model_used: str
    raw_response: str | None = None     # full LLM output before parsing


def _build_context_block(
    chunks: list[RetrievedChunk],
    drug_cards: list[str] | None = None,
) -> str:
    """Format retrieved chunks as numbered passages for the LLM context.

    `drug_cards` is an optional list of pre-formatted drug-reference strings
    (FDA labels / PubChem) that go in their own section above the literature.
    These are authoritative — the LLM cites them by drug name, not [N].
    """
    n = len(chunks)
    lines: list[str] = []

    if drug_cards:
        lines.append(
            "DRUG REFERENCES (one or more cards below). Each card's first line "
            "is formatted: 'DRUG REFERENCE — <NAME> (<SOURCE>)'. Cite by name "
            "and source, e.g. 'the FDA label for atorvastatin states…' or "
            "'per the Wikipedia entry on mephenoxalone…' — match the source "
            "tag on the card. Do NOT use [N] markers for these references; "
            "[N] is reserved for the literature passages below."
        )
        lines.append("")
        lines.append(
            "If a drug card lacks a specific field (e.g. no Mechanism listed), "
            "say so plainly — don't paraphrase the indication line as if it "
            "were a mechanism explanation. The retrieved literature passages "
            "below may have more detail; pull from them if relevant."
        )
        lines.append("")
        for card in drug_cards:
            lines.append(card)
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(
        f"You have exactly {n} retrieved passage{'s' if n != 1 else ''}, "
        f"numbered [1] through [{n}]. "
        f"Do NOT cite any number outside this range — "
        f"if you do, your answer will be rejected."
    )
    lines.append("")

    for i, chunk in enumerate(chunks, start=1):
        author_year = f"{chunk.authors_short}, {chunk.pub_year}" if chunk.pub_year else chunk.authors_short
        source_line = f"[{i}] Source: {author_year}. {chunk.journal or ''}. PMID:{chunk.pmid}"
        lines.append(source_line)
        lines.append(chunk.chunk_text)
        lines.append("")
    return "\n".join(lines)


def _parse_citation_indices(text: str) -> list[int]:
    """Extract [N] citation markers from LLM output."""
    return list({int(m) for m in re.findall(r"\[(\d+)\]", text)})


def _select_system_prompt(plain_language: bool) -> str:
    return PATIENT_SYSTEM_PROMPT if plain_language else GROUNDING_SYSTEM_PROMPT


class LLMProvider(ABC):
    @abstractmethod
    def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        plain_language: bool = False,
        drug_cards: list[str] | None = None,
    ) -> SynthesisResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ExtractiveProvider(LLMProvider):
    """No LLM: returns the most relevant sentences from the top chunks."""

    @property
    def name(self) -> str:
        return "extractive"

    def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        plain_language: bool = False,
        drug_cards: list[str] | None = None,
    ) -> SynthesisResult:
        if not chunks:
            return SynthesisResult(
                response_text="No relevant passages were found for this query.",
                citation_indices=[],
                cited_chunk_ids=[],
                model_used="extractive",
            )

        sentences: list[str] = []
        citation_indices: list[int] = []
        cited_chunk_ids: list[int] = []

        for i, chunk in enumerate(chunks[:5], start=1):  # top 5 chunks
            # Take first 2 sentences from each chunk as the extractive summary
            text = chunk.chunk_text.strip()
            split = text.replace("? ", "?|").replace(". ", ".|").replace("! ", "!|")
            chunk_sentences = [s.strip() for s in split.split("|") if len(s.strip()) > 30]
            selected = chunk_sentences[:2]
            if selected:
                sentences.append(f"[{i}] " + " ".join(selected))
                citation_indices.append(i)
                cited_chunk_ids.append(chunk.chunk_id)

        response = "\n\n".join(sentences)
        return SynthesisResult(
            response_text=response,
            citation_indices=citation_indices,
            cited_chunk_ids=cited_chunk_ids,
            model_used="extractive",
        )


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "biomistral",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        plain_language: bool = False,
        drug_cards: list[str] | None = None,
    ) -> SynthesisResult:
        context = _build_context_block(chunks, drug_cards=drug_cards)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _select_system_prompt(plain_language)},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            # 32K window — Gemma 4 supports long context; large enough that
            # 8+ chunks (each ~500 tokens) fit with room for system prompt,
            # question, and response. First request reloads the model (~10s).
            "options": {"num_ctx": 32768},
        }

        # Generous timeout: first call reloads model with new context window;
        # local 7B on M-series typically generates 30-60 tokens/sec.
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)) as client:
            response = client.post(f"{self._base_url}/api/chat", json=payload)
            if response.status_code != 200:
                # Surface Ollama's actual error message (e.g. "model 'X' not found")
                try:
                    detail = response.json().get("error", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(
                    f"Ollama {response.status_code}: {detail} "
                    f"(model={self._model!r}; check `ollama list`)"
                )
            data = response.json()

        raw = data["message"]["content"]
        indices = _parse_citation_indices(raw)
        cited_ids = [chunks[i - 1].chunk_id for i in indices if 1 <= i <= len(chunks)]

        return SynthesisResult(
            response_text=raw,
            citation_indices=indices,
            cited_chunk_ids=cited_ids,
            model_used=self.name,
            raw_response=raw,
        )


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        plain_language: bool = False,
        drug_cards: list[str] | None = None,
    ) -> SynthesisResult:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install anthropic package: pip install anthropic") from exc

        context = _build_context_block(chunks, drug_cards=drug_cards)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_select_system_prompt(plain_language),
            messages=[{"role": "user", "content": user_message}],
        )

        raw = message.content[0].text
        indices = _parse_citation_indices(raw)
        cited_ids = [chunks[i - 1].chunk_id for i in indices if 1 <= i <= len(chunks)]

        return SynthesisResult(
            response_text=raw,
            citation_indices=indices,
            cited_chunk_ids=cited_ids,
            model_used=self.name,
            raw_response=raw,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        plain_language: bool = False,
        drug_cards: list[str] | None = None,
    ) -> SynthesisResult:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("Install openai package: pip install openai") from exc

        context = _build_context_block(chunks, drug_cards=drug_cards)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _select_system_prompt(plain_language)},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content or ""
        indices = _parse_citation_indices(raw)
        cited_ids = [chunks[i - 1].chunk_id for i in indices if 1 <= i <= len(chunks)]

        return SynthesisResult(
            response_text=raw,
            citation_indices=indices,
            cited_chunk_ids=cited_ids,
            model_used=self.name,
            raw_response=raw,
        )


def get_provider(provider_spec: str | None = None) -> LLMProvider:
    """Factory: return the appropriate LLMProvider.

    `provider_spec` is either a bare provider name ("extractive", "ollama",
    "claude", "openai") or a provider/model override ("ollama/medgemma:4b",
    "claude/claude-haiku-4-5"). When no model is specified, the
    corresponding env var is used (OLLAMA_MODEL / CLAUDE_MODEL / OPENAI_MODEL).
    """
    spec = provider_spec or os.environ.get("LLM_PROVIDER", "extractive")
    head, _, model_override = spec.partition("/")
    head = head.strip().lower()
    model_override = model_override.strip() or None

    if head == "extractive":
        return ExtractiveProvider()

    if head == "ollama":
        return OllamaProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model_override or os.environ.get("OLLAMA_MODEL", "biomistral"),
        )

    if head == "claude":
        return ClaudeProvider(
            model=model_override or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        )

    if head == "openai":
        return OpenAIProvider(
            model=model_override or os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )

    raise ValueError(
        f"Unknown LLM provider: {head!r}. Choose: extractive, ollama[/<model>], "
        "claude[/<model>], openai[/<model>]"
    )
