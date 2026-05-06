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

GROUNDING_SYSTEM_PROMPT = """You are a biomedical research assistant with access to retrieved passages from peer-reviewed medical literature.

RULES:
1. Answer ONLY using the provided context passages.
2. Cite every factual claim with [N] where N is the passage number (1-indexed).
3. If the passages do not contain sufficient information to answer, respond with exactly: "The retrieved literature does not provide sufficient evidence to answer this question."
4. Do NOT use knowledge beyond what is in the provided passages.
5. Be concise and factual. Prefer quantitative data when available.
"""


@dataclass
class SynthesisResult:
    response_text: str
    citation_indices: list[int]         # 1-indexed passage numbers cited
    cited_chunk_ids: list[int]          # resolved chunk IDs from the DB
    model_used: str
    raw_response: str | None = None     # full LLM output before parsing


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered passages for the LLM context."""
    lines: list[str] = []
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


class LLMProvider(ABC):
    @abstractmethod
    def synthesize(self, query: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
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

    def synthesize(self, query: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
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

    def synthesize(self, query: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
        context = _build_context_block(chunks)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            # 8K window — meditron's default 2048 truncates long retrieval contexts.
            # First request reloads the model (~10s); subsequent ones are fast.
            "options": {"num_ctx": 8192},
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

    def synthesize(self, query: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install anthropic package: pip install anthropic") from exc

        context = _build_context_block(chunks)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=GROUNDING_SYSTEM_PROMPT,
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

    def synthesize(self, query: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("Install openai package: pip install openai") from exc

        context = _build_context_block(chunks)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
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


def get_provider(provider_name: str | None = None) -> LLMProvider:
    """Factory: return the appropriate LLMProvider from env or explicit name."""
    name = (provider_name or os.environ.get("LLM_PROVIDER", "extractive")).lower()

    if name == "extractive":
        return ExtractiveProvider()

    if name == "ollama":
        return OllamaProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_MODEL", "biomistral"),
        )

    if name == "claude":
        return ClaudeProvider(
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        )

    if name == "openai":
        return OpenAIProvider(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )

    raise ValueError(f"Unknown LLM provider: {name!r}. Choose: extractive, ollama, claude, openai")
