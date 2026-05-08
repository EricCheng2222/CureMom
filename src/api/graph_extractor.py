"""On-the-fly knowledge-graph extraction for the chat panel.

Each Q&A turn yields a small graph of entities + LLM-asserted relations.
Nodes come from biomedical NER over the question, the answer, and the
cited chunk texts. Edges come from a second LLM pass that emits JSON
triples like {subject, predicate, object, evidence_chunk_ids}.

Design constraints (matched to user's stated goal — visually grasp how
GH affects muscle growth across turns):
  * Edges must carry meaningful predicates (not just co-occurrence).
  * Edges that name entities NOT in the NER set are dropped — keeps the
    LLM from inventing new entities and hallucinating links.
  * Edges with no supporting chunk_ids are dropped — every assertion
    must point at something in the cited evidence.

The frontend merges per-turn payloads into a session-local graph state,
so this module returns just what changed in the current turn.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ─── Public payload shape ────────────────────────────────────────────────────

@dataclass
class GraphNode:
    id: str               # stable across turns: kb_id if present, else slug(label)
    label: str            # human-readable name
    type: str             # DISEASE / CHEMICAL / GENE_OR_GENE_PRODUCT / ANATOMY / SYMPTOM / PROCEDURE / OTHER
    citations: list[int] = field(default_factory=list)   # chunk_ids that mentioned this entity
    kb_id: str | None = None


@dataclass
class GraphEdge:
    id: str               # f"{source}|{predicate}|{target}"
    source: str
    target: str
    predicate: str
    citations: list[int] = field(default_factory=list)


@dataclass
class GraphPayload:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "label": n.label, "type": n.type,
                 "citations": n.citations, "kb_id": n.kb_id}
                for n in self.nodes
            ],
            "edges": [
                {"id": e.id, "source": e.source, "target": e.target,
                 "predicate": e.predicate, "citations": e.citations}
                for e in self.edges
            ],
        }


# ─── Slug helper for stable node ids ─────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")

def _slugify(text: str) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s or "node"


# ─── NER over query + answer + chunks ────────────────────────────────────────

# Map LABEL → canonical label kept by the chunk pipeline. We accept the
# already-collapsed labels from src.embeddings.ner_pipeline.LABEL_MAP plus
# a couple of fallbacks.
_KNOWN_TYPES = {
    "DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT",
    "ANATOMY", "SYMPTOM", "PROCEDURE", "CELL_TYPE", "ORGANISM",
}


def _gather_entities(
    query: str,
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, GraphNode]:
    """Run NER over the question, answer, and each chunk text.

    Returns a dict keyed by node id (slug or kb_id) so duplicates merge
    naturally and citations accumulate per node.
    """
    # Lazy import — keeps module import cheap, mirrors hipporag.extract_query_entities.
    from ..search.hipporag import extract_query_entities  # noqa: F401  (warms model cache)
    from ..embeddings.ner_pipeline import _NERRunner

    # Reuse hipporag's cached runner if it already exists; otherwise create a
    # local one. extract_query_entities() initialises and caches the runner
    # as a side-effect, so call it once on the query to ensure the singleton
    # is loaded.
    extract_query_entities(query)  # warms hipporag._query_ner_runner
    from ..search import hipporag as _hr
    runner: _NERRunner = _hr._query_ner_runner  # type: ignore[attr-defined]

    by_id: dict[str, GraphNode] = {}

    def _add(text: str, source_chunk_id: int) -> None:
        # paper_id/chunk_id are bookkeeping for the chunk pipeline; here we
        # only need the spans. Pass 0 for paper_id; pass the actual chunk_id
        # so we can attribute citations.
        for ent in runner.extract(text, paper_id=0, chunk_id=source_chunk_id):
            label = ent.entity_text.strip()
            if len(label) < 3:
                continue
            etype = ent.entity_type if ent.entity_type in _KNOWN_TYPES else "OTHER"
            nid = ent.kb_id or _slugify(label)
            node = by_id.get(nid)
            if node is None:
                by_id[nid] = GraphNode(
                    id=nid, label=label, type=etype, kb_id=ent.kb_id,
                    citations=[source_chunk_id] if source_chunk_id > 0 else [],
                )
            else:
                # Prefer the longer / better-cased label seen so far.
                if len(label) > len(node.label):
                    node.label = label
                if source_chunk_id > 0 and source_chunk_id not in node.citations:
                    node.citations.append(source_chunk_id)

    # 0 = "from query/answer, no specific chunk attribution"
    _add(query, 0)
    _add(answer, 0)
    for c in chunks:
        cid = int(c["id"])
        text = c.get("text") or ""
        if text:
            _add(text, cid)

    return by_id


# ─── LLM relation extraction ─────────────────────────────────────────────────

_RELATIONS_PROMPT = """You extract directed relations between biomedical entities.

You will be given:
  * a question
  * an answer that was synthesized from research papers
  * a list of allowed entities (you may ONLY use these as subjects/objects)
  * a list of evidence chunks, each with an id and text excerpt

Output ONLY a JSON object with a single key "relations" whose value is an array of triples:
  {
    "relations": [
      {"subject": "<entity from allowed list>",
       "predicate": "<1-3 words, lowercase: stimulates, inhibits, treats, causes, activates, regulates, is downstream of, ...>",
       "object": "<entity from allowed list>",
       "evidence_chunk_ids": [<one or more chunk ids that support this triple>]
      }
    ]
  }

Rules:
  - Both subject and object MUST appear verbatim (case-insensitive) in the allowed entities list.
  - Predicate must be a short verb phrase (max 3 words). Prefer biological mechanism verbs.
  - Every relation must cite at least one chunk id from the evidence list.
  - Do NOT invent new entities. Do NOT include relations you cannot evidence.
  - Return at most 12 relations. Skip the noisiest ones first.

Return ONLY the JSON object — no prose, no markdown fences."""


def _ollama_relations(
    query: str,
    answer: str,
    allowed_entities: list[str],
    chunks: list[dict[str, Any]],
    timeout_s: float = 45.0,
) -> list[dict[str, Any]]:
    """Call Ollama with format=json and return the parsed `relations` list.

    Mirrors the pattern in src/api/query_analyzer.py — same client, same
    JSON-mode flag, same timeout discipline.
    """
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_GRAPH_MODEL") or os.environ.get("OLLAMA_MODEL", "medgemma:4b")

    # Build the user payload. Keep chunk excerpts short — relation extraction
    # works fine on the first ~600 chars and saves a lot of tokens on long
    # full-text chunks.
    chunk_lines = []
    for c in chunks[:6]:
        text = (c.get("text") or "").replace("\n", " ").strip()
        if len(text) > 600:
            text = text[:600] + "…"
        chunk_lines.append(f"[chunk_id={c['id']}] {text}")

    user_msg = (
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Allowed entities ({len(allowed_entities)}): {', '.join(allowed_entities)}\n\n"
        "Evidence chunks:\n" + "\n".join(chunk_lines)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _RELATIONS_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "format": "json",
        "options": {"num_ctx": 8192, "temperature": 0.0},
    }

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        raw = r.json()["message"]["content"]

    return _parse_relations(raw)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_relations(raw: str) -> list[dict[str, Any]]:
    m = _JSON_RE.search(raw)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    rels = obj.get("relations") or []
    if not isinstance(rels, list):
        return []
    return rels


# ─── Filter & assemble ───────────────────────────────────────────────────────

def _build_edges(
    raw_relations: list[dict[str, Any]],
    nodes_by_label: dict[str, GraphNode],
    valid_chunk_ids: set[int],
) -> list[GraphEdge]:
    """Convert raw LLM triples into validated GraphEdge objects.

    Drops:
      * triples whose subject or object isn't in the NER node set
      * triples with no supporting chunk_ids in the provided set
      * exact duplicates (same source|predicate|target)
    """
    edges: dict[str, GraphEdge] = {}

    for r in raw_relations:
        if not isinstance(r, dict):
            continue
        subj = (r.get("subject") or "").strip().lower()
        obj  = (r.get("object")  or "").strip().lower()
        pred = (r.get("predicate") or "").strip().lower()
        if not subj or not obj or not pred or subj == obj:
            continue

        # Resolve to node ids by case-insensitive label match.
        s_node = nodes_by_label.get(subj)
        o_node = nodes_by_label.get(obj)
        if s_node is None or o_node is None:
            continue

        # Trim predicate to at most 3 words so the edge label stays readable.
        pred_words = pred.split()
        if len(pred_words) > 3:
            pred = " ".join(pred_words[:3])

        # Validate evidence: at least one chunk id must be in the request set.
        ev_raw = r.get("evidence_chunk_ids") or []
        if not isinstance(ev_raw, list):
            continue
        evidence: list[int] = []
        for x in ev_raw:
            try:
                cid = int(x)
            except (TypeError, ValueError):
                continue
            if cid in valid_chunk_ids:
                evidence.append(cid)
        if not evidence:
            continue

        eid = f"{s_node.id}|{pred}|{o_node.id}"
        existing = edges.get(eid)
        if existing:
            for cid in evidence:
                if cid not in existing.citations:
                    existing.citations.append(cid)
        else:
            edges[eid] = GraphEdge(
                id=eid, source=s_node.id, target=o_node.id,
                predicate=pred, citations=evidence,
            )

    return list(edges.values())


# ─── Public entry point ──────────────────────────────────────────────────────

def extract_graph(
    query: str,
    answer: str,
    chunks: list[dict[str, Any]],
) -> GraphPayload:
    """Extract a per-turn knowledge-graph payload.

    `chunks` is a list of {"id": int, "text": str} as returned by the
    frontend after a /api/v1/query response (one entry per cited chunk).

    The returned payload contains every entity found by NER and every
    LLM-asserted relation that survived the filters. It does NOT compute
    new_node_ids / new_edge_ids — the frontend merges with its own
    session-local graphState and tracks novelty there.
    """
    if not query or not answer:
        return GraphPayload(nodes=[], edges=[])

    nodes_by_id = _gather_entities(query, answer, chunks)
    if not nodes_by_id:
        logger.info("graph_extract: NER produced no entities; returning empty graph")
        return GraphPayload(nodes=[], edges=[])

    # For LLM filtering we match by lowercased label. If two NER spans
    # collapse to the same id but have different label cases, the longer
    # label won — so this dict is unambiguous for matching.
    nodes_by_label: dict[str, GraphNode] = {n.label.lower(): n for n in nodes_by_id.values()}
    allowed_labels = sorted({n.label for n in nodes_by_id.values()})

    valid_chunk_ids = {int(c["id"]) for c in chunks if "id" in c}

    try:
        raw_relations = _ollama_relations(
            query=query, answer=answer,
            allowed_entities=allowed_labels,
            chunks=chunks,
        )
    except Exception as exc:
        logger.warning("graph_extract: LLM relation call failed (%s); returning nodes only", exc)
        raw_relations = []

    edges = _build_edges(raw_relations, nodes_by_label, valid_chunk_ids)

    logger.info(
        "graph_extract: %d entities, %d raw relations, %d kept edges",
        len(nodes_by_id), len(raw_relations), len(edges),
    )

    return GraphPayload(nodes=list(nodes_by_id.values()), edges=edges)
