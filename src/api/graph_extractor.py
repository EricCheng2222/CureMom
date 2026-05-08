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
            label = ent.entity_text.strip().rstrip(',.;:')
            # Minimal pre-filter — only obvious junk that wastes LLM tokens:
            # too short to mean anything, no alphabetic content, or ends in
            # a punctuation dangler from a bad span boundary. Real
            # canonicalization (folding "muscle" → "skeletal muscle",
            # dropping vague terms) is delegated to the LLM downstream.
            if len(label) < 3:
                continue
            if not any(ch.isalpha() for ch in label):
                continue
            if label.endswith('/') or label.endswith('-') or label.endswith(','):
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

_GRAPH_PROMPT = """You build a small biomedical knowledge graph from a question, an answer, and supporting evidence chunks.

Your job has TWO parts:

PART 1 — Canonicalize the entity list.
You will be given a noisy list of NER-extracted entity candidates. Some are useful (e.g. "skeletal muscle", "growth hormone", "satellite cells", "IGF-1"). Some are fragments or duplicates ("muscles", "muscle", "growth", "hormone", "IGF"). Some are too vague ("life", "type", "physical").

Produce a clean canonical entity list:
  - Pick a SHORT, specific canonical label for each real concept (e.g. "skeletal muscle", not "muscle").
  - Fold duplicates and fragments under that canonical label as "aliases" (e.g. canonical "skeletal muscle" with aliases ["muscle", "muscles"]).
  - DROP entities that are too vague to be a graph node ("life", "type", "physical", "mass", "training" alone, "diet" alone, generic adjectives).
  - DROP entities that don't appear (in some form) in the candidate list — do not invent.
  - Pick ONE type from: CHEMICAL, DISEASE, GENE_OR_GENE_PRODUCT, ANATOMY, SYMPTOM, PROCEDURE, CELL_TYPE, ORGANISM, OTHER.

PART 2 — Extract directed relations between canonical entities.
Look for every distinct mechanism stated in the answer or chunks. Each relation has a subject, a 1-3 word lowercase predicate (e.g. "stimulates", "inhibits", "treats", "causes", "activates", "regulates", "reduces", "promotes"), an object, and the chunk_ids that evidence it.

Output ONLY this JSON shape (no prose, no markdown fences):

{
  "entities": [
    {"label": "<canonical label>", "type": "<TYPE>", "aliases": ["<surface form>", ...]},
    ...
  ],
  "relations": [
    {"subject": "<canonical label>", "predicate": "<verb phrase>", "object": "<canonical label>", "evidence_chunk_ids": [<int>, ...]},
    ...
  ]
}

Rules:
  - Subject/object in `relations` MUST exactly match a `label` in your `entities` list.
  - Each relation MUST cite at least one chunk_id from the provided evidence chunks.
  - Aim for COVERAGE — emit every distinct mechanism. Chains are good: if A activates B and B promotes C, emit both triples.
  - Up to 25 entities and 18 relations. Skip vague or unsupported ones first.

Example (illustrative, do not copy):
Input candidates: ["muscle", "muscles", "skeletal muscle", "growth", "growth hormone", "GH", "IGF-1", "IGF", "liver", "life", "type"]
Output:
{"entities":[
  {"label":"growth hormone","type":"CHEMICAL","aliases":["GH"]},
  {"label":"IGF-1","type":"GENE_OR_GENE_PRODUCT","aliases":["IGF"]},
  {"label":"liver","type":"ANATOMY","aliases":[]},
  {"label":"skeletal muscle","type":"ANATOMY","aliases":["muscle","muscles"]}
 ],
 "relations":[
  {"subject":"growth hormone","predicate":"stimulates","object":"liver","evidence_chunk_ids":[101]},
  {"subject":"liver","predicate":"produces","object":"IGF-1","evidence_chunk_ids":[101]},
  {"subject":"IGF-1","predicate":"promotes","object":"skeletal muscle","evidence_chunk_ids":[101,102]}
 ]
}"""


def _ollama_graph(
    query: str,
    answer: str,
    candidate_entities: list[str],
    chunks: list[dict[str, Any]],
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Call Ollama with format=json and return the parsed {entities, relations}.

    Single combined call: the LLM canonicalizes the noisy NER candidate list
    AND emits relations between the canonical entities. Mirrors the pattern
    in src/api/query_analyzer.py.
    """
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_GRAPH_MODEL") or os.environ.get("OLLAMA_MODEL", "medgemma:4b")

    chunk_lines = []
    for c in chunks[:6]:
        text = (c.get("text") or "").replace("\n", " ").strip()
        if len(text) > 600:
            text = text[:600] + "…"
        chunk_lines.append(f"[chunk_id={c['id']}] {text}")

    user_msg = (
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"NER candidate entities ({len(candidate_entities)}): {', '.join(candidate_entities)}\n\n"
        "Evidence chunks:\n" + "\n".join(chunk_lines)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _GRAPH_PROMPT},
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

    return _parse_graph(raw)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_graph(raw: str) -> dict[str, Any]:
    m = _JSON_RE.search(raw)
    if not m:
        return {"entities": [], "relations": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}
    ents = obj.get("entities") if isinstance(obj.get("entities"), list) else []
    rels = obj.get("relations") if isinstance(obj.get("relations"), list) else []
    return {"entities": ents, "relations": rels}


# ─── Filter & assemble ───────────────────────────────────────────────────────

def _build_canonical_nodes(
    raw_entities: list[dict[str, Any]],
    ner_nodes: dict[str, GraphNode],
) -> tuple[list[GraphNode], dict[str, str]]:
    """Build canonical GraphNodes from the LLM's `entities` output.

    Each LLM entity must ground in NER — the canonical label OR at least
    one alias must (case-insensitively) match an existing NER label.
    Citations from all matching NER hits are unioned into the canonical
    node.

    Returns (canonical_nodes, label_to_id_map) where label_to_id_map
    keys are lowercase canonical labels AND aliases — used by edge
    resolution to map LLM relation endpoints to node ids.
    """
    # Build NER label → GraphNode lookup (case-insensitive)
    ner_by_label: dict[str, GraphNode] = {n.label.lower(): n for n in ner_nodes.values()}

    canonical: dict[str, GraphNode] = {}     # id -> GraphNode
    label_to_id: dict[str, str] = {}         # lowercase label/alias -> canonical id

    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        label = (e.get("label") or "").strip()
        if not label:
            continue
        etype = e.get("type") or "OTHER"
        if etype not in _KNOWN_TYPES:
            etype = "OTHER"
        aliases_raw = e.get("aliases") or []
        aliases = [str(a).strip() for a in aliases_raw if isinstance(a, (str, int))]

        # Grounding: canonical label or any alias must appear in NER set.
        candidates = [label] + aliases
        matching_ner_labels = [c for c in candidates if c.lower() in ner_by_label]
        if not matching_ner_labels:
            continue   # ungrounded — drop

        # Aggregate citations from all matching NER hits, plus accept the
        # LLM's chosen type even if NER labelled it differently (LLM has
        # more semantic context).
        citations: list[int] = []
        kb_id: str | None = None
        seen_cites: set[int] = set()
        for nlabel in matching_ner_labels:
            nnode = ner_by_label[nlabel.lower()]
            for c in nnode.citations:
                if c not in seen_cites:
                    citations.append(c)
                    seen_cites.add(c)
            if kb_id is None and nnode.kb_id:
                kb_id = nnode.kb_id

        nid = kb_id or _slugify(label)
        if nid in canonical:
            # Merge into existing
            existing = canonical[nid]
            seen_cites = set(existing.citations)
            for c in citations:
                if c not in seen_cites:
                    existing.citations.append(c)
            for alias in candidates:
                label_to_id[alias.lower()] = nid
        else:
            canonical[nid] = GraphNode(
                id=nid, label=label, type=etype,
                citations=citations, kb_id=kb_id,
            )
            for alias in candidates:
                label_to_id[alias.lower()] = nid

    return list(canonical.values()), label_to_id


def _build_edges(
    raw_relations: list[dict[str, Any]],
    label_to_id: dict[str, str],
    valid_chunk_ids: set[int],
) -> list[GraphEdge]:
    """Convert raw LLM triples into validated GraphEdge objects.

    Drops:
      * triples whose subject or object isn't in the canonical entity set
        (matched via label OR any alias, case-insensitively)
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

        s_id = label_to_id.get(subj)
        o_id = label_to_id.get(obj)
        if s_id is None or o_id is None or s_id == o_id:
            continue

        pred_words = pred.split()
        if len(pred_words) > 3:
            pred = " ".join(pred_words[:3])

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

        eid = f"{s_id}|{pred}|{o_id}"
        existing = edges.get(eid)
        if existing:
            for cid in evidence:
                if cid not in existing.citations:
                    existing.citations.append(cid)
        else:
            edges[eid] = GraphEdge(
                id=eid, source=s_id, target=o_id,
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

    ner_nodes = _gather_entities(query, answer, chunks)
    if not ner_nodes:
        logger.info("graph_extract: NER produced no entities; returning empty graph")
        return GraphPayload(nodes=[], edges=[])

    candidate_labels = sorted({n.label for n in ner_nodes.values()})
    valid_chunk_ids = {int(c["id"]) for c in chunks if "id" in c}

    try:
        graph_obj = _ollama_graph(
            query=query, answer=answer,
            candidate_entities=candidate_labels,
            chunks=chunks,
        )
    except Exception as exc:
        logger.warning("graph_extract: LLM call failed (%s); returning empty graph", exc)
        return GraphPayload(nodes=[], edges=[])

    raw_entities = graph_obj.get("entities", [])
    raw_relations = graph_obj.get("relations", [])

    # Build the canonical entity set the LLM picked. Each entity must
    # ground in NER (label or alias matches an NER hit), which prevents
    # the LLM from inventing concepts the source text never mentioned.
    canonical_nodes, label_to_id = _build_canonical_nodes(raw_entities, ner_nodes)

    edges = _build_edges(raw_relations, label_to_id, valid_chunk_ids)

    # Drop isolated nodes — only show entities that participate in some
    # mechanism. A solo node isn't useful in a knowledge graph.
    connected_ids: set[str] = set()
    for e in edges:
        connected_ids.add(e.source)
        connected_ids.add(e.target)
    kept_nodes = [n for n in canonical_nodes if n.id in connected_ids]

    logger.info(
        "graph_extract: NER=%d candidates → LLM=%d entities (%d kept after filter), %d relations (%d kept edges)",
        len(ner_nodes), len(canonical_nodes), len(kept_nodes),
        len(raw_relations), len(edges),
    )

    return GraphPayload(nodes=kept_nodes, edges=edges)
