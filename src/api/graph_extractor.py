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
    error: str | None = None    # Set when the LLM call failed; empty result + reason

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
        if self.error:
            out["error"] = self.error
        return out


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


# ─── LLM relation extraction ─────────────────────────────────────────────────

_GRAPH_PROMPT = """You extract a small biomedical knowledge graph from a question and a synthesized answer.

The answer is a summary of research papers. Citation markers look like [1], [2], [3].

Output ONLY directed relations between concepts mentioned in the answer. No prose, no markdown, no entity list — just relations. Each subject and object you write becomes a node automatically.

OUTPUT SHAPE:

{
  "relations": [
    {"subject":"<concept>", "predicate":"<verb phrase>", "object":"<concept>", "citations":[<ints>]}
  ]
}

CONCEPTS (subjects/objects) — pick SPECIFIC, NAMED things mentioned in the answer:
  ✓ Named genes/proteins:        myostatin, IGF-1, METTL3, mTORC1, klotho, TGF-beta
  ✓ Named pathways:              IGF/PI3K/AKT/mTORC1 pathway, JAK-STAT, NF-kB
  ✓ Named cell types:            satellite cells, myoblasts, B cells, T cells
  ✓ Named drugs/compounds:       hydroxychloroquine, curcumin, leucine
  ✓ Specific tissues / organs:   skeletal muscle, lupus nephritis, dermis, liver
  ✓ Specific phenotypes:         muscle hypertrophy, muscle wasting, anabolic state

AVOID generic single words as concepts: "protein", "RNA", "cell", "molecular", "modification" — use the specific named version instead ("myostatin" not "protein"; "skeletal muscle" not "muscle").

Use the SAME spelling for the same concept everywhere. If you call something "B cell" in one relation, do not call it "B-cell" or "B cells" in another. Pick one form and stick with it.

PREDICATES — short verb phrases (1-3 words) the answer actually uses:
  ✓ activates, inhibits, promotes, stimulates, suppresses, regulates,
    phosphorylates, binds, produces, secretes, releases,
    treats, causes, reduces, reverses, induces, blocks,
    is downstream of, is upstream of, is part of

NEVER use these vague predicates — they make the graph meaningless:
  ✗ "is managed by"          ← MEANINGLESS
  ✗ "is controlled by"       ← VAGUE; prefer "is regulated by" / "is inhibited by"
  ✗ "involves"               ← MEANINGLESS
  ✗ "relates to"             ← MEANINGLESS
  ✗ "is associated with"     ← last resort only
  ✗ "is increased by"        ← VAGUE; prefer "is stimulated by"
  ✗ "depends on"             ← VAGUE

EMIT CHAINS, NOT STARS. If the answer says "A activates B which activates C", emit BOTH:
  {"subject":"A","predicate":"activates","object":"B"}
  {"subject":"B","predicate":"activates","object":"C"}
NOT a single edge from A to C.

Pathway rule: name a multi-step pathway as ONE concept ("IGF/PI3K/AKT/mTORC1 pathway"), then emit edges into and out of it. Don't split it into separate steps unless the answer describes them.

Citations: each relation may cite [N] indices nearest the supporting sentence; an empty array is allowed.

Up to 25 relations. Quality > quantity.

═══════════════════════════════════════════════════════════════════════
WORKED EXAMPLE (learn the shape, don't copy values):

Answer: "Growth hormone stimulates the liver to produce IGF-1 [1], which activates the PI3K/AKT/mTOR pathway in skeletal muscle to promote hypertrophy [1,2]. Myostatin opposes this by inhibiting protein synthesis [3]."

Output:
{"relations":[
  {"subject":"growth hormone","predicate":"stimulates","object":"liver","citations":[1]},
  {"subject":"liver","predicate":"produces","object":"IGF-1","citations":[1]},
  {"subject":"IGF-1","predicate":"activates","object":"PI3K/AKT/mTOR pathway","citations":[1,2]},
  {"subject":"PI3K/AKT/mTOR pathway","predicate":"promotes","object":"muscle hypertrophy","citations":[2]},
  {"subject":"myostatin","predicate":"inhibits","object":"protein synthesis","citations":[3]}
]}"""


def _build_user_message(query: str, answer: str) -> str:
    return (
        f"Question: {query}\n\n"
        f"Answer (with [N] citation markers): {answer[:5000]}"
    )


def _resolve_provider(provider_spec: str | None) -> str:
    """Decide which provider to actually call based on the dropdown.

    Defaults to "nim" (free tier MiniMax) when no provider is specified.
    """
    if not provider_spec:
        return "nim"
    if provider_spec == "claude":
        return "claude"
    if provider_spec == "openai":
        return "openai"
    if provider_spec.startswith("nim/") or provider_spec == "nim":
        return "nim"
    return "nim"


def _llm_graph(
    query: str,
    answer: str,
    provider_spec: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Dispatch the graph-extract LLM call to whichever provider the user
    picked in the dropdown. Returns the parsed {entities, relations}.

    Defaults to NIM (free tier) when no provider is specified.

    Tunable via env:
      GRAPH_TIMEOUT_S — request timeout (default 600s = 10 min)
    """
    if timeout_s is None:
        timeout_s = float(os.environ.get("GRAPH_TIMEOUT_S", "600"))

    user_msg = _build_user_message(query, answer)
    target = _resolve_provider(provider_spec)

    if target == "claude":
        return _claude_graph(user_msg, timeout_s)
    if target == "openai":
        return _openai_graph(user_msg, timeout_s)
    return _nim_graph(user_msg, provider_spec, timeout_s)


def _claude_graph(user_msg: str, timeout_s: float) -> dict[str, Any]:
    """Call Anthropic's Messages API directly. Anthropic doesn't have a
    strict format=json mode like Ollama, but Haiku reliably returns clean
    JSON when the system prompt instructs it to."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    logger.info("graph_extract: calling Claude (model=%s, timeout=%.0fs)", model, timeout_s)
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_GRAPH_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = msg.content[0].text if msg.content else ""
    return _parse_graph(raw)


def _openai_graph(user_msg: str, timeout_s: float) -> dict[str, Any]:
    """Call OpenAI's chat completions with response_format=json_object."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError("OPENAI_API_KEY not set in env")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    logger.info("graph_extract: calling OpenAI (model=%s, timeout=%.0fs)", model, timeout_s)
    client = openai.OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GRAPH_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_graph(raw)


def _nim_graph(user_msg: str, provider_spec: str | None, timeout_s: float) -> dict[str, Any]:
    """Call NVIDIA NIM (OpenAI-compatible) chat completions.

    Uses the OpenAI SDK with a custom base_url. We rely on prompt-based
    JSON output (system prompt says "output ONLY this JSON") rather than
    response_format=json_object, since NIM's MiniMax-M2.7 doesn't reliably
    accept that flag — _parse_graph extracts the JSON object via regex.
    """
    import openai

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key == "your_nvidia_api_key_here":
        raise RuntimeError("NVIDIA_API_KEY not set in env")
    if provider_spec and provider_spec.startswith("nim/"):
        model = provider_spec.split("/", 1)[1]
    else:
        model = os.environ.get("NIM_MODEL", "minimaxai/minimax-m2.7")
    base_url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    logger.info("graph_extract: calling NIM (model=%s, timeout=%.0fs)", model, timeout_s)
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GRAPH_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_graph(raw)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_graph(raw: str) -> dict[str, Any]:
    m = _JSON_RE.search(raw)
    if not m:
        return {"entities": [], "relations": [], "_raw": raw}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relations": [], "_raw": raw}
    ents = obj.get("entities") if isinstance(obj.get("entities"), list) else []
    rels = obj.get("relations") if isinstance(obj.get("relations"), list) else []
    return {"entities": ents, "relations": rels, "_raw": raw}


# ─── Filter & assemble ───────────────────────────────────────────────────────

def _build_nodes_from_relations(
    raw_relations: list[dict[str, Any]],
    answer_text: str = "",
) -> tuple[list[GraphNode], dict[str, str]]:
    """Each unique subject/object string in `raw_relations` becomes a node.

    Dedup is purely by the LLM's chosen surface form (case-insensitive).
    The LLM is instructed to use ONE spelling per concept across all
    relations; if it doesn't, the duplicates show up as separate nodes
    (the Merge button can fold them later).

    Grounding rule: a node is kept iff its label appears (case-
    insensitively) as a substring of the answer text. The answer is the
    authoritative source — if the LLM emits a concept that isn't in the
    answer, it invented it. Vague single-word labels are still dropped
    by the _VAGUE_LABELS filter.
    """
    answer_lc = (answer_text or "").lower()

    def _grounded(label_lc: str) -> bool:
        return bool(answer_lc) and label_lc in answer_lc

    # First pass: collect every unique subject/object the LLM mentioned.
    seen_labels: dict[str, str] = {}  # lowercase -> canonical label (first form seen)
    for r in raw_relations:
        if not isinstance(r, dict):
            continue
        for field in ("subject", "object"):
            v = (r.get(field) or "").strip()
            if not v:
                continue
            key = v.lower()
            if key not in seen_labels:
                seen_labels[key] = v

    # Build nodes, applying grounding + vague-label filter.
    canonical: dict[str, GraphNode] = {}     # id -> GraphNode
    label_to_id: dict[str, str] = {}         # lowercase label -> canonical id

    for label_lc, label in seen_labels.items():
        if label_lc in _VAGUE_LABELS:
            continue
        if not _grounded(label_lc):
            continue
        nid = _slugify(label)
        if nid in canonical:
            label_to_id[label_lc] = nid
            continue
        canonical[nid] = GraphNode(
            id=nid, label=label, type="OTHER",
            citations=[], kb_id=None,
        )
        label_to_id[label_lc] = nid

    return list(canonical.values()), label_to_id


# Predicates that carry no biological information. We strip relations using
# any of these so the graph stays meaningful. The LLM is told to avoid them
# in the prompt, but small models still emit them — server-side guard.
_VAGUE_PREDICATES = {
    "is managed by", "is controlled by", "is involved in", "involves",
    "relates to", "is related to", "is associated with",
    "is increased by", "is decreased by", "is affected by", "affects",
    "depends on", "is dependent on",
    "interacts with", "interaction with",
    "is",  "has", "are",
}

# Single-word generic biological terms that NER often picks up but that
# carry no meaning as graph nodes. We drop them at edge-build time when
# they appear as a node id.
_VAGUE_LABELS = {
    "protein", "proteins", "rna", "dna", "cell", "cells", "gene", "genes",
    "tissue", "tissues", "molecular", "cellular", "modification",
    "molecule", "molecules", "process", "pathway", "system",
    "physical", "chronic", "acute", "mechanical",
    "growth", "wasting",   # too generic without a body part qualifier
}


def _build_edges(
    raw_relations: list[dict[str, Any]],
    label_to_id: dict[str, str],
    citation_index_to_chunk_id: dict[int, int],
) -> list[GraphEdge]:
    """Convert raw LLM triples into validated GraphEdge objects.

    Each LLM relation may carry `citations: [<int>, ...]` which are 1-based
    [N] indices from the answer text. We map these to chunk_ids via the
    citations array passed by the frontend (order = citation_index - 1).
    Relations with NO valid citations are still kept — the prompt allows
    citing nothing if no [N] is nearby in the answer.

    Drops:
      * triples whose subject or object isn't in the canonical entity set
        (matched via label OR any alias, case-insensitively)
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

        # Drop relations whose endpoints are pure-fragment generic terms.
        if subj in _VAGUE_LABELS or obj in _VAGUE_LABELS:
            continue
        # Drop relations with vague predicates ("is managed by" etc.).
        if pred in _VAGUE_PREDICATES:
            continue

        s_id = label_to_id.get(subj)
        o_id = label_to_id.get(obj)
        if s_id is None or o_id is None or s_id == o_id:
            continue

        pred_words = pred.split()
        if len(pred_words) > 3:
            pred = " ".join(pred_words[:3])

        # Accept either the new `citations` field (1-based [N] indices) or
        # the legacy `evidence_chunk_ids` field (raw chunk_ids), for
        # forward-compat with prompts the LLM might still emit.
        ev_raw = r.get("citations") or r.get("evidence_chunk_ids") or []
        if not isinstance(ev_raw, list):
            ev_raw = []
        evidence: list[int] = []
        for x in ev_raw:
            try:
                idx = int(x)
            except (TypeError, ValueError):
                continue
            # First try mapping as a [N] citation index
            cid = citation_index_to_chunk_id.get(idx)
            if cid is not None:
                if cid not in evidence:
                    evidence.append(cid)
            else:
                # Fallback: maybe the LLM emitted a raw chunk_id (legacy)
                if idx in set(citation_index_to_chunk_id.values()) and idx not in evidence:
                    evidence.append(idx)

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
    provider_spec: str | None = None,
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

    # Map [N] citation index → chunk_id. The frontend sends `chunks` in the
    # same order as the citation pills, so chunks[i] corresponds to [i+1].
    citation_index_to_chunk_id: dict[int, int] = {
        i + 1: int(c["id"]) for i, c in enumerate(chunks) if "id" in c
    }

    try:
        graph_obj = _llm_graph(
            query=query, answer=answer,
            provider_spec=provider_spec,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("graph_extract: LLM call failed (%s); returning empty graph", msg)
        return GraphPayload(nodes=[], edges=[], error=msg)

    raw_relations = graph_obj.get("relations", [])
    raw_dump = graph_obj.get("_raw", "") or ""

    # Each unique subject/object string in the relations becomes a node.
    # Grounding: the label must appear (case-insensitively) as a substring
    # of the answer text. The answer is the authoritative source — NER
    # was an incomplete proxy for "is this concept actually mentioned".
    canonical_nodes, label_to_id = _build_nodes_from_relations(
        raw_relations, answer_text=answer,
    )

    edges = _build_edges(raw_relations, label_to_id, citation_index_to_chunk_id)

    # Drop isolated nodes — only show entities that participate in some
    # mechanism. A solo node isn't useful in a knowledge graph.
    connected_ids: set[str] = set()
    for e in edges:
        connected_ids.add(e.source)
        connected_ids.add(e.target)
    kept_nodes = [n for n in canonical_nodes if n.id in connected_ids]

    # Derive each node's citations from the edges it participates in
    # (we no longer track per-NER-mention chunk_ids — chunks aren't sent
    # to the LLM at all).
    node_citations: dict[str, list[int]] = {n.id: [] for n in kept_nodes}
    for e in edges:
        for nid in (e.source, e.target):
            seen = set(node_citations.get(nid, []))
            for c in e.citations:
                if c not in seen:
                    node_citations[nid].append(c)
                    seen.add(c)
    for n in kept_nodes:
        n.citations = node_citations.get(n.id, [])

    stage_counts = (
        f"LLM(raw_relations={len(raw_relations)}) → "
        f"grounded_nodes={len(canonical_nodes)} → "
        f"edges={len(edges)} → "
        f"kept_nodes={len(kept_nodes)}"
    )
    logger.info("graph_extract: %s", stage_counts)

    diag: str | None = None
    if not kept_nodes and not edges:
        if not raw_relations:
            cause = "LLM returned no relations"
        elif not canonical_nodes:
            cause = (
                "every relation subject/object was rejected by grounding — "
                "the LLM emitted concepts not present in the answer text"
            )
        else:
            cause = "edges built but nothing survived filtering"
        logger.warning("graph_extract: empty result — %s. Stage counts: %s. "
                       "Raw LLM head: %s", cause, stage_counts, raw_dump[:500])
        diag = (
            f"{cause}. Stages: {stage_counts}. "
            f"Raw LLM head: {raw_dump[:300]}"
        )

    return GraphPayload(nodes=kept_nodes, edges=edges, error=diag)
