"""HippoRAG-style entity graph + Personalized PageRank retrieval.

The entity_graph table stores undirected edges weighted by how often two
entities co-occur in the same chunk. At query time:

  1. Extract query entities (scispaCy on the question).
  2. Run Personalized PageRank seeded at those query entities.
  3. Each chunk gets a score = sum of PPR scores of entities it contains.
  4. Boost chunks accordingly and merge with BM25/dense rankings.

This excels at multi-hop questions ("drugs affecting complement pathway in
SLE patients with nephritis?") because PPR will surface entities that
co-occur with both 'complement', 'SLE', and 'nephritis' even if no single
chunk mentions all three.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


@dataclass
class EntityHit:
    text: str
    entity_type: str | None = None
    kb_id: str | None = None


def build_mesh_graph(conn: psycopg.Connection, batch_size: int = 5000) -> int:
    """Build the entity graph from MeSH co-occurrence on shared papers.

    This is a NER-free shortcut: MeSH descriptors are already curated,
    normalized biomedical entities attached to every PubMed paper. Two
    descriptors get an edge if they co-occur on the same paper; weight is
    the number of papers they share.

    Use this to bootstrap HippoRAG immediately instead of waiting for
    scispaCy NER. Run scispaCy later for finer-grained entities (chemicals
    / specific gene products that aren't in MeSH).
    """
    logger.info("Building entity graph from MeSH co-occurrences…")

    with conn.cursor() as cur:
        cur.execute("DELETE FROM entity_graph")

    # Aggregate MeSH pairs per paper, then count over all papers
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_graph (entity_a, entity_b, co_occurrences, paper_ids)
            SELECT
                LEAST(LOWER(m1.descriptor_name), LOWER(m2.descriptor_name)) AS entity_a,
                GREATEST(LOWER(m1.descriptor_name), LOWER(m2.descriptor_name)) AS entity_b,
                COUNT(DISTINCT pm1.paper_id) AS co_occurrences,
                ARRAY_AGG(DISTINCT pm1.paper_id) AS paper_ids
            FROM paper_mesh pm1
            JOIN paper_mesh pm2 ON pm1.paper_id = pm2.paper_id AND pm1.mesh_id < pm2.mesh_id
            JOIN mesh_terms m1 ON pm1.mesh_id = m1.id
            JOIN mesh_terms m2 ON pm2.mesh_id = m2.id
            WHERE m1.descriptor_name IS NOT NULL
              AND m2.descriptor_name IS NOT NULL
            GROUP BY entity_a, entity_b
            HAVING COUNT(DISTINCT pm1.paper_id) >= 2
            ON CONFLICT (entity_a, entity_b) DO UPDATE
                SET co_occurrences = EXCLUDED.co_occurrences,
                    paper_ids = EXCLUDED.paper_ids
            """
        )
        inserted = cur.rowcount
    conn.commit()
    logger.info("MeSH graph built: %d edges", inserted)
    return inserted


def merge_ner_into_graph(conn: psycopg.Connection) -> int:
    """Merge paper_entities co-occurrences into the existing entity_graph.

    Use this AFTER build_mesh_graph() if you want both MeSH and NER edges
    in one graph. Edges that already exist (from MeSH) are augmented with
    NER co-occurrence counts; new NER-only edges are inserted.
    """
    logger.info("Merging NER entities into entity_graph…")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, paper_id, LOWER(entity_text) AS entity
            FROM paper_entities
            WHERE chunk_id IS NOT NULL
            """
        )
        rows = cur.fetchall()
    if not rows:
        logger.warning("paper_entities is empty — nothing to merge.")
        return 0

    by_chunk: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_id"]].append(r)

    upserts = 0
    with conn.cursor() as cur:
        for chunk_id, ents in by_chunk.items():
            unique = list({e["entity"]: e["paper_id"] for e in ents}.items())
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    a, b = sorted([unique[i][0], unique[j][0]])
                    if a == b:
                        continue
                    cur.execute(
                        """
                        INSERT INTO entity_graph (entity_a, entity_b, co_occurrences, paper_ids)
                        VALUES (%s, %s, 1, ARRAY[%s,%s]::bigint[])
                        ON CONFLICT (entity_a, entity_b) DO UPDATE
                            SET co_occurrences = entity_graph.co_occurrences + 1,
                                paper_ids = (
                                    SELECT ARRAY(SELECT DISTINCT unnest(entity_graph.paper_ids || EXCLUDED.paper_ids))
                                )
                        """,
                        (a, b, unique[i][1], unique[j][1]),
                    )
                    upserts += 1
    conn.commit()
    return upserts


def update_entity_graph_for_papers(
    conn: psycopg.Connection,
    paper_ids: list[int],
) -> int:
    """Incrementally upsert entity_graph edges for a small set of papers.

    Cost is roughly O(papers × entities_per_chunk²). Use this from the
    ingestion pipeline after each paper finishes NER, instead of doing a full
    rebuild. Returns the number of edge upserts.
    """
    if not paper_ids:
        return 0

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, paper_id, LOWER(entity_text) AS entity
            FROM paper_entities
            WHERE paper_id = ANY(%s) AND chunk_id IS NOT NULL
            """,
            (paper_ids,),
        )
        rows = cur.fetchall()

    by_chunk: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_id"]].append(r)

    upserts = 0
    with conn.cursor() as cur:
        for chunk_id, ents in by_chunk.items():
            unique = list({e["entity"]: e["paper_id"] for e in ents}.items())
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    a, b = sorted([unique[i][0], unique[j][0]])
                    if a == b:
                        continue
                    cur.execute(
                        """
                        INSERT INTO entity_graph (entity_a, entity_b, co_occurrences, paper_ids)
                        VALUES (%s, %s, 1, ARRAY[%s,%s]::bigint[])
                        ON CONFLICT (entity_a, entity_b) DO UPDATE
                            SET co_occurrences = entity_graph.co_occurrences + 1,
                                paper_ids = (
                                    SELECT ARRAY(SELECT DISTINCT unnest(entity_graph.paper_ids || EXCLUDED.paper_ids))
                                )
                        """,
                        (a, b, unique[i][1], unique[j][1]),
                    )
                    upserts += 1
    conn.commit()
    return upserts


def build_entity_graph(conn: psycopg.Connection, batch_size: int = 5000) -> int:
    """Populate `entity_graph` from `paper_entities` co-occurrences.

    Two entities are 'connected' when they appear in the same chunk. Weight
    is the number of chunks they co-appear in.

    Returns number of edges (rows in entity_graph after rebuild).
    """
    logger.info("Building entity graph from paper_entities…")

    with conn.cursor() as cur:
        cur.execute("DELETE FROM entity_graph")

    # Normalize entity text to lowercase for grouping
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, paper_id, LOWER(entity_text) AS entity, entity_type
            FROM paper_entities
            WHERE chunk_id IS NOT NULL
            ORDER BY chunk_id
            """
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning("paper_entities is empty — run scripts/extract_entities.py first.")
        return 0

    by_chunk: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_id"]].append(r)

    # Edge counts and paper provenance
    edges: dict[tuple[str, str], dict] = {}
    for chunk_id, ents in by_chunk.items():
        # Dedupe entities within a chunk
        seen: dict[str, int] = {}
        for e in ents:
            seen.setdefault(e["entity"], e["paper_id"])
        unique = list(seen.items())
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a, b = sorted([unique[i][0], unique[j][0]])
                if a == b:
                    continue
                key = (a, b)
                slot = edges.setdefault(key, {"count": 0, "papers": set()})
                slot["count"] += 1
                slot["papers"].add(unique[i][1])
                slot["papers"].add(unique[j][1])

    logger.info("Inserting %d edges…", len(edges))
    with conn.cursor() as cur:
        items = list(edges.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            cur.executemany(
                """
                INSERT INTO entity_graph (entity_a, entity_b, co_occurrences, paper_ids)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (entity_a, entity_b) DO UPDATE
                    SET co_occurrences = EXCLUDED.co_occurrences,
                        paper_ids = EXCLUDED.paper_ids
                """,
                [(a, b, slot["count"], list(slot["papers"])) for (a, b), slot in batch],
            )
    conn.commit()
    return len(edges)


class HippoRAGRetriever:
    """In-memory entity graph + PPR retrieval."""

    def __init__(self, db_dsn: str, min_edge_weight: int = 2) -> None:
        self._db_dsn = db_dsn
        self._min_edge_weight = min_edge_weight
        self._graph = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("Install networkx: pip install networkx") from exc

        logger.info("Loading entity graph into memory…")
        self._graph = nx.Graph()
        with psycopg.connect(self._db_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT entity_a, entity_b, co_occurrences FROM entity_graph "
                    "WHERE co_occurrences >= %s",
                    (self._min_edge_weight,),
                )
                for a, b, w in cur.fetchall():
                    self._graph.add_edge(a, b, weight=w)
        logger.info(
            "Loaded entity graph: %d nodes, %d edges",
            self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )
        self._loaded = True

    def reload(self) -> None:
        """Force a reload of the in-memory graph (after a rebuild)."""
        self._loaded = False
        self._graph = None

    def personalized_pagerank(
        self,
        query_entities: list[str],
        damping: float = 0.85,
        max_iter: int = 100,
        top_n: int = 50,
    ) -> dict[str, float]:
        """Run PPR seeded at query entities. Returns {entity: score}."""
        import networkx as nx
        self._ensure_loaded()
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return {}

        seed_lower = [e.lower() for e in query_entities]
        valid_seeds = [e for e in seed_lower if e in self._graph]
        if not valid_seeds:
            return {}

        personalization = {n: 0.0 for n in self._graph.nodes}
        for s in valid_seeds:
            personalization[s] = 1.0

        try:
            scores = nx.pagerank(
                self._graph,
                alpha=damping,
                personalization=personalization,
                max_iter=max_iter,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR did not converge; returning best-effort partial scores.")
            scores = nx.pagerank(self._graph, alpha=damping, personalization=personalization, max_iter=max_iter, tol=1e-3, weight="weight")

        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n])

    def rerank_chunks(
        self,
        chunk_ids: list[int],
        query_entities: list[str],
        boost_weight: float = 1.0,
    ) -> dict[int, float]:
        """Score chunks by sum of PPR over their entities. Returns {chunk_id: boost}."""
        ppr = self.personalized_pagerank(query_entities)
        if not ppr or not chunk_ids:
            return {cid: 0.0 for cid in chunk_ids}

        with psycopg.connect(self._db_dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT chunk_id, LOWER(entity_text) AS entity FROM paper_entities "
                    "WHERE chunk_id = ANY(%s)",
                    (chunk_ids,),
                )
                rows = cur.fetchall()

        by_chunk: dict[int, set[str]] = defaultdict(set)
        for r in rows:
            by_chunk[r["chunk_id"]].add(r["entity"])

        return {
            cid: boost_weight * sum(ppr.get(e, 0.0) for e in by_chunk.get(cid, set()))
            for cid in chunk_ids
        }


def extract_query_entities(query: str) -> list[str]:
    """Extract entities from a query using scispaCy. Returns lowercased entity strings.

    Lazy-imports scispaCy so this module can still be imported when scispaCy
    isn't installed (graph operations work without NER).
    """
    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
    except Exception as exc:
        logger.warning("scispaCy unavailable for query NER (%s); falling back to keywords.", exc)
        return _fallback_keywords(query)

    doc = nlp(query)
    return [ent.text.lower() for ent in doc.ents]


def _fallback_keywords(query: str) -> list[str]:
    """Cheap fallback when scispaCy isn't present — returns content words."""
    import re
    stop = {"the", "is", "are", "and", "or", "of", "in", "on", "for", "to",
            "a", "an", "what", "how", "why", "does", "do", "with", "by"}
    return [w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", query.lower())
            if w not in stop]
