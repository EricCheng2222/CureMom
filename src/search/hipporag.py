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

import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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


def merge_ner_into_graph(conn: psycopg.Connection, batch_size: int = 10000) -> int:
    """Merge paper_entities co-occurrences into the existing entity_graph.

    Aggregates counts in Python first (so each (a,b) pair only requires one
    DB upsert), then bulk-upserts in batches.

    Use this AFTER build_mesh_graph() to add fine-grained NER entities on
    top of the MeSH skeleton.
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
    logger.info("Loaded %d entity rows; aggregating pairs in memory…", len(rows))

    # Group by chunk, then enumerate pairs and aggregate counts + paper sets
    by_chunk: dict[int, dict[str, int]] = defaultdict(dict)
    for r in rows:
        # dedupe entity text within a chunk
        by_chunk[r["chunk_id"]].setdefault(r["entity"], r["paper_id"])

    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    pair_papers: dict[tuple[str, str], set[int]] = defaultdict(set)
    for ents in by_chunk.values():
        items = list(ents.items())
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                ea, pa = items[i]
                eb, pb = items[j]
                a, b = (ea, eb) if ea < eb else (eb, ea)
                if a == b:
                    continue
                pair_count[(a, b)] += 1
                pair_papers[(a, b)].add(pa)
                pair_papers[(a, b)].add(pb)

    logger.info("Upserting %d unique entity pairs…", len(pair_count))

    pairs = list(pair_count.items())
    upserts = 0
    with conn.cursor() as cur:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            cur.executemany(
                """
                INSERT INTO entity_graph (entity_a, entity_b, co_occurrences, paper_ids)
                VALUES (%s, %s, %s, %s::bigint[])
                ON CONFLICT (entity_a, entity_b) DO UPDATE
                    SET co_occurrences = entity_graph.co_occurrences + EXCLUDED.co_occurrences,
                        paper_ids = (
                            SELECT ARRAY(SELECT DISTINCT unnest(entity_graph.paper_ids || EXCLUDED.paper_ids))
                        )
                """,
                [
                    (a, b, cnt, list(pair_papers[(a, b)]))
                    for ((a, b), cnt) in batch
                ],
            )
            conn.commit()
            upserts += len(batch)
            if upserts % (batch_size * 10) == 0 or upserts >= len(pairs):
                logger.info("  %d / %d pairs upserted", upserts, len(pairs))
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
    """In-memory entity graph + PPR retrieval.

    The graph is cached to disk as a pickle so uvicorn reloads don't
    pay the multi-minute Postgres-rehydration cost on every restart.

    Pickle invalidation is automatic: we compare the edge count
    embedded in the meta sidecar against `SELECT COUNT(*) FROM
    entity_graph WHERE co_occurrences >= min_edge_weight`. If they
    differ — which they will after `scripts/build_entity_graph.py`
    has run on new ingested papers — the pickle is rebuilt fresh.
    """

    # Absolute paths anchored to the repo root (parent of src/) so the cache
    # is found regardless of which CWD the importer runs in.
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    _PICKLE_PATH = _REPO_ROOT / "data" / "hipporag_graph.pkl"
    _META_PATH = _REPO_ROOT / "data" / "hipporag_graph.meta.json"

    # Class-level lock — serializes concurrent _ensure_loaded() calls so
    # multiple threadpool requests don't each spawn their own pickle.load
    # (which we observed: two threads pickle.loading the same 1.6 GB file
    # in parallel, GIL-thrashing and turning a 60s load into 13+ min).
    import threading as _threading_for_lock
    _load_lock = _threading_for_lock.Lock()

    def __init__(self, db_dsn: str, min_edge_weight: int = 2) -> None:
        self._db_dsn = db_dsn
        self._min_edge_weight = min_edge_weight
        self._graph = None
        self._loaded = False

    def _live_edge_count(self) -> int:
        """Count of qualifying edges in the live entity_graph table."""
        with psycopg.connect(self._db_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM entity_graph WHERE co_occurrences >= %s",
                (self._min_edge_weight,),
            )
            return int(cur.fetchone()[0])

    def _try_load_pickle(self, expected_rows: int):
        """Return the unpickled graph if cache is valid, else None.

        Validator compares Postgres ROW count vs the row count
        recorded in meta at last build (NOT NetworkX edge count —
        NX dedupes (a,b)/(b,a) and self-loops, so graph.number_of_edges()
        is always ≤ table COUNT(*)).
        """
        if not (self._PICKLE_PATH.exists() and self._META_PATH.exists()):
            return None
        try:
            meta = json.loads(self._META_PATH.read_text())
            cached_rows = meta.get("source_row_count")
            if cached_rows != expected_rows:
                logger.info(
                    "HippoRAG pickle stale (cached_rows=%s, live_rows=%d) — will rebuild",
                    cached_rows, expected_rows,
                )
                return None
            if meta.get("min_edge_weight") != self._min_edge_weight:
                logger.info("HippoRAG pickle min_edge_weight differs — will rebuild")
                return None
            t0 = time.monotonic()
            with self._PICKLE_PATH.open("rb") as f:
                graph = pickle.load(f)
            logger.info(
                "Loaded HippoRAG graph from pickle: %d nodes, %d edges in %.1fs",
                graph.number_of_nodes(), graph.number_of_edges(),
                time.monotonic() - t0,
            )
            return graph
        except Exception as exc:
            logger.warning("HippoRAG pickle unreadable (%s) — will rebuild", exc)
            return None

    def _save_pickle(self, graph, source_row_count: int) -> None:
        """Persist the graph + a meta sidecar.

        `source_row_count` is the Postgres row count of qualifying edges
        AT THE MOMENT the rebuild started — NOT graph.number_of_edges().
        These two are not equal because NetworkX dedupes (a,b)/(b,a)
        rows and self-loops, so the graph has fewer edges than the table
        has rows. The validator compares Postgres-row-count to
        Postgres-row-count, which is the only apples-to-apples check.
        """
        try:
            self._PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            # Wrap the file in a counter so we log progress every 200 MB —
            # otherwise a multi-GB pickle.dump shows nothing for minutes.
            class _ProgressFile:
                def __init__(self, real_file, log_every_bytes=200_000_000):
                    self._f = real_file
                    self._written = 0
                    self._next_mark = log_every_bytes
                    self._step = log_every_bytes
                def write(self, data):
                    n = self._f.write(data)
                    self._written += len(data)
                    if self._written >= self._next_mark:
                        logger.info("HippoRAG pickle: wrote %.0f MB so far…",
                                    self._written / 1_000_000)
                        self._next_mark += self._step
                    return n
                def close(self): return self._f.close()
                def __getattr__(self, name): return getattr(self._f, name)

            with self._PICKLE_PATH.open("wb") as raw:
                pf = _ProgressFile(raw)
                pickle.dump(graph, pf, protocol=pickle.HIGHEST_PROTOCOL)
            self._META_PATH.write_text(json.dumps({
                "source_row_count": source_row_count,
                "graph_edge_count": graph.number_of_edges(),
                "node_count": graph.number_of_nodes(),
                "min_edge_weight": self._min_edge_weight,
            }))
            logger.info(
                "Saved HippoRAG pickle: %d Postgres rows → %d NetworkX edges, %.1f MB in %.1fs",
                source_row_count, graph.number_of_edges(),
                self._PICKLE_PATH.stat().st_size / 1_000_000,
                time.monotonic() - t0,
            )
        except Exception as exc:
            logger.warning("HippoRAG pickle save failed (%s); not cached", exc)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Serialize concurrent loads. The 1.6 GB pickle.load is slow
        # enough that multiple threadpool requests racing each other
        # makes ALL of them slow due to GIL contention. With this lock
        # the first request loads (~60-120 s), the others wait, then
        # they all share the freshly-loaded graph.
        with self._load_lock:
            if self._loaded:
                return
            self._do_load()

    def _do_load(self) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("Install networkx: pip install networkx") from exc

        # 1. Check the live edge count cheaply (one indexed COUNT).
        live_edges = self._live_edge_count()

        # 2. Try the pickle cache first.
        cached = self._try_load_pickle(live_edges)
        if cached is not None:
            self._graph = cached
            self._loaded = True
            return

        # 3. Cold path: build from Postgres, then write the pickle.
        # Stream the cursor (server-side, named cursor) instead of fetchall
        # so we don't materialize 36M tuples in Python before we even start
        # adding edges — saves ~3-4 GB of transient memory and lets us log
        # progress as the loop runs.
        logger.info("Loading %d edges from Postgres into memory…", live_edges)
        t0 = time.monotonic()
        self._graph = nx.Graph()
        log_every = max(1, live_edges // 20)   # 20 progress lines total
        next_mark = log_every
        i = 0
        with psycopg.connect(self._db_dsn) as conn:
            with conn.cursor(name="hipporag_load") as cur:
                cur.execute(
                    "SELECT entity_a, entity_b, co_occurrences FROM entity_graph "
                    "WHERE co_occurrences >= %s",
                    (self._min_edge_weight,),
                )
                for row in cur:
                    a, b, w = row
                    self._graph.add_edge(a, b, weight=w)
                    i += 1
                    if i >= next_mark:
                        elapsed = time.monotonic() - t0
                        rate = i / max(elapsed, 0.001)
                        eta = (live_edges - i) / max(rate, 1)
                        logger.info(
                            "  %d / %d edges (%.1f%%, %.0fk/s, ~%.0fs left)",
                            i, live_edges, 100 * i / live_edges,
                            rate / 1000, eta,
                        )
                        next_mark += log_every
        logger.info(
            "Loaded entity graph: %d nodes, %d edges in %.1fs",
            self._graph.number_of_nodes(), self._graph.number_of_edges(),
            time.monotonic() - t0,
        )
        self._save_pickle(self._graph, source_row_count=live_edges)
        self._loaded = True

    def reload(self) -> None:
        """Force a reload of the in-memory graph (after a rebuild).

        Doesn't delete the pickle — the live-vs-cached edge-count check
        in _ensure_loaded handles invalidation automatically.
        """
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


# Singleton — same HF biomedical NER model as the chunk pipeline.
# Cached after first call so we don't reload the 110 MB model per query.
_query_ner_runner = None


def extract_query_entities(query: str) -> list[str]:
    """Extract entities from a query for HippoRAG PPR seeding.

    Uses the same HuggingFace biomedical NER model as the chunk pipeline
    (`d4data/biomedical-ner-all`). The model is loaded once on first call;
    subsequent calls reuse it.

    Raises RuntimeError if the model can't be loaded — callers (or the API
    layer above) should surface that error rather than silently degrade.
    """
    global _query_ner_runner
    if _query_ner_runner is None:
        from ..embeddings.ner_pipeline import _NERRunner
        logger.info("Loading HF NER model for query-time entity extraction…")
        _query_ner_runner = _NERRunner()

    extracted = _query_ner_runner.extract(query, paper_id=0, chunk_id=0)
    return [e.entity_text.lower() for e in extracted]
