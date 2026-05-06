# CureMom — Remaining Phases TODO

## Phase 2 — Vector Search + NER

### Setup
- [ ] Add `pgvector` extension to the running PostgreSQL instance (already in schema, needs `CREATE EXTENSION IF NOT EXISTS vector` confirmed on live DB)
- [ ] Install Phase 2 dependencies: `transformers`, `torch`, `sentence-transformers`, `scispacy`, `en_core_sci_lg`, `en_ner_bc5cdr_md`

### Chunking pipeline (`src/embeddings/chunk_pipeline.py`)
- [ ] Write `chunk_paper(paper_id)` — splits paper into chunks by section type
  - Abstract → single chunk
  - Introduction/Discussion/Methods → 512-token sliding window, 128-token overlap
  - Results → one chunk per paragraph
  - Never split across section boundaries
- [ ] Store chunks in `paper_chunks` (paper_id, section_id, chunk_index, source_type, start_char, end_char, paragraph_index, token_count)
- [ ] Write batch runner — iterate all papers missing chunks, process in batches of 500

### Embedding pipeline (`src/embeddings/chunk_pipeline.py`)
- [ ] Load `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- [ ] Write `embed_chunks(chunk_ids)` — batch inference (batch_size=32 GPU / 8 CPU), store 768-dim vectors in `paper_chunks.embedding`
- [ ] Make GPU/CPU detection automatic
- [ ] After bulk load, build HNSW index:
  ```sql
  CREATE INDEX idx_chunks_embedding ON paper_chunks
  USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
  ```
- [ ] Add CLI command: `python scripts/embed.py [--paper-ids ...] [--batch-size 32]`

### NER pipeline (`src/embeddings/ner_pipeline.py`)
- [ ] Load scispaCy models (`en_ner_bc5cdr_md` for DISEASE/CHEMICAL, `en_core_sci_lg` for general)
- [ ] Write `extract_entities(chunk_id)` — extract DISEASE, CHEMICAL, GENE_OR_GENE_PRODUCT, ORGANISM, CELL_TYPE
- [ ] Normalize entities to UMLS CUI / MeSH descriptor_ui via scispaCy linker
- [ ] Store in `paper_entities` (paper_id, chunk_id, entity_text, entity_type, kb_id, start_char, end_char)
- [ ] Add CLI command: `python scripts/extract_entities.py`

### Hybrid retrieval wiring
- [ ] Wire `dense_weight=0.5` path in `HybridRetriever.retrieve()` — currently stubbed but not called
- [ ] Add embedding model to the API lifespan context so query embeddings can be generated on the fly
- [ ] Expose `retrieval_strategy=hybrid` in query API (currently accepted but dense path requires embedding)
- [ ] A/B test BM25-only vs hybrid on 50 hand-written SLE queries — log precision@5 to a results file

---

## Phase 3 — LLM Synthesis

### Query complexity classifier (`src/api/classifier.py`)
- [ ] Write `classify_query(query: str) -> Literal["factual", "exploratory", "comparative"]`
  - Simple rule-based v1: comparative keywords ("vs", "compare", "better than") → comparative; question words + specific entity → factual; else exploratory
  - Upgrade to small local classifier (scikit-learn text classifier trained on query examples) if rule-based is insufficient
- [ ] Wire classifier into `POST /api/v1/query` — use it to auto-select provider if `llm_provider` not specified in request

### Citation verification (`src/api/citation_verifier.py`)
- [ ] Write `verify_citations(response_text, chunks)` — parse `[N]` markers, check each N is ≤ len(chunks)
- [ ] Optional: add NLI entailment check using `cross-encoder/nli-deberta-v3-small` — for each claim+cited chunk pair, score entailment; flag low-confidence citations in response metadata
- [ ] Return `citation_warnings` list in API response metadata when claims are weakly supported

### LLM provider hardening
- [ ] Add `anthropic` to requirements.txt (behind optional marker: `anthropic>=0.28.0`)
- [ ] Add `openai` to requirements.txt (optional: `openai>=1.30.0`)
- [ ] Add provider health-check endpoint: `GET /api/v1/llm/status` — reports which provider is active and whether it's reachable
- [ ] Handle LLM provider timeout gracefully — fall back to extractive if LLM call exceeds 30s
- [ ] Add streaming support for Claude/OpenAI providers (Server-Sent Events on the query endpoint)

### Prompt tuning
- [ ] Test grounding system prompt against 20 SLE queries using Claude — check for hallucinated claims
- [ ] Add few-shot examples to system prompt for comparative query type
- [ ] Add explicit "do not infer beyond stated data" instruction for numerical claims (efficacy %, p-values)

---

## Phase 4 — HippoRAG PPR Graph + SPLADE

### Entity co-occurrence graph (`src/search/hipporag.py`)
- [ ] Write `build_entity_graph()` — after NER, populate `entity_graph` table:
  - For each chunk: get all entity pairs that co-occur → upsert edge with co_occurrence_count + 1, append paper_id to paper_ids[]
  - Run after each NER batch, or as a nightly rebuild
- [ ] Write `personalized_pagerank(query_entities, damping=0.85, max_iter=100)` — using NetworkX or igraph:
  - Build in-memory graph from `entity_graph` table (load once, cache)
  - Start PPR from nodes matching query entities
  - Return top-N entity nodes by PPR score
- [ ] Write `hipporag_rerank(chunks, query)`:
  - Extract query entities with scispaCy
  - Run PPR to get high-scoring entities
  - Boost chunk scores if chunks contain high-PPR entities
  - Merge boosted scores with RRF scores from Phase 2
- [ ] Wire `HippoRAGRetriever` as an optional retrieval strategy: `retrieval_strategy=hipporag`
- [ ] Add CLI: `python scripts/build_entity_graph.py`
- [ ] Graph reload strategy — cache graph in memory, reload nightly or when `entity_graph` row count changes by >5%

### SPLADE sparse vectors
- [ ] Research: confirm `naver/splade-v3` or `prithivida/Splade_PP_en_v1` works well on biomedical text; alternatively use `naver/efficient-splade-VI-BT-large-query` + `...-doc`
- [ ] Add SPLADE inference to `src/embeddings/splade_pipeline.py`:
  - Encode all chunks → sparse vectors (dict of {token_id: weight})
  - Store as Elasticsearch `sparse_vector` field (ES 8.11+ required)
- [ ] Update Elasticsearch index mapping to add `sparse_vector` field for SPLADE
- [ ] Write `search_splade(query, top_k)` in `elasticsearch_client.py` using `sparse_vector` query
- [ ] Integrate SPLADE into retrieval pipeline: `retrieval_strategy=splade` or as the BM25 replacement in hybrid
- [ ] Benchmark SPLADE vs BM25 on SLE query test set — record results in `benchmarks/retrieval_comparison.md`
- [ ] Add CLI: `python scripts/encode_splade.py`

### Full Phase 4 pipeline
- [ ] Wire final pipeline: `SPLADE (ES) + PubMedBERT dense (pgvector) → RRF → HippoRAG PPR re-rank → top-5 → response builder`
- [ ] Expose as `retrieval_strategy=full` in the query API
- [ ] Document multi-hop reasoning example in README (e.g. "What drugs affect the complement pathway in SLE nephritis?")

---

## Infrastructure / Cross-cutting

### Elasticsearch sync pipeline (`scripts/sync_es.py`)
- [ ] Write script to sync papers from PostgreSQL → Elasticsearch index
  - Pull papers with `status=done` in ingestion_log, check if already in ES (by PMID), bulk-index missing ones
  - Run incrementally (only new/updated papers since last sync)
  - Add to docker-compose as a one-shot service or cron

### MeSH ontology loader (`scripts/load_mesh.py`)
- [ ] Download NLM MeSH XML: `https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/`
- [ ] Write parser for MeSH XML — extract descriptor_ui, descriptor_name, tree_numbers[], synonyms[] (entry terms)
- [ ] Bulk-load into `mesh_terms` table
- [ ] Schedule annual refresh (MeSH updates every January)

### Citation graph backfill (`scripts/resolve_citations.py`)
- [ ] Write script to resolve `cited_pmid_raw` in `citations` table → `cited_paper_id` for PMIDs already in DB
  - `UPDATE citations SET cited_paper_id = p.id FROM papers p WHERE citations.cited_pmid_raw = p.pmid AND citations.cited_paper_id IS NULL`
- [ ] Run after each ingestion batch

### PMC full-text ingestion (`scripts/fetch_pmc.py`)
- [ ] Download PMC OA file list CSV: `ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv`
- [ ] Filter for papers already in `papers` table that have a PMCID and `has_full_text=false`
- [ ] Download JATS XML for those papers via PMC FTP
- [ ] Parse with `jats_parser.py` → insert into `paper_sections`
- [ ] Update `papers.has_full_text = true`

### Testing
- [ ] Write unit tests for `pubmed_parser.py` — use fixture XML files in `tests/fixtures/`
- [ ] Write unit tests for `jats_parser.py`
- [ ] Write integration test for ingestion pipeline using a small PMID list (10 SLE papers)
- [ ] Write retrieval quality test: 50 hand-labeled SLE queries with expected PMIDs in top-10; assert precision@10 ≥ 0.7
- [ ] Add `pytest` and `pytest-asyncio` to requirements.txt

### Monitoring
- [ ] Add Prometheus metrics to FastAPI: query latency histogram, retrieval count, LLM provider calls
- [ ] Add `GET /metrics` endpoint (prometheus_client)
- [ ] Add `docker-compose.yml` service for Grafana + Prometheus (optional, dev only)

### Expand topic coverage (after Phase 1 is stable)
- [ ] Ingest priority-2 topics: `cutaneous_lupus`, `antiphospholipid`, `sjogrens`, `complement_system`
- [ ] Ingest priority-2 muscle topics: `resistance_training`, `mtor_signaling`, `amino_acid_muscle`
- [ ] Add new autoimmune topic: `"Rheumatoid Arthritis"[MeSH]` (high overlap with SLE research)
- [ ] Add new autoimmune topic: `"Myositis"[MeSH]`
- [ ] Add new autoimmune topic: `"Systemic Sclerosis"[MeSH]`
- [ ] Add new muscle topic: `"Sarcopenia"[MeSH]` (muscle loss — inverse of hypertrophy)
- [ ] Add new muscle topic: `"Exercise"[MeSH] AND "Gene Expression"[MeSH]` (transcriptomics of training)
