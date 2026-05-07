# CureMom — Medical Literature Knowledge Base

A system that ingests peer-reviewed medical papers from PubMed/PMC and answers queries with reasoned, citation-grounded responses. Built on first-principles drug discovery: the system starts from molecular biology and discovers compound→pathway→condition connections from evidence, rather than encoding assumptions about what treats what.

**Starting scope:** Systemic Lupus Erythematosus (SLE), autoimmune disease, and muscle physiology/hypertrophy.

---

## Architecture

```
PubMed API                   ChEMBL API (no key required)
(SLE + muscle MeSH queries)  (all clinical-phase human targets)
        ↓                             ↓
  Resumable Ingestion Pipeline   Compound-Target Layer
        ↓                             ↓
  PostgreSQL 16 + pgvector  ←—————————┘
        +
  Elasticsearch 8.x (BM25 full-text)
        ↓
  Hybrid Retrieval (BM25 → BM25+dense → HippoRAG PPR)
        ↓
  Swappable LLM Layer (extractive / Ollama / Claude / OpenAI)
        ↓
  FastAPI → structured JSON response + passage-level citations
```

**Phases (current state):**
- **Phase 1** ✅ live — BM25 retrieval, extractive responses, no LLM required
- **Phase 2** ✅ code complete; embedding run optional — section-aware chunking + PubMedBERT (768-dim, mean-pooled, MPS/CUDA/CPU auto), HuggingFace transformer NER (`d4data/biomedical-ner-all`)
- **Phase 3** ✅ live — pluggable LLM (extractive / Ollama / Claude / OpenAI), query-complexity classifier, citation verifier (catches hallucinated `[N]` indices and weakly-supported claims), `/llm/status` health endpoint
- **Phase 4** ✅ live — HippoRAG Personalized PageRank over a 5.27M-edge entity graph (built from MeSH descriptors merged with NER co-occurrences), SPLADE sparse-vector pipeline ready to encode

**Default retrieval strategy:** `full` = BM25 + dense (when embeddings present) + HippoRAG PPR rerank.

---

## First-Principles Drug Discovery

CureMom does **not** assume that compound X treats disease Y. Instead:

1. **Disease layer:** Ingest all papers on SLE mechanisms, lupus nephritis, complement biology, etc.
2. **Compound-pharmacology layer:** Ingest ChEMBL compound data for all ~3,000 human single-protein targets that have at least one clinical-phase compound — covering the entire druggable proteome, not a hand-picked list.
3. **Clinical evidence layer:** Ingest RCT and trial literature broadly.

The system then lets you query across all three layers. A compound like belimumab surfaces because it inhibits BAFF (TNFSF13B), and BAFF has documented roles in B cell survival — not because we hard-coded "belimumab treats SLE."

---

## Quickstart

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.11+
- [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) (free — enables 10 req/s vs 3 without)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set NCBI_EMAIL at minimum; NCBI_API_KEY for higher rate limits
```

### 3. Start services

```bash
docker compose up -d postgres elasticsearch
# Wait for both to be healthy (~30 seconds)
docker compose ps
```

### 4. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Run ChEMBL compound-target ingestion (optional but recommended first)

```bash
# Full druggable proteome — discovers ALL human single-protein targets
# with clinical compounds (~3,000 targets). Resumable — safe to interrupt.
python scripts/fetch_chembl.py --all-targets

# Approved drugs only (faster, ~500 targets)
python scripts/fetch_chembl.py --all-targets --min-phase 4

# Dry run — just shows which targets would be processed
python scripts/fetch_chembl.py --all-targets --dry-run

# Hand-curated seed list only (30 targets, runs in minutes)
python scripts/fetch_chembl.py --seed-only
```

### 6. Run paper ingestion (SLE + muscle)

```bash
# All priority-1 topics (SLE core, lupus nephritis, muscle hypertrophy, protein synthesis)
python scripts/ingest.py --priority 1

# Specific topics
python scripts/ingest.py --topics sle_core lupus_nephritis

# With date filter
python scripts/ingest.py --topics sle_core --date-from 2020/01/01

# Dry run — queues PMIDs without fetching
python scripts/ingest.py --topics sle_core --dry-run
```

The pipeline is **resumable** — kill it at any point and restart. Completed PMIDs are skipped automatically. Monitor progress:

```bash
curl http://localhost:8000/api/v1/ingestion/status
```

### 7. Start the API server

```bash
export PYTHONPATH=$(pwd)
uvicorn src.api.main:app --reload
# Docs: http://localhost:8000/docs
# UI:   http://localhost:8000/
```

### 8a. (Recommended) Pull PMC full text for OA papers

PubMed only exposes abstracts. For methods, results, discussion, etc., you
need to download JATS XML from PMC for the Open Access subset of papers
(roughly the ones with a `pmcid` set). Without this step, the LLM only
sees the abstract of each retrieved paper.

```bash
# Fetch full text for all eligible papers (PMCID set, has_full_text=false)
PYTHONPATH=. python scripts/fetch_pmc.py

# Smoke-test on a handful first:
PYTHONPATH=. python scripts/fetch_pmc.py --limit 100
```

Rate: 3 req/s without an NCBI API key (~12 hr for ~38K eligible papers),
10 req/s with one (~1.5 hr). Set `NCBI_API_KEY` in `.env` to use the faster path.

Verified: 5 papers → 12 + 7 + 12 + 3 + 1 sections (intro / methods / results /
discussion / conclusion / abstract / other), avg ~2,400 chars per section.

### 8b. (Phase 2 — recommended) Generate embeddings for hybrid retrieval

PubMedBERT inference adds 768-dim embeddings to each chunk in
`paper_chunks`. Without this, `retrieval_strategy=hybrid` and `full` raise
a 502 (no silent fallback).

```bash
# After step 8a, regenerate chunks for the new full-text sections
# (intro/methods/discussion → 512-token sliding window with 128 overlap;
#  results → one chunk per paragraph; abstract chunks already exist)
PYTHONPATH=. python scripts/embed.py --chunk-fulltext

# Embed all chunks lacking embeddings
PYTHONPATH=. python scripts/embed.py --batch-size 32

# After bulk load, build the HNSW index for fast vector search
PYTHONPATH=. python scripts/embed.py --index-only
```

The first run downloads `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
(~440 MB). Embedding cost: ~1 hour per 30K chunks on M-series MPS.

### 9. (Phase 2 — optional) Extract biomedical entities (for Phase 4 graph)

A HuggingFace transformer NER model (`d4data/biomedical-ner-all`, ~110 MB)
identifies diseases, chemicals, anatomy, and other biomedical entities in
each chunk. Output rows in `paper_entities` are the substrate for Phase 4's
HippoRAG entity-graph traversal.

We pivoted away from scispaCy because its current versions pin Python 3.10+,
while this project still runs on 3.9. The HF model uses the same
`transformers + torch` stack as the embeddings pipeline.

```bash
PYTHONPATH=. python scripts/extract_entities.py
# Limit to specific papers for incremental updates:
PYTHONPATH=. python scripts/extract_entities.py --paper-ids 1 2 3
# Try alternative models:
PYTHONPATH=. python scripts/extract_entities.py --model alvaroalon2/biobert_diseases_ner
```

Verified on M1 Max: ~125 ms/chunk on MPS. **71K chunks → 1.23M entities in ~33 min.**

### 10. (Phase 4 — optional) Build the entity graph for HippoRAG retrieval

The HippoRAG retriever runs Personalized PageRank over an entity co-occurrence
graph. There are three sources you can feed the graph:

```bash
# Default — fast, NER-free. Co-occurrence of MeSH descriptors on the same
# paper. Verified on 33K papers: ~52 seconds → 557K edges.
PYTHONPATH=. python scripts/build_entity_graph.py --source mesh

# Higher resolution but requires step 9 (NER) first. Uses paper_entities.
PYTHONPATH=. python scripts/build_entity_graph.py --source ner

# Recommended once NER has run: MeSH first, then merge NER on top.
# Verified: 33K papers + 71K chunks → 5.27M edges in ~3 min.
PYTHONPATH=. python scripts/build_entity_graph.py --source both
```

After rebuilding, hot-reload the in-memory graph in the running server
(no restart needed):

```bash
curl -X POST http://localhost:8000/api/v1/hipporag/reload
# → {"status":"reloaded","nodes":46190,"edges":1081723}
```

For a live system, prefer **incremental updates** as new papers are ingested
(milliseconds per paper) instead of repeated full rebuilds — call
`src.search.hipporag.update_entity_graph_for_papers(conn, [paper_ids])` after
each NER batch. Schedule a full rebuild nightly or weekly to compact the table.

Then call the API with `retrieval_strategy: "hipporag"` (entity-graph rerank
on top of BM25) or `"full"` (BM25 + dense + HippoRAG combined — the default).

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What drugs target the complement pathway in lupus nephritis?",
       "options":{"retrieval_strategy":"full","top_k":10,"llm_provider":"ollama"}}'
```

Multi-hop questions like the example above benefit most from `hipporag`/`full`
because PPR surfaces chunks that *bridge* the query entities ("complement",
"drug", "lupus", "nephritis") even if no single chunk mentions all four.
Verified live: returns 5 relevant papers in ~2.3s on the 1M-edge filtered graph.

### 11. (Phase 4 — optional) SPLADE sparse vectors

Learned sparse vectors that capture biomedical synonym expansion (e.g.
"antimalarial" ↔ "hydroxychloroquine") natively in Elasticsearch. Requires
ES 8.11+ (the `sparse_vector` field type).

```bash
PYTHONPATH=. python scripts/encode_splade.py
```

Each paper gets its chunks max-pooled into one sparse vector merged into the
existing ES doc. ~2 hours for 33K papers on M-series MPS.

---

## API Usage

### Query with citations

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the efficacy of hydroxychloroquine in SLE?",
    "query_type": "factual",
    "filters": {
      "pub_year_from": 2015,
      "publication_types": ["Randomized Controlled Trial", "Meta-Analysis"]
    },
    "options": {
      "top_k": 10,
      "retrieval_strategy": "full",
      "llm_provider": "ollama"
    }
  }'
```

`retrieval_strategy` is one of:
- `bm25` — Elasticsearch keyword scoring only (~20 ms; baseline)
- `hybrid` — BM25 + dense vector RRF fusion (degrades to BM25 if embeddings missing)
- `hipporag` — BM25 + HippoRAG entity-graph PPR rerank (~2 s)
- `full` — BM25 + dense + HippoRAG (default)

`llm_provider` is one of `extractive` / `ollama` / `claude` / `openai`. Defaults to whatever `LLM_PROVIDER` is in `.env`.

**Response structure:**
```json
{
  "query": "...",
  "response": "Hydroxychloroquine reduced disease activity... [1] ... [2]",
  "citations": [
    {
      "citation_index": 1,
      "pmid": "34567890",
      "title": "...",
      "authors": "Smith J, et al.",
      "journal": "Arthritis & Rheumatology",
      "year": 2021,
      "publication_types": ["Meta-Analysis"],
      "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/34567890/",
      "chunk": {
        "section": "results",
        "text": "HCQ reduced SLEDAI score by 2.1 points...",
        "start_char": 1240,
        "end_char": 1580
      },
      "relevance_score": 0.94
    }
  ],
  "metadata": {
    "retrieval_strategy": "bm25",
    "model_used": "extractive",
    "latency_ms": 180,
    "query_type": "factual",
    "citation_warnings": []
  }
}
```

`citation_warnings` flags `[N]` markers in the response that:
- reference an out-of-range chunk (`severity: "invalid"` — usually a hallucination), or
- have low lexical overlap with the cited chunk (`severity: "weak"` — manually verify).

`query_type` is one of `factual` / `exploratory` / `comparative`. Use
`GET /api/v1/query/classify?q=…` to classify without retrieving.

### Other endpoints

```bash
# Check which LLM providers are configured / reachable
curl "http://localhost:8000/api/v1/llm/status"

# Classify a query (factual | exploratory | comparative)
curl "http://localhost:8000/api/v1/query/classify?q=Is+HCQ+better+than+MTX"

# Reload HippoRAG entity graph after rebuild (no server restart)
curl -X POST "http://localhost:8000/api/v1/hipporag/reload"

# Search papers
curl "http://localhost:8000/api/v1/papers/search?q=lupus+nephritis+treatment&pub_year_from=2018"

# Paper detail
curl "http://localhost:8000/api/v1/papers/34567890"

# Similar papers
curl "http://localhost:8000/api/v1/papers/34567890/similar"

# Citation graph
curl "http://localhost:8000/api/v1/papers/34567890/cited_by"

# MeSH autocomplete
curl "http://localhost:8000/api/v1/mesh/suggest?q=lupus"

# Ingestion progress
curl http://localhost:8000/api/v1/ingestion/status

# Stats
curl http://localhost:8000/api/v1/stats
```

---

## Swapping the LLM Provider

Set `LLM_PROVIDER` in `.env`:

| Value | Description | Requirements |
|---|---|---|
| `extractive` | No LLM — returns top sentences with citations | None (default) |
| `ollama` | Local LLM via Ollama | Ollama running + model pulled |
| `claude` | Anthropic Claude API | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI API | `OPENAI_API_KEY` |

**Ollama setup:**
```bash
# Install Ollama: https://ollama.com
ollama pull meditron       # medical fine-tune of Llama (~4 GB)
# Set in .env: LLM_PROVIDER=ollama, OLLAMA_MODEL=meditron
```

LLM providers can also be specified per request via `options.llm_provider` in the query body.

---

## Project Structure

```
src/
  ingestion/
    topics.py             — SLE/autoimmune + muscle MeSH query definitions (3-layer)
    targets.py            — Hand-curated molecular target seed list (30 targets)
    chembl_client.py      — ChEMBL REST client: compounds, targets, get_all_clinical_targets()
    pubmed_parser.py      — PubMed XML (EFetch) parser
    jats_parser.py        — PMC full-text JATS XML parser
    pipeline.py           — Resumable ingestion pipeline (state machine via ingestion_log)
  db/
    schema.sql            — Full PostgreSQL schema
  search/
    elasticsearch_client.py  — ES index setup + BM25 search
    hybrid_retriever.py      — RRF fusion: BM25 + pgvector (Phase 2)
    mesh_expander.py         — MeSH hierarchy query expansion
    hipporag.py              — Entity co-occurrence graph PPR retrieval (Phase 4)
  embeddings/
    chunk_pipeline.py     — Section-aware chunking + PubMedBERT inference (Phase 2)
    ner_pipeline.py       — HuggingFace transformer biomedical NER (Phase 2)
    splade_pipeline.py    — SPLADE sparse-vector encoding for ES (Phase 4)
  api/
    main.py               — FastAPI app + all endpoints
    response_builder.py   — Structured response + passage-level citation provenance
    llm_providers.py      — Swappable LLM (Extractive / Ollama / Claude / OpenAI)
    classifier.py         — Query complexity classifier (Phase 3)
    citation_verifier.py  — Citation [N] parser + lexical-overlap check (Phase 3)
scripts/
  ingest.py               — CLI: run the ingestion pipeline (PubMed metadata + abstracts)
  fetch_chembl.py         — CLI: ChEMBL compound-target data + PubMed queuing
  fetch_pmc.py            — CLI: full-text JATS XML for OA papers → paper_sections
  sync_elasticsearch.py   — CLI: backfill PostgreSQL papers → Elasticsearch
  embed.py                — CLI: chunk full text + embed with PubMedBERT (Phase 2)
  extract_entities.py     — CLI: HF transformer NER on chunks (Phase 2)
  build_entity_graph.py   — CLI: MeSH/NER entity graph builder (Phase 4)
  encode_splade.py        — CLI: SPLADE sparse vectors → ES (Phase 4)
docker-compose.yml        — postgres+pgvector, elasticsearch, app
Dockerfile
requirements.txt
TODO.md                   — Remaining phases (2–4) task list
```

---

## Ingestion Topics

### Autoimmune / SLE

| Topic | Description | Priority |
|---|---|---|
| `sle_core` | Systemic Lupus Erythematosus (MeSH) | 1 |
| `lupus_nephritis` | Lupus nephritis | 1 |
| `cutaneous_lupus` | Cutaneous lupus | 2 |
| `antiphospholipid` | Antiphospholipid syndrome | 2 |
| `sjogrens` | Sjögren's syndrome | 2 |
| `complement_system` | Complement pathway in autoimmune disease | 2 |
| `autoimmune_broad` | Broad autoimmune + SLE/lupus | 3 |

### Muscle Physiology

| Topic | Description | Priority |
|---|---|---|
| `muscle_hypertrophy` | Skeletal muscle hypertrophy — growth, adaptation, signalling | 1 |
| `protein_synthesis` | Muscle protein synthesis — mTOR, ribosomes, amino acid signalling | 1 |
| `resistance_training` | Resistance training effects on muscle mass and protein turnover | 2 |
| `mtor_signaling` | mTORC1 pathway in muscle anabolism | 2 |
| `amino_acid_muscle` | Amino acid regulation of MPS (leucine, EAAs, whey) | 2 |

### Compound Pharmacology (Layer 2, assumption-free)

| Topic | Description |
|---|---|
| `hydroxychloroquine_pharma` | HCQ pharmacology, mechanism, and off-target effects |
| `belimumab_pharma` | Belimumab pharmacology |
| `rituximab_pharma` | Rituximab pharmacology |
| `voclosporin_pharma` | Voclosporin pharmacology |
| `creatine_pharma` | Creatine pharmacology and metabolism |
| `amino_acids_pharma` | Essential amino acid pharmacology |
| `anabolic_hormones_pharma` | Testosterone, IGF-1, GH pharmacology |

Add new topics in `src/ingestion/topics.py`. Add new molecular targets in `src/ingestion/targets.py` (or use `--all-targets` to pull the full druggable proteome from ChEMBL automatically).

---

## ChEMBL Integration

CureMom queries ChEMBL for compound-target interaction data. Rather than a hard-coded drug list, it discovers compounds from molecular biology:

```
fetch_chembl.py
  → ChEMBL /mechanism endpoint (all drug mechanism records)
  → unique target ChEMBL IDs
  → filter: Homo sapiens, SINGLE PROTEIN targets only
  → for each target: fetch all active compounds (IC50/Ki/Kd ≤ 10 µM, confidence ≥ 6)
  → for each compound: ESearch PubMed by compound name → queue PMIDs
```

This surfaces compounds we've never heard of that happen to strongly inhibit a target relevant to a pathway — the same compound might also show up in SLE literature, creating a data-driven connection.

**Resumability:** If interrupted, re-running skips targets that already have compound-target rows in the database.

---

## Roadmap

See [`TODO.md`](TODO.md) for the full task list.

**Done:**
- ✅ **Phase 1:** BM25 retrieval + extractive responses + citation provenance
- ✅ **Phase 2 (code):** PubMedBERT embedder + section-aware chunking; HF transformer NER (`d4data/biomedical-ner-all`); RRF hybrid path wired
- ✅ **Phase 3:** Pluggable LLM (extractive / Ollama / Claude / OpenAI); `[N]` citation parser; query classifier; citation verifier; `/llm/status` health endpoint
- ✅ **Phase 4:** HippoRAG entity graph (5.27M edges from MeSH + NER co-occurrence) with NetworkX Personalized PageRank rerank — no Neo4j; SPLADE sparse-vector pipeline ready

**Pending offline runs (heavy compute):**
- Run `scripts/embed.py` — generate 768-dim PubMedBERT vectors over chunks (~1 hr); enables real `hybrid` and `full` strategies (currently degrade to BM25 + HippoRAG)
- Run `scripts/encode_splade.py` — populate Elasticsearch `sparse_vector` field (~2 hr)

**Next:**
- Streaming responses for Claude / OpenAI (Server-Sent Events)
- Cross-encoder re-rank on top-20 (e.g. `ms-marco-MiniLM-L-12-v2`)
- Expand corpus: RA, myositis, systemic sclerosis, sarcopenia, exercise transcriptomics

---

## Design Notes

- The full PubMed corpus (35M papers) is intentionally not downloaded. Expand scope by adding topics to `src/ingestion/topics.py`.
- Citation provenance is tracked at chunk level (section + paragraph + char offsets), not just paper level.
- All LLM responses are grounded: every claim requires an inline `[N]` citation to an ingested chunk.
- ChEMBL queries are by **molecular target** (e.g. JAK1), not by compound name — this avoids encoding prior beliefs about which drugs work.
- `targets.py` is a 30-target seed list for fast local testing. For production, use `--all-targets` to cover the full ~3,000-target clinical druggable proteome.
