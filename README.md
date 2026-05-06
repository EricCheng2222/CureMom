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

**Phases:**
- **Phase 1 (complete):** BM25 + extractive responses, no LLM required
- **Phase 2:** PubMedBERT embeddings + hybrid retrieval (RRF fusion)
- **Phase 3:** Pluggable LLM synthesis (Ollama, Claude, OpenAI)
- **Phase 4:** HippoRAG entity graph + SPLADE (no Neo4j needed)

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

### 8. (Phase 2 — optional) Generate embeddings for hybrid retrieval

After abstract chunks are populated by the ingestion pipeline, run PubMedBERT
inference to add 768-dim embeddings to each chunk. Without this, the
"Hybrid (BM25 + dense)" mode in the UI silently falls back to BM25-only.

```bash
# Embed all chunks lacking embeddings (uses MPS/CUDA/CPU automatically)
PYTHONPATH=. python scripts/embed.py --batch-size 32

# After bulk load, build the HNSW index for fast vector search
PYTHONPATH=. python scripts/embed.py --index-only
```

The first run downloads `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
(~440 MB) and takes ~1 hour for 33K chunks on M-series Macs (faster on GPU).

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
      "retrieval_strategy": "bm25"
    }
  }'
```

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
    "latency_ms": 180
  }
}
```

### Other endpoints

```bash
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
    ner_pipeline.py       — scispaCy NER + entity linking (Phase 2)
  api/
    main.py               — FastAPI app + all endpoints
    response_builder.py   — Structured response + passage-level citation provenance
    llm_providers.py      — Swappable LLM (Extractive / Ollama / Claude / OpenAI)
    classifier.py         — Query complexity classifier (Phase 3)
    citation_verifier.py  — Citation [N] parser + NLI entailment check (Phase 3)
scripts/
  ingest.py               — CLI: run the ingestion pipeline
  fetch_chembl.py         — CLI: ChEMBL compound-target data + PubMed queuing
  embed.py                — CLI: embed chunks with PubMedBERT (Phase 2)
  extract_entities.py     — CLI: scispaCy NER on chunks (Phase 2)
  sync_es.py              — CLI: sync PostgreSQL → Elasticsearch index
  build_entity_graph.py   — CLI: build entity co-occurrence graph (Phase 4)
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

- **Phase 2:** PubMedBERT embeddings, scispaCy NER, hybrid BM25+dense retrieval (RRF)
- **Phase 3:** LLM synthesis fully wired, `[N]` citation parser, query complexity classifier
- **Phase 4:** HippoRAG entity graph (PostgreSQL PPR, no Neo4j), SPLADE sparse vectors in Elasticsearch
- **Expand:** Additional autoimmune topics (RA, myositis, systemic sclerosis), sarcopenia, exercise transcriptomics

---

## Design Notes

- The full PubMed corpus (35M papers) is intentionally not downloaded. Expand scope by adding topics to `src/ingestion/topics.py`.
- Citation provenance is tracked at chunk level (section + paragraph + char offsets), not just paper level.
- All LLM responses are grounded: every claim requires an inline `[N]` citation to an ingested chunk.
- ChEMBL queries are by **molecular target** (e.g. JAK1), not by compound name — this avoids encoding prior beliefs about which drugs work.
- `targets.py` is a 30-target seed list for fast local testing. For production, use `--all-targets` to cover the full ~3,000-target clinical druggable proteome.
