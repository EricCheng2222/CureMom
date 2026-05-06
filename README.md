# CureMom — Medical Literature Knowledge Base

A system that ingests peer-reviewed medical papers from PubMed/PMC and answers queries with reasoned, citation-grounded responses. Starting scope: Systemic Lupus Erythematosus (SLE) and autoimmune literature.

## Architecture

```
PubMed API (SLE-targeted MeSH queries)
        ↓
  Resumable Ingestion Pipeline
        ↓
  PostgreSQL 16 + pgvector  ←→  Elasticsearch 8.x (BM25)
        ↓
  Hybrid Retrieval (BM25 → BM25+dense → HippoRAG PPR)
        ↓
  Swappable LLM Layer (extractive / Ollama / Claude / OpenAI)
        ↓
  FastAPI — structured JSON response + passage-level citations
```

**Phases:**
- **Phase 1 (now):** BM25 + extractive responses, no LLM required
- **Phase 2:** PubMedBERT embeddings + hybrid retrieval (RRF)
- **Phase 3:** Pluggable LLM synthesis (Ollama, Claude, OpenAI)
- **Phase 4:** HippoRAG entity graph + SPLADE (no Neo4j)

---

## Quickstart

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.11+
- [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) (free — enables 10 req/s vs 3)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set NCBI_API_KEY and NCBI_EMAIL at minimum
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

### 5. Run ingestion (SLE papers first)

```bash
# Discover and queue SLE core PMIDs (dry run — no fetch yet)
python scripts/ingest.py --topics sle_core --dry-run

# Full ingestion of SLE core + lupus nephritis
python scripts/ingest.py --topics sle_core lupus_nephritis

# All priority-1 topics
python scripts/ingest.py --priority 1

# Filter by date
python scripts/ingest.py --topics sle_core --date-from 2020/01/01

# List all available topics
python scripts/ingest.py --help
```

The pipeline is **resumable** — killing it mid-run and restarting is safe. Already-ingested PMIDs are skipped. Check progress with:

```bash
curl http://localhost:8000/api/v1/ingestion/status
```

### 6. Start the API server

```bash
export PYTHONPATH=$(pwd)
uvicorn src.api.main:app --reload
# API docs: http://localhost:8000/docs
```

Or via Docker:
```bash
docker compose up app
```

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

### Search papers

```bash
curl "http://localhost:8000/api/v1/papers/search?q=lupus+nephritis+treatment&pub_year_from=2018"
```

### MeSH autocomplete

```bash
curl "http://localhost:8000/api/v1/mesh/suggest?q=lupus"
```

### Ingestion status

```bash
curl http://localhost:8000/api/v1/ingestion/status
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

Per-request override (query API `options.llm_provider` field).

**Ollama setup:**
```bash
# Install Ollama: https://ollama.com
ollama pull biomistral   # or: meditron:7b
# Set in .env: LLM_PROVIDER=ollama, OLLAMA_MODEL=biomistral
```

---

## Project Structure

```
src/
  ingestion/
    topics.py           — SLE/autoimmune MeSH query definitions
    pubmed_parser.py    — PubMed XML (EFetch) parser
    jats_parser.py      — PMC full-text JATS XML parser
    pipeline.py         — Resumable ingestion pipeline
  db/
    schema.sql          — PostgreSQL schema (papers, chunks, mesh, entities, etc.)
  search/
    elasticsearch_client.py  — ES index setup + BM25 search
    hybrid_retriever.py      — RRF fusion: BM25 + pgvector
    mesh_expander.py         — MeSH hierarchy query expansion
  embeddings/           — Phase 2: PubMedBERT chunking + NER (coming)
  api/
    main.py             — FastAPI app + all endpoints
    response_builder.py — Structured response + citation provenance
    llm_providers.py    — Swappable LLM provider (ABC + Extractive/Ollama/Claude/OpenAI)
scripts/
  ingest.py             — CLI for running the ingestion pipeline
docker-compose.yml
Dockerfile
```

---

## Ingestion Topics

| Name | Description | Priority |
|---|---|---|
| `sle_core` | Systemic Lupus Erythematosus core | 1 |
| `lupus_nephritis` | Lupus nephritis | 1 |
| `cutaneous_lupus` | Cutaneous lupus | 2 |
| `antiphospholipid` | Antiphospholipid syndrome | 2 |
| `sjogrens` | Sjögren's syndrome | 2 |
| `complement_system` | Complement system in autoimmune disease | 2 |
| `autoimmune_broad` | Broad autoimmune + SLE/lupus | 3 |

Add new topics in `src/ingestion/topics.py`.

---

## Roadmap

- **Phase 2:** PubMedBERT embeddings (`src/embeddings/`), scispaCy NER, hybrid BM25+dense retrieval
- **Phase 3:** LLM synthesis fully wired, citation `[N]` parser, query complexity classifier
- **Phase 4:** HippoRAG entity graph (PostgreSQL-native PPR), SPLADE sparse vectors in Elasticsearch
- **Later:** Expand to additional autoimmune topics, broader MeSH coverage

---

## Notes

- The full PubMed corpus (35M papers) is intentionally not downloaded. Expand scope by adding topics to `src/ingestion/topics.py` and re-running the pipeline.
- Citation provenance is tracked at the chunk level (paragraph + char offsets), not just paper level.
- All responses include the exact passage text that supports each claim.
