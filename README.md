# CureMom — Medical Literature Knowledge Base

> ## ⚠️ NOT MEDICAL ADVICE — RESEARCH PROTOTYPE
>
> **This software is an experimental research tool. Read this before using it.**
>
> - **Not a medical device. Not for clinical use.** Outputs are AI-generated summaries of public research papers, not professional medical opinions. **Never** make a treatment, dosing, or diagnostic decision based on what this system says. **Always consult a licensed healthcare professional** for any medical question that affects you or someone you care about.
> - **LLM outputs can be wrong.** The synthesis layer (Anthropic Claude / OpenAI / NVIDIA NIM) can hallucinate, misread evidence, miscite papers, or omit critical contraindications. The `[N]` citation markers do not guarantee that the cited paper actually supports the claim.
> - **The corpus is incomplete and historical.** We index a slice of PubMed/PMC at a point in time. Newer research, retractions, drug-label changes, and safety signals after that snapshot are not reflected.
> - **Drug references come from openFDA + Wikipedia.** Coverage is not exhaustive, may be out of date, and does not include every approved indication, contraindication, or interaction. Region-specific approvals and prescribing information can differ.
> - **No warranty.** Provided as-is, no fitness for any purpose, no liability for any decision made on the basis of its output. Logs, queries, and any data you put into a public deployment may be visible to whoever operates that deployment — treat it as **public**, not as PHI.
> - **Drug names mentioned ≠ endorsement.** A medication appearing in an answer means the literature mentions it, not that it is appropriate, safe, or available for your situation.
>
> If you or someone you know may be in a medical emergency, call your local emergency number immediately.

---

A system that ingests peer-reviewed medical papers from PubMed/PMC and answers queries with reasoned, citation-grounded responses. Built on first-principles drug discovery: the system starts from molecular biology and discovers compound→pathway→condition connections from evidence, rather than encoding assumptions about what treats what.

**Starting scope:** Systemic Lupus Erythematosus (SLE), autoimmune disease, muscle physiology/hypertrophy, ichthyosis, and creatine pharmacology + clinical evidence.

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
  Swappable LLM Layer (extractive / Anthropic Claude / OpenAI / NVIDIA NIM)
        ↓
  FastAPI ─ async-job + polling endpoints (compute-bound)
        │       POST /query/async → {job_id}
        │       GET  /query/job/{id} → {status, stage, payload?}   ← Cache-Control: no-store
        ↓
   serveo / Cloudflare Tunnel ─ HTTPS edge with ~10 s response cap
        ↓
   Browser ─ Cytoscape.js knowledge-graph panel, polls every 150 ms → 1.5 s
```

**Phases (current state):**
- **Phase 1** ✅ live — BM25 retrieval, extractive responses, no LLM required
- **Phase 2** ✅ live — section-aware chunking + PubMedBERT (768-dim) embeddings on **all 866K chunks** (abstract + intro/methods/results/discussion from 34,596 OA full-text papers), HuggingFace transformer NER (`d4data/biomedical-ner-all`) over **1.23M entities**, HNSW vector index built
- **Phase 3** ✅ live — pluggable LLM (Anthropic Claude / OpenAI / NVIDIA NIM / extractive) with per-request model selection in the UI; the dropdown auto-populates from `/llm/status` based on which API keys are configured. Patient-mode prompt with clickable follow-up suggestions, query-complexity classifier, citation verifier (catches hallucinated `[N]` indices and weakly-supported claims).
- **Phase 4** ✅ live — HippoRAG Personalized PageRank over a **5.27M-edge entity graph** (built from MeSH descriptors merged with NER co-occurrences), SPLADE sparse-vector pipeline ready to encode
- **Async-job runtime** ✅ live — every compute-bound endpoint (`/query`, `/graph_extract`, `/graph_dedup`) is fronted by a POST that returns a `job_id` and a GET that polls the in-memory job store. Each HTTP exchange stays well under any tunnel's wall-clock cap; the LLM call runs on a daemon thread for as long as it needs. Poll responses carry `Cache-Control: no-store` so proxies can't shadow a "pending" forever. See **Runtime Architecture** below.
- **Drug reference layer** ✅ live — **1,719 FDA drug labels** (openFDA) + Wikipedia fallback for older/discontinued drugs. An LLM query analyzer routes each request: drug-name questions trigger a forward FDA lookup; effect/condition questions ("drugs for muscle relaxation") trigger reverse FTS with LLM-expanded clinical synonyms. Drug lookup runs **concurrently** with retrieval (both depend only on the effective query) so the slower of the two — not their sum — gates the LLM call.
- **Multi-turn conversation** ✅ live — patient chat sends a rolling history (last 6 turns) with each request. Past Q+A appear bare (no [N] markers, no boilerplate, no drug cards re-injected); only the current turn carries full retrieval context. Pronouns ("its side effects") resolve via the analyzer + retrieval-side query fusion.
- **Live knowledge-graph panel** ✅ live — split-page chat with a Cytoscape.js canvas on the right that grows turn-by-turn. The LLM emits directed relations from the question + answer plus a `types` map classifying each concept into Drug / Disease / Gene / Anatomy / Symptom / Other; each unique subject/object becomes a node, colored by type to match the legend. Grounding: labels must appear as a substring of the answer text (answer is the authoritative source). Vague predicates ("is managed by", "involves") and generic single-word labels ("protein", "RNA") are dropped server-side. Web-like fcose layout (debounced), pan/zoom/fit controls, **node-search** input, click popover with citation pills + "Ask about this" + "Remove", **Merge** button calls the LLM to dedup equivalent entities ("B-cell" ≡ "B cell" ≡ "B cells"). The QA dropdown choice drives graph extraction + dedup: pick `claude` and the whole pipeline goes through Anthropic; pick `nim` and it goes through NVIDIA NIM (free tier MiniMax-M2.7). Partial-output recovery: when the LLM hits `max_tokens` mid-JSON the parser salvages every complete relation; the panel flags "Graph partial — truncated …" so the user knows to shorten the question or raise the cap. An inflight token discards the older result when the user fires a new question before the previous graph finishes — only the latest answer's graph is applied.
- **Ichthyosis corpus** ✅ ingested — 15,613 papers across 7 ichthyosis-related MeSH topics (core, lamellar, X-linked, harlequin/EHK, genetics, treatment, skin barrier biology). Embedding/NER/full-text pipelines running.
- **Creatine corpus** 🟡 partial — 10,432 PubMed records ingested + abstracts embedded; full-text JATS XML fetched for 3,186 OA papers (2026-05-10). Section-aware chunking + embedding of the full text **not yet run** — those papers currently surface in retrieval at abstract level only.

**Default retrieval strategy:** `full` = BM25 + dense + HippoRAG PPR rerank.

**Default LLM provider:** `nim/minimaxai/minimax-m2.7` (NVIDIA NIM free tier, ~40 RPM cap). Falls back to `claude` (Haiku 4.5) or `openai` (gpt-4o) if those keys are configured. Per-request override via the dropdown in the UI. Configure via `LLM_PROVIDER`, `NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` in `.env`.

---

## Runtime Architecture

The same query that returns in 800 ms locally takes 20–60 s end-to-end on Sonnet through a public tunnel — and serveo (a common free option) closes any HTTP response that hasn't finished within ~10 s, no matter how much keep-alive traffic the server emits. So compute-bound endpoints don't return their result inline; they spawn a background job and let the client poll.

```
┌──────────────────────────────────────────────────────────────────────┐
│  POST /api/v1/{query,graph_extract,graph_dedup}/async (or POST /…)   │
│       → {"job_id":"abc123"}                returns in <1 s            │
│                                                                      │
│  daemon thread:  work() → _update(stage=…) → _update(status="done")  │
│                                                                      │
│  GET  /api/v1/{…}/job/{id}                ← every 150 ms then 1.5 s   │
│       → 200 + {status,stage?,payload?}     Cache-Control: no-store   │
│       → 404 + {detail}                     Cache-Control: no-store   │
└──────────────────────────────────────────────────────────────────────┘
```

**Why each piece is there:**

- **`Cache-Control: no-store, no-cache, must-revalidate` on every poll response (and the 404 path too).** Without this, serveo's HTTP/2 edge caches the first `{"status":"pending"}` and re-serves it for the entire polling window — the client polls for 10 minutes, never sees "done", and the user thinks the graph is broken. The 404 path is wrapped for the same reason (a cached 404 from one job could phantom-block a future job_id collision).
- **First poll fires at 150 ms, not 1500 ms.** A fast LLM call (Haiku on a small graph) routinely finishes in <500 ms; the prior 1500 ms first-poll delay was visible to the user. Subsequent polls fall back to the configured cadence.
- **Per-poll `AbortSignal.timeout(8000)`.** A single hung fetch (tunnel dropped mid-response) used to stall the overall polling deadline because the wall-clock check ran *between* polls. Each fetch is now self-canceling; the loop retries cleanly.
- **Per-key concurrent-job semaphore (`_KEY_JOB_LIMIT = 3`).** Bounds memory + LLM spend per API key. The fourth concurrent job from one key returns `429 too many concurrent jobs` instead of spawning another daemon thread.
- **In-process auth-key cache (60 s TTL).** Without it, every poll opens a fresh `psycopg.connect()` to validate the same `X-API-Key`. At 1.5 s polling intervals over a 30 s job that's ~20 DB connections per job. The cache is invalidated on revoke; negative results are not cached so a freshly minted key isn't blocked by an earlier 401.
- **Inflight token on the frontend.** If the user fires Q2 while Q1's `graph_extract` is still polling, Q1's eventual result is discarded (token mismatch) rather than re-shuffling the canvas after Q2 already merged.
- **Layout debounce (250 ms).** Cytoscape's fcose layout takes ~120 ms; back-to-back `merge()` + `applyMergeGroups()` collapse to one layout instead of running them in series.
- **In-memory job store (`_jobs` dict, 1200 s TTL, hard cap 4096).** No Redis, no SQLite — fine for the current single-worker uvicorn deployment. The trade-off: jobs are lost on server restart, so polling clients get a 404 and have to re-ask. Going multi-worker is a deliberate later step that requires moving `_jobs` to shared storage.
- **HippoRAG + PubMedBERT loaders behind a double-checked lock.** First-callers race to instantiate the heavy singleton; the lock ensures only one load happens.

The legacy `POST /api/v1/query` (synchronous) and `POST /api/v1/query/stream` (SSE) endpoints still exist for callers that haven't migrated, but they break through any tunnel with a wall-clock cap. New integrations should use the async + polling pair.

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
  -H 'X-API-Key: <your-key>' \
  -d '{"query":"What drugs target the complement pathway in lupus nephritis?",
       "options":{"retrieval_strategy":"full","top_k":20,"llm_provider":"nim"}}'
```

Multi-hop questions like the example above benefit most from `hipporag`/`full`
because PPR surfaces chunks that *bridge* the query entities ("complement",
"drug", "lupus", "nephritis") even if no single chunk mentions all four.
Verified live: returns 5 relevant papers in ~2.3s on the 1M-edge filtered graph.

### 11. (Recommended) Pull FDA drug labels into the lookup table

OpenFDA's Drug Label API gives ~1.7K commonly-prescribed modern drugs with
indications, mechanism of action, pharmacology, dosing, contraindications,
warnings, and interactions. Wikipedia fills in older/discontinued drugs
(e.g. mephenoxalone) that the FDA no longer hosts current SPLs for.

```bash
PYTHONPATH=. python scripts/fetch_fda_drugs.py --max-pages 50
```

After this populates `fda_drugs`, every query that mentions a drug (NER-
detected) gets that drug's structured info prepended to the LLM context
as authoritative reference data — separate from the citation-bearing
literature passages. Verified on `mephenoxalone` (Wikipedia fallback) and
`atorvastatin` (FDA path), both surface correctly in `metadata.drug_cards`.

### 12. (Phase 4 — optional) SPLADE sparse vectors

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

### Async query with citations (recommended)

The async + polling pair is what the UI uses and what survives the tunnel cap. The work runs on a daemon thread; each round-trip stays under any proxy timeout.

```bash
# 1. Kick off the job — returns immediately with a job_id
curl -X POST http://localhost:8000/api/v1/query/async \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{
    "query": "What is the efficacy of hydroxychloroquine in SLE?",
    "options": {"top_k": 20, "retrieval_strategy": "full", "llm_provider": "nim"}
  }'
# → {"job_id":"a1b2c3d4..."}

# 2. Poll until done (every ~1.5s; first poll can fire immediately)
curl http://localhost:8000/api/v1/query/job/a1b2c3d4... \
  -H "X-API-Key: <your-key>"
# → {"status":"pending","stage":"synthesizing","model":"claude-haiku-4-5"}
# → {"status":"done","payload":{ ...the full response below... }}
```

The poll response carries `Cache-Control: no-store, no-cache, must-revalidate` — don't strip these in any intermediate proxy.

The same shape works for `/graph_extract` and `/graph_dedup`:

```bash
curl -X POST http://localhost:8000/api/v1/graph_extract \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query":"...","answer":"...","chunks":[{"id":1,"text":"..."}],"llm_provider":"nim"}'
# → {"job_id":"..."}
# then poll /api/v1/graph_extract/job/{job_id} until status=done

curl -X POST http://localhost:8000/api/v1/graph_dedup \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"labels":["B-cell","B cell","B cells","IGF-1","IGF1"],"llm_provider":"claude"}'
# → {"job_id":"..."}
# then poll /api/v1/graph_dedup/job/{job_id} until status=done
```

The fourth concurrent job from one key returns `429 too many concurrent jobs (max 3)`.

### Synchronous query (legacy)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{"query":"...","options":{"top_k":20,"retrieval_strategy":"full","llm_provider":"nim"}}'
```

This response shape is what the async job's `payload` returns; only use the sync endpoint locally or behind a proxy with no wall-clock cap.

`retrieval_strategy` is one of:
- `bm25` — Elasticsearch keyword scoring only (~20 ms; baseline)
- `hybrid` — BM25 + dense vector RRF fusion (degrades to BM25 if embeddings missing)
- `hipporag` — BM25 + HippoRAG entity-graph PPR rerank (~2 s)
- `full` — BM25 + dense + HippoRAG (default)

`llm_provider` is one of `extractive` / `claude` / `openai` / `nim`. Defaults to whatever `LLM_PROVIDER` is in `.env`. Per-request override accepts a `<provider>/<model>` form for fine-grained control (e.g. `nim/minimaxai/minimax-m2.7`, `claude/claude-haiku-4-5-20251001`).

All write endpoints (`/query`, `/query/async`, `/graph_extract`, `/graph_dedup`, `/keys/*`) require an `X-API-Key` header. See **Public deployment** below.

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
| `extractive` | No LLM — returns top sentences with citations | None (always works) |
| `nim` | NVIDIA NIM (OpenAI-compatible, free tier) | `NVIDIA_API_KEY`; default model `minimaxai/minimax-m2.7` (set via `NIM_MODEL`) |
| `claude` | Anthropic Claude API | `ANTHROPIC_API_KEY`; default model Haiku 4.5 (set via `CLAUDE_MODEL`) |
| `openai` | OpenAI API | `OPENAI_API_KEY`; default `gpt-4o` (set via `OPENAI_MODEL`) |

LLM providers can also be specified per request via `options.llm_provider` in the query body, or picked from the dropdown in the UI (auto-populated from `/llm/status`). The same dropdown choice drives graph extraction and graph dedup so the answer + the knowledge graph + the merge come from one model.

The drug-aware query analyzer always uses Claude Haiku for speed and JSON reliability, regardless of the QA dropdown choice.

---

## Public Deployment

The app is designed to run behind a Cloudflare Tunnel with an API-key gate.

### 1. API-key auth

`/query`, `/graph_extract`, `/graph_dedup`, and `/keys/*` all require `X-API-Key`. Two-tier key model:

- **Admin keys** can mint unlimited child keys via `POST /api/v1/keys/generate`.
- **Each non-admin key** can mint exactly one child key (good for friend-of-a-friend invitations without unbounded fan-out).

Bootstrap (one-time):

```bash
PYTHONPATH=. python3 -c "
import psycopg, os
from src.api.auth import init_keys_table, bootstrap_admin_key
dsn = f\"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@localhost:5432/{os.environ['POSTGRES_DB']}\"
with psycopg.connect(dsn) as conn:
    init_keys_table(conn)
    print('ADMIN_KEY:', bootstrap_admin_key(conn))
"
```

The admin key prints to stdout once. Save it. To pin a specific value across restarts, set `INITIAL_ADMIN_KEY` in `.env` before the first run. Keys live in the `api_keys` table; revoke a key by `UPDATE api_keys SET is_revoked = TRUE WHERE id = X`.

The frontend prompts for an API key on first use, stores it in `localStorage`, and sends `X-API-Key` automatically. The "Share access" button in the consumer sidebar mints a child key.

### 2. Pick a tunnel

**Option A — serveo (zero-signup SSH reverse tunnel).** Free, no account, ~10 s wall-clock cap on any individual HTTP response (which is exactly why the async-poll architecture exists — see **Runtime Architecture**). serveo drops connections periodically with no SLA; the helper script wraps the ssh in a respawn loop so a flap recovers in seconds without manual intervention.

```bash
# Foreground (Ctrl-C stops it):
./scripts/tunnel.sh                  # forwards localhost:8000 as curemom.serveo
SUBDOMAIN=mylab ./scripts/tunnel.sh 9000

# Background:
nohup ./scripts/tunnel.sh > /tmp/curemom-tunnel.log 2>&1 &
```

The script applies the SSH options we settled on: `ServerAliveInterval=30`, `ServerAliveCountMax=3`, `ExitOnForwardFailure=yes`, accept-new host keys. Public URL prints on the first successful SSH session.

**Option B — Cloudflare Tunnel (named, free, no wall-clock cap).** Slower to set up (requires a Cloudflare account) but the URL is stable and there's no 10 s response cap.

```bash
brew install cloudflared

# 1. Sign in (browser opens; uses a free Cloudflare account)
cloudflared tunnel login

# 2. Create the tunnel — generates a credentials JSON in ~/.cloudflared/
cloudflared tunnel create curemom

# 3. Run it (URL form: bypasses config file, sufficient for our case)
cloudflared tunnel run --url http://localhost:8000 curemom
```

Cloudflare prints a `*.cfargotunnel.com` URL. Pair it with a domain via `cloudflared tunnel route dns curemom curemom.example.com` to get a clean URL. Optionally add **Cloudflare Access** (free for ≤50 users) for an SSO layer in front of the API key gate. For ephemeral use without a Cloudflare account: `cloudflared tunnel --url http://localhost:8000` gives a one-shot `*.trycloudflare.com` URL.

### 3. Launching the server

```bash
# Foreground — best for development.
cd /path/to/CureMom
python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level info

# Add --reload to auto-restart on save (caveat: drops in-flight jobs → polling
# clients see 404 on next poll, have to re-ask):
python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level info --reload

# Background:
nohup python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level info > /tmp/curemom.log 2>&1 &
```

The server is single-worker by design — `_jobs` is a process-local dict, so going multi-worker requires moving the job store to Redis/SQLite first. For 1–5 concurrent users this is fine; the threadpool serves polls and the LLM call runs on a daemon thread.

### 4. Operational notes

- The `.env` file (with API keys) never leaves the host machine — it's gitignored and read by uvicorn directly.
- The `/llm/status` endpoint is **not** auth-gated — it's used by the frontend to populate the provider dropdown before the user enters a key. It only reports availability + model strings, never key material.
- Restart uvicorn to pick up server-side changes (when not running with `--reload`): `pkill -f "uvicorn src.api.main"` then relaunch. The SSH tunnel can stay up — it'll briefly 502 during the gap, then reconnect to the new uvicorn.
- Revocations: set `is_revoked = TRUE` in `api_keys` and call `src.api.auth.invalidate_key_cache(<key>)` (or restart) so the 60-s key cache doesn't keep accepting the dead key.

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
    main.py               — FastAPI app + endpoints + async job store + per-key semaphore
    auth.py               — X-API-Key dependency + 60 s in-process cache (admin / child-key model)
    response_builder.py   — Structured response + passage-level citation provenance
    llm_providers.py      — Swappable LLM (Extractive / Claude / OpenAI / NVIDIA NIM)
    classifier.py         — Query complexity classifier (Phase 3)
    citation_verifier.py  — Citation [N] parser + lexical-overlap check (Phase 3)
    drug_lookup.py        — FDA/openFDA + Wikipedia drug card lookup (concurrent with retrieval)
    query_analyzer.py     — Claude Haiku JSON analyzer (drug/effect routing, synonym expansion)
    graph_extractor.py    — LLM relation extraction (claude/openai/nim) + partial-JSON recovery
    graph_merger.py       — LLM entity-dedup for the Merge button
scripts/
  ingest.py               — CLI: run the ingestion pipeline (PubMed metadata + abstracts)
  fetch_chembl.py         — CLI: ChEMBL compound-target data + PubMed queuing
  fetch_pmc.py            — CLI: full-text JATS XML for OA papers → paper_sections
  sync_elasticsearch.py   — CLI: backfill PostgreSQL papers → Elasticsearch
  embed.py                — CLI: chunk full text + embed with PubMedBERT (Phase 2)
  extract_entities.py     — CLI: HF transformer NER on chunks (Phase 2)
  build_entity_graph.py   — CLI: MeSH/NER entity graph builder (Phase 4)
  encode_splade.py        — CLI: SPLADE sparse vectors → ES (Phase 4)
  tunnel.sh               — serveo reverse-tunnel wrapper with respawn-on-drop
tests/
  test_graph_parser.py             — _parse_graph + partial-JSON recovery (10 cases)
  test_job_poll_headers.py         — Cache-Control: no-store on every poll path
  test_auth_cache.py               — require_api_key TTL cache (hit / invalidate / expiry / 401 not cached)
  test_job_semaphore.py            — _KEY_JOB_LIMIT = 3 enforcement + concurrent contention
  test_graph_input_truncation.py   — extract_graph surfaces "truncated input:" when answer > 5 KB
  test_tunnel_integration.py       — opt-in (CUREMOM_TUNNEL_URL + CUREMOM_API_KEY) end-to-end against the live tunnel
frontend/
  index.html              — split-view chat + graph panel
  app.js                  — async polling (QA + graph), inflight token, AbortSignal-timed fetches
  graph.js                — Cytoscape.js + fcose, applyMergeGroups, debounced _scheduleLayout
  style.css               — dark theme + responsive split
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
- ✅ **Phase 3:** Cloud-only LLM dispatch (extractive / Anthropic Claude / OpenAI / NVIDIA NIM); `[N]` citation parser; query classifier (Claude Haiku); citation verifier; `/llm/status` health endpoint
- ✅ **Phase 4:** HippoRAG entity graph (~5M+ edges from MeSH + NER co-occurrence) with NetworkX Personalized PageRank rerank — no Neo4j; SPLADE sparse-vector pipeline ready
- ✅ **Async-job runtime:** every compute-bound endpoint runs as POST→job_id + GET-poll-until-done with `Cache-Control: no-store` on every poll path; per-key concurrent-job semaphore (limit 3, 429 on overflow); in-process API-key cache (60 s TTL); double-checked locks around the heavy lazy singletons (PubMedBERT, HippoRAG)
- ✅ **Live knowledge-graph panel:** Cytoscape.js canvas; relations-only LLM extraction with answer-text grounding (no NER at query time); web-like fcose layout (debounced); type-based node coloring (Drug / Disease / Gene / Anatomy / Symptom / Other); click popover with Ask/Remove; **Merge** button for LLM-driven dedup of equivalent entities; **node search** in the topbar; provider dispatch — picks Anthropic / OpenAI / NIM based on the QA dropdown; partial-output recovery + truncation surfacing; inflight token discards stale `graph_extract` results when the user fires a new question before the previous graph returns
- ✅ **Drug lookup parallelized with retrieval:** the two stages run on separate threads instead of in series, so drug-mentioning queries no longer pay the full ~1–3 s FDA round-trip on top of retrieval
- ✅ **Tunnel auto-reconnect:** `scripts/tunnel.sh` wraps the serveo SSH command in a respawn loop with backoff so a drop recovers in seconds without manual intervention
- ✅ **Adaptive retrieval top-k:** retriever pulls a 100-candidate pool and returns top 10% (floor 5, cap 20) instead of a fixed top-k=10/12; broad queries get more chunks, narrow factual lookups get fewer
- ✅ **Public-deployment auth:** X-API-Key on /query / /graph_extract / /graph_dedup; admin keys mint unlimited child keys, each non-admin key mints exactly one. Bootstrap admin key created on first run via `auth.bootstrap_admin_key`.
- ✅ **Corpora ingested:** SLE + muscle physiology + ichthyosis (15,613 papers, 7 MeSH topics) + creatine (10,432 papers, pharmacology + clinical)
- ✅ **Test suite:** 29 in-process cases (graph parser, poll headers, auth cache, job semaphore, input truncation) + 9 opt-in tunnel cases gated on `CUREMOM_TUNNEL_URL`

**Pending offline runs (heavy compute):**
- Run `scripts/embed.py --chunk-fulltext` on the creatine corpus — section-aware chunks for the 2,586 OA full-text papers are not yet generated (only abstract chunks exist; queries surface abstract-level only)
- Run `scripts/encode_splade.py` — populate Elasticsearch `sparse_vector` field (~2 hr)

**Next:**
- Move `_jobs` to Redis/SQLite so uvicorn can run with `--workers N` without losing in-flight jobs
- Cross-encoder re-rank on top-20 (e.g. `ms-marco-MiniLM-L-12-v2`)
- Expand corpus: RA, myositis, systemic sclerosis, sarcopenia, exercise transcriptomics

---

## Frontend (Tailwind CSS, no SPA)

The frontend is vanilla HTML/JS plus a Tailwind v4 stylesheet. There's no bundler, no JSX, no framework runtime — just a single CLI step that compiles `frontend/src.css` (edit this) → `frontend/style.css` (committed; served as-is by uvicorn).

```bash
cd frontend
npm install               # one-time
npm run dev               # watch mode — edit src.css, style.css regenerates on save
npm run build             # one-shot minified build before commit
```

Why Tailwind v4 here:
- `@theme inline` in `src.css` mirrors the existing CSS variables, so utility classes (`bg-accent`, `text-secondary`, `bg-node-disease`, …) automatically follow the `prefers-color-scheme` light/dark flip we already had.
- New components in `index.html` can use utility classes directly. The existing 1,600 lines of bespoke CSS (`.msg-ai`, `.citation-pill`, `.graph-panel`, …) keep working untouched — they live in the cascade below the `@theme` block.
- `graph.js` reads `--node-chemical / --node-disease / …` via `getComputedStyle` on init; those raw vars stay on `:root` so the graph palette never had to migrate.
- No bundler step: deploy is still "edit + push" once `npm run build` regenerates `style.css`.

When editing the frontend:
1. Edit `frontend/src.css` (or add Tailwind utility classes directly in `index.html`).
2. Run `npm run build` in `frontend/`.
3. Bump the `style.css?v=NN` query string in `index.html` so cached browsers re-fetch.
4. Commit both `src.css` and the regenerated `style.css`.

---

## Tests

`pytest tests/` runs the in-process suite end-to-end against a `TestClient` (no Postgres / Elasticsearch / network needed). The opt-in tunnel suite is the only piece that hits the real wire:

```bash
# In-process suite (29 cases, <1s):
pytest tests/

# End-to-end through the live tunnel — set both env vars first:
CUREMOM_TUNNEL_URL=https://...serveousercontent.com \
CUREMOM_API_KEY=<your-key> \
pytest tests/test_tunnel_integration.py -v
```

Coverage:

| File | What it pins down |
|---|---|
| `test_graph_parser.py` | `_parse_graph` survives every shape of malformed LLM JSON: clean, fenced, truncated mid-relation, truncated after complete relations (salvage + `_truncated` flag), invalid type label, wrong-shape fields. |
| `test_job_poll_headers.py` | Every poll path — pending, done, and 404 unknown-job — returns `Cache-Control: no-store, no-cache, must-revalidate`. Without this the production "graph never shows" bug regresses. |
| `test_auth_cache.py` | Second `require_api_key` call within 60 s skips the DB; invalidation drops the entry; expiry forces a re-fetch; failed lookups (401) are NOT cached so a freshly minted key isn't blocked. |
| `test_job_semaphore.py` | A single key can take exactly 3 slots; the 4th `_acquire_key_slot` returns `False` (route handler turns it into 429); released slots are reusable; keys are isolated; 20 concurrent threads acquire exactly 3. |
| `test_graph_input_truncation.py` | A >5000-char answer sets `payload.error` starting with `truncated input:` so the frontend regex catches it. A short answer doesn't trigger any truncation note. |
| `test_tunnel_integration.py` | Opt-in. Exercises serveo + the live LLM with the SLE question, the graph_extract flow, and the merge flow; confirms `Cache-Control` headers actually survive the proxy. |

---

## Design Notes

- The full PubMed corpus (35M papers) is intentionally not downloaded. Expand scope by adding topics to `src/ingestion/topics.py`.
- Citation provenance is tracked at chunk level (section + paragraph + char offsets), not just paper level.
- All LLM responses are grounded: every claim requires an inline `[N]` citation to an ingested chunk.
- The knowledge-graph LLM is **never trusted to invent entities**: every subject/object label must appear as a case-insensitive substring of the answer text, and vague predicates ("is managed by", "involves") + generic single-word labels ("protein", "RNA") are dropped server-side. The "answer is the authoritative source" rule means the graph can never assert relations the user didn't see in the prose.
- Compute-bound endpoints are **always** async + polling — never inline, never SSE on long calls. SSE was tried first and broke through serveo at the 10 s mark; the polling pattern is what every long-running cloud API (OpenAI runs, HF endpoints, Vertex LRO) settled on for the same reason.
- The job store is **deliberately in-memory** for now. A Redis/SQLite migration is a one-day change but unnecessary until we go multi-worker. The current trade-off (lost on restart → polling clients see 404) is acceptable for ≤5 concurrent users.
- ChEMBL queries are by **molecular target** (e.g. JAK1), not by compound name — this avoids encoding prior beliefs about which drugs work.
- `targets.py` is a 30-target seed list for fast local testing. For production, use `--all-targets` to cover the full ~3,000-target clinical druggable proteome.
