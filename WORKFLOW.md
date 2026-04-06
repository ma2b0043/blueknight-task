# System Workflow — Complete Blueprint

A step-by-step breakdown of how every user query is processed, which files are called, which functions execute, and how data flows through the system.

---

## Table of Contents

1. [System Boot (Server Startup)](#1-system-boot)
2. [Offline Data Pipeline (One-Time Setup)](#2-offline-data-pipeline)
3. [Endpoint 1: POST /agent/refine](#3-endpoint-1-post-agentrefine)
4. [Endpoint 2: POST /search/run](#4-endpoint-2-post-searchrun)
5. [File Map](#5-file-map)
6. [Complete Call Graph](#6-complete-call-graph)

---

## 1. System Boot

**What happens when you run `uvicorn app.main:app`:**

```
Terminal: uvicorn app.main:app
         │
         ▼
┌─ app/main.py ─────────────────────────────────────────────┐
│                                                            │
│  1. Configure logging                                      │
│     - "blueknight" logger → DEBUG level                    │
│     - Console handler → INFO (minimal)                     │
│     - File handler → DEBUG (pipeline.log)                  │
│     - Suppress noisy libs (httpx, transformers, torch)     │
│                                                            │
│  2. Create FastAPI app                                     │
│                                                            │
│  3. @app.on_event("startup")                               │
│     ├── VectorStoreClient.get_instance()                   │
│     │   └── Loads FAISS index + metadata from disk         │
│     │       File: app/services/vector_store.py             │
│     │       Reads: data/faiss.index + data/companies.pkl   │
│     │                                                      │
│     └── warmup_embedder()                                  │
│         └── Loads BAAI/bge-small-en-v1.5 into memory       │
│             File: app/services/embedder.py                 │
│                                                            │
│  Server ready on http://localhost:8000                      │
└────────────────────────────────────────────────────────────┘
```

**Why:** Pre-loading the FAISS index (~1000 vectors) and embedding model (~33M params) at startup eliminates cold-start latency on the first request. Without this, the first query would take ~6 seconds.

---

## 2. Offline Data Pipeline

**Run once before the server starts. Converts raw data → searchable index.**

```
Step 1: python scripts/prepare_data.py
┌─ scripts/prepare_data.py ─────────────────────────────────┐
│                                                            │
│  Input:  company_1000_data.xlsx (root directory)           │
│  Output: data/companies.csv                                │
│                                                            │
│  What it does:                                             │
│  1. Read xlsx with pandas + openpyxl                       │
│  2. Rename columns:                                        │
│     "Consolidated ID" → "id"                               │
│     "Company Name"    → "company_name"                     │
│     "Country"         → "country"                          │
│     "Long Offering"   → "long_offering"                    │
│  3. Convert id to string, drop rows with empty offering    │
│  4. Write CSV to data/companies.csv                        │
└────────────────────────────────────────────────────────────┘

Step 2: python scripts/build_index.py
┌─ scripts/build_index.py ──────────────────────────────────┐
│                                                            │
│  Input:  data/companies.csv                                │
│  Output: data/faiss.index + data/companies.pkl             │
│                                                            │
│  What it does:                                             │
│  1. Load CSV into pandas DataFrame                         │
│  2. Load BAAI/bge-small-en-v1.5 model                     │
│  3. Embed every company's long_offering text               │
│     → 1000 companies × 384 dimensions = 384,000 floats    │
│     → Normalized vectors (unit length)                     │
│  4. Create FAISS IndexFlatIP (inner product = cosine sim)  │
│  5. Add all vectors to index                               │
│  6. Save index → data/faiss.index                          │
│  7. Extract metadata (id, name, country, offering)         │
│  8. Pickle metadata → data/companies.pkl                   │
│                                                            │
│  Takes ~2 minutes on M1 Mac                                │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Endpoint 1: POST /agent/refine

**The main endpoint. Takes a natural language query and iteratively refines it.**

### 3.1 — Request Arrives

```
User sends:
POST /agent/refine
{
  "message": "Fintech companies not focused on payments",
  "base_query": null,
  "history": [],
  "max_iterations": 3
}
```

```
app/main.py → refine()
│
│  Pydantic auto-validates request into RefineRequest
│  (app/schemas.py — auto-generates trace_id UUID)
│
└── Creates QueryRefinerAgent() and calls agent.refine(request)
    File: app/services/refiner.py
```

### 3.2 — Refinement Loop Begins

```
app/services/refiner.py → QueryRefinerAgent.refine()
│
│  Initialize tracking variables:
│  - best_query = None
│  - best_quality = -1
│  - prev_quality = None
│  - prev_summary = None
│
└── FOR iteration = 1 to max_iterations:
    │
    │  ┌─────────────────────────────────────────┐
    │  │  STEP A: Call LLM to Refine Query       │
    │  │  STEP B: Run Search Pipeline            │
    │  │  STEP C: Evaluate Results               │
    │  │  STEP D: Decide — Stop or Continue      │
    │  └─────────────────────────────────────────┘
```

### 3.3 — STEP A: LLM Call (Query Refinement)

```
QueryRefinerAgent._call_llm()
│
├── _build_user_prompt()
│   │
│   │  Iteration 1 prompt:
│   │  ┌────────────────────────────────────────────────┐
│   │  │ User query: Fintech companies not focused on   │
│   │  │ payments                                        │
│   │  └────────────────────────────────────────────────┘
│   │
│   │  Iteration 2+ prompt (includes feedback):
│   │  ┌────────────────────────────────────────────────┐
│   │  │ User query: Fintech companies not focused on   │
│   │  │ payments                                        │
│   │  │                                                 │
│   │  │ Previous structured query: {...}                │
│   │  │                                                 │
│   │  │ Previous iteration results summary:             │
│   │  │ - Result count: 50                              │
│   │  │ - Top score: 0.72                               │
│   │  │ - Score spread (top 5): 0.03                    │
│   │  │ - Filter drop ratio: 3.0%                       │
│   │  │ - Drop reasons: {"exclude_term": 6}             │
│   │  │                                                 │
│   │  │ Please refine the query to improve results.     │
│   │  └────────────────────────────────────────────────┘
│
├── Anthropic API call
│   │  Model: claude-sonnet-4-20250514
│   │  System prompt: "You are a search query optimizer..."
│   │  Max tokens: 512
│   │  File: app/config.py (REFINER_MODEL, ANTHROPIC_API_KEY)
│   │
│   └── LLM returns raw text like:
│       {
│         "query_text": "fintech financial technology...",
│         "geography": [],
│         "exclusions": ["payments", "payment processing"]
│       }
│
├── parse_json_contract(raw_text)
│   │  File: app/utils/json_contract.py
│   │
│   │  Parsing strategy (in order):
│   │  1. Strip markdown fences (```json ... ```)
│   │  2. Try direct JSON.parse
│   │  3. Regex extract first {...} block, try parse
│   │  4. If all fail → {"_parse_error": True, "raw": "..."}
│   │
│   └── Returns: {"query_text": "...", "geography": [...], "exclusions": [...]}
│
└── _normalise(parsed, fallback_text)
    │
    │  Merge with defaults — ensures no field is ever missing:
    │  - query_text: use parsed value, or fallback to user message
    │  - geography: ensure list of strings
    │  - exclusions: ensure list of strings
    │
    └── Returns: QueryPayload(
          query_text="fintech financial technology...",
          geography=[],
          exclusions=["payments", "payment processing"]
        )
```

### 3.4 — STEP B: Run Search Pipeline

```
The refiner calls SearchPipeline.run() directly (not via HTTP).
This is the same logic as POST /search/run.

See Section 4 below for the complete 3-stage breakdown.

Input:  SearchRequest(query=refined_query, top_k_raw=200, top_k_final=50)
Output: SearchResponse(results=[...], total=50, diagnostics={...})
```

### 3.5 — STEP C: Evaluate Result Quality

```
_compute_quality(search_response) → float (0.0 to 1.0)
│
│  Four components:
│
│  count_ratio = min(result_count / 10, 1.0)
│  │  Example: 50 results → min(50/10, 1.0) = 1.0
│  │  Rationale: Want at least 10 results
│
│  top_score = best result's final re-ranked score
│  │  Example: 0.72
│  │  Rationale: Best match quality
│
│  spread_score = score difference between #1 and #5 result
│  │  Example: 0.72 - 0.69 = 0.03
│  │  Rationale: Good spread = diverse, confident ranking
│
│  filter_health = 1.0 - (filtered_count / raw_count)
│  │  Example: 1.0 - (6/200) = 0.97
│  │  Rationale: Low filter drops = query aligned with corpus
│
│  quality = 0.30 × 1.0      = 0.300  (count)
│          + 0.30 × 0.72     = 0.216  (top score)
│          + 0.20 × 0.03     = 0.006  (spread)
│          + 0.20 × 0.97     = 0.194  (filter health)
│          ─────────────────────────
│          = 0.716
│
└── Returns: 0.716
```

### 3.6 — STEP D: Termination Decision

```
_evaluate(iteration, search_response, prev_quality) → EvalResult
│
│  Six rules checked in order:
│
│  Rule 1: quality >= 0.75?
│  │  → STOP "Excellent result quality"
│  │  Why: Results are already great, don't waste iterations
│
│  Rule 2: iteration >= 2 AND improvement < 0.02?
│  │  → STOP "Plateau detected"
│  │  Why: Quality isn't improving, further refinement won't help
│
│  Rule 3: reranked_count < 3?
│  │  → CONTINUE "Near-empty results"
│  │  Why: Too few results, query is probably too narrow
│
│  Rule 4: filter_drop_ratio > 80%?
│  │  → CONTINUE "Query misaligned with corpus"
│  │  Why: Most candidates are being filtered out
│
│  Rule 5: top_score < 0.40?
│  │  → CONTINUE "Weak top score"
│  │  Why: Best match is poor, try a different approach
│
│  Rule 6: iteration == 1 AND quality < 0.75?
│  │  → CONTINUE "Always try >= 2 iterations"
│  │  Why: Spec requires loop behavior, not single-pass
│
│  If no rule triggered → STOP "Acceptable quality"
│
└── Returns: EvalResult(quality=0.716, should_stop=False, reason="...")
```

### 3.7 — Loop Continues or Returns

```
IF should_stop == True OR iteration == max_iterations:
│
│  Track the best result across all iterations
│  (highest quality score wins)
│
└── Return RefineResponse:
    {
      "refined_query": {
        "query_text": "fintech financial technology...",
        "geography": [],
        "exclusions": ["payments", "payment processing"]
      },
      "rationale": "Stopped after 2 iteration(s) — Plateau detected",
      "actions": [{"id": "show_results", "label": "Show results"}],
      "iterations_used": 2,
      "meta": {
        "trace_id": "47ab605e-...",
        "final_quality": 0.716,
        "result_count": 50
      }
    }

IF should_stop == False:
│
│  Feed back diagnostics to LLM in next iteration:
│  - Previous query JSON
│  - Result count, top score, score spread
│  - Filter drop ratio and reasons
│
└── Go back to STEP A with enriched prompt
```

### 3.8 — Full Refine Flow (Visual)

```
User: "Fintech companies not focused on payments"
  │
  ▼
╔══════════════════ ITERATION 1 ══════════════════╗
║                                                  ║
║  LLM Input:  "Fintech companies not focused..."  ║
║       ↓                                          ║
║  LLM Output: {query_text, geography, exclusions} ║
║       ↓                                          ║
║  Search Pipeline → 200 raw → 194 filtered → 50  ║
║       ↓                                          ║
║  Quality: 0.74 → Rule 6 fires → CONTINUE        ║
║  (always try ≥2 iterations)                      ║
║                                                  ║
╠══════════════════ ITERATION 2 ══════════════════╣
║                                                  ║
║  LLM Input:  original query + prev diagnostics   ║
║       ↓                                          ║
║  LLM Output: refined {query_text, exclusions}    ║
║       ↓                                          ║
║  Search Pipeline → 200 raw → 192 filtered → 50  ║
║       ↓                                          ║
║  Quality: 0.71 → Rule 2 fires → STOP            ║
║  (plateau: 0.71 vs 0.74, improvement < 0.02)    ║
║                                                  ║
╚══════════════════════════════════════════════════╝
  │
  ▼
Return best iteration (iteration 1, quality 0.74)
```

---

## 4. Endpoint 2: POST /search/run

**The retrieval engine. Three-stage pipeline: recall → filter → rank.**

### 4.1 — Request Arrives

```
User sends (or refiner calls internally):
POST /search/run
{
  "query": {
    "query_text": "sales enablement platform B2B",
    "geography": ["United Kingdom", "Germany"],
    "exclusions": ["CRM", "marketing"]
  },
  "top_k_raw": 200,
  "top_k_final": 5,
  "offset": 0
}
```

```
app/main.py → search_run()
│
└── Creates SearchPipeline() and calls pipeline.run(request)
    File: app/services/search_pipeline.py
```

### 4.2 — Stage 1: Vector Recall

```
SearchPipeline.run() → Stage 1
│
├── retrieve(query="sales enablement platform B2B", top_k=200, trace_id=...)
│   File: app/services/retrieval_wrapper.py
│   │
│   ├── Acquire semaphore (max 5 concurrent searches)
│   │   Config: RETRIEVAL_CONCURRENCY_LIMIT = 5
│   │
│   ├── _retrieve_with_retry()
│   │   │  @retry decorator (tenacity):
│   │   │  - Retries on RetrievalError only
│   │   │  - Max 3 attempts
│   │   │  - Exponential backoff: 0.1s, 0.2s, 0.4s (max 2s)
│   │   │
│   │   └── mock_retrieve(query, top_k)
│   │       File: app/retrieval.py
│   │       │
│   │       ├── embed(query)
│   │       │   File: app/services/embedder.py
│   │       │   │
│   │       │   │  Model: BAAI/bge-small-en-v1.5
│   │       │   │  Input: "sales enablement platform B2B"
│   │       │   │  Output: [0.034, -0.012, 0.089, ...] (384 floats)
│   │       │   │  Normalized to unit length
│   │       │   │
│   │       │   └── Returns: list[float] (384 dimensions)
│   │       │
│   │       ├── VectorStoreClient.get_instance()
│   │       │   File: app/services/vector_store.py
│   │       │   └── Returns singleton (already loaded at startup)
│   │       │
│   │       ├── store.query(embedding, top_k=200)
│   │       │   │
│   │       │   │  1. Convert embedding → numpy float32 array
│   │       │   │  2. Normalize L2 (for cosine similarity)
│   │       │   │  3. FAISS index.search(vector, 200)
│   │       │   │     → Returns: distances[200], indices[200]
│   │       │   │  4. Map each index → metadata dict
│   │       │   │  5. Sanitize NaN values (pandas artifact)
│   │       │   │
│   │       │   └── Returns: [
│   │       │         {"id": "9856595", "company_name": "ComX",
│   │       │          "country": "Germany", "score": 0.816,
│   │       │          "long_offering": "ComX.io primarily..."},
│   │       │         ... (200 results)
│   │       │       ]
│   │       │
│   │       └── Convert to list[CompanyResult] dataclass objects
│   │
│   ├── asyncio.wait_for(timeout=10.0 seconds)
│   │
│   └── Release semaphore
│
└── Result: 200 CompanyResult objects, sorted by vector similarity
    Timing: logged as stage_latency_ms["vector_recall"]
```

### 4.3 — Stage 2: Post-Filter

```
SearchPipeline._post_filter(candidates=200, query, drop_reasons)
│
│  Pre-compute filter values:
│  - normalised_geos = {"united kingdom", "germany"}
│  - exclusions_lower = ["crm", "marketing"]
│
│  FOR each of 200 candidates:
│  │
│  ├── Filter 1: Geography Mismatch
│  │   │  _normalise_geo(candidate.country)
│  │   │  File: app/services/reranker.py
│  │   │
│  │   │  Aliases: uk→united kingdom, us→united states, etc.
│  │   │
│  │   │  Example: "United States" → "united states"
│  │   │           NOT in {"united kingdom", "germany"}
│  │   │           → DROP, increment drop_reasons["geography_mismatch"]
│  │   │
│  │   │  Example: "Germany" → "germany"
│  │   │           IN {"united kingdom", "germany"}
│  │   │           → KEEP
│  │   │
│  │   └── Dropped 167 companies (not UK or Germany)
│  │
│  ├── Filter 2: Exclusion Terms
│  │   │  Check: "crm" in long_offering.lower()?
│  │   │         "marketing" in long_offering.lower()?
│  │   │
│  │   │  Example: offering mentions "CRM integration"
│  │   │           → DROP, increment drop_reasons["exclude_term"]
│  │   │
│  │   └── Dropped 21 companies (mentioned CRM or marketing)
│  │
│  └── Filter 3: Low Vector Score
│      │  Threshold: 0.20 (from config.py)
│      │
│      │  Example: score = 0.15 → DROP
│      │  Example: score = 0.65 → KEEP
│      │
│      └── Dropped 0 companies (all above threshold)
│
└── Result: 200 → 12 candidates remain
    drop_reasons: {"geography_mismatch": 167, "exclude_term": 21}
    Timing: logged as stage_latency_ms["post_filter"]
```

### 4.4 — Stage 3: Re-rank

```
Reranker.rerank(candidates=12, query, top_k=5)
File: app/services/reranker.py
│
├── Tokenize query: "sales enablement platform b2b"
│   │  Remove stopwords → {"sales", "enablement", "platform", "b2b"}
│   └── _tokenize() strips stopwords from _STOPWORDS frozenset
│
├── FOR each of 12 candidates, compute 4 score components:
│   │
│   │  ┌─────────────────────────────────────────────────────────┐
│   │  │ Example: ComX (Germany)                                 │
│   │  │                                                         │
│   │  │ 1. vector_score = 0.816 (from FAISS)                   │
│   │  │    Clamped to [0.0, 1.0]                                │
│   │  │                                                         │
│   │  │ 2. keyword_score = _keyword_score()                     │
│   │  │    offering tokens: {"sales", "enablement", "platform", │
│   │  │                      "b2b", "saas", "automation", ...}  │
│   │  │    overlap = {"sales","enablement","platform","b2b"} = 4│
│   │  │    score = 4 / 4 = 1.0                                 │
│   │  │                                                         │
│   │  │ 3. geography_score = _geography_score()                 │
│   │  │    "germany" in {"united kingdom", "germany"} → 1.0    │
│   │  │    (0.5 if no geo filter, 0.0 if specified but no match)│
│   │  │                                                         │
│   │  │ 4. quality_score = _quality_score()                     │
│   │  │    word_count = 85 (between 50–400) → 1.0              │
│   │  │    (<50 words → penalized, >400 words → penalized)     │
│   │  │                                                         │
│   │  │ FINAL = 0.55 × 0.816  = 0.4488                         │
│   │  │       + 0.25 × 1.000  = 0.2500                         │
│   │  │       + 0.10 × 1.000  = 0.1000                         │
│   │  │       + 0.10 × 1.000  = 0.1000                         │
│   │  │       ────────────────────────                          │
│   │  │       = 0.8988                                          │
│   │  └─────────────────────────────────────────────────────────┘
│
├── Sort all 12 by final score descending
│
├── Take top 5 (top_k_final)
│
└── Return as list[SearchResult] with score_components visible:
    [
      {id: "9856595", company_name: "ComX", score: 0.8988,
       score_components: {vector: 0.816, keyword: 1.0, geography: 1.0, quality: 1.0}},
      ...
    ]
    Timing: logged as stage_latency_ms["rerank"]
```

### 4.5 — Response Assembled

```
SearchResponse:
{
  "results": [ ...5 SearchResult objects... ],
  "total": 5,
  "diagnostics": {
    "raw_count": 200,
    "filtered_count": 188,
    "reranked_count": 5,
    "drop_reasons": {
      "geography_mismatch": 167,
      "exclude_term": 21
    },
    "stage_latency_ms": {
      "vector_recall": 14,
      "post_filter": 0,
      "rerank": 0
    },
    "trace_id": "c23c8c62-..."
  }
}
```

---

## 5. File Map

```
blueknight-task/
│
├── app/
│   ├── main.py                    ← FastAPI app, endpoints, startup
│   ├── config.py                  ← All settings (API keys, paths, thresholds)
│   ├── schemas.py                 ← Pydantic models (request/response contracts)
│   ├── retrieval.py               ← mock_retrieve() — FAISS vector search
│   │
│   ├── services/
│   │   ├── refiner.py             ← QueryRefinerAgent — iterative LLM loop
│   │   ├── search_pipeline.py     ← SearchPipeline — 3-stage orchestrator
│   │   ├── reranker.py            ← Reranker — 4-component weighted scoring
│   │   ├── retrieval_wrapper.py   ← Resilient wrapper (retry, timeout, semaphore)
│   │   ├── embedder.py            ← Sentence-transformers embedding (local)
│   │   └── vector_store.py        ← FAISS singleton (load index + query)
│   │
│   └── utils/
│       ├── json_contract.py       ← Robust LLM JSON parser
│       └── logging.py             ← log_stage() context manager
│
├── scripts/
│   ├── prepare_data.py            ← xlsx → CSV conversion
│   └── build_index.py             ← CSV → FAISS index + metadata pickle
│
├── data/
│   ├── companies.csv              ← Company dataset (generated)
│   ├── faiss.index                ← FAISS vector index (generated)
│   └── companies.pkl              ← Company metadata (generated)
│
├── .env                           ← ANTHROPIC_API_KEY (not committed)
├── .env.example                   ← Template
├── .gitignore                     ← Protects .env, data files, __pycache__
├── requirements.txt               ← All Python dependencies
├── README.md                      ← Documentation + Subtask 3 essay
└── pipeline.log                   ← Debug trace file (generated at runtime)
```

---

## 6. Complete Call Graph

### POST /agent/refine — Full Chain

```
app/main.py::refine()
  └── app/services/refiner.py::QueryRefinerAgent.refine()
        │
        ├── [LOOP: 1 to max_iterations]
        │     │
        │     ├── refiner._call_llm()
        │     │     ├── refiner._build_user_prompt()
        │     │     ├── anthropic.messages.create()          ← Claude API
        │     │     ├── utils/json_contract.py::parse_json_contract()
        │     │     └── refiner._normalise()
        │     │           └── refiner._ensure_list()
        │     │
        │     ├── app/services/search_pipeline.py::SearchPipeline.run()
        │     │     │
        │     │     ├── STAGE 1: retrieval_wrapper.py::retrieve()
        │     │     │     ├── asyncio.Semaphore.acquire()
        │     │     │     ├── retrieval_wrapper._retrieve_with_retry()  ← tenacity
        │     │     │     │     └── retrieval.py::mock_retrieve()
        │     │     │     │           ├── embedder.py::embed()
        │     │     │     │           │     └── SentenceTransformer.encode()
        │     │     │     │           └── vector_store.py::VectorStoreClient.query()
        │     │     │     │                 └── faiss.IndexFlatIP.search()
        │     │     │     └── asyncio.wait_for(timeout=10s)
        │     │     │
        │     │     ├── STAGE 2: search_pipeline._post_filter()
        │     │     │     └── reranker.py::_normalise_geo()
        │     │     │
        │     │     └── STAGE 3: reranker.py::Reranker.rerank()
        │     │           ├── reranker._tokenize()
        │     │           ├── reranker._keyword_score()
        │     │           ├── reranker._geography_score()
        │     │           └── reranker._quality_score()
        │     │
        │     ├── refiner._compute_quality()
        │     ├── refiner._evaluate()               ← 6-rule termination
        │     └── refiner._get_prev_summary()        ← feedback for next iteration
        │
        └── Return RefineResponse
```

### POST /search/run — Full Chain

```
app/main.py::search_run()
  └── app/services/search_pipeline.py::SearchPipeline.run()
        │
        ├── STAGE 1: retrieval_wrapper.py::retrieve()
        │     └── (same chain as above)
        │
        ├── STAGE 2: search_pipeline._post_filter()
        │     └── reranker._normalise_geo()
        │
        ├── STAGE 3: reranker.py::Reranker.rerank()
        │     └── (same chain as above)
        │
        └── Return SearchResponse with Diagnostics
```

---

## Key Design Decisions Summary

| Decision | Why |
|---|---|
| FAISS IndexFlatIP (brute force) | 1000 vectors × 384 dims — brute force is <1ms, no need for approximate search |
| Local embeddings (BGE) | Free, no API key, no network latency, runs on M1 Mac in ~100ms |
| Claude Sonnet for LLM | Best balance of JSON reliability, speed, and cost |
| Refiner calls pipeline directly | HTTP self-call would deadlock single-worker uvicorn |
| Semaphore on retrieval | Prevents unbounded concurrent FAISS searches (spec requirement) |
| Tenacity retry | Handles the ~5% transient failure rate from spec |
| 6-rule termination | Covers all failure modes: excellent results, plateau, empty, misaligned, weak, and minimum-iterations |
| Heuristic reranker (not LLM) | Deterministic, fast (<1ms), transparent score components, no extra API calls |
| Structured JSON logging | Every stage emits {trace_id, stage, duration_ms, item_count} — latency spikes are identifiable from logs alone |
