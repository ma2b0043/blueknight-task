# Refiner and Reranker — M&A Company Matching System

An agentic search workflow for matching companies in an M&A setting. Users describe the kind of company they're looking for in natural language, and the system refines, retrieves, and re-ranks results iteratively.

---

## Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and add your Anthropic API key:

```bash
cp .env.example .env
# Edit .env with your key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Data Preparation

```bash
# Convert the xlsx dataset to CSV
python scripts/prepare_data.py

# Embed long_offering and build the FAISS index (~2 min for 1000 companies)
python scripts/build_index.py
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Architecture

```
User Query (natural language)
    │
    ▼
┌──────────────────────────────┐
│  POST /agent/refine          │  Iterative refinement loop
│  ┌─────────────────────┐     │
│  │  Claude LLM          │──→ Structured QueryPayload
│  └─────────────────────┘     │
│           │                   │
│           ▼                   │
│  ┌─────────────────────┐     │
│  │  POST /search/run    │     │  3-stage pipeline (called internally)
│  │  1. Vector recall    │     │
│  │  2. Post-filter      │     │
│  │  3. Re-rank          │     │
│  └─────────────────────┘     │
│           │                   │
│           ▼                   │
│  ┌─────────────────────┐     │
│  │  Evaluate quality    │──→ Stop or iterate
│  └─────────────────────┘     │
└──────────────────────────────┘
```

### Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Vector DB | FAISS (in-memory, `IndexFlatIP`) | 1000 vectors at 384 dims — brute force is sub-ms; no external service needed |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) | Top MTEB quality for size, 384 dims, runs locally via sentence-transformers — no API key needed |
| LLM | Claude Sonnet (`claude-sonnet-4-20250514`) | Excellent structured JSON output, fast, reliable |
| Framework | FastAPI + Pydantic | Already in starter kit; async support |
| Retry | tenacity | Production-grade retry with exponential backoff |

---

## Assumptions

1. **FAISS is sufficient** for 1000 companies. No need for distributed vector DBs.
2. **The refiner calls SearchPipeline directly** (not via HTTP) to avoid single-worker deadlock and serialisation overhead. The spec says "call POST /search/run internally" — I interpret this as using the same pipeline logic, not an HTTP self-call.
3. **BGE embeddings are normalised** (via `normalize_embeddings=True`), so inner product ≈ cosine similarity.
4. **Geography filtering is strict** (exact country match with aliases), because in M&A context, geography is usually a hard constraint, not a soft preference.
5. **Exclusion filtering is substring-based** on `long_offering`, which is conservative but avoids false negatives from tokenisation.
6. **The refiner uses `top_k_raw=200`** (not 1000) during iteration for faster evaluation. The final search can use the full 1000.
7. **Dataset composition**: The provided dataset is ~86% B2B sales/CRM SaaS companies. Queries for underrepresented verticals (logistics ~3%, manufacturing ~3%, fintech ~1.5%) will naturally return fewer and lower-scoring results. This is a data limitation, not a retrieval issue — the pipeline correctly surfaces the closest matches available in the corpus.

---

## Termination Condition Rationale

The refiner loop uses a **composite quality score** (0–1) evaluated by **6 deterministic rules**:

### Quality Score

```
quality = 0.30 × count_ratio        # min(result_count / 10, 1.0)
        + 0.30 × top_score           # best result's final score
        + 0.20 × spread_score        # score gradient across top-5
        + 0.20 × filter_health       # 1.0 − (filtered_count / raw_count)
```

### Rules (evaluated in order)

| # | Condition | Action | Rationale |
|---|-----------|--------|-----------|
| 1 | quality ≥ 0.75 | STOP | Results are excellent — no need to iterate |
| 2 | iteration ≥ 2 AND improvement < 0.02 | STOP | Plateau detected — further refinement won't help |
| 3 | reranked_count < 3 | CONTINUE | Near-empty results suggest the query is too narrow |
| 4 | filter_drop_ratio > 80% | CONTINUE | Most results are being filtered — query is misaligned |
| 5 | top_score < 0.40 | CONTINUE | Best match is weak — worth trying a different approach |
| 6 | iteration = 1 AND quality < 0.75 | CONTINUE | Always validate with a second pass unless results are excellent |

**Why this works:**
- Rule 6 ensures the system never exits after just one iteration (unless results are already excellent), satisfying the spec's explicit requirement.
- Rule 2 prevents wasting iterations when quality stabilises.
- Rules 3–5 identify specific failure modes that another LLM pass can fix (e.g., broadening a too-narrow query, adjusting geography, changing approach for weak matches).
- The previous iteration's diagnostics (drop reasons, score distribution) are fed back to the LLM so it can adjust intelligently.

---

## Post-Filter Signals

| Signal | Implementation | Why |
|---|---|---|
| `geography_mismatch` | Strict country match with alias normalisation (UK→United Kingdom, US→United States, etc.) | Geography is typically a hard constraint in M&A deal sourcing |
| `exclude_term` | Case-insensitive substring search in `long_offering` for each exclusion term | Respects negative intent (e.g. "not focused on payments") |
| `low_vector_score` | Drop candidates below 0.20 cosine similarity threshold | Removes clearly irrelevant tail noise from FAISS results |

---

## Re-Ranking Scoring Logic

Deterministic weighted heuristic — no additional LLM calls:

```
final_score = 0.55 × vector_score      # Cosine similarity from FAISS
            + 0.25 × keyword_score      # Query token overlap with long_offering
            + 0.10 × geography_boost    # Country match bonus
            + 0.10 × quality_score      # Offering length quality signal
```

**Why these weights:**
- **vector_score (0.55)**: Semantic similarity is the primary signal.
- **keyword_score (0.25)**: Catches exact industry terms (e.g. "logistics", "SaaS") that embeddings sometimes under-weight. Computed as `|query_tokens ∩ offering_tokens| / |query_tokens|` after stopword removal.
- **geography_boost (0.10)**: Rewards geo-matching companies. Neutral (0.5) when no geography is specified.
- **quality_score (0.10)**: Penalises very short (<50 words) or very long (>400 words) offerings, which tend to be less informative.

All components are returned in `score_components` for full transparency.

---

## Observability

- **trace_id** propagated through all stages (refiner → search → retrieval)
- **Structured JSON logs** emitted at every stage boundary with: `trace_id`, `stage`, `duration_ms`, `item_count`
- **Per-stage latency** in `diagnostics.stage_latency_ms`: `vector_recall`, `post_filter`, `rerank` — distinguishable from logs alone
- **Refiner evaluation logs** include `iteration`, `quality`, `should_stop`, `reason`

---

## Subtask 3 — Production Readiness

> *"The system serves 10,000 queries/day. Result relevance silently degrades — no errors are thrown, but users are getting poor matches. How would you detect this before users complain, and what is your first operational change?"*

Silent relevance degradation is insidious because every health check passes — the system *works*, it just returns worse results. The key insight: **the diagnostics we already emit contain the signal**.

**Detection strategy:**

1. **Score distribution monitoring** — Track daily P50/P90 of `top_score` and `score_spread` from existing `diagnostics`. A sustained downward drift in top-result scores indicates degradation even when no errors are thrown.

2. **Filter drop ratio trending** — If `filtered_count / raw_count` creeps upward over days, it means either the query distribution shifted or the corpus changed, causing misalignment between what users ask for and what the index contains.

3. **Re-query rate as implicit feedback** — If users repeatedly refine the same search (high `iterations_used` average or multiple sessions with similar queries), it signals dissatisfaction with initial results.

**First operational change:** Ship a lightweight dashboard tracking the 7-day moving average of `top_score` P50 with an alert at 2σ deviation. This requires zero additional instrumentation — the data already exists in our structured logs. It would have caught the degradation within 24–48 hours.

---

## What I Would Do With More Time

1. **Cross-encoder re-ranker** — Use a cross-encoder model (e.g. `ms-marco-MiniLM`) for Stage 3 instead of the heuristic. This would dramatically improve ranking quality at the cost of latency.
2. **Hybrid retrieval** — Combine dense (FAISS) and sparse (BM25) retrieval for better recall on exact industry terms.
3. **Caching layer** — Cache embeddings and search results for repeated queries (common in M&A workflows where advisors run variations of similar searches).
4. **Comprehensive test suite** — Unit tests for each pipeline stage, integration tests for the full flow, and property-based tests for the termination condition.
5. **Async embedding** — Run embedding inference in a thread pool to avoid blocking the event loop under concurrent load.
6. **Query expansion** — Use the LLM to generate synonym expansions for key domain terms before embedding.
