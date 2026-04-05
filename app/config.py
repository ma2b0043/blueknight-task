from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Explicit path to .env relative to project root
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env", override=True)

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = _BASE_DIR
DATA_DIR = BASE_DIR / "data"
COMPANIES_CSV_PATH = DATA_DIR / "companies.csv"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
COMPANIES_META_PATH = DATA_DIR / "companies_meta.pkl"

# ── API Keys ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Embedding (free, local via sentence-transformers) ───────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ── LLM ─────────────────────────────────────────────────────────────────
REFINER_MODEL = "claude-sonnet-4-20250514"

# ── Retrieval ───────────────────────────────────────────────────────────
MAX_RETRIEVAL_RETRIES = 3
RETRIEVAL_TIMEOUT_S = 10.0
RETRIEVAL_CONCURRENCY_LIMIT = 5

# ── Thresholds ──────────────────────────────────────────────────────────
LOW_VECTOR_SCORE_THRESHOLD = 0.20
