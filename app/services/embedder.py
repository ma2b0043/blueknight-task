"""Local embedding using sentence-transformers (BAAI/bge-small-en-v1.5).

Free, no API key required. Runs entirely on CPU/MPS.
Model downloads ~130MB on first use, then cached locally.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> list[float]:
    """Embed a single text string."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()


def warmup() -> None:
    """Pre-load the embedding model so the first real query is fast."""
    _get_model()
