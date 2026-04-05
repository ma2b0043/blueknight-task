"""FAISS-backed vector store client (singleton)."""

from __future__ import annotations

import pickle
from typing import Any

import faiss
import numpy as np

from app.config import COMPANIES_META_PATH, FAISS_INDEX_PATH


class VectorStoreClient:
    """Singleton abstraction over the FAISS index and company metadata."""

    _instance: VectorStoreClient | None = None

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> VectorStoreClient:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(COMPANIES_META_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        assert self.index.ntotal == len(self.metadata), (
            f"Index has {self.index.ntotal} vectors but metadata has {len(self.metadata)} entries"
        )

    async def upsert(self, items: list[dict[str, Any]]) -> None:
        """No-op — index is built offline via scripts/build_index.py."""
        pass

    def query(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Search FAISS index and return company dicts with scores."""
        assert self.index is not None, "Index not loaded — call get_instance() first"

        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            # Sanitise NaN values from pandas
            results.append({
                "id": str(meta.get("id", "")),
                "company_name": meta.get("company_name", "") if isinstance(meta.get("company_name"), str) else "",
                "country": meta.get("country", "") if isinstance(meta.get("country"), str) else "",
                "long_offering": meta.get("long_offering", "") if isinstance(meta.get("long_offering"), str) else "",
                "score": float(score),
            })
        return results
