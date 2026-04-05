from __future__ import annotations

from dataclasses import dataclass

from app.services.embedder import embed
from app.services.vector_store import VectorStoreClient


class RetrievalError(RuntimeError):
    """Transient retrieval failure."""


@dataclass
class CompanyResult:
    id: str
    company_name: str
    country: str
    long_offering: str
    score: float


def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    """
    Vector search over pre-embedded long_offering values.

    Embeds the query with OpenAI, searches the FAISS index, and returns
    CompanyResult objects. Keeps the original function signature so the
    rest of the pipeline does not need to change.
    """
    query_embedding = embed(query)
    store = VectorStoreClient.get_instance()
    raw_results = store.query(embedding=query_embedding, top_k=top_k)

    return [
        CompanyResult(
            id=r["id"],
            company_name=r["company_name"],
            country=r["country"],
            long_offering=r["long_offering"],
            score=r["score"],
        )
        for r in raw_results
    ]
