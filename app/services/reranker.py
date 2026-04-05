"""Deterministic re-ranker with weighted heuristic scoring.

Scoring formula:
    final_score = 0.55 * vector_score
                + 0.25 * keyword_score
                + 0.10 * geography_boost
                + 0.10 * quality_score

Components:
    vector_score   — Raw cosine similarity from FAISS (already 0–1 for normalised vectors).
    keyword_score  — |query_tokens ∩ offering_tokens| / |query_tokens|
                     Catches exact industry terms that embeddings may under-weight.
    geography_boost — 1.0 if company country matches any requested geography,
                      0.5 if no geography filter was specified (neutral),
                      0.0 if geography was specified but doesn't match.
    quality_score  — Penalises very short (<50 words) or very long (>400 words) offerings.
"""

from __future__ import annotations

import logging
import re

from app.retrieval import CompanyResult
from app.schemas import QueryPayload, SearchResult

logger = logging.getLogger("blueknight")

# Common English stopwords to exclude from keyword matching
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "i", "we", "you", "he", "she", "they", "me", "us", "him",
    "her", "them", "my", "our", "your", "his", "their",
})

# Geography aliases for normalisation
_GEO_ALIASES: dict[str, str] = {
    "uk": "united kingdom",
    "gb": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "us": "united states",
    "usa": "united states",
    "united states of america": "united states",
    "america": "united states",
    "uae": "united arab emirates",
    "de": "germany",
    "fr": "france",
}


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOPWORDS


def _normalise_geo(geo: str | float | None) -> str:
    if not geo or not isinstance(geo, str):
        return ""
    lowered = geo.strip().lower()
    return _GEO_ALIASES.get(lowered, lowered)


def _keyword_score(query_tokens: set[str], offering_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & offering_tokens)
    return min(overlap / len(query_tokens), 1.0)


def _geography_score(
    company_country: str, requested_geos: list[str]
) -> float:
    if not requested_geos:
        return 0.5  # Neutral when no geography filter
    normalised_company = _normalise_geo(company_country)
    normalised_requested = {_normalise_geo(g) for g in requested_geos}
    return 1.0 if normalised_company in normalised_requested else 0.0


def _quality_score(long_offering: str) -> float:
    word_count = len(long_offering.split())
    if word_count < 50:
        return max(0.3, word_count / 50)
    if word_count > 400:
        return max(0.5, 1.0 - (word_count - 400) / 400)
    return 1.0


class Reranker:
    """Deterministic weighted-heuristic re-ranker."""

    W_VECTOR = 0.55
    W_KEYWORD = 0.25
    W_GEOGRAPHY = 0.10
    W_QUALITY = 0.10

    def rerank(
        self,
        candidates: list[CompanyResult],
        query: QueryPayload,
        top_k: int,
    ) -> list[SearchResult]:
        query_tokens = _tokenize(query.query_text)
        logger.debug(f"RERANKER: scoring {len(candidates)} candidates, query_tokens={query_tokens}")
        scored: list[SearchResult] = []

        for c in candidates:
            offering_tokens = _tokenize(c.long_offering)

            v_score = max(0.0, min(c.score, 1.0))
            k_score = _keyword_score(query_tokens, offering_tokens)
            g_score = _geography_score(c.country, query.geography)
            q_score = _quality_score(c.long_offering)

            final = (
                self.W_VECTOR * v_score
                + self.W_KEYWORD * k_score
                + self.W_GEOGRAPHY * g_score
                + self.W_QUALITY * q_score
            )

            scored.append(
                SearchResult(
                    id=c.id,
                    company_name=c.company_name if isinstance(c.company_name, str) else "",
                    country=c.country if isinstance(c.country, str) else "",
                    score=round(final, 4),
                    score_components={
                        "vector": round(v_score, 4),
                        "keyword": round(k_score, 4),
                        "geography": round(g_score, 4),
                        "quality": round(q_score, 4),
                    },
                    long_offering=c.long_offering,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
