"""Three-stage search pipeline: vector recall → post-filter → re-rank.

Post-filter signals:
    1. geography_mismatch — If query specifies geography, drop companies whose
       country doesn't match (with alias normalisation: UK→United Kingdom, etc.).
    2. exclude_term — If query specifies exclusions, drop companies whose
       long_offering contains any exclusion term (case-insensitive substring).
    3. low_vector_score — Drop candidates below a minimum cosine similarity
       threshold to remove clearly irrelevant tail results.
"""

from __future__ import annotations

import logging

from app.config import LOW_VECTOR_SCORE_THRESHOLD
from app.retrieval import CompanyResult
from app.schemas import Diagnostics, QueryPayload, SearchRequest, SearchResponse
from app.services.reranker import Reranker, _normalise_geo
from app.services.retrieval_wrapper import retrieve
from app.utils.logging import log_stage

logger = logging.getLogger("blueknight")


class SearchPipeline:
    """Orchestrates the three-stage retrieval pipeline."""

    def __init__(self) -> None:
        self.reranker = Reranker()

    async def run(self, request: SearchRequest) -> SearchResponse:
        trace_id = request.trace_id
        query = request.query
        stage_latency: dict[str, int] = {}
        drop_reasons: dict[str, int] = {}

        logger.debug(f"[{trace_id}] SEARCH PIPELINE START query_text='{query.query_text}' geography={query.geography} exclusions={query.exclusions} top_k_raw={request.top_k_raw} top_k_final={request.top_k_final}")

        # ── Stage 1: Vector Recall ──────────────────────────────────────
        with log_stage(trace_id, "vector_recall") as ctx:
            candidates = await retrieve(
                query=query.query_text,
                top_k=request.top_k_raw,
                trace_id=trace_id,
            )
            ctx["item_count"] = len(candidates)
        stage_latency["vector_recall"] = ctx["duration_ms"]
        raw_count = len(candidates)
        if candidates:
            top3 = candidates[:3]
            logger.debug(f"[{trace_id}] VECTOR RECALL: {raw_count} candidates. Top-3: " + ", ".join(
                f"{c.company_name}({c.score:.3f})" for c in top3
            ))

        # ── Stage 2: Post-Filter ────────────────────────────────────────
        with log_stage(trace_id, "post_filter") as ctx:
            filtered = self._post_filter(candidates, query, drop_reasons)
            ctx["item_count"] = len(filtered)
        stage_latency["post_filter"] = ctx["duration_ms"]
        filtered_count = raw_count - len(filtered)
        logger.debug(f"[{trace_id}] POST-FILTER: {raw_count} → {len(filtered)} (dropped {filtered_count}). Drop reasons: {drop_reasons}")

        # ── Stage 3: Re-rank ────────────────────────────────────────────
        with log_stage(trace_id, "rerank") as ctx:
            reranked = self.reranker.rerank(
                candidates=filtered,
                query=query,
                top_k=request.top_k_final + request.offset,
            )
            # Apply offset pagination
            reranked = reranked[request.offset:]
            reranked = reranked[: request.top_k_final]
            ctx["item_count"] = len(reranked)
        stage_latency["rerank"] = ctx["duration_ms"]
        if reranked:
            logger.debug(f"[{trace_id}] RERANK TOP-5:")
            for i, r in enumerate(reranked[:5]):
                logger.debug(f"  #{i+1} {r.company_name} | final={r.score:.4f} | components={r.score_components}")

        return SearchResponse(
            results=reranked,
            total=len(reranked),
            diagnostics=Diagnostics(
                raw_count=raw_count,
                filtered_count=filtered_count,
                reranked_count=len(reranked),
                drop_reasons=drop_reasons,
                stage_latency_ms=stage_latency,
                trace_id=trace_id,
            ),
        )

    def _post_filter(
        self,
        candidates: list[CompanyResult],
        query: QueryPayload,
        drop_reasons: dict[str, int],
    ) -> list[CompanyResult]:
        """Apply post-retrieval filters. Mutates drop_reasons dict."""
        kept: list[CompanyResult] = []

        # Pre-compute normalised geographies
        normalised_geos = {_normalise_geo(g) for g in query.geography} if query.geography else set()
        exclusions_lower = [e.lower() for e in query.exclusions] if query.exclusions else []

        for c in candidates:
            # Filter 1: Geography mismatch
            if normalised_geos:
                company_geo = _normalise_geo(c.country)
                if company_geo not in normalised_geos:
                    drop_reasons["geography_mismatch"] = drop_reasons.get("geography_mismatch", 0) + 1
                    continue

            # Filter 2: Exclusion terms
            if exclusions_lower:
                offering_lower = c.long_offering.lower()
                excluded = False
                for term in exclusions_lower:
                    if term in offering_lower:
                        drop_reasons["exclude_term"] = drop_reasons.get("exclude_term", 0) + 1
                        excluded = True
                        break
                if excluded:
                    continue

            # Filter 3: Low vector score
            if c.score < LOW_VECTOR_SCORE_THRESHOLD:
                drop_reasons["low_vector_score"] = drop_reasons.get("low_vector_score", 0) + 1
                continue

            kept.append(c)

        return kept
