"""Query Refiner Agent — iterative LLM-driven refinement loop (Subtask 1).

Loop:
    1. Call Claude to refine user intent into a structured QueryPayload
    2. Parse + normalise the LLM output
    3. Run the SearchPipeline to evaluate result quality
    4. Apply deterministic termination condition
    5. Return the best query seen across all iterations

Termination condition (6 rules evaluated in order):
    1. quality >= 0.75                              → STOP  (excellent results)
    2. iteration >= 2 AND improvement < 0.02        → STOP  (plateau)
    3. reranked_count < 3                            → CONTINUE (near-empty)
    4. filter_drop_ratio > 0.80                      → CONTINUE (query misaligned)
    5. top_score < 0.40                              → CONTINUE (weak best match)
    6. iteration == 1 AND quality < 0.75             → CONTINUE (always try ≥2)
    default                                          → STOP
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from anthropic import Anthropic

from app.config import ANTHROPIC_API_KEY, REFINER_MODEL
from app.schemas import (
    Action,
    QueryPayload,
    RefineRequest,
    RefineResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.search_pipeline import SearchPipeline
from app.utils.json_contract import parse_json_contract
from app.utils.logging import log_stage

logger = logging.getLogger("blueknight")


# ── Helpers ─────────────────────────────────────────────────────────────


def default_actions() -> list[Action]:
    """Starter UI action contract."""
    return [Action(id="show_results", label="Show results", payload={})]


def default_query_payload() -> QueryPayload:
    """Default payload used for deterministic query shape."""
    return QueryPayload()


@dataclass
class EvalResult:
    """Outcome of evaluating one iteration's result quality."""

    quality: float
    should_stop: bool
    reason: str


# ── Evaluation logic ────────────────────────────────────────────────────


def _compute_quality(response: SearchResponse) -> float:
    """Compute a 0–1 composite quality score from search diagnostics."""
    diag = response.diagnostics
    results = response.results

    # Count ratio: do we have a reasonable number of results?
    count_ratio = min(len(results) / 10, 1.0) if results else 0.0

    # Top score: how good is the best match?
    top_score = results[0].score if results else 0.0

    # Spread: score gradient across top-5 (higher spread = good differentiation)
    if len(results) >= 2:
        top_5 = [r.score for r in results[:5]]
        spread = top_5[0] - top_5[-1]
        spread_score = min(spread / 0.3, 1.0)  # Normalise: 0.3 spread → 1.0
    else:
        spread_score = 0.0

    # Filter health: low filter-drop ratio means query is well-aligned
    if diag.raw_count > 0:
        filter_health = 1.0 - (diag.filtered_count / diag.raw_count)
    else:
        filter_health = 0.0

    return (
        0.30 * count_ratio
        + 0.30 * top_score
        + 0.20 * spread_score
        + 0.20 * filter_health
    )


def _evaluate(
    iteration: int,
    response: SearchResponse,
    prev_quality: float | None,
) -> EvalResult:
    """Apply the 6-rule termination condition."""
    quality = _compute_quality(response)
    diag = response.diagnostics

    # Rule 1: Excellent results
    if quality >= 0.75:
        return EvalResult(quality, True, f"Excellent result quality ({quality:.2f})")

    # Rule 2: Plateau detection
    if iteration >= 2 and prev_quality is not None:
        improvement = quality - prev_quality
        if improvement < 0.02:
            return EvalResult(
                quality, True,
                f"Plateau detected — quality {quality:.2f}, improvement {improvement:.3f} from previous iteration",
            )

    # Rule 3: Near-empty results
    if len(response.results) < 3:
        return EvalResult(
            quality, False,
            f"Only {len(response.results)} results returned — refining for broader recall",
        )

    # Rule 4: High filter-drop ratio
    if diag.raw_count > 0 and (diag.filtered_count / diag.raw_count) > 0.80:
        return EvalResult(
            quality, False,
            f"Filter drop ratio {diag.filtered_count}/{diag.raw_count} > 80% — query misaligned with corpus",
        )

    # Rule 5: Weak top score
    top_score = response.results[0].score if response.results else 0.0
    if top_score < 0.40:
        return EvalResult(
            quality, False,
            f"Best match score {top_score:.2f} is below 0.40 — retrying with refined query",
        )

    # Rule 6: Always try at least 2 iterations unless excellent
    if iteration == 1 and quality < 0.75:
        return EvalResult(
            quality, False,
            f"First iteration quality {quality:.2f} < 0.75 — running second pass for validation",
        )

    # Default: accept
    return EvalResult(quality, True, f"Acceptable quality ({quality:.2f}) — stopping")


# ── LLM interaction ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a search query refiner for an M&A company matching system.
Your job: convert a user's natural language company search into a structured JSON query.

You MUST respond with ONLY valid JSON (no markdown, no explanation) in this exact format:
{
  "query_text": "refined search query optimised for semantic matching against company descriptions",
  "geography": ["list of countries/regions mentioned or implied, empty if none"],
  "exclusions": ["terms to exclude from results, empty if none"]
}

Rules:
- query_text should be a clear, specific description optimised for matching against company bios
- Extract geography from context (e.g. "UK companies" → geography: ["United Kingdom"])
- Extract exclusion intent (e.g. "not focused on payments" → exclusions: ["payments"])
- If the query is vague or single-character, produce a broad but reasonable query_text
- Do NOT add geographies or exclusions unless the user's intent clearly implies them"""


def _build_user_prompt(
    message: str,
    base_query: QueryPayload | None,
    history: list[dict[str, Any]],
    iteration: int,
    prev_summary: dict[str, Any] | None,
) -> str:
    parts = [f"User query: {message}"]

    if base_query and base_query.query_text:
        parts.append(f"\nPrevious structured query: {base_query.model_dump_json()}")

    if history:
        parts.append(f"\nConversation history: {json.dumps(history[-3:])}")  # Last 3 turns

    if iteration > 1 and prev_summary:
        parts.append(
            f"\nPrevious iteration results summary (iteration {iteration - 1}):\n"
            f"- Result count: {prev_summary['result_count']}\n"
            f"- Top score: {prev_summary['top_score']:.3f}\n"
            f"- Score spread (top 5): {prev_summary['score_spread']:.3f}\n"
            f"- Filter drop ratio: {prev_summary['filter_drop_ratio']:.1%}\n"
            f"- Drop reasons: {json.dumps(prev_summary['drop_reasons'])}\n"
            f"\nPlease refine the query to improve results. "
            f"Consider adjusting query_text for better semantic matching, "
            f"or adjusting geography/exclusions based on drop reasons."
        )

    return "\n".join(parts)


def _get_prev_summary(response: SearchResponse) -> dict[str, Any]:
    results = response.results
    diag = response.diagnostics
    top_5 = [r.score for r in results[:5]]
    return {
        "result_count": len(results),
        "top_score": top_5[0] if top_5 else 0.0,
        "score_spread": (top_5[0] - top_5[-1]) if len(top_5) >= 2 else 0.0,
        "filter_drop_ratio": diag.filtered_count / diag.raw_count if diag.raw_count > 0 else 0.0,
        "drop_reasons": diag.drop_reasons,
    }


# ── Agent class ─────────────────────────────────────────────────────────


class QueryRefinerAgent:
    """Iterative refinement agent that calls Claude + SearchPipeline in a loop."""

    def __init__(self) -> None:
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.pipeline = SearchPipeline()

    async def refine(self, request: RefineRequest) -> RefineResponse:
        trace_id = request.trace_id
        best_query: QueryPayload = request.base_query or default_query_payload()
        best_quality = -1.0
        best_response: SearchResponse | None = None
        prev_quality: float | None = None
        prev_summary: dict[str, Any] | None = None
        rationale = ""

        logger.debug(f"[{trace_id}] === REFINE START === message='{request.message}', max_iterations={request.max_iterations}")

        for iteration in range(1, request.max_iterations + 1):
            with log_stage(trace_id, f"refine_iteration_{iteration}") as ctx:
                logger.debug(f"[{trace_id}] --- Iteration {iteration}/{request.max_iterations} ---")

                # Step 1: Call LLM to produce refined query
                refined_query = await self._call_llm(
                    message=request.message,
                    base_query=best_query if iteration > 1 else request.base_query,
                    history=request.history,
                    iteration=iteration,
                    prev_summary=prev_summary,
                    trace_id=trace_id,
                )
                logger.debug(f"[{trace_id}] LLM refined query: {refined_query.model_dump_json()}")

                # Step 2: Run search pipeline
                search_req = SearchRequest(
                    query=refined_query,
                    top_k_raw=200,  # Lighter recall for evaluation
                    top_k_final=50,
                    offset=0,
                    trace_id=trace_id,
                )
                search_resp = await self.pipeline.run(search_req)
                ctx["item_count"] = len(search_resp.results)

                # Step 3: Evaluate
                eval_result = _evaluate(iteration, search_resp, prev_quality)

                # Track best
                if eval_result.quality > best_quality:
                    best_quality = eval_result.quality
                    best_query = refined_query
                    best_response = search_resp

                prev_quality = eval_result.quality
                prev_summary = _get_prev_summary(search_resp)
                rationale = (
                    f"Stopped after {iteration} iteration(s) — {eval_result.reason}"
                )

                logger.info(json.dumps({
                    "trace_id": trace_id,
                    "stage": "refine_eval",
                    "iteration": iteration,
                    "quality": round(eval_result.quality, 3),
                    "should_stop": eval_result.should_stop,
                    "reason": eval_result.reason,
                }))

                if eval_result.should_stop:
                    break

        return RefineResponse(
            refined_query=best_query,
            rationale=rationale,
            actions=default_actions(),
            iterations_used=iteration,
            meta={
                "trace_id": trace_id,
                "final_quality": round(best_quality, 3),
                "result_count": len(best_response.results) if best_response else 0,
            },
        )

    async def _call_llm(
        self,
        message: str,
        base_query: QueryPayload | None,
        history: list[dict[str, Any]],
        iteration: int,
        prev_summary: dict[str, Any] | None,
        trace_id: str,
    ) -> QueryPayload:
        """Call Claude to produce a refined QueryPayload. Falls back on parse errors."""
        user_prompt = _build_user_prompt(
            message=message,
            base_query=base_query,
            history=history,
            iteration=iteration,
            prev_summary=prev_summary,
        )

        logger.debug(f"[{trace_id}] LLM PROMPT:\n{user_prompt}")

        try:
            resp = self.client.messages.create(
                model=REFINER_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = resp.content[0].text
            logger.debug(f"[{trace_id}] LLM RAW RESPONSE:\n{raw_text}")
            parsed = parse_json_contract(raw_text)
            logger.debug(f"[{trace_id}] LLM PARSED JSON: {json.dumps(parsed)}")
        except Exception as e:
            logger.warning(json.dumps({
                "trace_id": trace_id,
                "stage": "refine_llm",
                "error": str(e),
            }))
            parsed = {"_parse_error": True}

        return self._normalise(parsed, fallback_text=message)

    def _normalise(self, parsed: dict[str, Any], fallback_text: str) -> QueryPayload:
        """Merge parsed LLM output onto defaults. Never returns missing fields."""
        if parsed.get("_parse_error"):
            return QueryPayload(query_text=fallback_text)

        return QueryPayload(
            query_text=parsed.get("query_text", fallback_text) or fallback_text,
            geography=self._ensure_list(parsed.get("geography", [])),
            exclusions=self._ensure_list(parsed.get("exclusions", [])),
        )

    @staticmethod
    def _ensure_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(v) for v in val if v]
        if isinstance(val, str) and val:
            return [val]
        return []
