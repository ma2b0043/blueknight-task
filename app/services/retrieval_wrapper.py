"""Resilient wrapper around mock_retrieve with retry, timeout, and bounded concurrency."""

from __future__ import annotations

import asyncio
import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import (
    MAX_RETRIEVAL_RETRIES,
    RETRIEVAL_CONCURRENCY_LIMIT,
    RETRIEVAL_TIMEOUT_S,
)
from app.retrieval import CompanyResult, RetrievalError, mock_retrieve

logger = logging.getLogger("blueknight")

_semaphore = asyncio.Semaphore(RETRIEVAL_CONCURRENCY_LIMIT)


@retry(
    retry=retry_if_exception_type(RetrievalError),
    stop=stop_after_attempt(MAX_RETRIEVAL_RETRIES),
    wait=wait_exponential(multiplier=0.1, max=2),
    reraise=True,
)
def _retrieve_with_retry(query: str, top_k: int) -> list[CompanyResult]:
    return mock_retrieve(query, top_k)


async def retrieve(query: str, top_k: int, trace_id: str) -> list[CompanyResult]:
    """Async wrapper: bounded concurrency + timeout + retry around mock_retrieve."""
    async with _semaphore:
        loop = asyncio.get_event_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _retrieve_with_retry, query, top_k),
            timeout=RETRIEVAL_TIMEOUT_S,
        )
        return results
