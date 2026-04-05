"""Structured logging utilities for stage-level observability."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger("blueknight")


@contextmanager
def log_stage(trace_id: str, stage: str) -> Generator[dict[str, Any], None, None]:
    """Context manager that times a pipeline stage and emits structured JSON log.

    Usage:
        with log_stage(trace_id, "vector_recall") as ctx:
            results = do_recall(...)
            ctx["item_count"] = len(results)
    """
    ctx: dict[str, Any] = {"item_count": 0}
    start = time.perf_counter()
    try:
        yield ctx
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000)
        ctx["duration_ms"] = duration_ms
        log_entry = {
            "trace_id": trace_id,
            "stage": stage,
            "duration_ms": duration_ms,
            "item_count": ctx.get("item_count", 0),
        }
        logger.info(json.dumps(log_entry))
