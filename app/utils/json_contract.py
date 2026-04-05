"""Robust JSON parser for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_contract(raw: str) -> dict[str, Any]:
    """Parse JSON from raw LLM output, handling fenced code blocks and malformed output.

    Returns a deterministic error shape on failure:
        {"_parse_error": True, "raw": "<truncated raw input>"}
    """
    cleaned = raw.strip()

    # Strip markdown fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first {...} block
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Deterministic error shape
    return {"_parse_error": True, "raw": raw[:500]}
