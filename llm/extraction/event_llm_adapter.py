from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from config.settings import (
    EVENT_INTELLIGENCE_LLM_MODEL,
    EVENT_INTELLIGENCE_LLM_PROVIDER,
    EVENT_INTELLIGENCE_LLM_TEMPERATURE,
    EVENT_INTELLIGENCE_LLM_TIMEOUT_SECONDS,
)
from llm.prompts.event_extraction_prompt import EVENT_EXTRACTION_PROMPT


_EVENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_type",
        "instrument_scope",
        "expected_direction",
        "directional_confidence",
        "vol_impact",
        "vol_confidence",
        "event_strength",
        "uncertainty_score",
        "gap_risk_score",
        "time_horizon",
        "catalyst_quality",
        "risk_flag",
        "summary",
    ],
    "properties": {
        "event_type": {"type": "string"},
        "instrument_scope": {"type": "string"},
        "expected_direction": {"type": "string"},
        "directional_confidence": {"type": "number"},
        "vol_impact": {"type": "string"},
        "vol_confidence": {"type": "number"},
        "event_strength": {"type": "number"},
        "uncertainty_score": {"type": "number"},
        "gap_risk_score": {"type": "number"},
        "time_horizon": {"type": "string"},
        "catalyst_quality": {"type": "string"},
        "risk_flag": {"type": "boolean"},
        "summary": {"type": "string"},
    },
}


def _extract_json_candidate(response: Any) -> str | None:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return None
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            candidate = getattr(block, "text", None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


@lru_cache(maxsize=8)
def _get_cached_openai_client(openai_ctor: Any, timeout_seconds: float):
    return openai_ctor(timeout=timeout_seconds)


def _extract_openai_payload(*, text: str) -> dict[str, Any] | None:
    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = _get_cached_openai_client(OpenAI, float(EVENT_INTELLIGENCE_LLM_TIMEOUT_SECONDS))
        response = client.responses.create(
            model=EVENT_INTELLIGENCE_LLM_MODEL,
            temperature=EVENT_INTELLIGENCE_LLM_TEMPERATURE,
            input=[
                {"role": "system", "content": EVENT_EXTRACTION_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "event_intelligence_payload",
                    "strict": True,
                    "schema": _EVENT_JSON_SCHEMA,
                },
            },
        )
    except Exception:
        return None

    candidate = _extract_json_candidate(response)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def llm_extract_event_payload(*, text: str, llm_enabled: bool = False) -> dict[str, Any] | None:
    """Extract strict JSON event payload from configured provider when enabled."""
    if not llm_enabled:
        return None
    if EVENT_INTELLIGENCE_LLM_PROVIDER == "OPENAI":
        return _extract_openai_payload(text=text)
    return None
