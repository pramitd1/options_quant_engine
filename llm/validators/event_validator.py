from __future__ import annotations

from nlp.schemas.event_schema import EventIntelligenceRecord, validate_event_record


def validate_llm_event_payload(payload: dict) -> EventIntelligenceRecord:
    """Validate raw LLM JSON against engine-safe event schema."""
    return validate_event_record(payload)
