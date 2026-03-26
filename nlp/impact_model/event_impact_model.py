from __future__ import annotations

from nlp.schemas.event_schema import EventIntelligenceRecord


def apply_event_impact_model(event: EventIntelligenceRecord) -> EventIntelligenceRecord:
    """Adjust extracted event payload with options-specific impact priors."""
    time_horizon = event.time_horizon
    catalyst_quality = event.catalyst_quality
    event_strength = event.event_strength

    if event.event_type in {"earnings_result", "guidance_revision"}:
        time_horizon = "1_3_sessions"
        catalyst_quality = "high"
        event_strength = min(1.0, event_strength + 0.15)
    elif event.event_type in {"regulatory_action", "litigation_adverse_order"}:
        time_horizon = "1_2_weeks"
        catalyst_quality = "high"
        event_strength = min(1.0, event_strength + 0.2)
    elif event.event_type in {"management_change", "rating_action"}:
        time_horizon = "intraday"
        catalyst_quality = "medium"
    elif event.event_type == "rumor_unconfirmed_report":
        time_horizon = "intraday"
        catalyst_quality = "low"
        event_strength = min(1.0, event_strength + 0.1)

    return EventIntelligenceRecord(
        event_type=event.event_type,
        instrument_scope=event.instrument_scope,
        expected_direction=event.expected_direction,
        directional_confidence=event.directional_confidence,
        vol_impact=event.vol_impact,
        vol_confidence=event.vol_confidence,
        event_strength=event_strength,
        uncertainty_score=event.uncertainty_score,
        gap_risk_score=event.gap_risk_score,
        time_horizon=time_horizon,
        catalyst_quality=catalyst_quality,
        risk_flag=event.risk_flag,
        summary=event.summary,
        source=event.source,
        symbols=event.symbols,
        event_timestamp=event.event_timestamp,
        event_age_minutes=event.event_age_minutes,
    )
