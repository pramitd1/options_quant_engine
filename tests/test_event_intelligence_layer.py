from __future__ import annotations

import pandas as pd

from decision_policy.event_overlay import apply_event_overlay
from features.event_features.aggregator import aggregate_event_features
from macro.engine_adjustments import compute_macro_news_adjustments
import nlp.extraction.structured_extractor as extractor_mod
from nlp.extraction.structured_extractor import extract_structured_event
from nlp.schemas.event_schema import validate_event_record


def test_event_schema_validation_clamps_and_normalizes_labels():
    item = validate_event_record(
        {
            "event_type": "not_known",
            "instrument_scope": "weird",
            "expected_direction": "up",
            "directional_confidence": 2.5,
            "vol_impact": "boom",
            "vol_confidence": -1,
            "event_strength": 5,
            "uncertainty_score": 1.2,
            "gap_risk_score": -0.2,
            "time_horizon": "forever",
            "catalyst_quality": "elite",
            "summary": "",
        }
    )
    assert item.event_type == "unknown"
    assert item.instrument_scope == "mixed"
    assert item.expected_direction == "mixed"
    assert item.vol_impact == "mixed"
    assert item.directional_confidence == 1.0
    assert item.vol_confidence == 0.0
    assert item.time_horizon == "1_3_sessions"
    assert item.summary == "No summary available"


def test_directional_vs_volatility_are_extracted_separately():
    event = extract_structured_event(
        text="ABC posts strong earnings beat but management warns of elevated volatility in near term",
        source="unit_test",
    )
    assert event is not None
    assert event.expected_direction in {"bullish", "mixed"}
    assert event.vol_impact in {"expansion", "mixed"}


def test_uncertainty_and_gap_risk_high_for_regulatory_litigation_events():
    event = extract_structured_event(
        text="SEBI initiates probe and tribunal issues adverse order against XYZ",
        source="unit_test",
    )
    assert event is not None
    assert event.event_type in {"regulatory_action", "litigation_adverse_order"}
    assert event.uncertainty_score >= 0.75
    assert event.gap_risk_score >= 0.7


def test_missing_text_handling_returns_none_and_neutral_features():
    assert extract_structured_event(text=None) is None
    state = aggregate_event_features([])
    assert state.event_count == 0
    assert state.decayed_event_signal == 0.0
    assert state.explanation_lines


def test_malformed_timestamp_does_not_break_extraction():
    event = extract_structured_event(
        text="SEBI initiates probe into XYZ disclosures",
        source="unit_test",
        timestamp="not-a-timestamp",
    )
    assert event is not None
    assert event.event_timestamp is None


def test_feature_aggregation_builds_reusable_scores():
    bullish = extract_structured_event(text="DEF wins large order and raises guidance", source="unit_test")
    bearish = extract_structured_event(text="GHI faces litigation penalty and regulatory scrutiny", source="unit_test")
    assert bullish is not None and bearish is not None

    features = aggregate_event_features([bullish, bearish], direction_hint="CALL")
    assert 0.0 <= features.bullish_event_score <= 1.0
    assert 0.0 <= features.bearish_event_score <= 1.0
    assert 0.0 <= features.vol_expansion_score <= 1.0
    assert 0.0 <= features.event_uncertainty_score <= 1.0
    assert 0.0 <= features.contradictory_event_penalty <= 1.0


def test_event_overlay_suppresses_on_extreme_uncertainty():
    decision = apply_event_overlay(
        direction="CALL",
        event_features={
            "bullish_event_score": 0.3,
            "bearish_event_score": 0.5,
            "vol_expansion_score": 0.9,
            "event_uncertainty_score": 0.95,
            "gap_risk_score": 0.9,
            "contradictory_event_penalty": 0.8,
            "catalyst_alignment_score": 0.2,
        },
        enabled=True,
    )
    assert decision.suppress_signal is True
    assert decision.size_multiplier <= 0.4
    assert "event_overlay_signal_suppressed" in decision.reasons


def test_macro_adjustments_include_event_overlay_explainability():
    adjustments = compute_macro_news_adjustments(
        direction="PUT",
        macro_news_state={
            "macro_regime": "RISK_OFF",
            "macro_sentiment_score": -25,
            "volatility_shock_score": 72,
            "news_confidence_score": 80,
            "event_lockdown_flag": False,
            "neutral_fallback": False,
            "event_intelligence_enabled": True,
            "event_features": {
                "bullish_event_score": 0.1,
                "bearish_event_score": 0.7,
                "vol_expansion_score": 0.8,
                "event_uncertainty_score": 0.6,
                "gap_risk_score": 0.55,
                "contradictory_event_penalty": 0.2,
                "catalyst_alignment_score": 0.65,
            },
        },
    )
    assert "event_overlay_reasons" in adjustments
    assert isinstance(adjustments["event_overlay_reasons"], list)
    assert adjustments["event_overlay_size_multiplier"] <= 1.1


def test_llm_confidence_fallback_to_rule_path(monkeypatch):
    def _fake_llm_extract_event_payload(*, text: str, llm_enabled: bool = False):
        return {
            "event_type": "earnings_result",
            "instrument_scope": "single_stock",
            "expected_direction": "bullish",
            "directional_confidence": 0.2,
            "vol_impact": "neutral",
            "vol_confidence": 0.2,
            "event_strength": 0.2,
            "uncertainty_score": 0.1,
            "gap_risk_score": 0.1,
            "time_horizon": "1_3_sessions",
            "catalyst_quality": "medium",
            "risk_flag": False,
            "summary": "low confidence llm output",
        }

    monkeypatch.setattr(extractor_mod, "llm_extract_event_payload", _fake_llm_extract_event_payload)
    item = extract_structured_event(
        text="TCS beats earnings estimates strongly",
        llm_enabled=True,
        source="unit_test",
    )
    assert item is not None
    # Should use rule fallback, not low-confidence llm source.
    assert item.source != "llm_openai"


def test_recency_decay_and_symbol_routing_reduces_stale_irrelevant_events():
    as_of = pd.Timestamp("2026-03-26T13:00:00+05:30")
    old_event = validate_event_record(
        {
            "event_type": "regulatory_action",
            "instrument_scope": "single_stock",
            "expected_direction": "bearish",
            "directional_confidence": 1.0,
            "vol_impact": "expansion",
            "vol_confidence": 1.0,
            "event_strength": 1.0,
            "uncertainty_score": 0.9,
            "gap_risk_score": 0.9,
            "time_horizon": "1_3_sessions",
            "catalyst_quality": "high",
            "risk_flag": True,
            "summary": "legacy adverse event",
            "symbols": ["UNRELATED"],
            "event_timestamp": "2026-03-20T10:00:00+05:30",
        }
    )
    fresh_index_event = validate_event_record(
        {
            "event_type": "macro_event_sector_index",
            "instrument_scope": "index",
            "expected_direction": "bearish",
            "directional_confidence": 0.8,
            "vol_impact": "expansion",
            "vol_confidence": 0.8,
            "event_strength": 0.8,
            "uncertainty_score": 0.7,
            "gap_risk_score": 0.7,
            "time_horizon": "intraday",
            "catalyst_quality": "high",
            "risk_flag": True,
            "summary": "fresh index shock",
            "event_timestamp": "2026-03-26T12:30:00+05:30",
        }
    )

    state = aggregate_event_features(
        [old_event, fresh_index_event],
        direction_hint="CALL",
        underlying_symbol="NIFTY",
        as_of=as_of,
    )
    assert state.routed_event_relevance_score > 0.5
    assert state.routed_event_count >= 1
    assert state.bearish_event_score >= state.bullish_event_score
