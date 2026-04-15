from app.terminal_output import (
    _describe_effective_strength_gate,
    _format_probability_display,
    _format_trigger_for_display,
    _render_data_usability_diagnostics,
    _render_dealer_gamma_levels,
    _render_provider_health_compact_detail,
    _summarize_confidence_guards,
)
from contextlib import redirect_stdout
from io import StringIO


def test_format_probability_display_preserves_small_nonzero_values() -> None:
    assert _format_probability_display(0.0) == "0%"
    assert _format_probability_display(0.004) == "<1%"
    assert _format_probability_display(0.056) == "5.6%"
    assert _format_probability_display(0.42) == "42%"


def test_summarize_confidence_guards_explains_execution_caps() -> None:
    note = _summarize_confidence_guards(
        [
            "status_watchlist_or_blocked",
            "provider_health_weak",
            "explicit_no_trade_reason",
        ]
    )

    assert note == (
        "capped by weak provider health; capped by explicit no-trade reason; "
        "execution-gated by blocked/watchlist status (setup may still be strong)"
    )


def test_format_trigger_for_display_applies_noise_buffer_for_tight_breakouts() -> None:
    trade = {"spot": 99.98}
    rendered = _format_trigger_for_display("decisive move above 100.00", trade)
    assert rendered.startswith("decisive move above 104.98 (noise-buffered from 100.00)")
    assert "[+5.0pts / +5.00%]" in rendered


def test_describe_effective_strength_gate_uses_readable_regime_separator() -> None:
    trade = {
        "min_trade_strength_threshold": 60,
        "data_quality_status": "CAUTION",
        "confirmation_status": "STRONG_CONFIRMATION",
        "regime_threshold_adjustments": [
            "Configured NEGATIVE_GAMMA threshold and sizing adjustment",
            "Adjusted for VOL_EXPANSION: composite +2, size 0.85x",
        ],
    }

    description = _describe_effective_strength_gate(trade)

    assert description == (
        "60 (base~60; conf:none; regime:Configured NEGATIVE_GAMMA threshold and sizing adjustment, "
        "Adjusted for VOL_EXPANSION: composite +2, size 0.85x)"
    )


def test_summarize_confidence_guards_uses_constrained_by_when_cap_not_applied() -> None:
    """When no cap was binding (score already below all thresholds), use 'constrained by'."""
    note = _summarize_confidence_guards(
        ["provider_health_weak", "data_quality_caution", "direction_unresolved"],
        cap_applied=False,
    )

    assert note == (
        "constrained by weak provider health; constrained by caution data quality; "
        "constrained by unresolved direction"
    )


def test_summarize_confidence_guards_direction_root_suppresses_downstream_with_cap_applied() -> None:
    """direction_unresolved suppresses downstream confirmation/no-trade guards."""
    note = _summarize_confidence_guards(
        [
            "direction_unresolved",
            "confirmation_no_direction",
            "explicit_no_trade_reason",
            "status_watchlist_or_blocked",
        ],
        cap_applied=True,
    )

    # Only the root cause should appear; downstream guards suppressed.
    assert note == "capped by unresolved direction"


def test_render_dealer_gamma_levels_flip_drift_mentions_previous_snapshot() -> None:
    trade = {
        "spot": 23944.7,
        "gamma_flip": 23933.1,
        "gamma_clusters": [24000, 24500],
        "dealer_flow_state": "HEDGING_NEUTRAL",
        "gamma_exposure_greeks": 74000.0,
        "gamma_flip_drift": {
            "drift": 1133.0,
            "drift_direction": "RISING",
            "prev_flip": 22800.1,
        },
    }

    with StringIO() as buffer, redirect_stdout(buffer):
        _render_dealer_gamma_levels(trade)
        output = buffer.getvalue()

    assert "flip_drift" in output
    assert "vs prev snapshot" in output


def test_render_provider_health_compact_detail_uses_reason_when_no_unmet_reasons() -> None:
    trade = {
        "provider_health": {
            "summary_status": "WEAK",
            "source": "NSE",
        },
        "provider_health_override_diagnostics": {
            "eligible": False,
            "reason": "override_disabled",
            "fail_reasons": [],
        }
    }

    with StringIO() as buffer, redirect_stdout(buffer):
        _render_provider_health_compact_detail(trade)
        output = buffer.getvalue()

    assert "override  : eligible=False; unmet=override_disabled" in output


def test_render_data_usability_diagnostics_shows_usability_and_weights() -> None:
    trade = {
        "analytics_usable": True,
        "execution_suggestion_usable": False,
        "tradable_data": {
            "status": "ANALYTICS_ONLY",
            "score": 0.41,
            "reasons": ["crossed_quotes_high", "quote_outliers_high"],
            "crossed_or_locked_ratio": 0.18,
            "quote_outlier_ratio": 0.11,
        },
        "feature_reliability_weights": {
            "gamma": 0.95,
            "flow": 0.72,
            "surface": 0.31,
        },
    }

    with StringIO() as buffer, redirect_stdout(buffer):
        _render_data_usability_diagnostics(trade, verbose=True)
        output = buffer.getvalue()

    assert "DATA USABILITY" in output
    assert "analytics_usable" in output
    assert "execution_suggestion_usable" in output
    assert "ANALYTICS_ONLY" in output
    assert "feature_weights" in output
