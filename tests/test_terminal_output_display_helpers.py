from app.terminal_output import (
    _describe_effective_strength_gate,
    _format_probability_display,
    _summarize_confidence_guards,
)


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
        "capped by weak provider health; capped by blocked/watchlist status; "
        "capped by explicit no-trade reason"
    )


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
