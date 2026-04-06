from __future__ import annotations

from engine.trading_support.signal_state import decide_direction


def test_direction_microstructure_friction_can_suppress_marginal_call_setup():
    direction_clean, source_clean, *_ = decide_direction(
        final_flow_signal="BULLISH_FLOW",
        dealer_pos=None,
        vol_regime="NORMAL_VOL",
        spot_vs_flip="AT_FLIP",
        gamma_regime="NEGATIVE_GAMMA",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="NONE",
        vanna_regime=None,
        charm_regime=None,
        provider_health_summary="GOOD",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.90,
        core_one_sided_quote_ratio=0.05,
        core_quote_integrity_health="GOOD",
    )

    direction_fragile, source_fragile, *_ = decide_direction(
        final_flow_signal="BULLISH_FLOW",
        dealer_pos=None,
        vol_regime="NORMAL_VOL",
        spot_vs_flip="AT_FLIP",
        gamma_regime="NEGATIVE_GAMMA",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="NONE",
        vanna_regime=None,
        charm_regime=None,
        provider_health_summary="WEAK",
        provider_health_blocking_status="BLOCK",
        core_effective_priced_ratio=0.25,
        core_one_sided_quote_ratio=0.75,
        core_quote_integrity_health="WEAK",
    )

    assert direction_clean == "CALL"
    assert direction_fragile is None
    assert source_clean and "HEDGING_BIAS" in source_clean
    assert source_fragile is None


def test_direction_source_marks_microstructure_friction_when_direction_survives():
    direction, source, *_ = decide_direction(
        final_flow_signal="BULLISH_FLOW",
        dealer_pos="Short Gamma",
        vol_regime="VOL_EXPANSION",
        spot_vs_flip="ABOVE_FLIP",
        gamma_regime="NEGATIVE_GAMMA",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="GAMMA_SQUEEZE",
        vanna_regime="POSITIVE_VANNA",
        charm_regime="POSITIVE_CHARM",
        provider_health_summary="CAUTION",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.48,
        core_one_sided_quote_ratio=0.28,
        core_quote_integrity_health="CAUTION",
    )

    assert direction == "CALL"
    assert source is not None
    assert "MICROSTRUCTURE_FRICTION" in source
