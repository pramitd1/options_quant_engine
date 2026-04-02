from __future__ import annotations

from config.policy_resolver import temporary_parameter_pack
from engine.trading_support.signal_state import decide_direction


def test_decide_direction_marks_expansion_mode_on_bullish_breakout_reversal():
    direction, source, _, _, expansion_mode, expansion_direction, breakout_evidence = decide_direction(
        final_flow_signal="BULLISH_FLOW",
        dealer_pos="Short Gamma",
        vol_regime="VOL_EXPANSION",
        spot_vs_flip="ABOVE_FLIP",
        gamma_regime="NEGATIVE_GAMMA",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="GAMMA_SQUEEZE",
        vanna_regime="POSITIVE_VANNA",
        charm_regime="POSITIVE_CHARM",
        spot=23180.0,
        resistance_wall=23120.0,
        support_wall=22980.0,
        vacuum_state="BREAKOUT_ZONE",
        intraday_range_pct=0.72,
        intraday_gamma_state="VOL_EXPANSION",
        previous_direction="PUT",
        reversal_age=4,
        hybrid_move_probability=0.68,
        macro_news_state={
            "macro_regime": "RISK_ON",
            "news_confidence_score": 72.0,
            "macro_sentiment_score": 8.0,
            "india_macro_bias": 6.0,
            "neutral_fallback": False,
            "warnings": [],
        },
        global_risk_state={
            "global_risk_features": {
                "usdinr_change_24h": -0.15,
                "currency_shock_score": 0.0,
            }
        },
    )

    assert direction == "CALL"
    assert "BREAKOUT_STRUCTURE" in source
    assert expansion_mode is True
    assert expansion_direction == "CALL"
    assert breakout_evidence >= 1.25


def test_asymmetric_flipback_guard_blocks_weak_immediate_reflip_without_expansion():
    kwargs = dict(
        final_flow_signal="BEARISH_FLOW",
        dealer_pos="Long Gamma",
        vol_regime="NORMAL_VOL",
        spot_vs_flip="AT_FLIP",
        gamma_regime="NEUTRAL_GAMMA",
        hedging_bias="NEUTRAL",
        gamma_event="NONE",
        vanna_regime="POSITIVE_VANNA",
        charm_regime=None,
        volume_pcr_atm=1.35,
        previous_direction="CALL",
        reversal_age=1,
        hybrid_move_probability=0.49,
        spot=23005.0,
        resistance_wall=23080.0,
        support_wall=22940.0,
        vacuum_state="NEUTRAL",
        intraday_range_pct=0.12,
        intraday_gamma_state="NEUTRAL",
    )

    with temporary_parameter_pack(
        "baseline_v1",
        overrides={
            "trade_strength.direction_thresholds.min_score": 1.4,
            "trade_strength.runtime_thresholds.asymmetric_flipback_guard_steps": 0,
        },
    ):
        no_guard = decide_direction(**kwargs)

    with temporary_parameter_pack(
        "baseline_v1",
        overrides={
            "trade_strength.direction_thresholds.min_score": 1.4,
            "trade_strength.runtime_thresholds.asymmetric_flipback_guard_steps": 3,
            "trade_strength.runtime_thresholds.asymmetric_flipback_margin_surcharge": 0.35,
            "trade_strength.runtime_thresholds.asymmetric_flipback_score_surcharge": 0.20,
        },
    ):
        guarded = decide_direction(**kwargs)

    assert no_guard[0] == "PUT"
    assert guarded[0] is None


def test_usdinr_depreciation_adds_fx_pressure_vote_to_put_direction():
    direction, source, _, _, expansion_mode, expansion_direction, _ = decide_direction(
        final_flow_signal="BEARISH_FLOW",
        dealer_pos="Long Gamma",
        vol_regime="NORMAL_VOL",
        spot_vs_flip="BELOW_FLIP",
        gamma_regime="NEUTRAL_GAMMA",
        hedging_bias="NEUTRAL",
        gamma_event="NONE",
        previous_direction="CALL",
        reversal_age=5,
        hybrid_move_probability=0.61,
        spot=22940.0,
        resistance_wall=23100.0,
        support_wall=22980.0,
        vacuum_state="NEUTRAL",
        intraday_range_pct=0.31,
        intraday_gamma_state="VOL_EXPANSION",
        global_risk_state={
            "global_risk_features": {
                "usdinr_change_24h": 0.92,
                "currency_shock_score": 0.5,
            }
        },
    )

    assert direction == "PUT"
    assert "FX_PRESSURE" in source
    assert expansion_direction == "PUT" or expansion_mode is False