from __future__ import annotations

from engine.trading_support.historical_context import build_historical_context
from engine.trading_support.statistical_market_context import build_statistical_market_context


def _artifact() -> dict:
    return {
        "artifact_version": "unit_statistical_artifact",
        "source_run_id": "unit_run",
        "source_report_pdf": "unit.pdf",
        "baseline": {
            "daily_range_median_bps": 100.0,
            "fwd_abs_move_1d_mean_bps": 80.0,
        },
        "numeric_bucket_priors": {
            "india_vix_level": [
                {
                    "bucket": "low",
                    "lower": 10.0,
                    "upper": 15.0,
                    "n": 200,
                    "expected_abs_move_bps": 48.0,
                    "abs_move_delta_vs_base_bps": -32.0,
                    "expected_range_bps": 78.0,
                    "mean_return_bps": 2.0,
                    "hit_positive": 0.52,
                },
                {
                    "bucket": "high",
                    "lower": 24.0,
                    "upper": 90.0,
                    "n": 220,
                    "expected_abs_move_bps": 155.0,
                    "abs_move_delta_vs_base_bps": 75.0,
                    "expected_range_bps": 260.0,
                    "mean_return_bps": 10.0,
                    "hit_positive": 0.54,
                },
            ],
            "pcr_oi": [
                {
                    "bucket": "high",
                    "lower": 1.3,
                    "upper": 2.5,
                    "n": 120,
                    "expected_abs_move_bps": 92.0,
                    "abs_move_delta_vs_base_bps": 12.0,
                    "expected_range_bps": 150.0,
                    "mean_return_bps": 6.0,
                    "hit_positive": 0.55,
                }
            ],
        },
        "categorical_bucket_priors": {
            "trend_20d_bucket": {
                "selloff": {
                    "n": 160,
                    "mean_return_bps": 18.0,
                    "hit_positive": 0.62,
                    "expected_abs_move_bps": 160.0,
                    "abs_move_delta_vs_base_bps": 80.0,
                    "expected_range_bps": 270.0,
                }
            }
        },
        "macro_context": {
            "shock_priors": {
                "sp500_change_24h": {
                    "bottom_decile": {
                        "n": 120,
                        "threshold": -1.5,
                        "mean_return_bps": -35.0,
                        "hit_positive": 0.38,
                        "expected_abs_move_bps": 140.0,
                        "abs_move_delta_vs_base_bps": 60.0,
                        "expected_range_bps": 230.0,
                    },
                    "middle_80": {
                        "n": 800,
                        "mean_return_bps": 3.0,
                        "hit_positive": 0.53,
                        "expected_abs_move_bps": 76.0,
                        "abs_move_delta_vs_base_bps": -4.0,
                        "expected_range_bps": 120.0,
                    },
                },
                "oil_change_24h": {
                    "bottom_decile": {
                        "n": 100,
                        "threshold": -2.0,
                        "mean_return_bps": -12.0,
                        "hit_positive": 0.44,
                        "expected_abs_move_bps": 125.0,
                        "abs_move_delta_vs_base_bps": 45.0,
                        "expected_range_bps": 205.0,
                    }
                },
            },
            "interaction_priors": {
                "macro_commodity_bucket=commodity_down|trend_20d_bucket=selloff": {
                    "interaction": "macro_commodity_bucket=commodity_down|trend_20d_bucket=selloff",
                    "left_feature": "macro_commodity_bucket",
                    "left_bucket": "commodity_down",
                    "right_feature": "trend_20d_bucket",
                    "right_bucket": "selloff",
                    "n": 90,
                    "mean_return_bps": -45.0,
                    "hit_positive": 0.42,
                    "expected_abs_move_bps": 175.0,
                    "abs_move_delta_vs_base_bps": 95.0,
                    "expected_range_bps": 300.0,
                }
            },
        },
        "application_rules": {
            "min_bucket_n": 50,
            "expanded_abs_move_delta_bps": 20.0,
            "high_abs_move_delta_bps": 50.0,
            "compressed_abs_move_delta_bps": -20.0,
            "directional_mean_edge_bps": 8.0,
            "directional_hit_edge": 0.56,
            "conflict_hit_edge": 0.46,
            "tail_risk_size_cap": 0.80,
            "elevated_risk_size_cap": 0.90,
            "macro_expanded_abs_move_delta_bps": 35.0,
            "macro_high_abs_move_delta_bps": 70.0,
            "macro_directional_mean_edge_bps": 10.0,
            "macro_directional_hit_edge": 0.57,
            "macro_conflict_hit_edge": 0.46,
            "macro_tail_risk_size_cap": 0.85,
            "macro_elevated_risk_size_cap": 0.90,
            "macro_conflict_size_cap": 0.85,
            "macro_max_score_adjustment": 3,
            "macro_max_probability_adjustment": 0.02,
        },
    }


def test_statistical_market_context_applies_range_and_direction_priors():
    context = build_statistical_market_context(
        spot=23400.0,
        direction="CALL",
        weekday=0,
        artifact=_artifact(),
        market_state={
            "atm_iv": 29.5,
            "days_to_expiry": 2,
            "open_interest_pcr": 1.60,
            "support_wall": 23350.0,
            "resistance_wall": 23500.0,
            "ta_features": {"indicators": {"ret_20d_bps": -620.0}},
        },
        global_risk_state={"global_risk_features": {"india_vix_level": 29.68}},
    )

    assert context["version"] == "statistical_market_context_v1"
    assert context["artifact_version"] == "unit_statistical_artifact"
    assert context["source_run_id"] == "unit_run"
    assert context["expected_range_prior"] == "EXPANDED_TAIL_RISK"
    assert context["expected_range_bps"] == 260.0
    assert context["directional_followthrough_prior"] == "CALL"
    assert context["score_adjustment"] > 0
    assert context["trade_strength_threshold_adjustment"] < 0
    assert context["size_multiplier"] == 0.8
    assert context["hold_time_hint"] == "ALLOW_LONGER_BUT_SIZE_DOWN"
    assert "statistical_range_expanded_tail_risk" in context["reasons"]
    assert "statistical_directional_prior_aligned_trend_20d_bucket" in context["reasons"]


def test_statistical_market_context_keeps_near_atm_volume_pcr_named_as_volume():
    context = build_statistical_market_context(
        spot=23400.0,
        direction="CALL",
        artifact=_artifact(),
        market_state={
            "atm_iv": 20.0,
            "open_interest_pcr": 1.10,
            "volume_pcr": 0.92,
            "volume_pcr_atm": 0.74,
        },
    )

    assert context["feature_values"]["pcr_oi"] == 1.10
    assert context["feature_values"]["pcr_volume"] == 0.92
    assert context["feature_values"]["near_atm_pcr_volume"] == 0.74
    assert context["feature_values"]["near_atm_pcr_oi"] is None


def test_statistical_market_context_penalizes_conflicting_direction():
    context = build_statistical_market_context(
        spot=23400.0,
        direction="PUT",
        artifact=_artifact(),
        market_state={
            "atm_iv": 29.5,
            "open_interest_pcr": 1.60,
            "ta_features": {"indicators": {"ret_20d_bps": -620.0}},
        },
        global_risk_state={"global_risk_features": {"india_vix_level": 29.68}},
    )

    assert context["directional_followthrough_prior"] == "CALL"
    assert context["score_adjustment"] < 0
    assert context["probability_adjustment"] < 0
    assert context["trade_strength_threshold_adjustment"] > 0
    assert context["size_multiplier"] <= 0.85
    assert "statistical_directional_prior_conflicts_trend_20d_bucket" in context["reasons"]


def test_statistical_market_context_applies_macro_sub_context():
    context = build_statistical_market_context(
        spot=23400.0,
        direction="PUT",
        artifact=_artifact(),
        market_state={
            "atm_iv": 14.0,
            "open_interest_pcr": 1.10,
            "ta_features": {"indicators": {"ret_20d_bps": -620.0}},
        },
        global_risk_state={
            "global_risk_features": {
                "india_vix_level": 14.0,
                "sp500_change_24h": -2.1,
                "oil_change_24h": -2.5,
            }
        },
    )

    macro = context["macro_context"]
    assert macro["applied"] is True
    assert macro["macro_range_prior"] == "EXPANDED_TAIL_RISK"
    assert macro["macro_directional_prior"] == "PUT"
    assert macro["macro_factor_buckets"]["macro_risk_bucket"] == "risk_off"
    assert macro["macro_factor_buckets"]["macro_commodity_bucket"] == "commodity_down"
    assert context["expected_range_prior"] == "EXPANDED_TAIL_RISK"
    assert context["expected_range_bps"] == 300.0
    assert macro["score_adjustment"] > 0
    assert "macro_statistical_directional_prior_aligned_macro_commodity_bucket=commodity_down|trend_20d_bucket=selloff" in context["reasons"]


def test_historical_context_exposes_statistical_market_context_component():
    context = build_historical_context(
        spot=23400.0,
        direction="CALL",
        market_state={
            "atm_iv": 29.5,
            "days_to_expiry": 2,
            "open_interest_pcr": 1.60,
            "support_wall": 23350.0,
            "resistance_wall": 23500.0,
            "ta_features": {"indicators": {"ret_20d_bps": -620.0}},
        },
        global_risk_state={"global_risk_features": {"india_vix_level": 29.68}},
    )

    statistical = context["statistical_market_context"]
    assert statistical["applied"] is True
    assert statistical["expected_range_prior"] in {"EXPANDED", "EXPANDED_TAIL_RISK"}
    assert "statistical_market_context" in context["live_modifiers"]["components"]
    assert any(str(note).startswith("stat_range=") for note in context["primary_notes"])
