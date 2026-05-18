from __future__ import annotations

from engine.trading_support.historical_context import build_historical_context


def _test_interaction_artifact():
    return {
        "artifact_version": "test_prior_artifact_v1",
        "source_run_id": "test_run",
        "bucket_thresholds": {
            "pcr_oi": [
                {"label": "low", "upper": 0.80},
                {"label": "mid", "upper": 1.20},
                {"label": "high", "upper": None},
            ],
            "trend_20d": [
                {"label": "selloff", "upper": -500.0},
                {"label": "flat", "upper": 500.0},
                {"label": "surge", "upper": None},
            ],
            "expiry": [
                {"label": "0-1d", "upper": 1.0},
                {"label": "2-3d", "upper": 3.0},
                {"label": "4d+", "upper": None},
            ],
        },
        "interactions": {
            "expiry_x_pcr": {
                "target": "fwd_ret_1d_bps",
                "rows": {
                    "2-3d|high": {"n": 130, "mean_bps": 18.0, "hit_up": 0.64, "abs_mean_bps": 55.0},
                },
            },
            "india_vix_x_trend": {
                "target": "fwd_ret_1d_bps",
                "rows": {
                    "high|selloff": {"n": 150, "mean_bps": 21.0, "hit_up": 0.63, "abs_mean_bps": 88.0},
                },
            },
            "weekday_x_vix": {
                "target": "next_day_range_bps",
                "rows": {
                    "Monday|high": {"n": 160, "mean_bps": 266.0, "hit_up": 1.0, "abs_mean_bps": 266.0},
                },
            },
        },
    }


def test_historical_context_builds_live_priors_and_direction_fallback():
    context = build_historical_context(
        spot=23400.0,
        market_state={
            "atm_iv": 29.5,
            "volume_pcr_atm": 1.42,
            "max_pain_dist": 85.0,
            "support_wall": 23350.0,
            "resistance_wall": 23500.0,
        },
        global_risk_state={
            "global_risk_features": {
                "india_vix_level": 29.68,
                "sp500_change_24h": -0.9,
                "nasdaq_change_24h": -1.2,
                "us_vix_change_24h": 6.5,
            }
        },
    )

    assert context["version"] == "historical_context_v1"
    assert context["decision_mode"] == "LIVE_APPLIED"
    assert context["apply_to_live_decision"] is True
    assert context["volatility_context"]["bucket"] == "HIGH"
    assert context["volatility_context"]["expected_range_bps"] == 262.25
    assert context["global_directional_prior"]["prior_direction"] == "PUT"
    assert context["global_directional_prior"]["prior_score"] < 0
    assert context["direction_override"] == "PUT"
    assert context["score_adjustment"] > 0
    assert context["probability_adjustment"] > 0
    assert context["trade_strength_threshold_adjustment"] > 0
    assert context["size_multiplier"] < 1.0
    assert context["pcr_context"]["interpretation"] == "support_or_pinning_context_not_automatic_bearish_signal"
    assert "high_pcr_as_pinning_not_bearish" in context["primary_notes"]


def test_historical_context_uses_vix_drop_as_call_evidence_only_when_drop_is_extreme():
    neutral_context = build_historical_context(
        spot=23400.0,
        market_state={"atm_iv": 16.0},
        global_risk_state={
            "global_risk_features": {
                "india_vix_level": 16.0,
                "us_vix_change_24h": -2.0,
            }
        },
    )
    call_context = build_historical_context(
        spot=23400.0,
        market_state={"atm_iv": 16.0},
        global_risk_state={
            "global_risk_features": {
                "india_vix_level": 16.0,
                "us_vix_change_24h": -6.0,
            }
        },
    )

    assert neutral_context["global_directional_prior"]["prior_direction"] == "NEUTRAL"
    assert call_context["global_directional_prior"]["prior_direction"] == "CALL"
    assert call_context["global_directional_prior"]["prior_score"] > 0
    assert call_context["direction_override"] is None


def test_historical_context_applies_generated_interaction_priors():
    context = build_historical_context(
        spot=23400.0,
        direction="CALL",
        weekday=0,
        artifact=_test_interaction_artifact(),
        market_state={
            "atm_iv": 29.5,
            "days_to_expiry": 2,
            "open_interest_pcr": 1.60,
            "volume_pcr_atm": 0.60,
            "ta_features": {"indicators": {"ret_20d_bps": -620.0}},
        },
        global_risk_state={"global_risk_features": {"india_vix_level": 29.68}},
    )

    interaction_context = context["interaction_context"]
    assert context["prior_artifact_version"] == "test_prior_artifact_v1"
    assert context["prior_artifact_source_run_id"] == "test_run"
    assert context["pcr_context"]["basis"] == "OPEN_INTEREST"
    assert context["pcr_context"]["value"] == 1.6
    assert context["pcr_context"]["volume_atm_value"] == 0.6
    assert interaction_context["matched_count"] == 3
    assert interaction_context["bucket_state"]["expiry_bucket"] == "2-3d"
    assert interaction_context["bucket_state"]["pcr_oi_bucket"] == "high"
    assert interaction_context["bucket_state"]["pcr_basis"] == "OPEN_INTEREST"
    assert interaction_context["bucket_state"]["india_vix_bucket"] == "high"
    assert interaction_context["bucket_state"]["trend_20d_bucket"] == "selloff"
    assert interaction_context["bucket_state"]["weekday"] == "Monday"
    assert interaction_context["score_adjustment"] > 0
    assert interaction_context["probability_adjustment"] > 0
    assert "expiry_x_pcr_aligned_call" in interaction_context["reasons"]
    assert "india_vix_x_trend_aligned_call" in interaction_context["reasons"]
    assert "high_range_weekday_vix_interaction" in interaction_context["reasons"]
    assert context["live_modifiers"]["components"]["interaction_context"]["matched_count"] == 3
    assert "interactions=3" in context["primary_notes"]


def test_historical_context_falls_back_to_volume_pcr_when_oi_pcr_is_missing():
    context = build_historical_context(
        spot=23400.0,
        direction="PUT",
        market_state={"atm_iv": 16.0, "volume_pcr_atm": 1.42},
        global_risk_state={"global_risk_features": {"india_vix_level": 16.0}},
        artifact=_test_interaction_artifact(),
    )

    assert context["pcr_context"]["basis"] == "VOLUME_ATM"
    assert context["pcr_context"]["source_warning"] == "historical_finding_was_oi_pcr_live_value_may_be_volume_pcr"


def test_historical_context_penalizes_direction_when_global_prior_conflicts():
    context = build_historical_context(
        spot=23400.0,
        direction="CALL",
        market_state={
            "atm_iv": 29.5,
            "volume_pcr_atm": 1.42,
            "max_pain_dist": 85.0,
            "support_wall": 23350.0,
            "resistance_wall": 23460.0,
        },
        global_risk_state={
            "global_risk_features": {
                "india_vix_level": 29.68,
                "sp500_change_24h": -0.9,
                "nasdaq_change_24h": -1.2,
                "us_vix_change_24h": 6.5,
            }
        },
    )

    assert context["global_directional_prior"]["prior_direction"] == "PUT"
    assert context["live_modifiers"]["components"]["global_directional_prior"]["score_adjustment"] < 0
    assert context["score_adjustment"] < 0
    assert context["probability_adjustment"] < 0
    assert context["trade_strength_threshold_adjustment"] > 0
    assert context["size_multiplier"] < 1.0
