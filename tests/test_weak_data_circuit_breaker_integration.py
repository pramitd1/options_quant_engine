from __future__ import annotations

import copy

from config.signal_policy import TRADE_RUNTIME_THRESHOLDS
from data.option_chain_validation import validate_option_chain
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot
from engine import signal_engine as se
from engine.signal_engine import generate_trade


def test_generate_trade_routes_to_watchlist_when_weak_data_breaker_triggers(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot_raw = spot_snapshot.get("spot")
    assert spot_raw is not None
    spot = float(spot_raw)
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    thresholds = dict(TRADE_RUNTIME_THRESHOLDS)
    thresholds["provider_health_caution_blocks_trade"] = 0
    thresholds["min_composite_score"] = 0
    monkeypatch.setattr(se, "get_trade_runtime_thresholds", lambda: thresholds)

    monkeypatch.setattr(
        se,
        "_evaluate_weak_data_circuit_breaker",
        lambda **kwargs: (
            True,
            {
                "enabled": True,
                "triggered": True,
                "trigger_count": 3,
                "trigger_reasons": ["provider_health_fragile", "confirmation_not_strong", "proxy_ratio_above_cap"],
            },
        ),
    )

    monkeypatch.setattr(
        se,
        "_compute_signal_state",
        lambda **kwargs: {
            "direction": "CALL",
            "direction_source": "test_fixture",
            "trade_strength": 90,
            "scoring_breakdown": {},
            "confirmation": {
                "status": "CONFIRMED",
                "veto": False,
                "reasons": [],
                "breakdown": {},
                "score_adjustment": 0,
            },
        },
    )

    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )

    trade = generate_trade(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        option_chain_validation=option_chain_validation,
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )

    assert isinstance(trade, dict)
    assert trade.get("trade_status") == "WATCHLIST"
    assert trade.get("no_trade_reason_code") == "WEAK_DATA_CIRCUIT_BREAKER"
    assert (trade.get("weak_data_circuit_breaker") or {}).get("triggered") is True
    assert (trade.get("weak_data_circuit_breaker_shadow") or {}).get("triggered") is True


def test_generate_trade_routes_to_watchlist_when_execution_suggestion_unusable(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot["spot"])
    option_chain_validation = validate_option_chain(option_chain, spot=spot)
    option_chain_validation = copy.deepcopy(option_chain_validation)
    option_chain_validation["analytics_usable"] = True
    option_chain_validation["execution_suggestion_usable"] = False
    option_chain_validation["tradable_data"] = {
        "status": "ANALYTICS_ONLY",
        "score": 0.42,
        "reasons": ["crossed_quotes_high"],
    }
    option_chain_validation.setdefault("provider_health", {})["summary_status"] = "GOOD"
    option_chain_validation["provider_health"]["trade_blocking_status"] = "PASS"

    thresholds = dict(TRADE_RUNTIME_THRESHOLDS)
    thresholds["provider_health_caution_blocks_trade"] = 0
    thresholds["min_trade_strength"] = 0
    thresholds["min_composite_score"] = 0
    monkeypatch.setattr(se, "get_trade_runtime_thresholds", lambda: thresholds)

    monkeypatch.setattr(
        se,
        "_evaluate_weak_data_circuit_breaker",
        lambda **kwargs: (False, {"enabled": True, "triggered": False, "trigger_reasons": []}),
    )

    monkeypatch.setattr(
        se,
        "_compute_signal_state",
        lambda **kwargs: {
            "direction": "CALL",
            "direction_source": "test_fixture",
            "trade_strength": 90,
            "scoring_breakdown": {},
            "confirmation": {
                "status": "STRONG_CONFIRMATION",
                "veto": False,
                "reasons": [],
                "breakdown": {},
                "score_adjustment": 0,
            },
        },
    )

    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )

    trade = generate_trade(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        option_chain_validation=option_chain_validation,
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )

    assert trade.get("trade_status") == "WATCHLIST"
    assert trade.get("no_trade_reason_code") == "EXECUTION_DATA_UNUSABLE"
    assert "tradable-data gate" in str(trade.get("message") or "")


def test_generate_trade_includes_iv_surface_residual_penalty_in_scoring(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot["spot"])
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    original_collect_market_state = se._collect_market_state

    def _collect_market_state_with_penalty(*args, **kwargs):
        state = original_collect_market_state(*args, **kwargs)
        state["iv_surface_residual_status"] = "DEGRADED"
        state["iv_surface_residual_penalty_score"] = 9
        state["iv_surface_residual_penalty_reasons"] = ["residual_rmse_high"]
        return state

    monkeypatch.setattr(se, "_collect_market_state", _collect_market_state_with_penalty)

    thresholds = dict(TRADE_RUNTIME_THRESHOLDS)
    thresholds["provider_health_caution_blocks_trade"] = 0
    monkeypatch.setattr(se, "get_trade_runtime_thresholds", lambda: thresholds)

    monkeypatch.setattr(
        se,
        "_evaluate_weak_data_circuit_breaker",
        lambda **kwargs: (False, {"enabled": True, "triggered": False, "trigger_reasons": []}),
    )

    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )

    trade = generate_trade(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        option_chain_validation=option_chain_validation,
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )

    assert trade.get("iv_surface_residual_status") == "DEGRADED"
    assert trade.get("iv_surface_residual_penalty_score") == 9
    assert (trade.get("scoring_breakdown") or {}).get("iv_surface_residual_penalty") == -9


def test_generate_trade_feature_reliability_overlay_reduces_scores(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot["spot"])
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    strong_validation = copy.deepcopy(option_chain_validation)
    strong_validation["feature_reliability_weights"] = {
        "flow": 1.0,
        "vol_surface": 1.0,
        "greeks": 1.0,
        "liquidity": 1.0,
        "macro": 1.0,
    }

    fragile_validation = copy.deepcopy(option_chain_validation)
    fragile_validation["feature_reliability_weights"] = {
        "flow": 0.35,
        "vol_surface": 0.25,
        "greeks": 0.30,
        "liquidity": 0.28,
        "macro": 0.80,
    }

    thresholds = dict(TRADE_RUNTIME_THRESHOLDS)
    thresholds["provider_health_caution_blocks_trade"] = 0
    thresholds["min_trade_strength"] = 0
    thresholds["min_composite_score"] = 0
    monkeypatch.setattr(se, "get_trade_runtime_thresholds", lambda: thresholds)

    monkeypatch.setattr(
        se,
        "_evaluate_weak_data_circuit_breaker",
        lambda **kwargs: (False, {"enabled": True, "triggered": False, "trigger_reasons": []}),
    )

    monkeypatch.setattr(
        se,
        "_compute_signal_state",
        lambda **kwargs: {
            "direction": "CALL",
            "direction_source": "test_fixture",
            "trade_strength": 86,
            "scoring_breakdown": {},
            "confirmation": {
                "status": "STRONG_CONFIRMATION",
                "veto": False,
                "reasons": [],
                "breakdown": {},
                "score_adjustment": 0,
            },
        },
    )

    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )

    base_kwargs = dict(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )

    strong_trade = generate_trade(option_chain_validation=strong_validation, **base_kwargs)
    fragile_trade = generate_trade(option_chain_validation=fragile_validation, **base_kwargs)

    assert strong_trade.get("trade_strength") > fragile_trade.get("trade_strength")
    assert strong_trade.get("runtime_composite_score") > fragile_trade.get("runtime_composite_score")
    assert fragile_trade.get("feature_reliability_status") == "FRAGILE"
    assert fragile_trade.get("feature_reliability_penalty_score", 0) > 0
    assert (fragile_trade.get("scoring_breakdown") or {}).get("feature_reliability_penalty", 0) < 0


def test_generate_trade_scales_overlay_adjustments_by_feature_reliability(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot["spot"])
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    strong_validation = copy.deepcopy(option_chain_validation)
    strong_validation["feature_reliability_weights"] = {
        "flow": 1.0,
        "vol_surface": 1.0,
        "greeks": 1.0,
        "liquidity": 1.0,
        "macro": 1.0,
    }
    fragile_validation = copy.deepcopy(option_chain_validation)
    fragile_validation["feature_reliability_weights"] = {
        "flow": 0.40,
        "vol_surface": 0.25,
        "greeks": 0.35,
        "liquidity": 0.30,
        "macro": 0.80,
    }

    thresholds = dict(TRADE_RUNTIME_THRESHOLDS)
    thresholds["provider_health_caution_blocks_trade"] = 0
    thresholds["min_trade_strength"] = 0
    thresholds["min_composite_score"] = 0
    monkeypatch.setattr(se, "get_trade_runtime_thresholds", lambda: thresholds)
    monkeypatch.setattr(
        se,
        "_evaluate_weak_data_circuit_breaker",
        lambda **kwargs: (False, {"enabled": True, "triggered": False, "trigger_reasons": []}),
    )
    monkeypatch.setattr(
        se,
        "_compute_signal_state",
        lambda **kwargs: {
            "direction": "CALL",
            "direction_source": "test_fixture",
            "trade_strength": 70,
            "scoring_breakdown": {},
            "confirmation": {
                "status": "STRONG_CONFIRMATION",
                "veto": False,
                "reasons": [],
                "breakdown": {},
                "score_adjustment": 10,
            },
        },
    )
    monkeypatch.setattr(
        se,
        "compute_macro_news_adjustments",
        lambda **kwargs: {
            "macro_confirmation_adjustment": 0,
            "event_overlay_score_adjustment": 0,
            "macro_adjustment_score": 0,
            "volatility_shock_score": 0,
            "event_overlay_probability_multiplier": 1.0,
            "event_overlay_suppress_signal": False,
            "event_lockdown_flag": False,
            "macro_regime": "MACRO_NEUTRAL",
            "macro_sentiment_score": 0,
            "news_confidence_score": 0,
            "macro_position_size_multiplier": 1.0,
            "macro_adjustment_reasons": [],
            "event_overlay_reasons": [],
        },
    )
    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )
    monkeypatch.setattr(
        se,
        "derive_gamma_vol_trade_modifiers",
        lambda *args, **kwargs: {
            "base_adjustment_score": 6,
            "alignment_adjustment_score": 6,
            "effective_adjustment_score": 12,
            "adjustment_reasons": [],
            "gamma_vol_acceleration_score": 20,
            "squeeze_risk_state": "LOW_ACCELERATION_RISK",
            "directional_convexity_state": "UPSIDE_SQUEEZE_RISK",
            "upside_squeeze_risk": 0.0,
            "downside_airpocket_risk": 0.0,
            "overnight_convexity_risk": 0.0,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_convexity_contained",
            "overnight_convexity_penalty": 0,
            "overnight_convexity_boost": 0,
        },
    )
    monkeypatch.setattr(
        se,
        "derive_dealer_pressure_trade_modifiers",
        lambda *args, **kwargs: {
            "base_adjustment_score": 4,
            "alignment_adjustment_score": 4,
            "effective_adjustment_score": 8,
            "adjustment_reasons": [],
            "dealer_hedging_pressure_score": 25,
            "dealer_flow_state": "UPSIDE_HEDGING_ACCELERATION",
            "upside_hedging_pressure": 0.0,
            "downside_hedging_pressure": 0.0,
            "pinning_pressure_score": 0.0,
            "overnight_hedging_risk": 0.0,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_hedging_contained",
            "overnight_dealer_pressure_penalty": 0,
            "overnight_dealer_pressure_boost": 0,
        },
    )
    monkeypatch.setattr(
        se,
        "derive_option_efficiency_trade_modifiers",
        lambda *args, **kwargs: {
            "effective_adjustment_score": 9,
            "expected_move_points": 120.0,
            "expected_move_pct": 1.2,
            "expected_move_quality": "DIRECT",
            "target_reachability_score": 60,
            "premium_efficiency_score": 62,
            "strike_efficiency_score": 64,
            "option_efficiency_score": 63,
            "option_efficiency_adjustment_score": 9,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_option_efficiency_contained",
            "overnight_option_efficiency_penalty": 0,
            "strike_moneyness_bucket": "ATM",
            "strike_distance_from_spot": 0.0,
            "payoff_efficiency_hint": "balanced",
        },
    )

    base_kwargs = dict(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )
    strong_trade = generate_trade(option_chain_validation=strong_validation, **base_kwargs)
    fragile_trade = generate_trade(option_chain_validation=fragile_validation, **base_kwargs)

    strong_scoring = strong_trade.get("scoring_breakdown") or {}
    fragile_scoring = fragile_trade.get("scoring_breakdown") or {}

    assert fragile_scoring.get("confirmation_filter_score", 0) < strong_scoring.get("confirmation_filter_score", 0)
    assert fragile_scoring.get("gamma_vol_adjustment_score", 0) < strong_scoring.get("gamma_vol_adjustment_score", 0)
    assert fragile_scoring.get("dealer_pressure_adjustment_score", 0) < strong_scoring.get("dealer_pressure_adjustment_score", 0)
    assert fragile_scoring.get("option_efficiency_adjustment_score", 0) < strong_scoring.get("option_efficiency_adjustment_score", 0)


def test_generate_trade_scales_strike_candidate_hook_by_reliability(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot["spot"])
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    strong_validation = copy.deepcopy(option_chain_validation)
    strong_validation["feature_reliability_weights"] = {
        "flow": 1.0,
        "vol_surface": 1.0,
        "greeks": 1.0,
        "liquidity": 1.0,
        "macro": 1.0,
    }
    fragile_validation = copy.deepcopy(option_chain_validation)
    fragile_validation["feature_reliability_weights"] = {
        "flow": 0.85,
        "vol_surface": 0.25,
        "greeks": 0.45,
        "liquidity": 0.30,
        "macro": 0.80,
    }

    thresholds = dict(TRADE_RUNTIME_THRESHOLDS)
    thresholds["provider_health_caution_blocks_trade"] = 0
    thresholds["min_trade_strength"] = 0
    thresholds["min_composite_score"] = 0
    monkeypatch.setattr(se, "get_trade_runtime_thresholds", lambda: thresholds)
    monkeypatch.setattr(
        se,
        "_evaluate_weak_data_circuit_breaker",
        lambda **kwargs: (False, {"enabled": True, "triggered": False, "trigger_reasons": []}),
    )
    monkeypatch.setattr(
        se,
        "_compute_signal_state",
        lambda **kwargs: {
            "direction": "CALL",
            "direction_source": "test_fixture",
            "trade_strength": 88,
            "scoring_breakdown": {},
            "confirmation": {
                "status": "STRONG_CONFIRMATION",
                "veto": False,
                "reasons": [],
                "breakdown": {},
                "score_adjustment": 0,
            },
        },
    )
    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )
    monkeypatch.setattr(
        se,
        "score_option_efficiency_candidate",
        lambda *args, **kwargs: {
            "score_adjustment": 3,
            "option_efficiency_score": 72,
            "strike_efficiency_score": 70,
            "premium_efficiency_score": 68,
            "expected_move_points": 120.0,
            "expected_move_quality": "DIRECT",
            "strike_moneyness_bucket": "ATM",
        },
    )

    base_kwargs = dict(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )

    strong_trade = generate_trade(option_chain_validation=strong_validation, **base_kwargs)
    fragile_trade = generate_trade(option_chain_validation=fragile_validation, **base_kwargs)

    strong_candidate = (strong_trade.get("ranked_strike_candidates") or [])[0]
    fragile_candidate = (fragile_trade.get("ranked_strike_candidates") or [])[0]

    assert strong_candidate.get("score_adjustment_raw") == 3
    assert (strong_candidate.get("score_breakdown") or {}).get("option_efficiency_score_adjustment") == 3
    assert fragile_candidate.get("score_adjustment_raw") == 3
    assert (fragile_candidate.get("score_breakdown") or {}).get("option_efficiency_score_adjustment", 0) < 3
    assert fragile_candidate.get("strike_reliability_weight", 1.0) < strong_candidate.get("strike_reliability_weight", 1.0)
    assert fragile_candidate.get("strike_reliability_delta", 0) < 0
