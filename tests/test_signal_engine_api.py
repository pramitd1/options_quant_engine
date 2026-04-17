from __future__ import annotations

import pandas as pd
import pytest

import engine.signal_engine as signal_engine
import engine.trading_engine as trading_engine
from engine.trading_support import market_state as market_state_mod


def test_trading_engine_facade_matches_signal_engine():
    assert trading_engine.generate_trade is signal_engine.generate_trade


def test_generate_trade_passes_days_to_expiry_into_market_state(monkeypatch):
    captured = {"days_to_expiry": None}

    class _StopReview(Exception):
        pass

    def _fake_normalize_option_chain(option_chain, spot=None, valuation_time=None):
        return option_chain

    def _fake_collect_market_state(df, spot, symbol=None, prev_df=None, days_to_expiry=None):
        captured["days_to_expiry"] = days_to_expiry
        raise _StopReview()

    monkeypatch.setattr(signal_engine, "normalize_option_chain", _fake_normalize_option_chain)
    monkeypatch.setattr(signal_engine, "_collect_market_state", _fake_collect_market_state)

    option_chain = pd.DataFrame(
        {
            "strikePrice": [22500],
            "OPTION_TYP": ["CE"],
            "lastPrice": [100.0],
        }
    )

    with pytest.raises(_StopReview):
        signal_engine.generate_trade(
            symbol="NIFTY",
            spot=22500.0,
            option_chain=option_chain,
            option_chain_validation={"selected_expiry": "2026-03-28 10:00:00"},
            valuation_time="2026-03-27 10:00:00",
        )

    assert captured["days_to_expiry"] == 1.0


def test_collect_market_state_emits_timing_breakdown(monkeypatch):
    monkeypatch.setattr(
        market_state_mod,
        "_call_first",
        lambda module, candidate_names, *args, default=None, **kwargs: default,
    )
    monkeypatch.setattr(
        market_state_mod,
        "summarize_greek_exposures",
        lambda df: {
            "delta_exposure": None,
            "gamma_exposure_greeks": None,
            "theta_exposure": None,
            "vega_exposure": None,
            "rho_exposure": None,
            "vanna_exposure": None,
            "charm_exposure": None,
            "vanna_regime": None,
            "charm_regime": None,
        },
    )
    monkeypatch.setattr(market_state_mod, "normalize_flow_signal", lambda flow, smart: "NEUTRAL_FLOW")
    monkeypatch.setattr(market_state_mod, "classify_spot_vs_flip_for_symbol", lambda symbol, spot, flip: "AT_FLIP")
    monkeypatch.setattr(market_state_mod, "build_dealer_liquidity_map", lambda **kwargs: {})

    state = market_state_mod._collect_market_state(
        pd.DataFrame({"strikePrice": [22000], "OPTION_TYP": ["CE"]}),
        22000.0,
        symbol="NIFTY",
    )

    timings = state["market_state_timings"]
    assert timings["total_ms"] >= 0.0
    assert "gamma_exposure" in timings["step_ms"]
    assert "greek_exposures" in timings["step_ms"]
    assert len(timings["slowest_steps"]) > 0


def test_feature_reliability_overlay_penalizes_fragile_inputs():
    overlay = signal_engine._compute_feature_reliability_overlay(
        {
            "feature_reliability_weights": {
                "flow": 0.42,
                "vol_surface": 0.28,
                "greeks": 0.39,
                "liquidity": 0.34,
                "macro": 0.80,
            }
        }
    )

    assert overlay["status"] == "FRAGILE"
    assert overlay["aggregate_score"] < 70.0
    assert overlay["trade_strength_penalty"] == 0
    assert overlay["runtime_composite_penalty"] == 0
    assert "vol_surface_low_reliability" in overlay["reasons"]


def test_collect_market_state_includes_iv_hv_analytics(monkeypatch):
    monkeypatch.setattr(
        market_state_mod,
        "_call_first",
        lambda module, candidate_names, *args, default=None, **kwargs: default,
    )
    monkeypatch.setattr(
        market_state_mod,
        "summarize_greek_exposures",
        lambda df: {
            "delta_exposure": None,
            "gamma_exposure_greeks": None,
            "theta_exposure": None,
            "vega_exposure": None,
            "rho_exposure": None,
            "vanna_exposure": None,
            "charm_exposure": None,
            "vanna_regime": None,
            "charm_regime": None,
        },
    )
    monkeypatch.setattr(market_state_mod, "normalize_flow_signal", lambda flow, smart: "NEUTRAL_FLOW")
    monkeypatch.setattr(market_state_mod, "classify_spot_vs_flip_for_symbol", lambda symbol, spot, flip: "AT_FLIP")
    monkeypatch.setattr(market_state_mod, "build_dealer_liquidity_map", lambda **kwargs: {})

    chain = pd.DataFrame(
        {
            "strikePrice": [22000, 22000],
            "STRIKE_PR": [22000, 22000],
            "OPTION_TYP": ["CE", "PE"],
            "IV": [12.0, 13.0],
            "EXPIRY_DT": ["2099-12-31", "2099-12-31"],
            "hist_vol_20d": [18.0, 18.0],
        }
    )

    state = market_state_mod._collect_market_state(chain, 22000.0, symbol="NIFTY")

    assert "iv_hv_regime" in state
    assert state["iv_hv_regime"] in {"IV_CHEAP", "IV_FAIR", "IV_RICH", "UNAVAILABLE"}


def test_advisory_sizing_stays_separate_from_signal_payload():
    payload = {"number_of_lots": 4}

    sizing = signal_engine._derive_advisory_size_recommendation(
        payload,
        confidence_score=72.0,
        global_risk_size_cap=0.8,
        at_flip_size_cap=1.0,
        macro_size_multiplier=1.0,
        freshness_size_cap=1.0,
        reversal_stage="CONFIRMED_REVERSAL",
        expansion_mode=False,
        expansion_direction=None,
        direction="CALL",
        runtime_thresholds={},
        regime_thresholds={"position_size_multiplier": 1.0},
        gamma_regime="NEGATIVE_GAMMA",
    )

    assert sizing["advisory_only"] is True
    assert payload["number_of_lots"] == 4
    assert sizing["advisory_lots"] <= 4


def test_advisory_sizing_uses_portfolio_heat_in_allocation_ladder():
    payload = {
        "number_of_lots": 4,
        "trade_strength": 81,
        "signal_success_probability": 0.66,
        "portfolio_book_heat_score": 78,
        "portfolio_book_heat_label": "HOT",
    }

    sizing = signal_engine._derive_advisory_size_recommendation(
        payload,
        confidence_score=71.0,
        global_risk_size_cap=1.0,
        at_flip_size_cap=1.0,
        macro_size_multiplier=1.0,
        freshness_size_cap=1.0,
        reversal_stage="CONFIRMED_REVERSAL",
        expansion_mode=False,
        expansion_direction=None,
        direction="CALL",
        runtime_thresholds={},
        regime_thresholds={"position_size_multiplier": 1.0},
        gamma_regime="POSITIVE_GAMMA",
    )

    assert sizing["portfolio_priority_bucket"] in {"MEDIUM_PRIORITY", "LOW_PRIORITY"}
    assert sizing["portfolio_allocation_tier"] in {"TACTICAL", "SMALL"}
    assert sizing["effective_size_cap"] <= 0.55
    assert sizing["advisory_lots"] < 4


def test_advisory_sizing_promotes_high_priority_when_book_heat_is_cool():
    payload = {
        "number_of_lots": 4,
        "trade_strength": 88,
        "signal_success_probability": 0.74,
        "portfolio_book_heat_score": 18,
        "portfolio_book_heat_label": "COOL",
    }

    sizing = signal_engine._derive_advisory_size_recommendation(
        payload,
        confidence_score=78.0,
        global_risk_size_cap=1.0,
        at_flip_size_cap=1.0,
        macro_size_multiplier=1.0,
        freshness_size_cap=1.0,
        reversal_stage="CONFIRMED_REVERSAL",
        expansion_mode=False,
        expansion_direction=None,
        direction="CALL",
        runtime_thresholds={},
        regime_thresholds={"position_size_multiplier": 1.0},
        gamma_regime="NEGATIVE_GAMMA",
    )

    assert sizing["portfolio_priority_bucket"] == "HIGH_PRIORITY"
    assert sizing["portfolio_allocation_tier"] == "CORE"
    assert sizing["portfolio_capital_fraction_max"] >= 0.2


def test_advisory_sizing_prefers_lower_premium_contracts():
    common = {
        "number_of_lots": 4,
        "trade_strength": 84,
        "signal_success_probability": 0.68,
        "portfolio_book_heat_score": 20,
        "portfolio_book_heat_label": "COOL",
    }
    cheap_payload = {
        **common,
        "spot": 22000.0,
        "entry_price": 78.0,
        "premium_efficiency_score": 78,
    }
    expensive_payload = {
        **common,
        "spot": 22000.0,
        "entry_price": 285.0,
        "premium_efficiency_score": 46,
    }

    cheap = signal_engine._derive_advisory_size_recommendation(
        cheap_payload,
        confidence_score=76.0,
        global_risk_size_cap=1.0,
        at_flip_size_cap=1.0,
        macro_size_multiplier=1.0,
        freshness_size_cap=1.0,
        reversal_stage="CONFIRMED_REVERSAL",
        expansion_mode=False,
        expansion_direction=None,
        direction="CALL",
        runtime_thresholds={},
        regime_thresholds={"position_size_multiplier": 1.0},
        gamma_regime="NEGATIVE_GAMMA",
    )
    expensive = signal_engine._derive_advisory_size_recommendation(
        expensive_payload,
        confidence_score=76.0,
        global_risk_size_cap=1.0,
        at_flip_size_cap=1.0,
        macro_size_multiplier=1.0,
        freshness_size_cap=1.0,
        reversal_stage="CONFIRMED_REVERSAL",
        expansion_mode=False,
        expansion_direction=None,
        direction="CALL",
        runtime_thresholds={},
        regime_thresholds={"position_size_multiplier": 1.0},
        gamma_regime="NEGATIVE_GAMMA",
    )

    assert cheap["portfolio_priority_score"] > expensive["portfolio_priority_score"]
    assert cheap["effective_size_cap"] >= expensive["effective_size_cap"]


def test_probability_overlay_deflates_fragile_risk_off_context():
    overlay = signal_engine._build_signal_probability_overlay(
        direction="CALL",
        adjusted_trade_strength=88,
        call_probability=0.82,
        put_probability=0.18,
        market_state={
            "iv_hv_regime": "IV_RICH",
            "global_risk_state": "RISK_OFF",
            "provider_health_summary": "WEAK",
            "data_quality_status": "CAUTION",
        },
        runtime_thresholds={},
    )

    assert overlay["directional_signal_probability"] == 0.82
    assert overlay["signal_success_probability"] < 0.75


def test_probability_overlay_shrinks_when_live_calibration_and_proxy_health_are_bad():
    baseline = signal_engine._build_signal_probability_overlay(
        direction="CALL",
        adjusted_trade_strength=88,
        call_probability=0.82,
        put_probability=0.18,
        market_state={
            "iv_hv_regime": "IV_FAIR",
            "global_risk_state": "MACRO_NEUTRAL",
            "provider_health_summary": "GOOD",
            "data_quality_status": "GOOD",
            "option_efficiency_status": "AVAILABLE",
            "ranked_strike_proxy_ratio": 0.0,
            "live_calibration_gate": {"verdict": "GO"},
            "live_directional_gate": {"verdict": "GO"},
        },
        runtime_thresholds={},
    )
    stressed = signal_engine._build_signal_probability_overlay(
        direction="CALL",
        adjusted_trade_strength=88,
        call_probability=0.82,
        put_probability=0.18,
        market_state={
            "iv_hv_regime": "IV_FAIR",
            "global_risk_state": "MACRO_NEUTRAL",
            "provider_health_summary": "GOOD",
            "data_quality_status": "GOOD",
            "option_efficiency_status": "UNAVAILABLE_NEUTRALIZED",
            "ranked_strike_proxy_ratio": 1.0,
            "live_calibration_gate": {
                "verdict": "BLOCK",
                "ece": 0.2868,
                "top_decile_overconfidence": 0.2463,
            },
            "live_directional_gate": {
                "verdict": "BLOCK",
                "stickiness_1step": 0.9746,
                "direction_imbalance": 0.3389,
                "flip_lag_penalty": 0.4943,
            },
        },
        runtime_thresholds={},
    )

    assert stressed["signal_success_probability"] < baseline["signal_success_probability"]
    assert stressed["probability_neutral_shrink"] > baseline["probability_neutral_shrink"]
    assert "live_calibration_block" in stressed["probability_shrink_reasons"]
    assert "proxy_heavy_structure" in stressed["probability_shrink_reasons"]


def test_portfolio_concentration_context_ignores_watchlist_only_history(monkeypatch):
    history = pd.DataFrame(
        {
            "symbol": ["NIFTY"] * 6,
            "direction": ["CALL"] * 6,
            "trade_status": ["WATCHLIST"] * 6,
            "tradeability_score": [68, 70, 69, 71, 67, 72],
        }
    )

    monkeypatch.setattr(signal_engine, "_load_recent_outcome_history_frame", lambda: (history, "test"))

    context = signal_engine._compute_portfolio_concentration_context(
        payload={"symbol": "NIFTY", "direction": "CALL"},
        runtime_thresholds={
            "enable_portfolio_concentration_guard": 1,
            "portfolio_concentration_lookback_signals": 6,
        },
    )

    assert context["recent_signal_count"] == 0
    assert context["same_direction_count"] == 0
    assert context["reason"] == "no_recent_executed_signals"


def test_portfolio_concentration_context_ignores_stale_prior_day_trades(monkeypatch):
    history = pd.DataFrame(
        {
            "symbol": ["NIFTY"] * 8,
            "direction": ["CALL"] * 8,
            "trade_status": ["TRADE", "TRADE", "WATCHLIST", "WATCHLIST", "WATCHLIST", "WATCHLIST", "WATCHLIST", "WATCHLIST"],
            "signal_timestamp": [
                "2026-04-16T10:00:00+05:30",
                "2026-04-16T10:05:00+05:30",
                "2026-04-17T09:20:00+05:30",
                "2026-04-17T09:25:00+05:30",
                "2026-04-17T09:30:00+05:30",
                "2026-04-17T09:35:00+05:30",
                "2026-04-17T09:40:00+05:30",
                "2026-04-17T09:45:00+05:30",
            ],
        }
    )

    monkeypatch.setattr(signal_engine, "_load_recent_outcome_history_frame", lambda: (history, "test"))

    context = signal_engine._compute_portfolio_concentration_context(
        payload={
            "symbol": "NIFTY",
            "direction": "CALL",
            "valuation_time": "2026-04-17T10:00:00+05:30",
        },
        runtime_thresholds={
            "enable_portfolio_concentration_guard": 1,
            "portfolio_concentration_lookback_signals": 6,
        },
    )

    assert context["recent_signal_count"] == 0
    assert context["same_direction_count"] == 0
    assert context["reason"] == "no_recent_executed_signals"


def test_trade_promotion_governor_blocks_risk_off_weak_provider_context():
    verdict = signal_engine._evaluate_trade_promotion_governor(
        payload={
            "trade_strength": 82,
            "runtime_composite_score": 76,
            "signal_success_probability": 0.64,
            "confirmation_status": "CONFIRMED",
            "data_quality_status": "GOOD",
            "provider_health_summary": "WEAK",
            "global_risk_state": "RISK_OFF",
            "live_calibration_gate": {"verdict": "PASS"},
            "live_directional_gate": {"verdict": "PASS"},
        },
        runtime_thresholds={},
    )

    assert verdict["verdict"] == "BLOCK"
    assert any("fragile" in reason or "risk_off" in reason for reason in verdict.get("reasons", []))


def test_intraday_exit_bias_caps_fragile_context_holds():
    rec, hold_max, reasons = signal_engine._apply_intraday_exit_bias(
        recommended_hold_minutes=110,
        max_hold_minutes=160,
        runtime_thresholds={},
        expansion_mode=False,
        expansion_direction=None,
        direction="CALL",
        reversal_stage="NONE",
        provider_health_summary="WEAK",
        data_quality_status="CAUTION",
        global_risk_state="RISK_OFF",
        gamma_regime="POSITIVE_GAMMA",
    )

    assert rec <= 45
    assert hold_max <= 45
    assert any("exit_bias" in reason or "fragile" in reason for reason in reasons)


def test_trade_promotion_governor_blocks_early_reversal_candidate_until_confirmed():
    verdict = signal_engine._evaluate_trade_promotion_governor(
        payload={
            "trade_strength": 88,
            "runtime_composite_score": 80,
            "signal_success_probability": 0.72,
            "confirmation_status": "CONFIRMED",
            "data_quality_status": "GOOD",
            "provider_health_summary": "GOOD",
            "global_risk_state": "RISK_OFF",
            "reversal_stage": "EARLY_REVERSAL_CANDIDATE",
            "fast_reversal_alert_level": "HIGH",
            "live_calibration_gate": {"verdict": "PASS"},
            "live_directional_gate": {"verdict": "PASS"},
        },
        runtime_thresholds={},
    )

    assert verdict["verdict"] == "BLOCK"
    assert any("reversal" in reason for reason in verdict.get("reasons", []))
