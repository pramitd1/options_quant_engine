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
