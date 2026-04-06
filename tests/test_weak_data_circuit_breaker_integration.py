from __future__ import annotations

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
