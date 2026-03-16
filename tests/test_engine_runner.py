from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import app.engine_runner as engine_runner


class _HeadlineRecord:
    def __init__(self, title: str) -> None:
        self.title = title

    def to_dict(self) -> dict:
        return {"title": self.title}


class _HeadlineService:
    def fetch(self, *, symbol, as_of, replay_mode):
        return SimpleNamespace(
            provider_name="test_provider",
            fetched_at=as_of,
            latest_headline_at=as_of,
            is_stale=False,
            data_available=True,
            neutral_fallback=False,
            stale_after_minutes=15,
            issues=[],
            warnings=[],
            provider_metadata={"symbol": symbol, "replay_mode": replay_mode},
            records=[_HeadlineRecord(f"{symbol} headline")],
        )


def _option_chain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strikePrice": 22000,
                "OPTION_TYP": "CE",
                "lastPrice": 110.0,
                "openInterest": 1200,
                "changeinOI": 50,
                "impliedVolatility": 14.5,
                "EXPIRY_DT": "2026-03-26",
            }
        ]
    )


def _spot_snapshot() -> dict:
    return {
        "spot": 22050.0,
        "day_open": 22010.0,
        "day_high": 22100.0,
        "day_low": 21980.0,
        "prev_close": 22020.0,
        "timestamp": "2026-03-15T09:20:00+05:30",
        "lookback_avg_range_pct": 0.9,
        "validation": {"is_valid": True, "is_stale": False},
    }


def test_run_engine_snapshot_replay_requires_snapshot_files(monkeypatch):
    monkeypatch.setattr(
        engine_runner,
        "latest_replay_snapshot_paths",
        lambda symbol, replay_dir: (None, None),
    )

    result = engine_runner.run_engine_snapshot(
        symbol="NIFTY",
        mode="REPLAY",
        source="NSE",
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=50,
        max_capital=100000,
        headline_service=_HeadlineService(),
    )

    assert result["ok"] is False
    assert "Replay mode requires both a spot snapshot" in result["error"]


def test_run_engine_snapshot_uses_promotion_state_and_logs_shadow(monkeypatch):
    comparisons = []
    captures = []

    class _SignalCaptureSink:
        def apply(self, **kwargs):
            captures.append(kwargs["result_payload"])
            kwargs["result_payload"]["signal_capture_status"] = "CAPTURED"

    class _ShadowSink:
        def apply(self, **kwargs):
            comparisons.append(
                {
                    "baseline_pack_name": kwargs["baseline_pack_name"],
                    "shadow_pack_name": kwargs["shadow_pack_name"],
                }
            )
            kwargs["result_payload"]["shadow_mode_active"] = True
            kwargs["result_payload"]["shadow_pack_name"] = kwargs["shadow_pack_name"]
            kwargs["result_payload"]["shadow_log_status"] = "CAPTURED"

    monkeypatch.setattr(
        engine_runner,
        "get_promotion_runtime_context",
        lambda: {"live_pack": "prod_pack", "shadow_pack": "cand_pack"},
    )
    monkeypatch.setattr(engine_runner, "load_spot_snapshot", lambda path: _spot_snapshot())
    monkeypatch.setattr(engine_runner, "load_option_chain_snapshot", lambda path: _option_chain_frame())
    monkeypatch.setattr(engine_runner, "evaluate_scheduled_event_risk", lambda symbol, as_of: {"macro_event_risk_score": 0})
    monkeypatch.setattr(engine_runner, "build_global_market_snapshot", lambda symbol, as_of: {"vix": 12.0})
    monkeypatch.setattr(engine_runner, "resolve_selected_expiry", lambda option_chain: "2026-03-26")
    monkeypatch.setattr(engine_runner, "filter_option_chain_by_expiry", lambda option_chain, expiry: option_chain)
    monkeypatch.setattr(
        engine_runner,
        "validate_option_chain",
        lambda option_chain: {
            "is_valid": True,
            "is_stale": False,
            "selected_expiry": "2026-03-26",
            "provider_health": {"summary_status": "GOOD"},
        },
    )

    def fake_eval(*, parameter_pack_name, **kwargs):
        trade = {
            "signal_id": f"{parameter_pack_name}-signal",
            "ranked_strike_candidates": [{"strike": 22000, "score": 0.81}],
        }
        return {
            "parameter_pack_name": parameter_pack_name,
            "macro_news_state": {"macro_bias": "NEUTRAL"},
            "global_risk_state": {"global_risk_state": "GLOBAL_NEUTRAL"},
            "trade": trade,
        }

    monkeypatch.setattr(engine_runner, "_evaluate_snapshot_for_pack", fake_eval)

    result = engine_runner.run_engine_snapshot(
        symbol="NIFTY",
        mode="REPLAY",
        source="NSE",
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=50,
        max_capital=100000,
        replay_spot="spot.json",
        replay_chain="chain.json",
        use_promotion_state=True,
        headline_service=_HeadlineService(),
        signal_capture_sink=_SignalCaptureSink(),
        shadow_evaluation_sink=_ShadowSink(),
    )

    assert result["ok"] is True
    assert result["authoritative_parameter_pack"] == "prod_pack"
    assert result["shadow_mode_active"] is True
    assert result["shadow_pack_name"] == "cand_pack"
    assert result["signal_capture_status"] == "CAPTURED"
    assert result["shadow_log_status"] == "CAPTURED"
    assert result["trade"]["signal_id"] == "prod_pack-signal"
    assert result["execution_trade"]["signal_id"] == "prod_pack-signal"
    assert comparisons[0]["shadow_pack_name"] == "cand_pack"
    assert len(captures) == 1


def test_run_engine_snapshot_closes_managed_router(monkeypatch):
    class _Router:
        def __init__(self):
            self.closed = False

        def get_option_chain(self, symbol):
            return _option_chain_frame()

        def close(self):
            self.closed = True

    router = _Router()

    monkeypatch.setattr(engine_runner, "DataSourceRouter", lambda source: router)
    monkeypatch.setattr(engine_runner, "get_spot_snapshot", lambda symbol: _spot_snapshot())
    monkeypatch.setattr(engine_runner, "evaluate_scheduled_event_risk", lambda symbol, as_of: {"macro_event_risk_score": 0})
    monkeypatch.setattr(engine_runner, "build_global_market_snapshot", lambda symbol, as_of: {"vix": 13.0})
    monkeypatch.setattr(engine_runner, "resolve_selected_expiry", lambda option_chain: "2026-03-26")
    monkeypatch.setattr(engine_runner, "filter_option_chain_by_expiry", lambda option_chain, expiry: option_chain)
    monkeypatch.setattr(
        engine_runner,
        "validate_option_chain",
        lambda option_chain: {
            "is_valid": True,
            "is_stale": False,
            "selected_expiry": "2026-03-26",
            "provider_health": {"summary_status": "GOOD"},
        },
    )
    monkeypatch.setattr(
        engine_runner,
        "_evaluate_snapshot_for_pack",
        lambda *, parameter_pack_name, **kwargs: {
            "parameter_pack_name": parameter_pack_name or "default_pack",
            "macro_news_state": {"macro_bias": "NEUTRAL"},
            "global_risk_state": {"global_risk_state": "GLOBAL_NEUTRAL"},
            "trade": None,
        },
    )

    result = engine_runner.run_engine_snapshot(
        symbol="NIFTY",
        mode="LIVE",
        source="NSE",
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=50,
        max_capital=100000,
        capture_signal_evaluation=False,
        headline_service=_HeadlineService(),
    )

    assert result["ok"] is True
    assert router.closed is True
