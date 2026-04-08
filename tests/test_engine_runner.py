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
        "resolve_replay_snapshot_paths",
        lambda symbol, replay_dir, source_label: {
            "spot_path": None,
            "chain_path": None,
            "selection_reason": "no_valid_chain_snapshot",
            "source_label": source_label,
            "skipped_chain_files": [],
        },
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
        lambda option_chain, **kwargs: {
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
    monkeypatch.setattr(engine_runner, "get_spot_snapshot", lambda symbol, **kwargs: _spot_snapshot())
    monkeypatch.setattr(engine_runner, "evaluate_scheduled_event_risk", lambda symbol, as_of: {"macro_event_risk_score": 0})
    monkeypatch.setattr(engine_runner, "build_global_market_snapshot", lambda symbol, as_of: {"vix": 13.0})
    monkeypatch.setattr(engine_runner, "resolve_selected_expiry", lambda option_chain: "2026-03-26")
    monkeypatch.setattr(engine_runner, "filter_option_chain_by_expiry", lambda option_chain, expiry: option_chain)
    monkeypatch.setattr(
        engine_runner,
        "validate_option_chain",
        lambda option_chain, **kwargs: {
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


def test_run_engine_snapshot_replay_uses_source_aware_selection_and_emits_diagnostics(monkeypatch):
    captured = {}

    def _fake_resolve(symbol, *, replay_dir, source_label):
        captured["symbol"] = symbol
        captured["replay_dir"] = replay_dir
        captured["source_label"] = source_label
        return {
            "spot_path": "spot.json",
            "chain_path": "chain.csv",
            "selection_reason": "latest_valid_for_source",
            "source_label": source_label,
            "skipped_chain_files": [{"path": "bad.csv", "reason": "empty_file"}],
        }

    monkeypatch.setattr(engine_runner, "resolve_replay_snapshot_paths", _fake_resolve)
    monkeypatch.setattr(engine_runner, "load_spot_snapshot", lambda path: _spot_snapshot())
    monkeypatch.setattr(engine_runner, "load_option_chain_snapshot", lambda path: _option_chain_frame())
    monkeypatch.setattr(
        engine_runner,
        "run_preloaded_engine_snapshot",
        lambda **kwargs: {
            "ok": True,
            "mode": kwargs["mode"],
            "source": kwargs["source"],
            "symbol": kwargs["symbol"],
            "replay_paths": kwargs.get("replay_paths"),
        },
    )

    result = engine_runner.run_engine_snapshot(
        symbol="NIFTY",
        mode="REPLAY",
        source="ICICI",
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=50,
        max_capital=100000,
        replay_dir="debug_samples",
        save_live_snapshots=False,
    )

    assert result["ok"] is True
    assert captured["source_label"] == "ICICI"
    assert result["replay_paths"]["spot"] == "spot.json"
    assert result["replay_paths"]["chain"] == "chain.csv"
    assert result["replay_selection"]["selection_reason"] == "latest_valid_for_source"
    assert result["replay_selection"]["skipped_chain_files"][0]["reason"] == "empty_file"


def test_run_preloaded_engine_snapshot_fail_closed_when_overlay_construction_fails(monkeypatch):
    monkeypatch.setenv("RUNTIME_FAIL_CLOSED_ON_OVERLAY_FAILURE", "1")

    def _failing_macro_news_state(**kwargs):
        raise RuntimeError("macro failure")

    monkeypatch.setattr(engine_runner, "build_macro_news_state", _failing_macro_news_state)
    monkeypatch.setattr(
        engine_runner,
        "_prepare_snapshot_context",
        lambda **kwargs: {
            "spot_snapshot": _spot_snapshot(),
            "spot_validation": {"is_valid": True, "is_stale": False},
            "spot": 22050.0,
            "day_open": 22010.0,
            "day_high": 22100.0,
            "day_low": 21980.0,
            "prev_close": 22020.0,
            "spot_timestamp": "2026-03-15T09:20:00+05:30",
            "lookback_avg_range_pct": 0.9,
            "macro_event_state": {"macro_event_risk_score": 0, "event_lockdown_flag": False},
            "headline_state": _HeadlineService().fetch(symbol="NIFTY", as_of="2026-03-15T09:20:00+05:30", replay_mode=False),
            "global_market_snapshot": {"data_available": True, "warnings": [], "issues": []},
            "option_chain_validation": {"is_valid": True, "is_stale": False, "selected_expiry": "2026-03-26"},
            "option_chain": _option_chain_frame(),
            "option_chain_frame": _option_chain_frame(),
        },
    )

    result = engine_runner.run_preloaded_engine_snapshot(
        symbol="NIFTY",
        mode="LIVE",
        source="NSE",
        spot_snapshot=_spot_snapshot(),
        option_chain=_option_chain_frame(),
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=50,
        max_capital=100000,
        capture_signal_evaluation=False,
        headline_service=_HeadlineService(),
    )

    assert result["ok"] is False
    assert result["trade"] is None
    assert result["global_risk_state"].get("overlay_fail_closed_blocked") is True
    assert "macro_news_state_construction_failed" in (result["global_risk_state"].get("overlay_failure_reasons") or [])


def test_run_engine_snapshot_returns_error_when_live_option_chain_is_empty(monkeypatch):
    class _Router:
        def __init__(self):
            self.closed = False

        def get_option_chain(self, symbol):
            return pd.DataFrame()

        def close(self):
            self.closed = True

    router = _Router()

    monkeypatch.setattr(engine_runner, "DataSourceRouter", lambda source: router)
    monkeypatch.setattr(engine_runner, "get_spot_snapshot", lambda symbol, **kwargs: _spot_snapshot())

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

    assert result["ok"] is False
    assert result["reason"] == "DATA_UNAVAILABLE_OPTION_CHAIN"
    assert "Option chain empty/invalid" in result["error"]
    assert router.closed is True


def test_run_engine_snapshot_continues_when_spot_history_append_raises_value_error(monkeypatch):
    class _Router:
        def __init__(self):
            self.closed = False

        def get_option_chain(self, symbol):
            return _option_chain_frame()

        def close(self):
            self.closed = True

    router = _Router()

    monkeypatch.setattr(engine_runner, "DataSourceRouter", lambda source: router)
    monkeypatch.setattr(engine_runner, "get_spot_snapshot", lambda symbol, **kwargs: _spot_snapshot())
    monkeypatch.setattr(engine_runner, "append_spot_observation", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad spot history")))
    monkeypatch.setattr(engine_runner, "evaluate_scheduled_event_risk", lambda symbol, as_of: {"macro_event_risk_score": 0})
    monkeypatch.setattr(engine_runner, "build_global_market_snapshot", lambda symbol, as_of: {"vix": 13.0})
    monkeypatch.setattr(engine_runner, "resolve_selected_expiry", lambda option_chain: "2026-03-26")
    monkeypatch.setattr(engine_runner, "filter_option_chain_by_expiry", lambda option_chain, expiry: option_chain)
    monkeypatch.setattr(
        engine_runner,
        "validate_option_chain",
        lambda option_chain, **kwargs: {
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
