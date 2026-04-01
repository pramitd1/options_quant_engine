import pandas as pd
from io import StringIO
from contextlib import redirect_stdout
import json
from pathlib import Path

from app.terminal_output import (
    _format_oi_change_value,
    _render_market_summary_levels_table,
    _resolve_top_oi_levels,
    _persist_oi_inference_artifact,
    render_snapshot,
)


def _current_zerodha_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strikePrice": [22950, 23000, 22950, 23000],
            "OPTION_TYP": ["CE", "CE", "PE", "PE"],
            "openInterest": [1200, 900, 1500, 1100],
            "changeinOI": [0, 0, 0, 0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07", "2026-04-07", "2026-04-07"],
            "source": ["ZERODHA", "ZERODHA", "ZERODHA", "ZERODHA"],
        }
    )


def _previous_zerodha_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strikePrice": [22950, 23000, 22950, 23000],
            "OPTION_TYP": ["CE", "CE", "PE", "PE"],
            "openInterest": [1000, 950, 1300, 1200],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07", "2026-04-07", "2026-04-07"],
            "source": ["ZERODHA", "ZERODHA", "ZERODHA", "ZERODHA"],
        }
    )


def test_resolve_top_oi_levels_derives_snapshot_oi_change_for_zerodha() -> None:
    trade = {
        "selected_expiry": "2026-04-07",
        "spot": 22906.75,
        "prev_close": 22331.4,
        "previous_chain_frame": _previous_zerodha_chain(),
    }

    call_rows, put_rows = _resolve_top_oi_levels(trade, _current_zerodha_chain(), top_n=2)

    call_by_strike = {
        int(strike): (chg_oi, inference, uses_proxy, confidence, reason_code, horizon_signature)
        for strike, _oi, chg_oi, inference, uses_proxy, confidence, reason_code, horizon_signature, _debug in call_rows
    }
    put_by_strike = {
        int(strike): (chg_oi, inference, uses_proxy, confidence, reason_code, horizon_signature)
        for strike, _oi, chg_oi, inference, uses_proxy, confidence, reason_code, horizon_signature, _debug in put_rows
    }

    assert call_by_strike[22950][0] == 200.0
    assert call_by_strike[22950][1] == "BUY_BUILDUP"
    assert call_by_strike[22950][2] is True
    assert put_by_strike[22950][0] == 200.0
    assert put_by_strike[22950][1] == "WRITE_BUILDUP"
    assert put_by_strike[22950][2] is True
    assert 0.0 <= call_by_strike[22950][3] <= 1.0
    assert 0.0 <= put_by_strike[22950][3] <= 1.0
    assert call_by_strike[22950][4] == "PROXY_ONLY"
    assert put_by_strike[22950][4] == "PROXY_ONLY"
    assert call_by_strike[22950][5].startswith("1m:")
    assert put_by_strike[22950][5].startswith("1m:")


def test_resolve_top_oi_levels_keeps_non_zerodha_native_change() -> None:
    current_chain = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1200, 1500],
            "changeinOI": [45, -60],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["NSE", "NSE"],
        }
    )
    trade = {
        "selected_expiry": "2026-04-07",
        "spot": 22906.75,
        "prev_close": 22331.4,
        "previous_chain_frame": _previous_zerodha_chain(),
    }

    call_rows, put_rows = _resolve_top_oi_levels(trade, current_chain, top_n=1)

    assert call_rows[0][2] == 45.0
    assert call_rows[0][3] == "BUY_BUILDUP"
    assert call_rows[0][4] is False
    assert 0.0 <= call_rows[0][5] <= 1.0
    assert call_rows[0][6] == "PROXY_ONLY"
    assert put_rows[0][2] == -60.0
    assert put_rows[0][3] == "LONG_UNWIND"
    assert put_rows[0][4] is False
    assert 0.0 <= put_rows[0][5] <= 1.0
    assert put_rows[0][6] == "PROXY_ONLY"


def test_format_oi_change_value_marks_snapshot_proxy() -> None:
    assert _format_oi_change_value(200.0, uses_snapshot_proxy=True) == "+200*"
    assert _format_oi_change_value(-1500.0, uses_snapshot_proxy=True) == "-1.5K*"
    assert _format_oi_change_value(45.0, uses_snapshot_proxy=False) == "+45"


def test_render_market_summary_levels_table_shows_snapshot_proxy_note_only_when_used() -> None:
    proxy_rows = [(22950.0, 1200.0, 200.0, "BUY_BUILDUP", True, 0.77, "PROXY_ONLY", "1m:0 3m:0 5m:0", {})]
    native_rows = [(23000.0, 1500.0, 45.0, "BUY_BUILDUP", False, 0.62, "PROXY_ONLY", "1m:0 3m:0 5m:0", {})]

    with StringIO() as buffer, redirect_stdout(buffer):
        _render_market_summary_levels_table(
            spot=22906.75,
            resistances=[],
            supports=[],
            call_oi=proxy_rows,
            put_oi=[],
        )
        proxy_output = buffer.getvalue()

    assert "+200*" in proxy_output
    assert "Zerodha snapshot OI delta proxy" in proxy_output

    with StringIO() as buffer, redirect_stdout(buffer):
        _render_market_summary_levels_table(
            spot=22906.75,
            resistances=[],
            supports=[],
            call_oi=native_rows,
            put_oi=[],
        )
        native_output = buffer.getvalue()

    assert "+45" in native_output
    assert "+45*" not in native_output
    assert "Zerodha snapshot OI delta proxy" not in native_output
    assert "inference confidence decomposes 1m/3m/5m premium baselines" in native_output
    assert "PROXY_ONLY" in native_output
    assert "1m:0 3m:0 5m:0" in native_output


def test_resolve_top_oi_levels_prefers_premium_delta_over_underlying_proxy() -> None:
    current_chain = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1200, 1500],
            "changeinOI": [80, -60],
            "lastPrice": [90.0, 65.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    previous_chain = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1100, 1550],
            "lastPrice": [110.0, 55.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    trade = {
        "selected_expiry": "2026-04-07",
        # Underlying is up, so old proxy logic would map CE +OI to BUY_BUILDUP.
        # Premium delta is down for CE and up for PE, which should dominate now.
        "spot": 22950.0,
        "prev_close": 22300.0,
        "previous_chain_frame": previous_chain,
    }

    call_rows, put_rows = _resolve_top_oi_levels(trade, current_chain, top_n=1)

    assert call_rows[0][3] == "WRITE_BUILDUP"
    assert put_rows[0][3] == "SHORT_COVERING"
    assert call_rows[0][6] == "PREMIUM_WEAK_CONFLICT"
    assert put_rows[0][6] == "PREMIUM_WEAK_CONFLICT"


def test_resolve_top_oi_levels_prefers_rolling_baseline_over_previous_snapshot() -> None:
    current_chain = pd.DataFrame(
        {
            "strikePrice": [22950],
            "OPTION_TYP": ["CE"],
            "openInterest": [1200],
            "changeinOI": [80],
            "lastPrice": [90.0],
            "EXPIRY_DT": ["2026-04-07"],
            "source": ["ICICI"],
        }
    )
    # previous_chain_frame implies premium went UP (would map to BUY_BUILDUP for +OI)
    previous_chain = pd.DataFrame(
        {
            "strikePrice": [22950],
            "OPTION_TYP": ["CE"],
            "openInterest": [1100],
            "lastPrice": [80.0],
            "EXPIRY_DT": ["2026-04-07"],
            "source": ["ICICI"],
        }
    )
    # premium_baseline_chain_frame implies premium went DOWN (should dominate)
    premium_baseline = pd.DataFrame(
        {
            "strikePrice": [22950],
            "OPTION_TYP": ["CE"],
            "openInterest": [1100],
            "lastPrice": [110.0],
            "EXPIRY_DT": ["2026-04-07"],
            "source": ["ICICI"],
        }
    )
    trade = {
        "selected_expiry": "2026-04-07",
        "spot": 22950.0,
        "prev_close": 22300.0,
        "previous_chain_frame": previous_chain,
        "premium_baseline_chain_frame": premium_baseline,
    }

    call_rows, _ = _resolve_top_oi_levels(trade, current_chain, top_n=1)
    assert call_rows[0][3] == "WRITE_BUILDUP"
    assert call_rows[0][6] == "PREMIUM_WEAK_CONFLICT"


def test_resolve_top_oi_levels_uses_multi_horizon_reason_codes() -> None:
    current_chain = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1400, 1550],
            "changeinOI": [120, -90],
            "lastPrice": [120.0, 70.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    baseline_1m = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1300, 1600],
            "lastPrice": [90.0, 88.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    baseline_3m = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1250, 1620],
            "lastPrice": [75.0, 98.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    baseline_5m = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1225, 1650],
            "lastPrice": [60.0, 110.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    trade = {
        "selected_expiry": "2026-04-07",
        "spot": 22980.0,
        "prev_close": 22800.0,
        "premium_baseline_chain_frames": {
            "1m": baseline_1m,
            "3m": baseline_3m,
            "5m": baseline_5m,
        },
    }

    call_rows, put_rows = _resolve_top_oi_levels(trade, current_chain, top_n=1)

    assert call_rows[0][3] == "BUY_BUILDUP"
    assert call_rows[0][6] == "PREMIUM_STRONG_AGREE"
    assert put_rows[0][3] == "LONG_UNWIND"
    assert put_rows[0][6] == "PREMIUM_STRONG_AGREE"


def test_persist_oi_inference_artifact_writes_jsonl_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    call_rows = [
        (
            22950.0,
            1200.0,
            80.0,
            "BUY_BUILDUP",
            False,
            0.78,
            "PREMIUM_STRONG_AGREE",
            "1m:+ 3m:+ 5m:+",
            {
                "premium_change_1m": 8.0,
                "premium_change_3m": 18.0,
                "premium_change_5m": 26.0,
            },
        )
    ]
    put_rows = []

    _persist_oi_inference_artifact(
        result={"symbol": "NIFTY", "source": "ICICI", "mode": "LIVE"},
        trade={"direction": "BUY", "trade_status": "TRADE"},
        spot_summary={"timestamp": "2026-04-01T10:00:00+05:30", "spot": 22950.0, "prev_close": 22800.0},
        call_oi=call_rows,
        put_oi=put_rows,
        signal_capture_policy="ALL_SIGNALS",
        capture_enabled=True,
    )

    artifact_dir = Path("research/artifacts/oi_inference/2026-04-01")
    matches = list(artifact_dir.glob("multi_horizon_inference_pid*.jsonl"))
    assert matches
    artifact_file = matches[0]
    assert artifact_dir.exists()
    assert artifact_file.exists()
    lines = artifact_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == 2
    assert payload["symbol"] == "NIFTY"
    assert payload["rows"][0]["reason_code"] == "PREMIUM_STRONG_AGREE"
    assert payload["rows"][0]["horizon_signature"] == "1m:+ 3m:+ 5m:+"


def test_render_snapshot_captures_oi_artifacts_in_standard_mode(monkeypatch) -> None:
    called = {"count": 0}

    def _fake_persist(**kwargs):
        called["count"] += 1

    monkeypatch.setattr("app.terminal_output._persist_oi_inference_artifact", _fake_persist)

    option_chain = pd.DataFrame(
        {
            "strikePrice": [22950, 23000],
            "OPTION_TYP": ["CE", "PE"],
            "openInterest": [1200, 1500],
            "changeinOI": [80, -60],
            "lastPrice": [90.0, 65.0],
            "EXPIRY_DT": ["2026-04-07", "2026-04-07"],
            "source": ["ICICI", "ICICI"],
        }
    )
    trade = {
        "symbol": "NIFTY",
        "selected_expiry": "2026-04-07",
        "spot": 22950.0,
        "prev_close": 22300.0,
        "direction": "BUY",
    }
    result = {
        "symbol": "NIFTY",
        "source": "ICICI",
        "mode": "LIVE",
        "option_chain_frame": option_chain,
        "option_chain_rows": len(option_chain),
    }

    render_snapshot(
        "STANDARD",
        result=result,
        spot_summary={"spot": 22950.0, "prev_close": 22300.0, "timestamp": "2026-04-01T10:00:00+05:30"},
        spot_validation={"is_valid": True},
        option_chain_validation={"is_valid": True},
        macro_event_state={},
        macro_news_state={},
        global_risk_state={},
        global_market_snapshot={},
        headline_state={},
        trade=trade,
        execution_trade=trade,
        signal_capture_policy="ALL_SIGNALS",
        capture_oi_inference_artifacts=True,
    )

    assert called["count"] == 1
