import pandas as pd
from io import StringIO
from contextlib import redirect_stdout

from app.terminal_output import _format_oi_change_value, _render_market_summary_levels_table, _resolve_top_oi_levels


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

    call_by_strike = {int(strike): (chg_oi, inference, uses_proxy) for strike, _oi, chg_oi, inference, uses_proxy in call_rows}
    put_by_strike = {int(strike): (chg_oi, inference, uses_proxy) for strike, _oi, chg_oi, inference, uses_proxy in put_rows}

    assert call_by_strike[22950][0] == 200.0
    assert call_by_strike[22950][1] == "BUY_BUILDUP"
    assert call_by_strike[22950][2] is True
    assert put_by_strike[22950][0] == 200.0
    assert put_by_strike[22950][1] == "WRITE_BUILDUP"
    assert put_by_strike[22950][2] is True


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
    assert put_rows[0][2] == -60.0
    assert put_rows[0][3] == "LONG_UNWIND"
    assert put_rows[0][4] is False


def test_format_oi_change_value_marks_snapshot_proxy() -> None:
    assert _format_oi_change_value(200.0, uses_snapshot_proxy=True) == "+200*"
    assert _format_oi_change_value(-1500.0, uses_snapshot_proxy=True) == "-1.5K*"
    assert _format_oi_change_value(45.0, uses_snapshot_proxy=False) == "+45"


def test_render_market_summary_levels_table_shows_snapshot_proxy_note_only_when_used() -> None:
    proxy_rows = [(22950.0, 1200.0, 200.0, "BUY_BUILDUP", True)]
    native_rows = [(23000.0, 1500.0, 45.0, "BUY_BUILDUP", False)]

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
    assert "proxy based on OI change plus underlying move vs prev_close" in native_output
