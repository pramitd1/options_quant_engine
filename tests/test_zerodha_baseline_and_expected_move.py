from io import StringIO
from contextlib import redirect_stdout

import pandas as pd

from app.terminal_output import _format_expected_move_display, _render_market_summary_levels_table
from main import _select_zerodha_oi_baseline


def _frame(value: int) -> pd.DataFrame:
    return pd.DataFrame({
        "strikePrice": [22950],
        "OPTION_TYP": ["CE"],
        "openInterest": [value],
        "source": ["ZERODHA"],
    })


def test_select_zerodha_oi_baseline_prefers_five_minute_window() -> None:
    current_ts = pd.Timestamp("2026-04-01 09:46:09+05:30")
    history = [
        (pd.Timestamp("2026-04-01 09:39:00+05:30"), _frame(1000)),
        (pd.Timestamp("2026-04-01 09:41:00+05:30"), _frame(1100)),
        (pd.Timestamp("2026-04-01 09:44:00+05:30"), _frame(1200)),
    ]

    baseline, label = _select_zerodha_oi_baseline(history, current_ts)

    assert baseline is not None
    assert label == "5m rolling"
    assert float(baseline.iloc[0]["openInterest"]) == 1100.0


def test_select_zerodha_oi_baseline_falls_back_to_prior_snapshot() -> None:
    current_ts = pd.Timestamp("2026-04-01 09:17:00+05:30")
    history = [
        (pd.Timestamp("2026-04-01 09:15:00+05:30"), _frame(1000)),
        (pd.Timestamp("2026-04-01 09:16:50+05:30"), _frame(1050)),
    ]

    baseline, label = _select_zerodha_oi_baseline(history, current_ts)

    assert baseline is not None
    assert label == "prior snapshot"
    assert float(baseline.iloc[0]["openInterest"]) == 1050.0


def test_format_expected_move_display_separates_straddle_and_model_when_they_diverge() -> None:
    rendered = _format_expected_move_display(
        spot=22860.4,
        straddle_points=626.0,
        straddle_pct=2.74,
        expected_move_up=23486.4,
        expected_move_down=22234.4,
        model_pct=3.44,
    )

    assert "straddle +-" not in rendered
    assert "straddle ±626 pts  [22234 - 23486]  (2.74%)" in rendered
    assert "model ±786 pts" in rendered
    assert "(3.44%)" in rendered


def test_format_expected_move_display_keeps_single_view_when_close() -> None:
    rendered = _format_expected_move_display(
        spot=22860.4,
        straddle_points=626.0,
        straddle_pct=2.74,
        expected_move_up=23486.4,
        expected_move_down=22234.4,
        model_pct=2.90,
    )

    assert "model ±" not in rendered
    assert "straddle ±626 pts" in rendered


def test_render_market_summary_note_mentions_rolling_baseline_when_proxy_used() -> None:
    proxy_rows = [(22950.0, 1200.0, 200.0, "BUY_BUILDUP", True)]

    with StringIO() as buffer, redirect_stdout(buffer):
        _render_market_summary_levels_table(
            spot=22906.75,
            resistances=[],
            supports=[],
            call_oi=proxy_rows,
            put_oi=[],
        )
        output = buffer.getvalue()

    assert "5m rolling baseline when available" in output
    assert "+200*" in output
