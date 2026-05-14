from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import app.streamlit_app as streamlit_app


def test_load_signal_drift_dashboard_state_builds_alert_from_trend(tmp_path: Path):
    trend_dashboard = {
        "report_type": "signal_drift_trend_dashboard",
        "generated_at": "2026-04-08T00:00:00+00:00",
        "run_count": 1,
        "lookback_runs": 1,
        "trend_assessment": "WATCH",
        "status_counts": {"WATCH": 1},
        "lookback_summary": {"avg_hit_rate_delta": -0.03},
        "latest": {
            "generated_at": "2026-04-08T00:00:00+00:00",
            "report_name": "unit_signal_drift",
            "monitor_status": "WATCH",
            "warning_count": 1,
            "recent_hit_rate_60m": 0.55,
            "hit_rate_delta": -0.03,
            "avg_return_delta_bps": -5.0,
        },
    }
    (tmp_path / "latest_signal_drift_trend.json").write_text(json.dumps(trend_dashboard), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "generated_at": "2026-04-08T00:00:00+00:00",
                "report_name": "unit_signal_drift",
                "monitor_status": "WATCH",
                "hit_rate_delta": -0.03,
                "avg_return_delta_bps": -5.0,
                "warning_count": 1,
            }
        ]
    ).to_csv(tmp_path / "signal_drift_trend_history.csv", index=False)

    state = streamlit_app._load_signal_drift_dashboard_state(tmp_path)

    assert state["alert_summary"]["ops_status"] == "WATCH"
    assert state["alert_summary"]["latest_run"]["report_name"] == "unit_signal_drift"
    assert state["trend_history"].shape[0] == 1
    assert state["paths"]["alert_summary"] == tmp_path / "latest_signal_drift_alert.json"


def test_prepare_drift_trend_chart_frame_sorts_and_keeps_numeric_columns():
    history = pd.DataFrame(
        [
            {
                "generated_at": "2026-04-09T00:00:00+00:00",
                "hit_rate_delta": "-0.10",
                "avg_return_delta_bps": "-15.0",
                "warning_count": "3",
            },
            {
                "generated_at": "2026-04-08T00:00:00+00:00",
                "hit_rate_delta": "0.02",
                "avg_return_delta_bps": "4.5",
                "warning_count": "0",
            },
        ]
    )

    chart = streamlit_app._prepare_drift_trend_chart_frame(history)

    assert chart.index.is_monotonic_increasing
    assert chart["hit_rate_delta"].tolist() == [0.02, -0.10]
    assert chart["warning_count"].tolist() == [0, 3]
