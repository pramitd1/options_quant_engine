from __future__ import annotations

import pandas as pd

from scripts.daily_readiness_dashboard import detect_daily_readiness_anomalies, summarize_daily_readiness


def test_summarize_daily_readiness_computes_qualified_and_suppression_metrics():
    frame = pd.DataFrame(
        [
            {
                "signal_timestamp": "2026-05-09T10:00:00Z",
                "trade_status": "TRADE",
                "direction": "CALL",
                "runtime_composite_score": 72.0,
                "confidence_score": 78.0,
            },
            {
                "signal_timestamp": "2026-05-09T11:00:00Z",
                "trade_status": "WATCHLIST",
                "direction": "PUT",
                "runtime_composite_score": 58.0,
                "confidence_score": 44.0,
            },
            {
                "signal_timestamp": "2026-05-10T09:30:00Z",
                "trade_status": "TRADE",
                "direction": "PUT",
                "runtime_composite_score": 82.0,
                "confidence_score": 91.0,
            },
        ]
    )

    summary = summarize_daily_readiness(frame, thresholds={
        "daily_min_qualified_signals": 2,
        "daily_min_composite_score_75th_pctl": 65.0,
        "daily_min_suppression_rate_pct": 85.0,
        "daily_call_put_ratio_target": 0.50,
        "daily_min_average_confidence": 50.0,
    })

    assert len(summary) == 2
    assert summary.loc[summary["signal_date"] == "2026-05-09", "qualified_signals"].iloc[0] == 1
    assert summary.loc[summary["signal_date"] == "2026-05-09", "suppression_rate_pct"].iloc[0] == 50.0


def test_detect_daily_readiness_anomalies_flags_low_volume_and_low_confidence():
    summary = pd.DataFrame(
        [
            {
                "signal_date": "2026-05-09",
                "qualified_signals": 1,
                "runtime_composite_score_75th_pctile": 60.0,
                "suppression_rate_pct": 80.0,
                "directional_balance": 0.0,
                "call_put_ratio": 0.5,
                "average_confidence_score": 48.0,
            }
        ]
    )

    anomalies = detect_daily_readiness_anomalies(summary, thresholds={
        "daily_min_qualified_signals": 2,
        "daily_min_composite_score_75th_pctl": 65.0,
        "daily_min_suppression_rate_pct": 85.0,
        "daily_call_put_ratio_target": 0.50,
        "daily_min_average_confidence": 50.0,
    })

    assert len(anomalies) == 1
    assert "qualified_signal_volume_below_threshold" in anomalies[0]["issues"]
    assert "average_confidence_below_threshold" in anomalies[0]["issues"]
