from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_runtime_score_calibrator import (
    _compute_calibration_drift_metrics,
    _compute_calibration_objective_metrics,
    _fit_isotonic_calibrator,
)
from main import _compute_calibration_gate_verdict, _compute_stickiness_gate_verdict


def test_calibration_objective_metrics_are_reported():
    raw = np.linspace(10, 90, 400)
    targets = (raw > 50).astype(float)
    calibrator, _ = _fit_isotonic_calibrator(raw.tolist(), targets.tolist())

    metrics = _compute_calibration_objective_metrics(raw.tolist(), targets.tolist(), calibrator)

    assert metrics["objective_score"] is not None
    assert metrics["brier_score"] is not None
    assert metrics["ece"] is not None
    assert metrics["top_decile_overconfidence"] is not None
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_calibration_drift_metrics_detect_shift_between_prior_and_recent():
    raw = np.linspace(5, 95, 500)
    targets = np.where(raw < 60, 0.7, 0.3).astype(float)
    targets[-100:] = np.where(raw[-100:] < 60, 0.3, 0.7)

    df = pd.DataFrame(
        {
            "signal_timestamp": pd.date_range("2026-01-01", periods=len(raw), freq="5min"),
            "composite_signal_score": raw,
            "correct_60m": targets,
        }
    )
    calibrator, _ = _fit_isotonic_calibrator(raw.tolist(), targets.tolist())

    drift = _compute_calibration_drift_metrics(df, raw.tolist(), targets.tolist(), calibrator)

    assert drift["drift_samples_recent"] > 0
    assert drift["drift_samples_prior"] > 0
    assert drift["calibration_gap_abs_delta"] is not None
    assert drift["brier_delta"] is not None


def test_live_calibration_gate_downgrades_stale_completed_trade_history(tmp_path):
    old_trade_times = pd.date_range("2026-04-01 09:15:00+05:30", periods=120, freq="5min")
    recent_watch_times = pd.date_range("2026-04-17 09:15:00+05:30", periods=20, freq="5min")

    old_trades = pd.DataFrame(
        {
            "signal_timestamp": old_trade_times,
            "trade_status": ["TRADE"] * len(old_trade_times),
            "correct_60m": [0] * 90 + [1] * 30,
            "hybrid_move_probability": [0.85] * len(old_trade_times),
            "outcome_status": ["COMPLETE"] * len(old_trade_times),
        }
    )
    recent_watch = pd.DataFrame(
        {
            "signal_timestamp": recent_watch_times,
            "trade_status": ["WATCHLIST"] * len(recent_watch_times),
            "correct_60m": [None] * len(recent_watch_times),
            "hybrid_move_probability": [0.52] * len(recent_watch_times),
            "outcome_status": [""] * len(recent_watch_times),
        }
    )
    dataset = pd.concat([old_trades, recent_watch], ignore_index=True)
    path = tmp_path / "signals_dataset_cumul.csv"
    dataset.to_csv(path, index=False)

    result = _compute_calibration_gate_verdict(
        dataset_path=path,
        cache_state={},
        lookback_trades=100,
        max_ece=0.18,
        max_brier=0.24,
        max_top_decile_overconfidence=0.20,
        min_completed_trades=80,
    )

    assert result["verdict"] == "CAUTION"
    assert result["reason"] == "stale_completed_trade_history"


def test_live_directional_gate_ignores_repeated_same_direction_snapshot_spam(tmp_path):
    rows = []
    base = pd.Timestamp("2026-04-17 09:15:00+05:30")
    for i in range(60):
        rows.append({
            "signal_timestamp": base + pd.Timedelta(seconds=15 * i),
            "direction": "CALL",
            "correct_5m": 1,
            "correct_15m": 1,
            "correct_30m": 1,
        })
    switch_base = base + pd.Timedelta(minutes=25)
    for i in range(6):
        rows.append({
            "signal_timestamp": switch_base + pd.Timedelta(seconds=15 * i),
            "direction": "PUT",
            "correct_5m": 1,
            "correct_15m": 1,
            "correct_30m": 1,
        })

    path = tmp_path / "signals_dataset_cumul.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = _compute_stickiness_gate_verdict(
        dataset_path=path,
        cache_state={},
        max_stickiness=0.90,
        max_imbalance=0.20,
        max_flip_lag_penalty=0.35,
    )

    assert result["verdict"] != "BLOCK"
    assert result["red_alerts"] <= 1
