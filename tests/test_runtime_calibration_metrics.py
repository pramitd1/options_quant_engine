from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_runtime_score_calibrator import (
    _compute_calibration_drift_metrics,
    _compute_calibration_objective_metrics,
    _fit_isotonic_calibrator,
)


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
