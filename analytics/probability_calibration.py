"""
Module: probability_calibration.py

Purpose:
    Apply Platt scaling to convert raw engine signal strength scores into
    well-calibrated probability estimates.

Role in the System:
    Part of the analytics layer (research-facing).  The live engine currently
    uses heuristic probability estimates; this module provides a data-driven
    calibration layer that can be trained offline and served at runtime.

Key Concepts:
    Platt scaling fits a logistic regression model on top of a classifier's raw
    scores:  P(Y=1 | score) = sigmoid(A * score + B), where A and B are fit by
    MLE on held-out calibration data.  This corrects over-confident or
    under-confident raw scores.

Downstream Usage:
    1. Offline training: call ``fit_platt_calibration`` with a dataset of
       (signal_strength, outcome) pairs → saves calibrator to JSON.
    2. Runtime inference: call ``calibrate_probability`` with a raw score and
       the loaded calibrator → returns a calibrated probability.
    3. Research diagnostics: call ``calibration_report`` to produce a summary
       of reliability (bucket-level expected vs actual win rates).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Default path for the trained calibrator JSON
_DEFAULT_CALIBRATOR_PATH = (
    Path(__file__).resolve().parent.parent / "models_store" / "platt_calibrator.json"
)


# ---------------------------------------------------------------------------
# Sigmoid / logistic helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _sigmoid_arr(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(float), -500, 500)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_platt_calibration(
    scores: list[float],
    outcomes: list[int],
    *,
    n_iterations: int = 2000,
    lr: float = 0.01,
    save_path: Path | None = None,
) -> dict:
    """Fit Platt scaling parameters A and B using gradient descent on log-loss.

    Parameters
    ----------
    scores:
        Raw signal strength scores (e.g. 0–100 from the engine).
    outcomes:
        Binary outcomes: 1 = direction correct, 0 = direction wrong.
    n_iterations:
        Gradient descent steps.
    lr:
        Learning rate.
    save_path:
        If provided, serialises the fitted calibrator to this JSON path.

    Returns
    -------
    dict with keys: A, B, log_loss_before, log_loss_after, n_samples
    """
    s = np.array(scores, dtype=float)
    y = np.array(outcomes, dtype=float)
    n = len(s)

    if n < 10:
        raise ValueError(f"Need at least 10 samples to fit Platt calibration; got {n}")

    # Normalise scores to ~[0, 1] for stable optimisation
    s_norm = s / 100.0

    # Initial parameters
    A = float(np.std(s_norm) * 4.0 + 1e-6)
    B = 0.0

    def _log_loss(A_, B_):
        p = _sigmoid_arr(A_ * s_norm + B_)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    ll_before = _log_loss(A, B)

    for _ in range(n_iterations):
        p = _sigmoid_arr(A * s_norm + B)
        err = p - y
        grad_A = float(np.mean(err * s_norm))
        grad_B = float(np.mean(err))
        A -= lr * grad_A
        B -= lr * grad_B

    ll_after = _log_loss(A, B)

    calibrator = {
        "A": round(float(A), 6),
        "B": round(float(B), 6),
        "score_scale": 100.0,
        "n_samples": n,
        "log_loss_before": round(ll_before, 6),
        "log_loss_after": round(ll_after, 6),
    }

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(calibrator, indent=2))
        log.info("platt_calibration: calibrator saved to %s", save_path)

    return calibrator


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_calibrator(path: Path | None = None) -> dict | None:
    """Load a saved Platt calibrator from JSON.  Returns None if unavailable."""
    p = Path(path) if path else _DEFAULT_CALIBRATOR_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        log.warning("platt_calibration: failed to load calibrator — %s", exc)
        return None


def calibrate_probability(
    raw_score: float,
    calibrator: dict | None = None,
    *,
    calibrator_path: Path | None = None,
) -> float:
    """Convert a raw engine signal strength score to a calibrated probability.

    Falls back to a linear heuristic (score / 100) when no calibrator is
    available, so the engine degrades gracefully.

    Parameters
    ----------
    raw_score:
        Signal strength score (0–100).
    calibrator:
        Pre-loaded calibrator dict.  If None, tries to load from disk.
    calibrator_path:
        Override path for the calibrator JSON file.

    Returns
    -------
    float: Probability estimate in [0, 1].
    """
    cal = calibrator or load_calibrator(calibrator_path)

    if cal is None:
        # Linear fallback: treat 50 as the break-even point
        return float(np.clip(raw_score / 100.0, 0.0, 1.0))

    A = float(cal.get("A", 1.0))
    B = float(cal.get("B", 0.0))
    scale = float(cal.get("score_scale", 100.0))
    s_norm = float(raw_score) / scale
    return round(_sigmoid(A * s_norm + B), 4)


# ---------------------------------------------------------------------------
# Diagnostics / calibration report
# ---------------------------------------------------------------------------

def calibration_report(
    scores: list[float],
    outcomes: list[int],
    *,
    calibrator: dict | None = None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Produce a reliability diagram summary as a DataFrame.

    Each row covers a bucket of predicted probabilities and shows:
    - ``bucket_mid``: midpoint of the probability bucket
    - ``n``: number of samples
    - ``predicted_mean``: mean calibrated probability in bucket
    - ``actual_win_rate``: fraction of samples with outcome == 1
    - ``calibration_error``: |predicted_mean - actual_win_rate|

    Parameters
    ----------
    scores:
        Raw signal scores.
    outcomes:
        Binary outcomes (1 = correct, 0 = wrong).
    calibrator:
        Platt calibrator dict; if None, loads from default path.
    n_buckets:
        Number of equal-width probability buckets.
    """
    s_arr = np.array(scores, dtype=float)
    y_arr = np.array(outcomes, dtype=float)

    cal = calibrator or load_calibrator()
    probs = np.array([calibrate_probability(float(s), cal) for s in s_arr])

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    rows = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_buckets - 1 else probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred_mean = float(probs[mask].mean())
        actual_rate = float(y_arr[mask].mean())
        rows.append({
            "bucket_mid": round((lo + hi) / 2, 3),
            "n": n,
            "predicted_mean": round(pred_mean, 4),
            "actual_win_rate": round(actual_rate, 4),
            "calibration_error": round(abs(pred_mean - actual_rate), 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        ece = float((df["n"] * df["calibration_error"]).sum() / df["n"].sum())
        log.info("platt_calibration: ECE = %.4f over %d buckets", ece, len(df))

    return df
