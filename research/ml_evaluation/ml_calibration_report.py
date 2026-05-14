"""
ML Calibration Report
======================
Analyzes LogReg calibration quality by ml_confidence_bucket.

Computes per-bucket:
  - Predicted probability (avg ml_confidence_score)
  - Actual hit rate at 60m
  - Calibration gap (|predicted - actual|)

Also computes Expected Calibration Error (ECE).

RESEARCH ONLY — does not affect production decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary


def build_calibration_report(df: pd.DataFrame) -> dict:
    """
    Build calibration analysis report grouped by ml_confidence_bucket.

    Parameters
    ----------
    df : pd.DataFrame
        Extended signals dataset with ml_confidence_score and ml_confidence_bucket.

    Returns
    -------
    dict with bucket_analysis, ECE, and reliability metrics.
    """
    if "ml_confidence_score" not in df.columns or "ml_confidence_bucket" not in df.columns:
        return {"error": "ML confidence columns not found in dataset"}

    quality_df = apply_quality_label_view(df)
    scored = quality_df[quality_df["ml_confidence_score"].notna()].copy()
    if scored.empty:
        return {"error": "No signals with ml_confidence_score available"}

    scored["correct_60m_num"] = pd.to_numeric(scored.get("correct_60m"), errors="coerce")

    bucket_order = ["Q1_lowest", "Q2_low", "Q3_mid", "Q4_high", "Q5_highest"]
    bucket_results = []
    ece_numerator = 0.0
    ece_denominator = 0

    for bucket in bucket_order:
        subset = scored[scored["ml_confidence_bucket"] == bucket]
        if subset.empty:
            continue

        n = len(subset)
        labeled_subset = subset[subset["correct_60m_num"].notna()]
        n_labeled = int(len(labeled_subset))
        avg_conf = _safe_mean(labeled_subset["ml_confidence_score"]) if n_labeled else None
        actual_hit = _safe_mean(labeled_subset["correct_60m_num"]) if n_labeled else None

        gap = None
        if avg_conf is not None and actual_hit is not None:
            gap = abs(avg_conf - actual_hit)
            ece_numerator += gap * n_labeled
            ece_denominator += n_labeled

        bucket_results.append({
            "bucket": bucket,
            "n": n,
            "n_labeled_60m": n_labeled,
            "avg_confidence_score": _rnd(avg_conf),
            "actual_hit_rate_60m": _rnd(actual_hit),
            "calibration_gap": _rnd(gap),
        })

    ece = round(ece_numerator / max(ece_denominator, 1), 4) if ece_denominator > 0 else None

    # Reliability diagram data (10 bins)
    reliability = _build_reliability_data(scored)

    return {
        "model": "LogReg_ElasticNet_v1",
        "role": "calibration",
        "n_scored": len(scored),
        "label_quality_summary": label_quality_summary(df),
        "bucket_analysis": bucket_results,
        "expected_calibration_error": ece,
        "reliability_diagram": reliability,
    }


def _build_reliability_data(df: pd.DataFrame) -> list[dict]:
    """Build 10-bin reliability diagram data for calibration visualization."""
    bins = np.linspace(0, 1, 11)
    df = df.copy()
    df["conf_bin"] = pd.cut(df["ml_confidence_score"], bins=bins, include_lowest=True)
    df["correct_60m_num"] = pd.to_numeric(df.get("correct_60m"), errors="coerce")

    results = []
    for interval in df["conf_bin"].cat.categories:
        subset = df[df["conf_bin"] == interval]
        if subset.empty:
            continue
        labeled_subset = subset[subset["correct_60m_num"].notna()]
        results.append({
            "bin_low": round(interval.left, 2),
            "bin_high": round(interval.right, 2),
            "n": len(subset),
            "n_labeled_60m": int(len(labeled_subset)),
            "avg_predicted": _rnd(_safe_mean(labeled_subset["ml_confidence_score"])),
            "avg_actual": _rnd(_safe_mean(labeled_subset["correct_60m_num"])),
        })
    return results


def _safe_mean(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _rnd(val, digits=4):
    if val is None:
        return None
    return round(val, digits)
