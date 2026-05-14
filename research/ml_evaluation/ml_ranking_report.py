"""
ML Ranking Report
==================
Analyzes GBT model ranking performance by ml_rank_score quintiles.

Computes per-quintile:
  - Hit rate at 60m horizon
  - Average signed return (bps)
  - Average rank score
  - Tradeability (% with trade_status == TRADE)

RESEARCH ONLY — does not affect production decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary


def build_ranking_report(df: pd.DataFrame) -> dict:
    """
    Build ranking analysis report grouped by ml_rank_bucket.

    Parameters
    ----------
    df : pd.DataFrame
        Extended signals dataset with ml_rank_score and ml_rank_bucket columns.

    Returns
    -------
    dict with quintile_analysis, spread, and monotonicity results.
    """
    if "ml_rank_score" not in df.columns or "ml_rank_bucket" not in df.columns:
        return {"error": "ML rank columns not found in dataset"}

    quality_df = apply_quality_label_view(df)
    scored = quality_df[quality_df["ml_rank_score"].notna()].copy()
    if scored.empty:
        return {"error": "No signals with ml_rank_score available"}

    scored["correct_60m_num"] = pd.to_numeric(scored.get("correct_60m"), errors="coerce")
    scored["return_60m_bps"] = pd.to_numeric(scored.get("signed_return_60m_bps"), errors="coerce")
    scored["return_120m_bps"] = pd.to_numeric(scored.get("signed_return_120m_bps"), errors="coerce")
    scored["is_trade"] = scored.get("trade_status", "").astype(str).str.upper().str.strip() == "TRADE"

    # Group by quintile bucket
    bucket_order = ["Q1_lowest", "Q2_low", "Q3_mid", "Q4_high", "Q5_highest"]
    quintile_results = []

    for bucket in bucket_order:
        subset = scored[scored["ml_rank_bucket"] == bucket]
        if subset.empty:
            continue

        n = len(subset)
        n_labeled = int(subset["correct_60m_num"].notna().sum())
        hit_rate_60m = _safe_mean(subset["correct_60m_num"])
        avg_return_60m = _safe_mean(subset["return_60m_bps"])
        avg_return_120m = _safe_mean(subset["return_120m_bps"])
        avg_rank_score = _safe_mean(subset["ml_rank_score"])
        tradeability = subset["is_trade"].mean() if not subset.empty else None

        quintile_results.append({
            "bucket": bucket,
            "n": n,
            "n_labeled_60m": n_labeled,
            "hit_rate_60m": _rnd(hit_rate_60m),
            "avg_signed_return_60m_bps": _rnd(avg_return_60m),
            "avg_signed_return_120m_bps": _rnd(avg_return_120m),
            "avg_rank_score": _rnd(avg_rank_score),
            "tradeability_pct": _rnd(tradeability),
        })

    # Compute spread and monotonicity
    hit_rates = [q["hit_rate_60m"] for q in quintile_results if q["hit_rate_60m"] is not None]
    spread = round(hit_rates[-1] - hit_rates[0], 4) if len(hit_rates) >= 2 else None
    monotonic = _is_loosely_monotonic(hit_rates) if len(hit_rates) >= 3 else None

    return {
        "model": "GBT_shallow_v1",
        "role": "ranking",
        "n_scored": len(scored),
        "label_quality_summary": label_quality_summary(df),
        "quintile_analysis": quintile_results,
        "spread": spread,
        "monotonic": monotonic,
    }


def _safe_mean(series: pd.Series):
    """Compute mean ignoring NaN, return None if all NaN."""
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _rnd(val, digits=4):
    if val is None:
        return None
    return round(val, digits)


def _is_loosely_monotonic(values: list) -> bool:
    """Check if a list is loosely monotonically increasing (allow 1 violation)."""
    if len(values) < 3:
        return True
    violations = sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])
    return violations <= 1
