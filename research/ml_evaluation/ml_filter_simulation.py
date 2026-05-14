"""
ML Filter & Sizing Simulation Report
======================================
Simulates hypothetical performance improvements from:
  1. Filtering out bottom N% of signals by ml_rank_score
  2. Position sizing based on ml_confidence_score

CRITICAL: This is pure simulation — no real trades are affected.

RESEARCH ONLY — does not affect production decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ml_models.ml_config import FILTER_PERCENTILES, SIZING_BUCKETS
from research.ml_models.ml_inference import compute_size_multiplier
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary


def build_filter_simulation_report(df: pd.DataFrame) -> dict:
    """
    Build filter and sizing simulation report.

    Parameters
    ----------
    df : pd.DataFrame
        Extended signals dataset with ML columns.

    Returns
    -------
    dict with filter_results, sizing_simulation, and summary.
    """
    if "ml_rank_score" not in df.columns:
        return {"error": "ml_rank_score not found in dataset"}

    raw_df = df.copy()
    df = apply_quality_label_view(df)
    df["correct_60m_num"] = pd.to_numeric(df.get("correct_60m"), errors="coerce")
    df["return_60m_bps"] = pd.to_numeric(df.get("signed_return_60m_bps"), errors="coerce")
    df["return_120m_bps"] = pd.to_numeric(df.get("signed_return_120m_bps"), errors="coerce")

    # Only evaluate scored signals
    scored = df[df["ml_rank_score"].notna()].copy()
    if scored.empty:
        return {"error": "No scored signals available"}

    # Baseline (all scored signals)
    baseline = _compute_perf(scored, "all_signals")

    # 1. Filter simulations
    filter_results = [baseline]
    for pct in FILTER_PERCENTILES:
        threshold = np.percentile(scored["ml_rank_score"].dropna(), pct)
        filtered = scored[scored["ml_rank_score"] >= threshold]
        result = _compute_perf(filtered, f"remove_bottom_{pct}pct")
        result["filter_pct"] = pct
        result["threshold"] = round(float(threshold), 4)
        # Compute improvement over baseline
        if baseline["hit_rate_60m"] is not None and result["hit_rate_60m"] is not None:
            result["hit_rate_improvement_pct"] = round(
                (result["hit_rate_60m"] - baseline["hit_rate_60m"]) / max(baseline["hit_rate_60m"], 1e-9) * 100, 2
            )
        if baseline["avg_return_bps"] is not None and result["avg_return_bps"] is not None:
            result["return_improvement_bps"] = round(result["avg_return_bps"] - baseline["avg_return_bps"], 2)
        filter_results.append(result)

    # 2. Sizing simulation
    sizing_sim = _sizing_simulation(scored)

    # 3. Summary
    best_filter = max(
        [f for f in filter_results if f.get("filter_pct")],
        key=lambda x: x.get("hit_rate_60m") or 0,
        default={},
    )

    summary = {
        "baseline_hit_rate": baseline.get("hit_rate_60m"),
        "best_filter_hit_rate": best_filter.get("hit_rate_60m"),
        "best_filter_pct": best_filter.get("filter_pct"),
        "sizing_improvement_pct": sizing_sim.get("sizing_improvement_pct"),
    }

    return {
        "label_quality_summary": label_quality_summary(raw_df),
        "filter_results": filter_results,
        "sizing_simulation": sizing_sim,
        "summary": summary,
    }


def _compute_perf(df: pd.DataFrame, label: str) -> dict:
    """Compute basic performance metrics for a signal subset."""
    n = len(df)
    n_labeled = int(df["correct_60m_num"].notna().sum())
    hit_rate = _safe_mean(df["correct_60m_num"])
    avg_return = _safe_mean(df["return_60m_bps"])
    avg_return_120 = _safe_mean(df["return_120m_bps"])

    return {
        "label": label,
        "n_kept": n,
        "n_labeled_60m": n_labeled,
        "hit_rate_60m": _rnd(hit_rate),
        "avg_return_bps": _rnd(avg_return),
        "avg_return_120m_bps": _rnd(avg_return_120),
    }


def _sizing_simulation(df: pd.DataFrame) -> dict:
    """
    Simulate ML-based position sizing using ml_confidence_score.

    Computes hypothetical adjusted returns and drawdown impact.
    """
    if "ml_confidence_score" not in df.columns:
        return {"error": "ml_confidence_score not found"}

    scored = df[df["ml_confidence_score"].notna() & df["return_60m_bps"].notna()].copy()
    if scored.empty:
        return {"error": "No signals with both confidence score and returns"}

    # Compute size multipliers
    scored["size_multiplier"] = scored["ml_confidence_score"].apply(compute_size_multiplier)

    # Baseline returns (equal sizing)
    baseline_returns = scored["return_60m_bps"].values
    baseline_cumulative = np.cumsum(baseline_returns)
    baseline_dd = _max_drawdown(baseline_cumulative)

    # ML-sized returns
    sized_returns = scored["return_60m_bps"].values * scored["size_multiplier"].values
    sized_cumulative = np.cumsum(sized_returns)
    sized_dd = _max_drawdown(sized_cumulative)

    baseline_avg = float(baseline_returns.mean())
    sized_avg = float(sized_returns.mean())

    improvement = (
        round((sized_avg - baseline_avg) / max(abs(baseline_avg), 1e-9) * 100, 2)
        if baseline_avg != 0 else 0.0
    )

    # Per-bucket breakdown
    bucket_breakdown = []
    for low, high, mult in SIZING_BUCKETS:
        mask = (scored["ml_confidence_score"] >= low) & (scored["ml_confidence_score"] < high)
        subset = scored[mask]
        if subset.empty:
            continue
        bucket_breakdown.append({
            "confidence_range": f"{low:.2f}-{high:.2f}",
            "multiplier": mult,
            "n": len(subset),
            "avg_return_bps": _rnd(_safe_mean(subset["return_60m_bps"])),
            "sized_avg_return_bps": _rnd(float((subset["return_60m_bps"] * mult).mean())),
        })

    return {
        "n_signals": len(scored),
        "baseline_avg_return_bps": round(baseline_avg, 2),
        "ml_sized_avg_return_bps": round(sized_avg, 2),
        "sizing_improvement_pct": improvement,
        "baseline_total_return_bps": round(float(baseline_cumulative[-1]), 2) if len(baseline_cumulative) > 0 else 0,
        "ml_sized_total_return_bps": round(float(sized_cumulative[-1]), 2) if len(sized_cumulative) > 0 else 0,
        "baseline_max_dd_bps": round(baseline_dd, 2),
        "ml_sized_max_dd_bps": round(sized_dd, 2),
        "bucket_breakdown": bucket_breakdown,
    }


def _max_drawdown(cumulative: np.ndarray) -> float:
    """Compute maximum drawdown from a cumulative return series."""
    if len(cumulative) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0


def _safe_mean(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _rnd(val, digits=4):
    if val is None:
        return None
    return round(val, digits)
