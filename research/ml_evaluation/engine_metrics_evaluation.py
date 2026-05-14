"""
Engine Metrics Evaluation
==========================
Computes Test AUC, ECE, Brier score, quintile spread, and other standard ML
metrics for the production engine's probability predictions on historical data.

Evaluates three engine prediction channels:
  1. hybrid_move_probability   (final blended output)
  2. rule_move_probability     (rule-based sub-component)
  3. ml_move_probability       (ML sub-component)

Uses the same methodology as the ML model evaluation for direct comparability.

RESEARCH ONLY — does not modify production logic.

Usage:
    python -m research.ml_evaluation.engine_metrics_evaluation
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary

logger = logging.getLogger(__name__)

BACKTEST_PARQUET = (
    Path(__file__).resolve().parents[1] / "signal_evaluation" / "backtest_signals_dataset.parquet"
)
BACKTEST_CSV = BACKTEST_PARQUET.with_suffix(".csv")
OUTPUT_DIR = Path(__file__).resolve().parent


def _load_dataset() -> pd.DataFrame:
    if BACKTEST_PARQUET.exists():
        return pd.read_parquet(BACKTEST_PARQUET)
    if BACKTEST_CSV.exists():
        return pd.read_csv(BACKTEST_CSV)
    raise FileNotFoundError(f"No backtest dataset at {BACKTEST_PARQUET} or {BACKTEST_CSV}")


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC without sklearn dependency (manual trapezoidal)."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score = mean((predicted - actual)^2)."""
    return float(np.mean((y_score - y_true) ** 2))


def _compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    """
    Expected Calibration Error with reliability diagram data.
    Returns (ece_value, list_of_bin_dicts).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    weighted_error = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({
                "bin_low": round(lo, 2), "bin_high": round(hi, 2),
                "n": 0, "avg_predicted": None, "avg_actual": None, "gap": None,
            })
            continue
        avg_pred = float(y_score[mask].mean())
        avg_actual = float(y_true[mask].mean())
        gap = abs(avg_pred - avg_actual)
        weighted_error += (n / total) * gap
        bins.append({
            "bin_low": round(lo, 2), "bin_high": round(hi, 2),
            "n": n,
            "avg_predicted": round(avg_pred, 4),
            "avg_actual": round(avg_actual, 4),
            "gap": round(gap, 4),
        })

    return round(weighted_error, 4), bins


def _compute_quintile_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    returns_bps: np.ndarray | None = None,
) -> dict:
    """
    Assign predictions to 5 quintile buckets and compute hit rate, return, spread.
    Uses the same bucket labels as the ML evaluation.
    """
    labels = ["Q1_lowest", "Q2_low", "Q3_mid", "Q4_high", "Q5_highest"]
    try:
        quintiles = pd.qcut(y_score, 5, labels=labels, duplicates="drop")
    except ValueError:
        # Fallback if too few unique values for 5 equal bins
        quintiles = pd.cut(y_score, 5, labels=labels[:5], duplicates="drop")

    df_q = pd.DataFrame({
        "quintile": quintiles,
        "actual": y_true,
        "predicted": y_score,
    })
    if returns_bps is not None:
        df_q["return_bps"] = returns_bps

    bucket_results = []
    hit_rates = []
    for label in labels:
        subset = df_q[df_q["quintile"] == label]
        n = len(subset)
        if n == 0:
            bucket_results.append({"bucket": label, "n": 0})
            continue
        hr = float(subset["actual"].mean())
        hit_rates.append(hr)
        entry = {
            "bucket": label,
            "n": n,
            "hit_rate_60m": round(hr, 4),
            "avg_predicted": round(float(subset["predicted"].mean()), 4),
        }
        if returns_bps is not None:
            entry["avg_return_bps"] = round(float(subset["return_bps"].mean()), 4)
        bucket_results.append(entry)

    # Check monotonicity
    monotonic = all(
        hit_rates[i] <= hit_rates[i + 1] for i in range(len(hit_rates) - 1)
    ) if len(hit_rates) >= 2 else False

    spread = round(hit_rates[-1] - hit_rates[0], 4) if len(hit_rates) >= 2 else 0.0

    return {
        "quintile_analysis": bucket_results,
        "spread": spread,
        "monotonic": monotonic,
        "top_quintile_hit": round(hit_rates[-1], 4) if hit_rates else None,
        "bottom_quintile_hit": round(hit_rates[0], 4) if hit_rates else None,
    }


def _evaluate_predictor(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    returns_bps: np.ndarray | None = None,
) -> dict:
    """Compute full suite of metrics for one predictor."""
    auc = _compute_auc(y_true, y_score)
    brier = _compute_brier(y_true, y_score)
    ece, reliability_bins = _compute_ece(y_true, y_score)
    quintile = _compute_quintile_analysis(y_true, y_score, returns_bps)

    # Log loss
    eps = 1e-15
    clipped = np.clip(y_score, eps, 1 - eps)
    log_loss = -float(np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))

    return {
        "predictor": name,
        "n_samples": int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
        "metrics": {
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "ece": ece,
            "log_loss": round(log_loss, 4),
        },
        "reliability_diagram": reliability_bins,
        **quintile,
    }


def run_engine_evaluation() -> dict:
    """
    Run full metrics evaluation for the production engine's predictions.

    Evaluates on:
      - ALL rows with outcomes (all signals where engine scored + outcome known)
      - TRADE-only rows (engine-selected trades with outcomes)

    Returns summary dict and writes JSON report.
    """
    raw_df = _load_dataset()
    logger.info("Loaded %d rows", len(raw_df))
    df = apply_quality_label_view(raw_df)

    # Identify rows with valid outcomes and predictions
    prob_cols = ["hybrid_move_probability", "rule_move_probability", "ml_move_probability"]
    for col in prob_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["correct_60m_num"] = pd.to_numeric(df.get("correct_60m"), errors="coerce")
    df["return_60m_bps"] = pd.to_numeric(df.get("signed_return_60m_bps"), errors="coerce")

    # Filter to rows with valid outcomes
    has_outcome = df["correct_60m_num"].notna()
    has_hybrid = df["hybrid_move_probability"].notna()
    valid = df[has_outcome & has_hybrid].copy()
    logger.info("Rows with outcomes + hybrid_move_probability: %d", len(valid))

    if valid.empty:
        return {"error": "No rows with both predictions and outcomes"}

    y_true = valid["correct_60m_num"].values.astype(float)
    returns = valid["return_60m_bps"].values if valid["return_60m_bps"].notna().all() else None

    # --- Evaluate each engine predictor ---
    results = {}
    for col in prob_cols:
        col_valid = valid[valid[col].notna()]
        if col_valid.empty:
            continue
        yt = col_valid["correct_60m_num"].values.astype(float)
        ys = col_valid[col].values.astype(float)
        ret = col_valid["return_60m_bps"].values if col_valid["return_60m_bps"].notna().all() else None
        results[col] = _evaluate_predictor(col, yt, ys, ret)

    # --- Also segment: trade-only evaluation for hybrid ---
    trade_mask = valid.get("trade_status", pd.Series(dtype=str)).astype(str).str.upper().str.strip() == "TRADE"
    trade_df = valid[trade_mask]
    if len(trade_df) > 0:
        yt_trade = trade_df["correct_60m_num"].values.astype(float)
        ys_trade = trade_df["hybrid_move_probability"].values.astype(float)
        ret_trade = trade_df["return_60m_bps"].values if trade_df["return_60m_bps"].notna().all() else None
        results["hybrid_trade_only"] = _evaluate_predictor(
            "hybrid_move_probability (TRADE only)", yt_trade, ys_trade, ret_trade
        )

    # --- Year-by-year stability for hybrid ---
    if "year" in valid.columns:
        valid["_year"] = pd.to_numeric(valid["year"], errors="coerce")
    elif "signal_timestamp" in valid.columns:
        valid["_year"] = pd.to_datetime(valid["signal_timestamp"], errors="coerce").dt.year
    elif "timestamp" in valid.columns:
        valid["_year"] = pd.to_datetime(valid["timestamp"], errors="coerce").dt.year
    elif "date" in valid.columns:
        valid["_year"] = pd.to_datetime(valid["date"], errors="coerce").dt.year
    else:
        valid["_year"] = None

    yearly = []
    if valid["_year"].notna().any():
        for year, grp in sorted(valid.groupby("_year")):
            if len(grp) < 10:
                continue
            yt_y = grp["correct_60m_num"].values.astype(float)
            ys_y = grp["hybrid_move_probability"].values.astype(float)
            auc_y = _compute_auc(yt_y, ys_y)
            brier_y = _compute_brier(yt_y, ys_y)
            ece_y, _ = _compute_ece(yt_y, ys_y)
            yearly.append({
                "year": int(year),
                "n": len(grp),
                "roc_auc": round(auc_y, 4),
                "brier": round(brier_y, 4),
                "ece": ece_y,
                "hit_rate": round(float(yt_y.mean()), 4),
            })

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_rows": int(len(df)),
        "rows_with_outcomes": int(len(valid)),
        "label_quality_summary": label_quality_summary(raw_df),
        "predictors": results,
        "yearly_stability": yearly,
    }

    # Save report
    out_path = OUTPUT_DIR / "engine_metrics_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Engine metrics report saved to %s", out_path)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = run_engine_evaluation()

    # Print summary
    print("\n" + "=" * 70)
    print("ENGINE METRICS EVALUATION — SUMMARY")
    print("=" * 70)
    for name, data in report.get("predictors", {}).items():
        m = data["metrics"]
        print(f"\n  {data['predictor']}:")
        print(f"    N={data['n_samples']}, pos_rate={data['pos_rate']}")
        print(f"    AUC={m['roc_auc']}, ECE={m['ece']}, Brier={m['brier_score']}, LogLoss={m['log_loss']}")
        print(f"    Spread={data['spread']}, Monotonic={data['monotonic']}")
        if data.get("top_quintile_hit") is not None:
            print(f"    Top Q={data['top_quintile_hit']}, Bottom Q={data['bottom_quintile_hit']}")

    if report.get("yearly_stability"):
        print(f"\n  Year-by-Year (hybrid):")
        for yr in report["yearly_stability"]:
            print(f"    {yr['year']}: AUC={yr['roc_auc']}, ECE={yr['ece']}, Brier={yr['brier']}, HitRate={yr['hit_rate']}")
