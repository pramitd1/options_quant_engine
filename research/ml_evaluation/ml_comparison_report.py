"""
ML vs Engine Comparison Report
================================
Compares engine-only performance against ML agreement/disagreement groups.

Computes:
  - Engine-only performance (all TRADE signals)
  - Performance when ML agrees with engine
  - Performance when ML disagrees with engine
  - Year-over-year breakdowns

RESEARCH ONLY — does not affect production decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary


def build_comparison_report(df: pd.DataFrame) -> dict:
    """
    Build ML vs Engine comparison report.

    Parameters
    ----------
    df : pd.DataFrame
        Extended signals dataset with ml_agreement_with_engine column.

    Returns
    -------
    dict with agreement-based performance breakdowns.
    """
    if "ml_agreement_with_engine" not in df.columns:
        return {"error": "ml_agreement_with_engine column not found"}

    raw_df = df.copy()
    df = apply_quality_label_view(df)
    df["correct_60m_num"] = pd.to_numeric(df.get("correct_60m"), errors="coerce")
    df["return_60m_bps"] = pd.to_numeric(df.get("signed_return_60m_bps"), errors="coerce")
    df["return_120m_bps"] = pd.to_numeric(df.get("signed_return_120m_bps"), errors="coerce")
    df["return_session_bps"] = pd.to_numeric(df.get("signed_return_session_close_bps"), errors="coerce")

    # Engine trades only (TRADE status)
    engine_trades = df[df["trade_status"].astype(str).str.upper().str.strip() == "TRADE"]

    # Agreement groups (within trades)
    ml_agree = df[df["ml_agreement_with_engine"] == "YES"]
    ml_disagree = df[df["ml_agreement_with_engine"] == "NO"]
    no_engine = df[df["ml_agreement_with_engine"] == "NO_ENGINE_SIGNAL"]

    summary = {
        "engine_n": len(engine_trades),
        "engine_hit_rate_60m": _safe_mean(engine_trades["correct_60m_num"]),
        "engine_avg_return_bps": _safe_mean(engine_trades["return_60m_bps"]),
        "engine_avg_return_120m_bps": _safe_mean(engine_trades["return_120m_bps"]),
        "ml_agree_n": len(ml_agree),
        "ml_agree_hit_rate_60m": _safe_mean(ml_agree["correct_60m_num"]),
        "ml_agree_avg_return_bps": _safe_mean(ml_agree["return_60m_bps"]),
        "ml_agree_avg_return_120m_bps": _safe_mean(ml_agree["return_120m_bps"]),
        "ml_disagree_n": len(ml_disagree),
        "ml_disagree_hit_rate_60m": _safe_mean(ml_disagree["correct_60m_num"]),
        "ml_disagree_avg_return_bps": _safe_mean(ml_disagree["return_60m_bps"]),
        "ml_disagree_avg_return_120m_bps": _safe_mean(ml_disagree["return_120m_bps"]),
        "no_engine_signal_n": len(no_engine),
        "no_engine_hit_rate_60m": _safe_mean(no_engine["correct_60m_num"]),
        "no_engine_avg_return_bps": _safe_mean(no_engine["return_60m_bps"]),
    }

    # Round all numeric values
    summary = {k: _rnd(v) if isinstance(v, float) else v for k, v in summary.items()}

    # Year-over-year breakdown
    yearly = _yearly_breakdown(df)

    # Regime-based breakdown
    regime_breakdown = _regime_breakdown(df)

    return {
        "label_quality_summary": label_quality_summary(raw_df),
        "summary": summary,
        "yearly_breakdown": yearly,
        "regime_breakdown": regime_breakdown,
    }


def _yearly_breakdown(df: pd.DataFrame) -> list[dict]:
    """Break down agreement performance by year."""
    df = df.copy()
    ts = pd.to_datetime(df.get("signal_timestamp"), errors="coerce")
    df["year"] = ts.dt.year

    results = []
    for year in sorted(df["year"].dropna().unique()):
        yr_df = df[df["year"] == year]
        agree = yr_df[yr_df["ml_agreement_with_engine"] == "YES"]
        disagree = yr_df[yr_df["ml_agreement_with_engine"] == "NO"]
        trades = yr_df[yr_df["trade_status"].astype(str).str.upper().str.strip() == "TRADE"]

        results.append({
            "year": int(year),
            "n_total": len(yr_df),
            "n_trades": len(trades),
            "engine_hit_rate_60m": _rnd(_safe_mean(trades["correct_60m_num"])),
            "ml_agree_hit_rate_60m": _rnd(_safe_mean(agree["correct_60m_num"])),
            "ml_disagree_hit_rate_60m": _rnd(_safe_mean(disagree["correct_60m_num"])),
            "engine_avg_return_bps": _rnd(_safe_mean(trades["return_60m_bps"])),
            "ml_agree_avg_return_bps": _rnd(_safe_mean(agree["return_60m_bps"])),
        })

    return results


def _regime_breakdown(df: pd.DataFrame) -> list[dict]:
    """Break down performance by signal regime for agreement analysis."""
    if "signal_regime" not in df.columns:
        return []

    results = []
    for regime in df["signal_regime"].dropna().unique():
        r_df = df[df["signal_regime"] == regime]
        agree = r_df[r_df["ml_agreement_with_engine"] == "YES"]
        disagree = r_df[r_df["ml_agreement_with_engine"] == "NO"]

        results.append({
            "regime": str(regime),
            "n": len(r_df),
            "ml_agree_n": len(agree),
            "ml_disagree_n": len(disagree),
            "agree_hit_rate_60m": _rnd(_safe_mean(agree["correct_60m_num"])),
            "disagree_hit_rate_60m": _rnd(_safe_mean(disagree["correct_60m_num"])),
        })

    return sorted(results, key=lambda x: x["n"], reverse=True)


def _safe_mean(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _rnd(val, digits=4):
    if val is None:
        return None
    return round(val, digits)
