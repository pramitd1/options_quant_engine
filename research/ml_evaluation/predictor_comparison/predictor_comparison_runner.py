"""
Predictor Method Comparison on Signal Cumulative Dataset
==========================================================
Evaluates all 5 predictor methods head-to-head on the cumulative signal
dataset to determine which probability estimate best predicts actual
outcomes.

Predictors
----------
1. blended       — 70/30 rule+ML blend (hybrid_move_probability)
2. pure_rule     — rule-only leg (rule_move_probability)
3. pure_ml       — ML-only leg (ml_move_probability)
4. research_dual — GBT rank + LogReg calibration overlay
5. decision_policy — dual_model + dual_threshold policy gate

Metrics
-------
- Hit rate (direction correctness at 60 min)
- Avg return (signed_return_60m_bps)
- Cumulative return
- Max drawdown
- Sharpe proxy
- Signal retention (how many pass ≥ 0.50 threshold)
- Probability-outcome correlation

Author: Pramit Dutta
Organization: Quant Engines

RESEARCH ONLY — no production logic is modified.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_SIG_EVAL_DIR = Path(__file__).resolve().parents[2] / "signal_evaluation"
_CUMUL_CSV = _SIG_EVAL_DIR / "signals_dataset_cumul.csv"
_BACKTEST_PARQUET = _SIG_EVAL_DIR / "backtest_signals_dataset.parquet"

# ── Outcome columns ─────────────────────────────────────────────────
HIT_COL = "correct_60m"
RETURN_COL = "signed_return_60m_bps"
RETURN_120_COL = "signed_return_120m_bps"
MFE_COL = "mfe_60m_bps"
MAE_COL = "mae_60m_bps"

# ── Dual-threshold policy constants (from policy_config) ────────────
DUAL_MIN_RANK = 0.40
DUAL_MIN_CONF = 0.50


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _rnd(v: float | None, decimals: int = 2) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return round(f, decimals) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_mean(s: pd.Series) -> float | None:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else None


def _safe_std(s: pd.Series) -> float | None:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.std()) if len(vals) > 1 else None


def _max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def _sharpe_proxy(returns: np.ndarray) -> float | None:
    if len(returns) < 2:
        return None
    mu = float(returns.mean())
    sd = float(returns.std())
    return round(mu / sd, 4) if sd > 1e-9 else None


# ═════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════

def _load_cumulative() -> pd.DataFrame:
    if not _CUMUL_CSV.exists():
        raise FileNotFoundError(f"Cumulative dataset not found: {_CUMUL_CSV}")
    return pd.read_csv(_CUMUL_CSV)


def _load_backtest() -> pd.DataFrame:
    if _BACKTEST_PARQUET.exists():
        return pd.read_parquet(_BACKTEST_PARQUET)
    raise FileNotFoundError(f"Backtest dataset not found: {_BACKTEST_PARQUET}")


def _ensure_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Run research ML batch inference to populate rank + confidence scores."""
    has_rank = "ml_rank_score" in df.columns and df["ml_rank_score"].notna().sum() > 0
    has_conf = "ml_confidence_score" in df.columns and df["ml_confidence_score"].notna().sum() > 0
    if has_rank and has_conf:
        return df
    from research.ml_models.ml_inference import infer_batch
    logger.info("Running ML batch inference to populate rank/confidence columns …")
    return infer_batch(df)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numerics, add ML columns, compute derived predictor columns."""
    df = _ensure_ml_columns(df)

    # Coerce all probability and outcome columns to numeric
    for col in [HIT_COL, RETURN_COL, RETURN_120_COL, MFE_COL, MAE_COL,
                "hybrid_move_probability", "rule_move_probability",
                "ml_move_probability", "move_probability",
                "ml_rank_score", "ml_confidence_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived predictor columns
    # research_dual_model: uses confidence_score as probability
    if "ml_confidence_score" in df.columns:
        df["pred_research_dual"] = df["ml_confidence_score"]
    else:
        df["pred_research_dual"] = np.nan

    # decision_policy: dual_threshold gate over research_dual
    if "ml_rank_score" in df.columns and "ml_confidence_score" in df.columns:
        rank = df["ml_rank_score"]
        conf = df["ml_confidence_score"]
        # ALLOW: rank >= 0.40 AND conf >= 0.50
        allow = (rank >= DUAL_MIN_RANK) & (conf >= DUAL_MIN_CONF)
        # DOWNGRADE: one passes, one fails
        rank_ok = rank >= DUAL_MIN_RANK
        conf_ok = conf >= DUAL_MIN_CONF
        downgrade = (rank_ok | conf_ok) & ~allow
        # BLOCK: both fail
        block = ~rank_ok & ~conf_ok

        # Effective probability: ALLOW → confidence, DOWNGRADE → 0.5×, BLOCK → 0
        effective = conf.copy()
        effective[downgrade] = conf[downgrade] * 0.5
        effective[block] = 0.0
        # Handle NaN: if either score is NaN, set to NaN (not evaluable)
        missing = rank.isna() | conf.isna()
        effective[missing] = np.nan

        df["pred_decision_policy"] = effective
        df["policy_decision"] = "ALLOW"
        df.loc[downgrade, "policy_decision"] = "DOWNGRADE"
        df.loc[block, "policy_decision"] = "BLOCK"
        df.loc[missing, "policy_decision"] = "MISSING"
    else:
        df["pred_decision_policy"] = np.nan
        df["policy_decision"] = "MISSING"

    # ev_sizing: uses same ml_confidence_score as research_dual (offline; CRT lookup
    # requires live market context and is not replayable from the signal row alone)
    if "ml_confidence_score" in df.columns:
        df["pred_ev_sizing"] = df["ml_confidence_score"]
    else:
        df["pred_ev_sizing"] = np.nan

    # research_rank_gate: block signals where rank < 0.55; otherwise use confidence
    _RANK_GATE_THRESHOLD = 0.55
    if "ml_rank_score" in df.columns and "ml_confidence_score" in df.columns:
        rank = df["ml_rank_score"]
        conf = df["ml_confidence_score"]
        gate_pass = rank >= _RANK_GATE_THRESHOLD
        effective_rg = conf.copy()
        effective_rg[~gate_pass] = 0.0
        effective_rg[rank.isna() | conf.isna()] = np.nan
        df["pred_research_rank_gate"] = effective_rg
    else:
        df["pred_research_rank_gate"] = np.nan

    # research_uncertainty_adjusted: discount confidence by disagreement + ambiguity
    if "ml_confidence_score" in df.columns and "hybrid_move_probability" in df.columns:
        conf = df["ml_confidence_score"]
        hybrid = df["hybrid_move_probability"]
        disagreement = (conf - hybrid).abs()
        ambiguity = 1.0 - (2.0 * conf - 1.0).abs()
        multiplier = 1.0 - 0.80 * disagreement - 0.35 * ambiguity
        multiplier = multiplier.clip(lower=0.0)
        effective_ua = conf * multiplier
        # block if effective probability too low
        effective_ua[effective_ua < 0.35] = 0.0
        effective_ua[conf.isna() | hybrid.isna()] = np.nan
        df["pred_research_uncertainty_adjusted"] = effective_ua
    else:
        df["pred_research_uncertainty_adjusted"] = np.nan

    return df


# ═════════════════════════════════════════════════════════════════════
# Predictor definitions
# ═════════════════════════════════════════════════════════════════════

PREDICTORS = {
    "blended":              {"col": "hybrid_move_probability",          "desc": "70/30 rule+ML blend (production)"},
    "pure_rule":            {"col": "rule_move_probability",             "desc": "Rule-based Bayesian prior only"},
    "pure_ml":              {"col": "ml_move_probability",               "desc": "ML model only"},
    "research_dual":        {"col": "pred_research_dual",                "desc": "GBT rank + LogReg calibration"},
    "decision_policy":      {"col": "pred_decision_policy",              "desc": "Dual-model + dual_threshold gate"},
    "ev_sizing":            {"col": "pred_ev_sizing",                    "desc": "EV-based sizing (offline: maps to research_dual; CRT requires live context)"},
    "research_rank_gate":   {"col": "pred_research_rank_gate",           "desc": "Dual-model + rank ≥ 0.55 gate (block below threshold)"},
    "uncertainty_adjusted": {"col": "pred_research_uncertainty_adjusted", "desc": "Dual-model + disagreement/ambiguity discount"},
}


# ═════════════════════════════════════════════════════════════════════
# Metric computation
# ═════════════════════════════════════════════════════════════════════

def _evaluate_predictor(
    df: pd.DataFrame,
    predictor_name: str,
    prob_col: str,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Evaluate a predictor on a dataset.

    - Signals with probability ≥ threshold are "TRADE" candidates
    - Compute hit rate, return, risk metrics for the TRADE subset
    """
    prob = df[prob_col] if prob_col in df.columns else pd.Series(dtype=float)
    hit = df[HIT_COL] if HIT_COL in df.columns else pd.Series(dtype=float)
    ret = df[RETURN_COL] if RETURN_COL in df.columns else pd.Series(dtype=float)

    total = len(df)
    evaluable = prob.notna() & hit.notna()
    n_evaluable = int(evaluable.sum())

    # TRADE subset: probability ≥ threshold AND has outcome
    trade_mask = evaluable & (prob >= threshold)
    n_trade = int(trade_mask.sum())

    # NO-TRADE subset (below threshold but evaluable)
    no_trade_mask = evaluable & (prob < threshold)
    n_no_trade = int(no_trade_mask.sum())

    result: dict[str, Any] = {
        "predictor": predictor_name,
        "prob_col": prob_col,
        "threshold": threshold,
        "total_signals": total,
        "evaluable": n_evaluable,
        "n_trade": n_trade,
        "n_no_trade": n_no_trade,
        "retention_pct": _rnd(n_trade / max(n_evaluable, 1) * 100),
    }

    # TRADE metrics
    if n_trade > 0:
        trade_hit = hit[trade_mask]
        trade_ret = ret[trade_mask].dropna()
        trade_ret_arr = trade_ret.values

        cum = np.cumsum(trade_ret_arr) if len(trade_ret_arr) > 0 else np.array([0.0])

        result["trade_hit_rate"] = _rnd(_safe_mean(trade_hit))
        result["trade_avg_return_bps"] = _rnd(_safe_mean(trade_ret))
        result["trade_cumulative_bps"] = _rnd(float(cum[-1])) if len(cum) > 0 else 0.0
        result["trade_max_dd_bps"] = _rnd(_max_drawdown(cum))
        result["trade_volatility_bps"] = _rnd(_safe_std(trade_ret))
        result["trade_sharpe"] = _sharpe_proxy(trade_ret_arr) if len(trade_ret_arr) > 1 else None

        # MFE / MAE
        if MFE_COL in df.columns:
            result["trade_avg_mfe_bps"] = _rnd(_safe_mean(df.loc[trade_mask, MFE_COL]))
        if MAE_COL in df.columns:
            result["trade_avg_mae_bps"] = _rnd(_safe_mean(df.loc[trade_mask, MAE_COL]))
    else:
        result["trade_hit_rate"] = None
        result["trade_avg_return_bps"] = None
        result["trade_cumulative_bps"] = None
        result["trade_max_dd_bps"] = None
        result["trade_volatility_bps"] = None
        result["trade_sharpe"] = None

    # NO-TRADE metrics (what we're filtering out)
    if n_no_trade > 0:
        result["no_trade_hit_rate"] = _rnd(_safe_mean(hit[no_trade_mask]))
        result["no_trade_avg_return_bps"] = _rnd(_safe_mean(ret[no_trade_mask]))
    else:
        result["no_trade_hit_rate"] = None
        result["no_trade_avg_return_bps"] = None

    # Probability-outcome correlation
    both_valid = prob.notna() & ret.notna()
    if both_valid.sum() > 5:
        result["prob_return_corr"] = _rnd(float(prob[both_valid].corr(ret[both_valid])), 4)
    else:
        result["prob_return_corr"] = None

    return result


def _evaluate_all(df: pd.DataFrame, dataset_label: str) -> list[dict]:
    """Run all predictors on a dataset and return comparison rows."""
    rows = []
    for name, cfg in PREDICTORS.items():
        r = _evaluate_predictor(df, name, cfg["col"])
        r["dataset"] = dataset_label
        r["description"] = cfg["desc"]
        rows.append(r)
    return rows


def _evaluate_threshold_sweep(
    df: pd.DataFrame,
    predictor_name: str,
    prob_col: str,
) -> list[dict]:
    """Sweep thresholds from 0.30 to 0.70 for one predictor."""
    rows = []
    for threshold in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        r = _evaluate_predictor(df, predictor_name, prob_col, threshold)
        rows.append(r)
    return rows


def _policy_decision_breakdown(df: pd.DataFrame) -> list[dict]:
    """Breakdown of decision_policy ALLOW/BLOCK/DOWNGRADE performance."""
    if "policy_decision" not in df.columns:
        return []

    hit = df[HIT_COL] if HIT_COL in df.columns else pd.Series(dtype=float)
    ret = df[RETURN_COL] if RETURN_COL in df.columns else pd.Series(dtype=float)

    rows = []
    for decision in ["ALLOW", "DOWNGRADE", "BLOCK", "MISSING"]:
        mask = (df["policy_decision"] == decision) & hit.notna()
        n = int(mask.sum())
        if n == 0:
            continue
        ret_arr = ret[mask].dropna().values
        cum = np.cumsum(ret_arr) if len(ret_arr) > 0 else np.array([0.0])
        rows.append({
            "decision": decision,
            "n": n,
            "hit_rate": _rnd(_safe_mean(hit[mask])),
            "avg_return_bps": _rnd(_safe_mean(ret[mask])),
            "cumulative_bps": _rnd(float(cum[-1])) if len(cum) > 0 else 0.0,
        })
    return rows


# ═════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════

def _generate_charts(
    cumul_results: list[dict],
    backtest_results: list[dict],
    threshold_sweeps: dict[str, list[dict]],
    policy_breakdown: list[dict],
) -> list[str]:
    """Generate comparison charts."""
    warnings.filterwarnings("ignore", category=UserWarning)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[str] = []

    # Color scheme
    colors = {
        "blended": "#2196F3",
        "pure_rule": "#FF9800",
        "pure_ml": "#9C27B0",
        "research_dual": "#4CAF50",
        "decision_policy": "#F44336",
        "ev_sizing": "#00BCD4",
        "research_rank_gate": "#8BC34A",
        "uncertainty_adjusted": "#FF5722",
    }

    # ── Chart 1: Head-to-head comparison (backtest dataset) ──────────
    for label, results in [("backtest", backtest_results), ("cumulative", cumul_results)]:
        if not results:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Predictor Comparison — {label.title()} Dataset", fontsize=14, fontweight="bold")

        names = [r["predictor"] for r in results]
        x = np.arange(len(names))
        w = 0.5
        c = [colors.get(n, "#888") for n in names]

        # Hit rate
        ax = axes[0, 0]
        vals = [r.get("trade_hit_rate") or 0 for r in results]
        ax.bar(x, vals, w, color=c, edgecolor="white")
        ax.set_ylabel("Hit Rate (TRADE signals)")
        ax.set_title("Hit Rate @ p ≥ 0.50")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

        # Avg return
        ax = axes[0, 1]
        vals = [r.get("trade_avg_return_bps") or 0 for r in results]
        ax.bar(x, vals, w, color=c, edgecolor="white")
        ax.set_ylabel("Avg Return (bps)")
        ax.set_title("Avg Return 60m (TRADE signals)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v + 0.3, f"{v:+.1f}", ha="center", fontsize=8)

        # Retention
        ax = axes[1, 0]
        vals = [r.get("retention_pct") or 0 for r in results]
        ax.bar(x, vals, w, color=c, edgecolor="white")
        ax.set_ylabel("Retention %")
        ax.set_title("Signal Retention (% passing threshold)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)

        # Sharpe
        ax = axes[1, 1]
        vals = [r.get("trade_sharpe") or 0 for r in results]
        ax.bar(x, vals, w, color=c, edgecolor="white")
        ax.set_ylabel("Sharpe Proxy")
        ax.set_title("Risk-Adjusted Return")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p = OUTPUT_DIR / f"predictor_comparison_{label}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 2: Threshold sweep for each predictor (backtest) ───────
    if threshold_sweeps:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Threshold Sweep — Backtest Dataset", fontsize=13, fontweight="bold")

        for pname, sweep in threshold_sweeps.items():
            thresholds = [r["threshold"] for r in sweep]
            hr = [r.get("trade_hit_rate") or 0 for r in sweep]
            ret = [r.get("trade_avg_return_bps") or 0 for r in sweep]
            c = colors.get(pname, "#888")
            axes[0].plot(thresholds, hr, marker="o", label=pname, color=c)
            axes[1].plot(thresholds, ret, marker="s", label=pname, color=c)

        axes[0].set_xlabel("Probability Threshold")
        axes[0].set_ylabel("Hit Rate")
        axes[0].set_title("Hit Rate vs Threshold")
        axes[0].legend(fontsize=8)
        axes[0].axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)

        axes[1].set_xlabel("Probability Threshold")
        axes[1].set_ylabel("Avg Return (bps)")
        axes[1].set_title("Avg Return vs Threshold")
        axes[1].legend(fontsize=8)
        axes[1].axhline(0, color="grey", ls="--", lw=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p = OUTPUT_DIR / "threshold_sweep.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 3: Trade vs No-Trade comparison ────────────────────────
    if backtest_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("TRADE vs NO-TRADE Performance — Backtest", fontsize=13, fontweight="bold")
        names = [r["predictor"] for r in backtest_results]
        x = np.arange(len(names))
        w2 = 0.35

        trade_hr = [r.get("trade_hit_rate") or 0 for r in backtest_results]
        no_trade_hr = [r.get("no_trade_hit_rate") or 0 for r in backtest_results]
        axes[0].bar(x - w2/2, trade_hr, w2, label="TRADE (p≥0.50)", color="#4CAF50", edgecolor="white")
        axes[0].bar(x + w2/2, no_trade_hr, w2, label="NO-TRADE (p<0.50)", color="#F44336", edgecolor="white")
        axes[0].set_ylabel("Hit Rate")
        axes[0].set_title("Hit Rate: TRADE vs NO-TRADE")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        axes[0].legend(fontsize=8)
        axes[0].axhline(0.5, color="grey", ls="--", lw=0.8)

        trade_ret = [r.get("trade_avg_return_bps") or 0 for r in backtest_results]
        no_trade_ret = [r.get("no_trade_avg_return_bps") or 0 for r in backtest_results]
        axes[1].bar(x - w2/2, trade_ret, w2, label="TRADE (p≥0.50)", color="#4CAF50", edgecolor="white")
        axes[1].bar(x + w2/2, no_trade_ret, w2, label="NO-TRADE (p<0.50)", color="#F44336", edgecolor="white")
        axes[1].set_ylabel("Avg Return (bps)")
        axes[1].set_title("Avg Return: TRADE vs NO-TRADE")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].axhline(0, color="grey", ls="--", lw=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p = OUTPUT_DIR / "trade_vs_no_trade.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 4: Policy decision breakdown ───────────────────────────
    if policy_breakdown:
        fig, ax = plt.subplots(figsize=(8, 5))
        dec_labels = [r["decision"] for r in policy_breakdown]
        dec_hr = [r.get("hit_rate") or 0 for r in policy_breakdown]
        dec_ret = [r.get("avg_return_bps") or 0 for r in policy_breakdown]
        dec_colors = {"ALLOW": "#4CAF50", "DOWNGRADE": "#FF9800", "BLOCK": "#F44336", "MISSING": "#9E9E9E"}

        x_d = np.arange(len(dec_labels))
        w2 = 0.35
        ax.bar(x_d - w2/2, dec_hr, w2, label="Hit Rate", color=[dec_colors.get(d, "#888") for d in dec_labels], edgecolor="white")
        ax2 = ax.twinx()
        ax2.bar(x_d + w2/2, dec_ret, w2, label="Avg Return (bps)", color=[dec_colors.get(d, "#888") for d in dec_labels], edgecolor="white", alpha=0.5)
        ax.set_ylabel("Hit Rate")
        ax2.set_ylabel("Avg Return (bps)")
        ax.set_title("Decision Policy Breakdown — Backtest")
        ax.set_xticks(x_d)
        ax.set_xticklabels([f"{d}\n(n={r['n']})" for d, r in zip(dec_labels, policy_breakdown)], fontsize=9)
        for i, (hr, ret) in enumerate(zip(dec_hr, dec_ret)):
            ax.text(i - w2/2, hr + 0.01, f"{hr:.2f}", ha="center", fontsize=8)
            ax2.text(i + w2/2, ret + 0.3, f"{ret:+.1f}", ha="center", fontsize=8)

        plt.tight_layout()
        p = OUTPUT_DIR / "policy_decision_breakdown.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 5: Risk-return scatter ─────────────────────────────────
    if backtest_results:
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in backtest_results:
            sp = r.get("trade_sharpe")
            ret = r.get("trade_avg_return_bps")
            if sp is not None and ret is not None:
                c = colors.get(r["predictor"], "#888")
                ax.scatter(sp, ret, s=150, c=c, edgecolors="white", zorder=3)
                ax.annotate(r["predictor"], (sp, ret), fontsize=9,
                            textcoords="offset points", xytext=(8, 5))
        ax.set_xlabel("Sharpe Proxy")
        ax.set_ylabel("Avg Return 60m (bps)")
        ax.set_title("Risk-Return Scatter — All Predictors (Backtest)")
        ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.3)
        ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.3)
        plt.tight_layout()
        p = OUTPUT_DIR / "risk_return_scatter.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


# ═════════════════════════════════════════════════════════════════════
# Report renderer
# ═════════════════════════════════════════════════════════════════════

def _render_report(
    cumul_results: list[dict],
    backtest_results: list[dict],
    threshold_sweeps: dict[str, list[dict]],
    policy_breakdown_cumul: list[dict],
    policy_breakdown_bt: list[dict],
    charts: list[str],
    cumul_stats: dict,
    bt_stats: dict,
) -> str:
    lines: list[str] = []
    _h = lines.append

    _h("# Predictor Method Comparison Report")
    _h(f"\n**Generated:** {datetime.now().isoformat()}")
    _h("**Author:** Pramit Dutta  |  **Organization:** Quant Engines")
    _h("\n---\n")

    # ── Dataset overview ─────────────────────────────────────────────
    _h("## 1. Datasets\n")
    _h("| Dataset | Total Rows | With Outcomes | Date Range |")
    _h("|---------|-----------|--------------|-----------|")
    _h(f"| Cumulative | {cumul_stats['total']:,} | {cumul_stats['with_outcomes']} | {cumul_stats['date_range']} |")
    _h(f"| Backtest | {bt_stats['total']:,} | {bt_stats['with_outcomes']:,} | {bt_stats['date_range']} |")
    _h("")

    # ── Predictor descriptions ───────────────────────────────────────
    _h("## 2. Predictor Methods\n")
    _h("| Predictor | Probability Source | Description |")
    _h("|-----------|-------------------|-------------|")
    for name, cfg in PREDICTORS.items():
        _h(f"| **{name}** | `{cfg['col']}` | {cfg['desc']} |")
    _h("")

    # ── Main comparison tables ───────────────────────────────────────
    for label, results in [("Backtest (7,404 signals)", backtest_results),
                           ("Cumulative (live signals)", cumul_results)]:
        _h(f"## 3. Performance — {label}\n")
        _h("Threshold: p ≥ 0.50 for TRADE signals\n")
        _h("| Predictor | Evaluable | TRADE | Retention | Hit Rate | Avg Ret (bps) | Cum Ret (bps) | Max DD (bps) | Sharpe | Prob-Ret Corr |")
        _h("|-----------|----------|-------|----------|----------|--------------|--------------|-------------|--------|--------------|")
        for r in results:
            cum = r.get('trade_cumulative_bps')
            cum_s = f"{cum:,.0f}" if isinstance(cum, (int, float)) and cum is not None else "—"
            dd = r.get('trade_max_dd_bps')
            dd_s = f"{dd:,.0f}" if isinstance(dd, (int, float)) and dd is not None else "—"
            _h(f"| **{r['predictor']}** "
               f"| {r['evaluable']:,} "
               f"| {r['n_trade']:,} "
               f"| {r.get('retention_pct', '—')}% "
               f"| {r.get('trade_hit_rate', '—')} "
               f"| {r.get('trade_avg_return_bps', '—')} "
               f"| {cum_s} "
               f"| {dd_s} "
               f"| {r.get('trade_sharpe', '—')} "
               f"| {r.get('prob_return_corr', '—')} |")
        _h("")

        # TRADE vs NO-TRADE
        _h(f"### Filtering Effectiveness — {label}\n")
        _h("| Predictor | TRADE HR | TRADE Ret | NO-TRADE HR | NO-TRADE Ret | Separation |")
        _h("|-----------|---------|----------|------------|-------------|-----------|")
        for r in results:
            t_hr = r.get("trade_hit_rate") or 0
            nt_hr = r.get("no_trade_hit_rate") or 0
            t_ret = r.get("trade_avg_return_bps") or 0
            nt_ret = r.get("no_trade_avg_return_bps") or 0
            sep = _rnd(t_ret - nt_ret)
            _h(f"| {r['predictor']} | {t_hr} | {t_ret:+.1f} | {nt_hr} | {nt_ret:+.1f} | **{sep:+.1f} bps** |")
        _h("")

    # ── Decision policy breakdown ────────────────────────────────────
    for label, breakdown in [("Backtest", policy_breakdown_bt), ("Cumulative", policy_breakdown_cumul)]:
        if breakdown:
            _h(f"## 4. Decision Policy Breakdown — {label}\n")
            _h("| Decision | N | Hit Rate | Avg Return (bps) | Cum Return (bps) |")
            _h("|----------|---|----------|-----------------|-----------------|")
            for r in breakdown:
                cum = r.get("cumulative_bps")
                cum_s = f"{cum:,.0f}" if isinstance(cum, (int, float)) and cum is not None else "—"
                _h(f"| {r['decision']} | {r['n']} | {r.get('hit_rate', '—')} | {r.get('avg_return_bps', '—')} | {cum_s} |")
            _h("")

    # ── Threshold sweep ──────────────────────────────────────────────
    if threshold_sweeps:
        _h("## 5. Threshold Sensitivity (Backtest)\n")
        for pname, sweep in threshold_sweeps.items():
            _h(f"### {pname}\n")
            _h("| Threshold | N Trade | Retention | Hit Rate | Avg Return | Sharpe |")
            _h("|-----------|---------|----------|----------|-----------|--------|")
            for r in sweep:
                _h(f"| {r['threshold']:.2f} | {r['n_trade']:,} | {r.get('retention_pct', '—')}% "
                   f"| {r.get('trade_hit_rate', '—')} | {r.get('trade_avg_return_bps', '—')} "
                   f"| {r.get('trade_sharpe', '—')} |")
            _h("")

    # ── Rankings ─────────────────────────────────────────────────────
    _h("## 6. Rankings (Backtest @ p ≥ 0.50)\n")
    if backtest_results:
        valid = [r for r in backtest_results if r.get("trade_hit_rate") is not None]
        if valid:
            by_hr = sorted(valid, key=lambda r: r.get("trade_hit_rate", 0), reverse=True)
            by_ret = sorted(valid, key=lambda r: r.get("trade_avg_return_bps", 0), reverse=True)
            by_sharpe = sorted(valid, key=lambda r: r.get("trade_sharpe", 0) or 0, reverse=True)
            by_cum = sorted(valid, key=lambda r: r.get("trade_cumulative_bps", 0) or 0, reverse=True)

            n_methods = len(valid)
            header_cols = " | ".join(f"#{i+1}" for i in range(n_methods))
            sep_cols = " | ".join("----" for _ in range(n_methods))
            _h(f"| Metric | {header_cols} |")
            _h(f"|--------|{sep_cols}|")
            _h(f"| Hit Rate | {'| '.join(r['predictor'] for r in by_hr)} |")
            _h(f"| Avg Return | {'| '.join(r['predictor'] for r in by_ret)} |")
            _h(f"| Sharpe | {'| '.join(r['predictor'] for r in by_sharpe)} |")
            _h(f"| Cum Return | {'| '.join(r['predictor'] for r in by_cum)} |")
    _h("")

    # ── Charts ───────────────────────────────────────────────────────
    if charts:
        _h("## 7. Charts\n")
        for c in charts:
            fname = Path(c).name
            _h(f"![{fname}]({fname})")
        _h("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════

def run_predictor_comparison() -> dict[str, Any]:
    """Full predictor comparison pipeline."""
    print("=" * 70)
    print("  PREDICTOR METHOD COMPARISON")
    print("=" * 70)

    # ── Load datasets ────────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")

    cumul = _load_cumulative()
    print(f"  Cumulative: {len(cumul):,} rows")

    backtest = _load_backtest()
    print(f"  Backtest:   {len(backtest):,} rows")

    # ── Prepare datasets (ML inference + derived columns) ────────────
    print("\n[2/6] Preparing datasets (ML inference if needed) …")
    cumul = _prepare(cumul)
    backtest = _prepare(backtest)

    # Dataset stats
    cumul_hit = pd.to_numeric(cumul.get(HIT_COL), errors="coerce")
    bt_hit = pd.to_numeric(backtest.get(HIT_COL), errors="coerce")
    cumul_stats = {
        "total": len(cumul),
        "with_outcomes": int(cumul_hit.notna().sum()),
        "date_range": f"{cumul['signal_date'].min()} to {cumul['signal_date'].max()}" if "signal_date" in cumul.columns else "—",
    }
    bt_stats = {
        "total": len(backtest),
        "with_outcomes": int(bt_hit.notna().sum()),
        "date_range": "2016–2025 (10 years simulated)",
    }
    print(f"  Cumulative: {cumul_stats['with_outcomes']} rows with outcomes")
    print(f"  Backtest:   {bt_stats['with_outcomes']:,} rows with outcomes")

    # ── Evaluate all predictors ──────────────────────────────────────
    print("\n[3/6] Evaluating predictors …")

    cumul_results = _evaluate_all(cumul, "cumulative")
    backtest_results = _evaluate_all(backtest, "backtest")

    print("\n  === BACKTEST (p ≥ 0.50) ===")
    for r in backtest_results:
        hr = r.get("trade_hit_rate") or 0
        ret = r.get("trade_avg_return_bps") or 0
        n = r.get("n_trade", 0)
        sp = r.get("trade_sharpe") or "—"
        print(f"  {r['predictor']:20s}  n={n:5,}  HR={hr:5.2f}  ret={ret:+7.1f}  sharpe={sp}")

    print("\n  === CUMULATIVE (p ≥ 0.50) ===")
    for r in cumul_results:
        hr = r.get("trade_hit_rate") or 0
        ret = r.get("trade_avg_return_bps") or 0
        n = r.get("n_trade", 0)
        sp = r.get("trade_sharpe") or "—"
        print(f"  {r['predictor']:20s}  n={n:5,}  HR={hr:5.2f}  ret={ret:+7.1f}  sharpe={sp}")

    # ── Threshold sweep (backtest only — enough data) ────────────────
    print("\n[4/6] Threshold sweep (backtest) …")
    threshold_sweeps: dict[str, list[dict]] = {}
    for name, cfg in PREDICTORS.items():
        threshold_sweeps[name] = _evaluate_threshold_sweep(backtest, name, cfg["col"])
    print(f"  {len(threshold_sweeps)} predictors × 7 thresholds")

    # ── Policy decision breakdown ────────────────────────────────────
    print("\n[5/6] Policy decision breakdown …")
    policy_breakdown_bt = _policy_decision_breakdown(backtest)
    policy_breakdown_cumul = _policy_decision_breakdown(cumul)
    for r in policy_breakdown_bt:
        print(f"  Backtest {r['decision']:12s}  n={r['n']:5,}  HR={r.get('hit_rate', 0):.2f}  ret={r.get('avg_return_bps', 0):+.1f}")
    for r in policy_breakdown_cumul:
        print(f"  Cumul    {r['decision']:12s}  n={r['n']:5,}  HR={r.get('hit_rate', 0):.2f}  ret={r.get('avg_return_bps', 0):+.1f}")

    # ── Charts ───────────────────────────────────────────────────────
    print("\n[6/6] Generating charts …")
    charts = _generate_charts(cumul_results, backtest_results, threshold_sweeps, policy_breakdown_bt)
    print(f"  {len(charts)} charts saved")

    # ── Report ───────────────────────────────────────────────────────
    report_md = _render_report(
        cumul_results, backtest_results, threshold_sweeps,
        policy_breakdown_cumul, policy_breakdown_bt,
        charts, cumul_stats, bt_stats,
    )
    report_path = OUTPUT_DIR / "predictor_comparison_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n  Report → {report_path}")

    # ── JSON results ─────────────────────────────────────────────────
    results = {
        "evaluation_date": datetime.now().isoformat(),
        "cumulative_stats": cumul_stats,
        "backtest_stats": bt_stats,
        "cumulative_results": cumul_results,
        "backtest_results": backtest_results,
        "threshold_sweeps": threshold_sweeps,
        "policy_breakdown_backtest": policy_breakdown_bt,
        "policy_breakdown_cumulative": policy_breakdown_cumul,
        "charts": charts,
    }
    json_path = OUTPUT_DIR / "predictor_comparison_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  JSON  → {json_path}")

    # ── CSV exports ──────────────────────────────────────────────────
    pd.DataFrame(backtest_results).to_csv(OUTPUT_DIR / "backtest_comparison.csv", index=False)
    pd.DataFrame(cumul_results).to_csv(OUTPUT_DIR / "cumulative_comparison.csv", index=False)
    for name, sweep in threshold_sweeps.items():
        pd.DataFrame(sweep).to_csv(OUTPUT_DIR / f"threshold_sweep_{name}.csv", index=False)

    print("\n" + "=" * 70)
    print("  COMPLETE — all artifacts saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)

    return results


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    run_predictor_comparison()
