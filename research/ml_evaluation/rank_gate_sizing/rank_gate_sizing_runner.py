"""
Rank-Gate + Confidence-Sizing Research Evaluation
===================================================
Tests a modified decision-policy framework where:

  • **Rank (GBT) is the ONLY filter** — signals below the Nth percentile
    rank threshold are blocked.  Three thresholds tested: 20 %, 30 %, 40 %.
  • **Confidence (LogReg) is used ONLY for position sizing** — never filters.
    Three tiers map confidence → size multiplier:
        low   (0.40–0.50)  → 0.5×
        medium(0.50–0.60)  → 1.0×
        high  (>0.60)      → 1.5×

Comparison baselines:
  • dual_threshold  (rank ≥ 0.40 AND confidence ≥ 0.50)
  • rank_filter_bottom_30pct (block bottom 30 % by rank)

Metrics produced:
  hit rate, avg return, cumulative return, max drawdown, Sharpe proxy,
  sized P&L (confidence-weighted), full comparison table.

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

# ── Local imports ────────────────────────────────────────────────────
from research.decision_policy.policy_config import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    PRIMARY_HIT_COL,
    PRIMARY_RETURN_COL,
    SECONDARY_RETURN_COL,
    MFE_COL,
    MAE_COL,
    SESSION_RETURN_COL,
)
from research.decision_policy.policy_engine import apply_policies
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_BACKTEST_DIR = Path(__file__).resolve().parents[2] / "signal_evaluation"
_PARQUET = _BACKTEST_DIR / "backtest_signals_dataset.parquet"
_CSV = _BACKTEST_DIR / "backtest_signals_dataset.csv"

# ── New sizing tiers (confidence → multiplier) ──────────────────────
# Confidence is NEVER a filter — only a size scaler.
SIZING_TIERS = [
    # (lower_inclusive, upper_exclusive, multiplier, label)
    (0.00, 0.40, 0.50, "very_low"),   # very low confidence → half size
    (0.40, 0.50, 0.50, "low"),        # low confidence      → half size
    (0.50, 0.60, 1.00, "medium"),     # medium confidence   → base size
    (0.60, 1.01, 1.50, "high"),       # high confidence     → aggressive
]

# ── Rank-only thresholds to test ────────────────────────────────────
RANK_GATE_PERCENTILES = [20, 30, 40]


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
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min()) if len(dd) > 0 else 0.0


def _sharpe_proxy(returns: np.ndarray) -> float | None:
    if len(returns) < 2:
        return None
    mu = float(returns.mean())
    sd = float(returns.std())
    return round(mu / sd, 4) if sd > 1e-9 else None


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    logger.info("Saved JSON → %s", path)


def _assign_size_tier(conf: float | None) -> tuple[float, str]:
    """Map a confidence score to (multiplier, tier_label)."""
    if conf is None or (isinstance(conf, float) and np.isnan(conf)):
        return 1.0, "unknown"
    for lo, hi, mult, label in SIZING_TIERS:
        if lo <= conf < hi:
            return mult, label
    return 1.0, "unknown"


# ═════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════

def _load_dataset() -> pd.DataFrame:
    if _PARQUET.exists():
        return pd.read_parquet(_PARQUET)
    if _CSV.exists():
        return pd.read_csv(_CSV)
    raise FileNotFoundError(f"Backtest dataset not found at {_PARQUET} or {_CSV}")


def _ensure_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ml_rank_score" in df.columns and "ml_confidence_score" in df.columns:
        return df
    from research.ml_models.ml_inference import infer_batch
    logger.info("ML columns missing — running batch inference …")
    return infer_batch(df)


def _prepare_dataset() -> pd.DataFrame:
    """Load dataset, ensure ML columns, apply existing policies for comparison."""
    df = _load_dataset()
    df = _ensure_ml_columns(df)
    quality_summary = label_quality_summary(df)

    # Apply existing policies (dual_threshold, rank_filter_30pct, etc.)
    df = apply_policies(df)
    df = apply_quality_label_view(df)
    df.attrs["label_quality_summary"] = quality_summary

    # Parse year
    if "signal_timestamp" in df.columns:
        ts = pd.to_datetime(df["signal_timestamp"], errors="coerce")
        df["_year"] = ts.dt.year

    # Coerce numerics
    for col in [PRIMARY_HIT_COL, PRIMARY_RETURN_COL, SECONDARY_RETURN_COL,
                MFE_COL, MAE_COL, SESSION_RETURN_COL,
                "ml_rank_score", "ml_confidence_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ═════════════════════════════════════════════════════════════════════
# Rank-gate + sizing-tier application
# ═════════════════════════════════════════════════════════════════════

def _apply_rank_gate_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each RANK_GATE_PERCENTILE, add columns:
      rank_gate_{pct}_decision  — ALLOW / BLOCK  (rank-only)
      rank_gate_{pct}_size_mult — from confidence sizing tiers
      rank_gate_{pct}_size_tier — label
    """
    rank = df["ml_rank_score"]
    conf = df["ml_confidence_score"]

    for pct in RANK_GATE_PERCENTILES:
        threshold = float(np.nanpercentile(rank.dropna(), pct))
        prefix = f"rank_gate_{pct}"

        # Decision: purely rank-based
        df[f"{prefix}_decision"] = np.where(rank >= threshold, DECISION_ALLOW, DECISION_BLOCK)
        df[f"{prefix}_threshold"] = threshold

        # Sizing: purely confidence-based (applied to ALL signals, including allowed)
        mults = []
        tiers = []
        for c in conf:
            m, t = _assign_size_tier(c)
            mults.append(m)
            tiers.append(t)
        df[f"{prefix}_size_mult"] = mults
        df[f"{prefix}_size_tier"] = tiers

    return df


# ═════════════════════════════════════════════════════════════════════
# Per-policy metric computation
# ═════════════════════════════════════════════════════════════════════

def _compute_metrics(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    *,
    size_mult: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Compute standard metrics for a signal subset.

    If *size_mult* is provided, also compute sized P&L.
    """
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "n": 0}

    hit = df.loc[mask, PRIMARY_HIT_COL]
    ret60 = df.loc[mask, PRIMARY_RETURN_COL]
    ret120 = df.loc[mask, SECONDARY_RETURN_COL] if SECONDARY_RETURN_COL in df.columns else pd.Series(dtype=float)
    mfe = df.loc[mask, MFE_COL] if MFE_COL in df.columns else pd.Series(dtype=float)
    mae = df.loc[mask, MAE_COL] if MAE_COL in df.columns else pd.Series(dtype=float)

    ret_arr = ret60.dropna().values
    cum = np.cumsum(ret_arr) if len(ret_arr) > 0 else np.array([0.0])

    row: dict[str, Any] = {
        "label": label,
        "n": n,
        "retention_pct": _rnd(n / len(df) * 100),
        "hit_rate": _rnd(_safe_mean(hit)),
        "avg_return_60m_bps": _rnd(_safe_mean(ret60)),
        "avg_return_120m_bps": _rnd(_safe_mean(ret120)),
        "avg_mfe_60m_bps": _rnd(_safe_mean(mfe)),
        "avg_mae_60m_bps": _rnd(_safe_mean(mae)),
        "cumulative_return_bps": _rnd(float(cum[-1])) if len(cum) else 0.0,
        "max_drawdown_bps": _rnd(_max_drawdown(cum)),
        "volatility_bps": _rnd(_safe_std(ret60)),
        "sharpe_proxy": _sharpe_proxy(ret_arr) if len(ret_arr) > 1 else None,
    }

    # Sized P&L
    if size_mult is not None:
        sized_ret = ret60[mask].dropna() * size_mult[mask].reindex(ret60[mask].dropna().index).fillna(1.0)
        sized_arr = sized_ret.values
        sized_cum = np.cumsum(sized_arr) if len(sized_arr) > 0 else np.array([0.0])
        row["sized_avg_return_bps"] = _rnd(float(sized_arr.mean())) if len(sized_arr) > 0 else None
        row["sized_cumulative_return_bps"] = _rnd(float(sized_cum[-1])) if len(sized_cum) > 0 else None
        row["sized_max_drawdown_bps"] = _rnd(_max_drawdown(sized_cum))
        row["sized_sharpe_proxy"] = _sharpe_proxy(sized_arr) if len(sized_arr) > 1 else None
        row["sizing_improvement_pct"] = _rnd(
            (float(sized_arr.mean()) - float(ret_arr.mean())) / max(abs(float(ret_arr.mean())), 1e-9) * 100
        ) if len(sized_arr) > 0 and len(ret_arr) > 0 else None

    return row


def _compute_tier_breakdown(
    df: pd.DataFrame,
    allow_mask: pd.Series,
    prefix: str,
) -> list[dict[str, Any]]:
    """Per-sizing-tier performance within ALLOW'd signals."""
    tier_col = f"{prefix}_size_tier"
    if tier_col not in df.columns:
        return []

    rows = []
    for _, _, mult, tier_label in SIZING_TIERS:
        tmask = allow_mask & (df[tier_col] == tier_label)
        n = int(tmask.sum())
        if n < 3:
            continue
        hit = df.loc[tmask, PRIMARY_HIT_COL]
        ret60 = df.loc[tmask, PRIMARY_RETURN_COL]
        rows.append({
            "tier": tier_label,
            "multiplier": mult,
            "n": n,
            "hit_rate": _rnd(_safe_mean(hit)),
            "avg_return_bps": _rnd(_safe_mean(ret60)),
        })
    return rows


# ═════════════════════════════════════════════════════════════════════
# Yearly stability
# ═════════════════════════════════════════════════════════════════════

def _yearly_stability(df: pd.DataFrame, mask: pd.Series, label: str) -> list[dict]:
    """Year-by-year hit rate and return for a subset."""
    if "_year" not in df.columns:
        return []
    hit = df[PRIMARY_HIT_COL]
    ret60 = df[PRIMARY_RETURN_COL]
    rows = []
    for yr in sorted(df.loc[mask, "_year"].dropna().unique()):
        ymask = mask & (df["_year"] == yr)
        n = int(ymask.sum())
        if n < 3:
            continue
        rows.append({
            "year": int(yr),
            "label": label,
            "n": n,
            "hit_rate": _rnd(_safe_mean(hit[ymask])),
            "avg_return_bps": _rnd(_safe_mean(ret60[ymask])),
        })
    return rows


# ═════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════

def _generate_charts(comparison: list[dict], yearly: list[dict], tier_data: dict) -> list[str]:
    """Generate comparison charts and return list of saved paths."""
    warnings.filterwarnings("ignore", category=UserWarning)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[str] = []

    # ── Chart 1: Performance comparison bar chart ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rank-Gate + Confidence-Sizing vs. Existing Policies", fontsize=14, fontweight="bold")

    labels = [r["label"] for r in comparison]
    x = np.arange(len(labels))
    w = 0.6

    # Hit rate
    ax = axes[0, 0]
    vals = [r.get("hit_rate") or 0 for r in comparison]
    colors = ["#2196F3" if "rank_gate" in r["label"] else "#FF9800" if "baseline" in r["label"] else "#4CAF50" for r in comparison]
    ax.bar(x, vals, w, color=colors, edgecolor="white")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Hit Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v + 0.005, f"{v:.2%}" if v < 1 else f"{v:.1f}%", ha="center", fontsize=7)

    # Avg return
    ax = axes[0, 1]
    vals = [r.get("avg_return_60m_bps") or 0 for r in comparison]
    ax.bar(x, vals, w, color=colors, edgecolor="white")
    ax.set_ylabel("Avg Return (bps)")
    ax.set_title("Average Return 60m")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)

    # Cumulative return
    ax = axes[1, 0]
    vals = [r.get("cumulative_return_bps") or 0 for r in comparison]
    ax.bar(x, vals, w, color=colors, edgecolor="white")
    ax.set_ylabel("Cumulative Return (bps)")
    ax.set_title("Cumulative Return")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v + 100, f"{v:,.0f}", ha="center", fontsize=7)

    # Max drawdown
    ax = axes[1, 1]
    vals = [abs(r.get("max_drawdown_bps") or 0) for r in comparison]
    ax.bar(x, vals, w, color=["#F44336"] * len(comparison), edgecolor="white", alpha=0.8)
    ax.set_ylabel("|Max Drawdown| (bps)")
    ax.set_title("Max Drawdown (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v + 10, f"{v:,.0f}", ha="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUTPUT_DIR / "performance_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # ── Chart 2: Sized vs unsized return comparison ──────────────────
    sized_rows = [r for r in comparison if r.get("sized_avg_return_bps") is not None]
    if sized_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_s = [r["label"] for r in sized_rows]
        x_s = np.arange(len(labels_s))
        unsized = [r.get("avg_return_60m_bps") or 0 for r in sized_rows]
        sized = [r.get("sized_avg_return_bps") or 0 for r in sized_rows]
        w2 = 0.35
        ax.bar(x_s - w2 / 2, unsized, w2, label="Unsized", color="#2196F3", edgecolor="white")
        ax.bar(x_s + w2 / 2, sized, w2, label="Confidence-Sized", color="#FF5722", edgecolor="white")
        ax.set_ylabel("Avg Return (bps)")
        ax.set_title("Unsized vs Confidence-Sized Avg Return")
        ax.set_xticks(x_s)
        ax.set_xticklabels(labels_s, rotation=25, ha="right", fontsize=9)
        ax.legend()
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        plt.tight_layout()
        p = OUTPUT_DIR / "sized_vs_unsized_return.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 3: Sizing tier breakdown (one per rank gate) ───────────
    for prefix, tiers in tier_data.items():
        if not tiers:
            continue
        fig, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Sizing Tier Breakdown — {prefix}", fontsize=12, fontweight="bold")

        tier_labels = [t["tier"] for t in tiers]
        x_t = np.arange(len(tier_labels))

        ax = axes2[0]
        hr = [t.get("hit_rate") or 0 for t in tiers]
        ax.bar(x_t, hr, 0.5, color=["#66BB6A", "#42A5F5", "#FFA726", "#EF5350"][:len(tiers)], edgecolor="white")
        ax.set_ylabel("Hit Rate")
        ax.set_title("Hit Rate by Sizing Tier")
        ax.set_xticks(x_t)
        ax.set_xticklabels(tier_labels, fontsize=9)
        for i, v in enumerate(hr):
            if v:
                ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)

        ax = axes2[1]
        ret = [t.get("avg_return_bps") or 0 for t in tiers]
        ax.bar(x_t, ret, 0.5, color=["#66BB6A", "#42A5F5", "#FFA726", "#EF5350"][:len(tiers)], edgecolor="white")
        ax.set_ylabel("Avg Return (bps)")
        ax.set_title("Avg Return by Sizing Tier")
        ax.set_xticks(x_t)
        ax.set_xticklabels(tier_labels, fontsize=9)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        for i, v in enumerate(ret):
            if v:
                ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        p = OUTPUT_DIR / f"tier_breakdown_{prefix}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 4: Yearly hit rate for new policies vs comparisons ─────
    if yearly:
        ydf = pd.DataFrame(yearly)
        unique_labels = ydf["label"].unique()

        fig, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Yearly Stability — Rank-Gate Policies", fontsize=12, fontweight="bold")

        cmap = plt.cm.Set2
        for idx, lbl in enumerate(unique_labels):
            sub = ydf[ydf["label"] == lbl].sort_values("year")
            axes3[0].plot(sub["year"], sub["hit_rate"], marker="o", label=lbl, color=cmap(idx / max(len(unique_labels), 1)))
            axes3[1].plot(sub["year"], sub["avg_return_bps"], marker="s", label=lbl, color=cmap(idx / max(len(unique_labels), 1)))

        axes3[0].set_title("Hit Rate by Year")
        axes3[0].set_ylabel("Hit Rate")
        axes3[0].legend(fontsize=7, loc="lower left")
        axes3[0].axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)

        axes3[1].set_title("Avg Return (bps) by Year")
        axes3[1].set_ylabel("Avg Return (bps)")
        axes3[1].legend(fontsize=7, loc="lower left")
        axes3[1].axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p = OUTPUT_DIR / "yearly_stability.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── Chart 5: Risk-return scatter — Sharpe vs Return ──────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in comparison:
        sp = r.get("sharpe_proxy")
        ret = r.get("avg_return_60m_bps")
        if sp is not None and ret is not None:
            c = "#2196F3" if "rank_gate" in r["label"] else "#FF9800" if "baseline" in r["label"] else "#4CAF50"
            ax.scatter(sp, ret, s=120, c=c, edgecolors="white", zorder=3)
            ax.annotate(r["label"], (sp, ret), fontsize=7, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Sharpe Proxy")
    ax.set_ylabel("Avg Return 60m (bps)")
    ax.set_title("Risk-Return Scatter")
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
    comparison: list[dict],
    tier_data: dict[str, list[dict]],
    yearly: list[dict],
    charts: list[str],
    verdict: dict[str, Any],
) -> str:
    """Produce a complete Markdown research report."""
    lines: list[str] = []
    _h = lines.append

    _h("# Rank-Gate + Confidence-Sizing Policy Evaluation")
    _h(f"\n**Generated:** {datetime.now().isoformat()}")
    _h("**Author:** Pramit Dutta  |  **Organization:** Quant Engines")
    _h("\n> **Thesis:** Remove confidence as a filtering gate. Use rank (GBT) "
       "as the sole quality filter. Repurpose confidence (LogReg) exclusively "
       "for position sizing — allocating more capital to higher-conviction signals.")
    _h("\n---\n")

    # ── Design ───────────────────────────────────────────────────────
    _h("## 1. Design\n")
    _h("### Rank-Gate (filtering)")
    _h("| Percentile | Logic |")
    _h("|------------|-------|")
    for pct in RANK_GATE_PERCENTILES:
        _h(f"| {pct}% | Block signals with rank score below the {pct}th percentile |")
    _h("\n### Confidence-Sizing (position sizing only)")
    _h("| Tier | Confidence Range | Size Multiplier |")
    _h("|------|-----------------|-----------------|")
    for lo, hi, mult, label in SIZING_TIERS:
        _h(f"| {label} | {lo:.2f}–{hi:.2f} | {mult}× |")
    _h("")

    # ── Comparison table ─────────────────────────────────────────────
    _h("## 2. Performance Comparison\n")
    _h("| Policy | N | Ret% | Hit Rate | Avg Ret (bps) | Cum Ret (bps) | Max DD (bps) | Sharpe | Sized Avg Ret | Sized Cum Ret | Sized DD | Sizing Δ% |")
    _h("|--------|---|------|----------|--------------|--------------|-------------|--------|--------------|--------------|---------|-----------|")
    for r in comparison:
        cum = r.get('cumulative_return_bps', '—')
        cum_s = f"{cum:,.0f}" if isinstance(cum, (int, float)) else str(cum)
        _h(f"| {r['label']} "
           f"| {r['n']:,} "
           f"| {r.get('retention_pct', '—')} "
           f"| {r.get('hit_rate', '—')} "
           f"| {r.get('avg_return_60m_bps', '—')} "
           f"| {cum_s} "
           f"| {r.get('max_drawdown_bps', '—')} "
           f"| {r.get('sharpe_proxy', '—')} "
           f"| {r.get('sized_avg_return_bps', '—')} "
           f"| {r.get('sized_cumulative_return_bps', '—')} "
           f"| {r.get('sized_max_drawdown_bps', '—')} "
           f"| {r.get('sizing_improvement_pct', '—')} |")
    _h("")

    # ── Tier breakdowns ──────────────────────────────────────────────
    _h("## 3. Sizing Tier Breakdown\n")
    for prefix, tiers in tier_data.items():
        _h(f"### {prefix}\n")
        if not tiers:
            _h("_No tier data (too few signals)._\n")
            continue
        _h("| Tier | Multiplier | N | Hit Rate | Avg Return (bps) |")
        _h("|------|-----------|---|----------|-----------------|")
        for t in tiers:
            _h(f"| {t['tier']} | {t['multiplier']}× | {t['n']} | {t.get('hit_rate', '—')} | {t.get('avg_return_bps', '—')} |")
        _h("")

    # ── Yearly stability ─────────────────────────────────────────────
    if yearly:
        _h("## 4. Yearly Stability\n")
        _h("| Year | Policy | N | Hit Rate | Avg Return (bps) |")
        _h("|------|--------|---|----------|-----------------|")
        for r in sorted(yearly, key=lambda x: (x["year"], x["label"])):
            _h(f"| {r['year']} | {r['label']} | {r['n']} | {r.get('hit_rate', '—')} | {r.get('avg_return_bps', '—')} |")
        _h("")

    # ── Verdict ──────────────────────────────────────────────────────
    _h("## 5. Verdict\n")
    _h(f"**Best unsized policy:** {verdict.get('best_unsized', '—')}")
    _h(f"**Best sized policy:** {verdict.get('best_sized', '—')}")
    _h(f"\n**Does rank-gate + confidence-sizing improve returns without increasing drawdown?**")
    _h(f"\n{verdict.get('conclusion', '—')}")
    _h("")

    # ── Charts ───────────────────────────────────────────────────────
    if charts:
        _h("## 6. Charts\n")
        for c in charts:
            fname = Path(c).name
            _h(f"![{fname}]({fname})")
        _h("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Verdict logic
# ═════════════════════════════════════════════════════════════════════

def _compute_verdict(comparison: list[dict]) -> dict[str, Any]:
    """Determine whether the new approach beats comparison baselines."""
    # Separate new policies from comparisons
    new_policies = [r for r in comparison if "rank_gate" in r["label"]]
    existing = [r for r in comparison if "rank_gate" not in r["label"] and "baseline" not in r["label"]]
    baselines = [r for r in comparison if "baseline" in r["label"]]

    # Best unsized: highest avg return among all
    valid = [r for r in comparison if r.get("avg_return_60m_bps") is not None and r["n"] > 0]
    best_unsized = max(valid, key=lambda r: r["avg_return_60m_bps"]) if valid else None

    # Best sized: highest sized avg return
    sized_valid = [r for r in comparison if r.get("sized_avg_return_bps") is not None]
    best_sized = max(sized_valid, key=lambda r: r["sized_avg_return_bps"]) if sized_valid else None

    # Compare best new vs best existing
    best_new = max(new_policies, key=lambda r: r.get("avg_return_60m_bps") or -999) if new_policies else None
    best_existing = max(existing, key=lambda r: r.get("avg_return_60m_bps") or -999) if existing else None

    parts: list[str] = []

    if best_new and best_existing:
        new_ret = best_new.get("avg_return_60m_bps") or 0
        exist_ret = best_existing.get("avg_return_60m_bps") or 0
        new_dd = abs(best_new.get("max_drawdown_bps") or 0)
        exist_dd = abs(best_existing.get("max_drawdown_bps") or 0)
        new_hr = best_new.get("hit_rate") or 0
        exist_hr = best_existing.get("hit_rate") or 0
        new_sharpe = best_new.get("sharpe_proxy") or 0
        exist_sharpe = best_existing.get("sharpe_proxy") or 0

        parts.append(f"**Best rank-gate policy: {best_new['label']}** — "
                     f"HR {new_hr:.2f}, avg return {new_ret:.1f} bps, "
                     f"max DD {best_new.get('max_drawdown_bps')} bps, Sharpe {new_sharpe}")
        parts.append(f"**Best existing policy: {best_existing['label']}** — "
                     f"HR {exist_hr:.2f}, avg return {exist_ret:.1f} bps, "
                     f"max DD {best_existing.get('max_drawdown_bps')} bps, Sharpe {exist_sharpe}")

        ret_better = new_ret >= exist_ret
        dd_better = new_dd <= exist_dd
        hr_better = new_hr >= exist_hr

        if ret_better and dd_better:
            parts.append("\n✅ **YES** — The rank-gate policy improves average return without increasing drawdown.")
        elif ret_better and not dd_better:
            parts.append(f"\n⚠️ **PARTIAL** — The rank-gate policy improves avg return (+{new_ret - exist_ret:.1f} bps) "
                         f"but max drawdown worsened ({new_dd:.0f} vs {exist_dd:.0f} bps).")
        elif not ret_better and dd_better:
            parts.append(f"\n⚠️ **PARTIAL** — The rank-gate policy reduces drawdown "
                         f"({new_dd:.0f} vs {exist_dd:.0f} bps) but avg return is lower "
                         f"({new_ret:.1f} vs {exist_ret:.1f} bps).")
        else:
            parts.append("\n❌ **NO** — The existing policy outperforms on both return and drawdown.")

        # Sizing impact
        if best_sized and "rank_gate" in best_sized["label"]:
            sized_ret = best_sized.get("sized_avg_return_bps") or 0
            unsized_ret = best_sized.get("avg_return_60m_bps") or 0
            delta = sized_ret - unsized_ret
            parts.append(f"\n**Confidence-sizing impact ({best_sized['label']}):** "
                         f"sized avg {sized_ret:.1f} bps vs unsized {unsized_ret:.1f} bps "
                         f"(Δ = {delta:+.1f} bps, {best_sized.get('sizing_improvement_pct', 0):+.1f}% improvement)")

    return {
        "best_unsized": best_unsized["label"] if best_unsized else None,
        "best_sized": best_sized["label"] if best_sized else None,
        "conclusion": "\n".join(parts),
    }


# ═════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════

def run_rank_gate_sizing_evaluation() -> dict[str, Any]:
    """
    Full evaluation pipeline:
    1. Load dataset + apply existing policies
    2. Apply rank-gate + sizing tiers for 20/30/40 percentiles
    3. Compute metrics for all policies
    4. Tier breakdowns
    5. Yearly stability
    6. Visualizations
    7. Verdict & report
    """
    print("=" * 70)
    print("  RANK-GATE + CONFIDENCE-SIZING EVALUATION")
    print("=" * 70)

    # 1. Load & prepare
    print("\n[1/7] Loading dataset …")
    df = _prepare_dataset()
    quality_summary = df.attrs.get("label_quality_summary", label_quality_summary(df))
    print(f"  Dataset: {len(df):,} signals, {len(df.columns)} columns")

    # 2. Apply rank-gate + sizing
    print("\n[2/7] Applying rank-gate + confidence-sizing policies …")
    df = _apply_rank_gate_sizing(df)

    for pct in RANK_GATE_PERCENTILES:
        prefix = f"rank_gate_{pct}"
        threshold = df[f"{prefix}_threshold"].iloc[0]
        n_allow = (df[f"{prefix}_decision"] == DECISION_ALLOW).sum()
        print(f"  {prefix}: threshold={threshold:.4f}, retained={n_allow:,} / {len(df):,} "
              f"({n_allow / len(df) * 100:.1f}%)")

    # 3. Compute metrics for all policies
    print("\n[3/7] Computing metrics …")
    comparison: list[dict] = []

    # Baseline
    baseline_mask = pd.Series(True, index=df.index)
    comparison.append(_compute_metrics(df, baseline_mask, "baseline_all"))

    # Existing policies for comparison
    for comp_policy in ["dual_threshold", "rank_filter_bottom_30pct"]:
        dcol = f"{comp_policy}_decision"
        if dcol in df.columns:
            mask = df[dcol] == DECISION_ALLOW
            comparison.append(_compute_metrics(df, mask, comp_policy))

    # New rank-gate + sizing policies
    for pct in RANK_GATE_PERCENTILES:
        prefix = f"rank_gate_{pct}"
        mask = df[f"{prefix}_decision"] == DECISION_ALLOW
        size_mult = df[f"{prefix}_size_mult"]
        comparison.append(_compute_metrics(df, mask, prefix, size_mult=size_mult))

    for r in comparison:
        hr = r.get("hit_rate", 0) or 0
        ret = r.get("avg_return_60m_bps", 0) or 0
        cum = r.get("cumulative_return_bps", 0) or 0
        dd = r.get("max_drawdown_bps", 0) or 0
        sp = r.get("sharpe_proxy", '—')
        print(f"  {r['label']:30s}  n={r['n']:5,}  HR={hr:5.2f}  "
              f"ret={ret:+7.1f}  cum={cum:+10,.0f}  DD={dd:+8,.0f}  Sharpe={sp}")

    # 4. Tier breakdowns
    print("\n[4/7] Sizing tier breakdowns …")
    tier_data: dict[str, list[dict]] = {}
    for pct in RANK_GATE_PERCENTILES:
        prefix = f"rank_gate_{pct}"
        mask = df[f"{prefix}_decision"] == DECISION_ALLOW
        tiers = _compute_tier_breakdown(df, mask, prefix)
        tier_data[prefix] = tiers
        for t in tiers:
            print(f"  {prefix} / {t['tier']:10s}  n={t['n']:5,}  HR={t.get('hit_rate', 0):.2f}  "
                  f"ret={t.get('avg_return_bps', 0):+.1f} bps")

    # 5. Yearly stability
    print("\n[5/7] Yearly stability …")
    yearly: list[dict] = []
    yearly.extend(_yearly_stability(df, baseline_mask, "baseline_all"))
    for comp_policy in ["dual_threshold", "rank_filter_bottom_30pct"]:
        dcol = f"{comp_policy}_decision"
        if dcol in df.columns:
            yearly.extend(_yearly_stability(df, df[dcol] == DECISION_ALLOW, comp_policy))
    for pct in RANK_GATE_PERCENTILES:
        prefix = f"rank_gate_{pct}"
        yearly.extend(_yearly_stability(df, df[f"{prefix}_decision"] == DECISION_ALLOW, prefix))
    print(f"  {len(yearly)} year × policy rows computed")

    # 6. Visualizations
    print("\n[6/7] Generating charts …")
    charts = _generate_charts(comparison, yearly, tier_data)
    print(f"  {len(charts)} charts saved")

    # 7. Verdict & report
    print("\n[7/7] Verdict & report …")
    verdict = _compute_verdict(comparison)
    report_md = _render_report(comparison, tier_data, yearly, charts, verdict)

    # Save all artifacts
    report_path = OUTPUT_DIR / "rank_gate_sizing_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  Report → {report_path}")

    results = {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_size": len(df),
        "label_quality_summary": quality_summary,
        "rank_gate_percentiles": RANK_GATE_PERCENTILES,
        "sizing_tiers": [{"lo": lo, "hi": hi, "mult": mult, "label": label}
                         for lo, hi, mult, label in SIZING_TIERS],
        "comparison": comparison,
        "tier_breakdowns": tier_data,
        "yearly_stability": yearly,
        "verdict": verdict,
        "charts": charts,
    }
    _save_json(results, OUTPUT_DIR / "rank_gate_sizing_results.json")

    # CSV export
    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUTPUT_DIR / "yearly_stability.csv", index=False)
    for prefix, tiers in tier_data.items():
        if tiers:
            pd.DataFrame(tiers).to_csv(OUTPUT_DIR / f"{prefix}_tier_breakdown.csv", index=False)

    print("\n" + "=" * 70)
    print("  COMPLETE — all artifacts saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)

    return results


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    run_rank_gate_sizing_evaluation()
