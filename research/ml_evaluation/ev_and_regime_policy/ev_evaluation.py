"""
EV Sizing Evaluation
=====================
Historical evaluation of EV-based sizing versus confidence-only sizing
and unsized baselines.  Produces comparison tables, bucket-level analysis,
yearly stability diagnostics, and charts.

Comparison axes
---------------
  1. Unsized baseline (all signals, multiplier=1.0)
  2. Rank-filtered baseline (top 70 % by rank, unsized)
  3. Rank-filtered + confidence-sized
  4. Rank-filtered + EV-sized

Author: Pramit Dutta
Organization: Quant Engines

RESEARCH ONLY — never imported by production engine paths.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.decision_policy.policy_config import (
    PRIMARY_HIT_COL,
    PRIMARY_RETURN_COL,
    SECONDARY_RETURN_COL,
    MFE_COL,
    MAE_COL,
    SESSION_RETURN_COL,
)

from research.ml_evaluation.ev_and_regime_policy.conditional_return_tables import (
    build_conditional_return_table,
    table_to_records,
)
from research.ml_evaluation.ev_and_regime_policy.ev_sizing_model import (
    assign_confidence_size,
    score_signals,
)
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = Path(__file__).resolve().parents[2] / "signal_evaluation"
_PARQUET = BACKTEST_DIR / "backtest_signals_dataset.parquet"
_CSV = BACKTEST_DIR / "backtest_signals_dataset.csv"

# Rank gate threshold — block bottom 30 % by rank (same as existing policy).
RANK_GATE_PERCENTILE = 30


# ── Helpers ──────────────────────────────────────────────────────────

def _rnd(v: float | None, d: int = 2) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return round(f, d) if np.isfinite(f) else None
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


def _sharpe_proxy(rets: np.ndarray) -> float | None:
    if len(rets) < 2:
        return None
    mu = float(rets.mean())
    sd = float(rets.std())
    return round(mu / sd, 4) if sd > 1e-9 else None


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    logger.info("Saved → %s", path)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved → %s", path)


# ── Data loading ─────────────────────────────────────────────────────

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


# ── Metrics computation ─────────────────────────────────────────────

def _metrics(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    *,
    size_col: str | None = None,
    total_n: int | None = None,
) -> dict[str, Any]:
    """Standard metrics block for a signal subset."""
    n = int(mask.sum())
    total = total_n or len(df)
    if n == 0:
        return {"label": label, "n": 0}

    hit = pd.to_numeric(df.loc[mask, PRIMARY_HIT_COL], errors="coerce")
    ret60 = pd.to_numeric(df.loc[mask, PRIMARY_RETURN_COL], errors="coerce")
    ret120 = pd.to_numeric(df.loc[mask, SECONDARY_RETURN_COL], errors="coerce") if SECONDARY_RETURN_COL in df.columns else pd.Series(dtype=float)
    mfe = pd.to_numeric(df.loc[mask, MFE_COL], errors="coerce") if MFE_COL in df.columns else pd.Series(dtype=float)
    mae = pd.to_numeric(df.loc[mask, MAE_COL], errors="coerce") if MAE_COL in df.columns else pd.Series(dtype=float)

    ret_arr = ret60.dropna().values
    cum = np.cumsum(ret_arr) if len(ret_arr) > 0 else np.array([0.0])

    row: dict[str, Any] = {
        "label": label,
        "n": n,
        "retention_pct": _rnd(n / total * 100),
        "hit_rate": _rnd(_safe_mean(hit)),
        "avg_return_60m_bps": _rnd(_safe_mean(ret60)),
        "avg_return_120m_bps": _rnd(_safe_mean(ret120)),
        "avg_mfe_60m_bps": _rnd(_safe_mean(mfe)),
        "avg_mae_60m_bps": _rnd(_safe_mean(mae)),
        "cumulative_return_bps": _rnd(float(cum[-1])),
        "max_drawdown_bps": _rnd(_max_drawdown(cum)),
        "volatility_bps": _rnd(_safe_std(ret60)),
        "sharpe_proxy": _sharpe_proxy(ret_arr),
    }

    # Sized return
    if size_col and size_col in df.columns:
        mult = pd.to_numeric(df.loc[mask, size_col], errors="coerce").fillna(1.0)
        sized_ret = ret60.dropna() * mult.reindex(ret60.dropna().index).fillna(1.0)
        sized_arr = sized_ret.values
        sized_cum = np.cumsum(sized_arr) if len(sized_arr) > 0 else np.array([0.0])
        row["sized_avg_return_bps"] = _rnd(float(sized_arr.mean())) if len(sized_arr) > 0 else None
        row["sized_cumulative_return_bps"] = _rnd(float(sized_cum[-1]))
        row["sized_max_drawdown_bps"] = _rnd(_max_drawdown(sized_cum))
        row["sized_sharpe_proxy"] = _sharpe_proxy(sized_arr)

    return row


def _yearly_stability(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    *,
    size_col: str | None = None,
) -> list[dict[str, Any]]:
    if "_year" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for yr in sorted(df.loc[mask, "_year"].dropna().unique()):
        ymask = mask & (df["_year"] == yr)
        n = int(ymask.sum())
        if n < 3:
            continue
        hit = pd.to_numeric(df.loc[ymask, PRIMARY_HIT_COL], errors="coerce")
        ret = pd.to_numeric(df.loc[ymask, PRIMARY_RETURN_COL], errors="coerce").dropna()
        entry: dict[str, Any] = {
            "year": int(yr), "label": label, "n": n,
            "hit_rate": _rnd(_safe_mean(hit)),
            "avg_return_bps": _rnd(float(ret.mean())) if len(ret) > 0 else None,
        }
        if size_col and size_col in df.columns:
            mult = pd.to_numeric(df.loc[ymask, size_col], errors="coerce").fillna(1.0)
            sized = ret * mult.reindex(ret.index).fillna(1.0)
            entry["sized_avg_return_bps"] = _rnd(float(sized.mean())) if len(sized) > 0 else None
        rows.append(entry)
    return rows


def _ev_bucket_breakdown(
    df: pd.DataFrame,
    mask: pd.Series,
) -> list[dict[str, Any]]:
    """Per-EV-bucket performance within allowed signals."""
    if "ev_bucket" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for bkt in df.loc[mask, "ev_bucket"].dropna().unique():
        bmask = mask & (df["ev_bucket"] == bkt)
        n = int(bmask.sum())
        if n < 3:
            continue
        hit = pd.to_numeric(df.loc[bmask, PRIMARY_HIT_COL], errors="coerce")
        ret = pd.to_numeric(df.loc[bmask, PRIMARY_RETURN_COL], errors="coerce")
        mult = pd.to_numeric(df.loc[bmask, "ev_size_multiplier"], errors="coerce").fillna(1.0)
        sized = ret.dropna() * mult.reindex(ret.dropna().index).fillna(1.0)
        rows.append({
            "ev_bucket": bkt,
            "n": n,
            "hit_rate": _rnd(_safe_mean(hit)),
            "avg_return_bps": _rnd(_safe_mean(ret)),
            "sized_avg_return_bps": _rnd(float(sized.mean())) if len(sized) > 0 else None,
            "avg_ev_raw": _rnd(float(df.loc[bmask, "ev_raw"].mean())),
            "avg_reliability": _rnd(float(df.loc[bmask, "ev_reliability_score"].mean())),
        })
    # Sort from negative → very_high
    order = {"negative": 0, "marginal": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
    rows.sort(key=lambda r: order.get(r["ev_bucket"], 99))
    return rows


# ── Charts ───────────────────────────────────────────────────────────

def _generate_charts(
    comparison: list[dict],
    yearly: list[dict],
    ev_buckets: list[dict],
) -> list[str]:
    warnings.filterwarnings("ignore", category=UserWarning)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[str] = []

    # ── 1. Performance comparison bar chart ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EV Sizing vs Confidence Sizing — Comparison", fontsize=14, fontweight="bold")

    labels = [r["label"] for r in comparison]
    x = np.arange(len(labels))
    w = 0.6

    def _color(lbl: str) -> str:
        if "ev_sized" in lbl:
            return "#9C27B0"
        if "confidence_sized" in lbl:
            return "#FF9800"
        if "rank_filtered" in lbl:
            return "#2196F3"
        return "#4CAF50"
    colors = [_color(r["label"]) for r in comparison]

    for idx, (metric, title, hline) in enumerate([
        ("hit_rate", "Hit Rate", 0.5),
        ("avg_return_60m_bps", "Avg Return 60m (bps)", 0),
        ("cumulative_return_bps", "Cumulative Return (bps)", None),
        ("sharpe_proxy", "Sharpe Proxy", 0),
    ]):
        ax = axes[idx // 2][idx % 2]
        vals = [r.get(metric) or 0 for r in comparison]
        ax.bar(x, vals, w, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=0.8, alpha=0.5)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v, f"{v:.2f}", ha="center", fontsize=7, va="bottom")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUTPUT_DIR / "ev_performance_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # ── 2. EV-sized vs confidence-sized bar comparison ───────────────
    sized_rows = [r for r in comparison if r.get("sized_avg_return_bps") is not None]
    if sized_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        sl = [r["label"] for r in sized_rows]
        xs = np.arange(len(sl))
        unsized = [r.get("avg_return_60m_bps") or 0 for r in sized_rows]
        sized = [r.get("sized_avg_return_bps") or 0 for r in sized_rows]
        w2 = 0.35
        ax.bar(xs - w2 / 2, unsized, w2, label="Unsized", color="#2196F3", edgecolor="white")
        ax.bar(xs + w2 / 2, sized, w2, label="Sized", color="#FF5722", edgecolor="white")
        ax.set_ylabel("Avg Return (bps)")
        ax.set_title("Unsized vs Sized Return")
        ax.set_xticks(xs)
        ax.set_xticklabels(sl, rotation=25, ha="right", fontsize=9)
        ax.legend()
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        plt.tight_layout()
        p = OUTPUT_DIR / "ev_sized_vs_unsized.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── 3. EV bucket performance ─────────────────────────────────────
    if ev_buckets:
        fig, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("EV Bucket Performance", fontsize=13, fontweight="bold")
        bl = [b["ev_bucket"] for b in ev_buckets]
        xb = np.arange(len(bl))
        palette = ["#F44336", "#FF9800", "#FFC107", "#4CAF50", "#2196F3", "#9C27B0"]

        # Hit rate
        ax = axes2[0]
        hr = [b.get("hit_rate") or 0 for b in ev_buckets]
        ax.bar(xb, hr, 0.6, color=palette[:len(bl)], edgecolor="white")
        ax.set_title("Hit Rate by EV Bucket")
        ax.set_ylabel("Hit Rate")
        ax.set_xticks(xb)
        ax.set_xticklabels(bl, fontsize=8)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)

        # Avg return
        ax = axes2[1]
        ret = [b.get("avg_return_bps") or 0 for b in ev_buckets]
        ax.bar(xb, ret, 0.6, color=palette[:len(bl)], edgecolor="white")
        ax.set_title("Avg Return (bps) by EV Bucket")
        ax.set_ylabel("bps")
        ax.set_xticks(xb)
        ax.set_xticklabels(bl, fontsize=8)
        ax.axhline(0, color="grey", ls="--", lw=0.8)

        # Sized return
        ax = axes2[2]
        sret = [b.get("sized_avg_return_bps") or 0 for b in ev_buckets]
        ax.bar(xb, sret, 0.6, color=palette[:len(bl)], edgecolor="white")
        ax.set_title("EV-Sized Avg Return (bps)")
        ax.set_ylabel("bps")
        ax.set_xticks(xb)
        ax.set_xticklabels(bl, fontsize=8)
        ax.axhline(0, color="grey", ls="--", lw=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        p = OUTPUT_DIR / "ev_bucket_performance.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── 4. Yearly stability ──────────────────────────────────────────
    if yearly:
        ydf = pd.DataFrame(yearly)
        unique_labels = ydf["label"].unique()
        fig, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Yearly Stability — EV Sizing vs Baselines", fontsize=12, fontweight="bold")
        cmap = plt.cm.Set2
        for idx2, lbl in enumerate(unique_labels):
            sub = ydf[ydf["label"] == lbl].sort_values("year")
            c2 = cmap(idx2 / max(len(unique_labels), 1))
            col = "sized_avg_return_bps" if "sized_avg_return_bps" in sub.columns and sub["sized_avg_return_bps"].notna().any() else "avg_return_bps"
            axes3[0].plot(sub["year"], sub["hit_rate"], marker="o", label=lbl, color=c2)
            axes3[1].plot(sub["year"], sub[col], marker="s", label=lbl, color=c2)
        axes3[0].set_title("Hit Rate by Year")
        axes3[0].set_ylabel("Hit Rate")
        axes3[0].legend(fontsize=7, loc="lower left")
        axes3[0].axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
        axes3[1].set_title("Avg Return (bps) by Year")
        axes3[1].set_ylabel("bps")
        axes3[1].legend(fontsize=7, loc="lower left")
        axes3[1].axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p = OUTPUT_DIR / "ev_yearly_stability.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


# ── Markdown report ──────────────────────────────────────────────────

def _generate_markdown(
    comparison: list[dict],
    yearly: list[dict],
    ev_buckets: list[dict],
    crt_meta: dict,
    charts: list[str],
) -> str:
    lines: list[str] = [
        "# EV-Based Sizing — Research Evaluation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "**Author:** Pramit Dutta  ",
        "**Organization:** Quant Engines  ",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "EV per signal is computed as:",
        "",
        "```",
        "EV = p_win × E[gain | win] − (1 − p_win) × |E[loss | loss]|",
        "```",
        "",
        "where p_win comes from the calibrated LogReg model and E[gain]/E[loss] are",
        "estimated from historical conditional return tables bucketed by",
        "(rank_bucket × confidence_bucket × gamma_regime).",
        "",
        f"- Conditional table cells: **{crt_meta.get('n_cells', 'N/A')}**",
        f"- Backed-off cells: **{crt_meta.get('n_backed_off', 'N/A')}**",
        f"- Smoothed cells: **{crt_meta.get('n_smoothed', 'N/A')}**",
        f"- Min samples per cell: **{crt_meta.get('min_samples', 'N/A')}**",
        f"- Smoothing weight: **{crt_meta.get('smoothing_weight', 'N/A')}**",
        "",
        "---",
        "",
        "## Performance Comparison",
        "",
    ]
    if comparison:
        hdr = "| Method | N | Retention | Hit Rate | Avg Ret 60m | Cum Ret | Max DD | Sharpe | Sized Avg Ret | Sized Cum |"
        sep = "|---|---|---|---|---|---|---|---|---|---|"
        lines.append(hdr)
        lines.append(sep)
        for r in comparison:
            lines.append(
                f"| {r.get('label', '')} | {r.get('n', '')} | {r.get('retention_pct', '')}% "
                f"| {r.get('hit_rate', '')} | {r.get('avg_return_60m_bps', '')} "
                f"| {r.get('cumulative_return_bps', '')} | {r.get('max_drawdown_bps', '')} "
                f"| {r.get('sharpe_proxy', '')} | {r.get('sized_avg_return_bps', '-')} "
                f"| {r.get('sized_cumulative_return_bps', '-')} |"
            )

    lines += ["", "---", "", "## EV Bucket Breakdown", ""]
    if ev_buckets:
        lines.append("| EV Bucket | N | Hit Rate | Avg Return | Sized Avg Return | Avg EV Raw | Reliability |")
        lines.append("|---|---|---|---|---|---|---|")
        for b in ev_buckets:
            lines.append(
                f"| {b['ev_bucket']} | {b['n']} | {b.get('hit_rate', '')} "
                f"| {b.get('avg_return_bps', '')} | {b.get('sized_avg_return_bps', '')} "
                f"| {b.get('avg_ev_raw', '')} | {b.get('avg_reliability', '')} |"
            )

    lines += ["", "---", "", "## Yearly Stability", ""]
    if yearly:
        ydf = pd.DataFrame(yearly)
        for lbl in ydf["label"].unique():
            sub = ydf[ydf["label"] == lbl].sort_values("year")
            lines.append(f"### {lbl}")
            lines.append("")
            lines.append("| Year | N | Hit Rate | Avg Return (bps) |")
            lines.append("|---|---|---|---|")
            for _, yr in sub.iterrows():
                lines.append(f"| {int(yr['year'])} | {yr['n']} | {yr.get('hit_rate', '')} | {yr.get('avg_return_bps', '')} |")
            lines.append("")

    if charts:
        lines += ["---", "", "## Charts", ""]
        for c in charts:
            fname = Path(c).name
            lines.append(f"![{fname}]({fname})")
            lines.append("")

    lines += [
        "---",
        "",
        "*RESEARCH ONLY — no production execution logic was modified.*",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════

def run_ev_sizing_evaluation() -> dict[str, Any]:
    """
    Run the full EV sizing evaluation pipeline.

    Returns a summary dict suitable for JSON serialization.
    """
    logger.info("=== EV Sizing Evaluation — start ===")

    # ── 1. Load and prepare dataset ──────────────────────────────────
    df = _load_dataset()
    df = _ensure_ml_columns(df)
    quality_summary = label_quality_summary(df)
    df = apply_quality_label_view(df)

    if "signal_timestamp" in df.columns:
        ts = pd.to_datetime(df["signal_timestamp"], errors="coerce")
        df["_year"] = ts.dt.year

    for col in [PRIMARY_HIT_COL, PRIMARY_RETURN_COL, SECONDARY_RETURN_COL,
                MFE_COL, MAE_COL, SESSION_RETURN_COL,
                "ml_rank_score", "ml_confidence_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    total_n = len(df)
    logger.info("Dataset loaded: %d signals", total_n)

    # ── 2. Build conditional return table ────────────────────────────
    crt = build_conditional_return_table(df)
    crt_records = table_to_records(crt)
    _save_json(crt_records, OUTPUT_DIR / "conditional_return_table.json")
    _save_csv(pd.DataFrame(crt_records), OUTPUT_DIR / "conditional_return_table.csv")

    # ── 3. Score signals with EV ─────────────────────────────────────
    df = score_signals(df, crt)

    # ── 4. Add confidence-only sizing for comparison ─────────────────
    df["confidence_size_multiplier"] = df["ml_confidence_score"].apply(assign_confidence_size)

    # ── 5. Define rank-gate mask ─────────────────────────────────────
    rank_threshold = float(np.nanpercentile(df["ml_rank_score"].dropna(), RANK_GATE_PERCENTILE))
    rank_mask = df["ml_rank_score"] >= rank_threshold
    all_mask = pd.Series(True, index=df.index)

    # ── 6. Compute comparison metrics ────────────────────────────────
    comparison: list[dict] = []

    # Baseline: all signals, unsized
    comparison.append(_metrics(df, all_mask, "baseline_all", total_n=total_n))

    # Rank-filtered, unsized
    comparison.append(_metrics(df, rank_mask, "rank_filtered_unsized", total_n=total_n))

    # Rank-filtered, confidence-sized
    comparison.append(_metrics(df, rank_mask, "rank_filtered_confidence_sized",
                               size_col="confidence_size_multiplier", total_n=total_n))

    # Rank-filtered, EV-sized
    comparison.append(_metrics(df, rank_mask, "rank_filtered_ev_sized",
                               size_col="ev_size_multiplier", total_n=total_n))

    # All signals, EV-sized (no rank gate)
    comparison.append(_metrics(df, all_mask, "all_ev_sized",
                               size_col="ev_size_multiplier", total_n=total_n))

    # ── 7. EV bucket breakdown ───────────────────────────────────────
    ev_buckets = _ev_bucket_breakdown(df, rank_mask)

    # ── 8. Yearly stability ──────────────────────────────────────────
    yearly: list[dict] = []
    yearly += _yearly_stability(df, all_mask, "baseline_all")
    yearly += _yearly_stability(df, rank_mask, "rank_filtered_unsized")
    yearly += _yearly_stability(df, rank_mask, "rank_filtered_confidence_sized",
                                size_col="confidence_size_multiplier")
    yearly += _yearly_stability(df, rank_mask, "rank_filtered_ev_sized",
                                size_col="ev_size_multiplier")

    # ── 9. Charts ────────────────────────────────────────────────────
    charts = _generate_charts(comparison, yearly, ev_buckets)

    # ── 10. Markdown report ──────────────────────────────────────────
    md = _generate_markdown(comparison, yearly, ev_buckets, crt.build_meta, charts)
    md_path = OUTPUT_DIR / "ev_sizing_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Saved → %s", md_path)

    # ── 11. JSON summary ─────────────────────────────────────────────
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_signals": total_n,
        "rank_gate_percentile": RANK_GATE_PERCENTILE,
        "rank_threshold": round(rank_threshold, 4),
        "label_quality_summary": quality_summary,
        "conditional_return_table": crt.build_meta,
        "comparison": comparison,
        "ev_bucket_breakdown": ev_buckets,
        "yearly_stability": yearly,
        "charts": [str(c) for c in charts],
    }
    _save_json(summary, OUTPUT_DIR / "ev_sizing_report.json")

    logger.info("=== EV Sizing Evaluation — complete ===")
    return summary
