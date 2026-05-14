"""
Regime-Switching Policy Evaluation
====================================
Backtests regime-switched policy assignment historically and compares
against static policies.  Includes a small interpretable search over
regime-to-policy mappings.

Outputs:
  • per-variant comparison table
  • regime-policy heatmap (regime × policy selection frequency)
  • yearly stability
  • static vs switched comparison charts
  • search-space summary with best variant

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
    DECISION_ALLOW,
    DECISION_BLOCK,
    PRIMARY_HIT_COL,
    PRIMARY_RETURN_COL,
    SECONDARY_RETURN_COL,
    MFE_COL,
    MAE_COL,
)

from research.ml_evaluation.ev_and_regime_policy.regime_switching_policy import (
    apply_regime_policy,
    generate_regime_map_variants,
    DEFAULT_REGIME_MAP,
    DEFAULT_FALLBACK_POLICY,
    GAMMA_COL,
    VOL_COL,
    MACRO_COL,
)
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = Path(__file__).resolve().parents[2] / "signal_evaluation"
_PARQUET = BACKTEST_DIR / "backtest_signals_dataset.parquet"
_CSV = BACKTEST_DIR / "backtest_signals_dataset.csv"


# ── Helpers (mirrored from ev_evaluation for self-containedness) ─────

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


def _sharpe_proxy(returns: np.ndarray) -> float | None:
    if len(returns) < 2:
        return None
    mu = float(returns.mean())
    sd = float(returns.std())
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


# ── Per-variant metrics ─────────────────────────────────────────────

def _variant_metrics(
    df: pd.DataFrame,
    label: str,
    *,
    decision_col: str = "regime_policy_decision",
    size_col: str | None = "regime_policy_size_mult",
    total_n: int | None = None,
) -> dict[str, Any]:
    """Compute metrics for signals that pass the variant's ALLOW filter."""
    mask = df[decision_col] == "ALLOW"
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

    # Sized return if available
    if size_col and size_col in df.columns:
        mult = pd.to_numeric(df.loc[mask, size_col], errors="coerce").fillna(1.0)
        sized_ret = ret60.dropna() * mult.reindex(ret60.dropna().index).fillna(1.0)
        sized_arr = sized_ret.values
        sized_cum = np.cumsum(sized_arr) if len(sized_arr) else np.array([0.0])
        row["sized_avg_return_bps"] = _rnd(float(sized_arr.mean())) if len(sized_arr) > 0 else None
        row["sized_cumulative_return_bps"] = _rnd(float(sized_cum[-1]))
        row["sized_max_drawdown_bps"] = _rnd(_max_drawdown(sized_cum))
        row["sized_sharpe_proxy"] = _sharpe_proxy(sized_arr)

    return row


def _yearly_metrics(
    df: pd.DataFrame,
    label: str,
    *,
    decision_col: str = "regime_policy_decision",
) -> list[dict[str, Any]]:
    if "_year" not in df.columns:
        return []
    mask = df[decision_col] == "ALLOW"
    rows: list[dict[str, Any]] = []
    for yr in sorted(df.loc[mask, "_year"].dropna().unique()):
        ymask = mask & (df["_year"] == yr)
        n = int(ymask.sum())
        if n < 3:
            continue
        hit = pd.to_numeric(df.loc[ymask, PRIMARY_HIT_COL], errors="coerce")
        ret = pd.to_numeric(df.loc[ymask, PRIMARY_RETURN_COL], errors="coerce").dropna()
        rows.append({
            "year": int(yr), "label": label, "n": n,
            "hit_rate": _rnd(_safe_mean(hit)),
            "avg_return_bps": _rnd(float(ret.mean())) if len(ret) > 0 else None,
        })
    return rows


def _regime_policy_heatmap(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Cross-tab of regime × selected policy (frequency + hit rate)."""
    if "selected_regime_policy" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for regime_col_name, regime_col in [(GAMMA_COL, GAMMA_COL), (VOL_COL, VOL_COL), (MACRO_COL, MACRO_COL)]:
        if regime_col not in df.columns:
            continue
        for regime_val in df[regime_col].dropna().unique():
            for pol in df["selected_regime_policy"].dropna().unique():
                mask = (df[regime_col] == regime_val) & (df["selected_regime_policy"] == pol)
                n = int(mask.sum())
                if n < 1:
                    continue
                hit = pd.to_numeric(df.loc[mask, PRIMARY_HIT_COL], errors="coerce")
                ret = pd.to_numeric(df.loc[mask, PRIMARY_RETURN_COL], errors="coerce")
                rows.append({
                    "regime_dimension": regime_col_name,
                    "regime_value": regime_val,
                    "policy": pol,
                    "n": n,
                    "hit_rate": _rnd(_safe_mean(hit)),
                    "avg_return_bps": _rnd(_safe_mean(ret)),
                })
    return rows


# ── Charts ───────────────────────────────────────────────────────────

def _generate_charts(
    search_results: list[dict],
    yearly: list[dict],
    heatmap_data: list[dict],
) -> list[str]:
    warnings.filterwarnings("ignore", category=UserWarning)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[str] = []

    # ── 1. Search comparison bar chart ───────────────────────────────
    if search_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Regime-Switching Policy Search — Comparison", fontsize=14, fontweight="bold")

        labels = [r["label"] for r in search_results]
        x = np.arange(len(labels))
        w = 0.6

        def _color(lbl: str) -> str:
            if "static" in lbl:
                return "#FF9800"
            if "ev" in lbl:
                return "#9C27B0"
            return "#2196F3"
        colors = [_color(l) for l in labels]

        for idx, (metric, title, hline) in enumerate([
            ("hit_rate", "Hit Rate", 0.5),
            ("avg_return_60m_bps", "Avg Return 60m (bps)", 0),
            ("cumulative_return_bps", "Cumulative Return (bps)", None),
            ("sharpe_proxy", "Sharpe Proxy", 0),
        ]):
            ax = axes[idx // 2][idx % 2]
            vals = [r.get(metric) or 0 for r in search_results]
            ax.bar(x, vals, w, color=colors, edgecolor="white")
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            if hline is not None:
                ax.axhline(hline, color="grey", ls="--", lw=0.8, alpha=0.5)
            for i, v in enumerate(vals):
                if v:
                    ax.text(i, v, f"{v:.2f}", ha="center", fontsize=6, va="bottom")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p = OUTPUT_DIR / "regime_policy_search_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # ── 2. Regime-policy heatmap ─────────────────────────────────────
    if heatmap_data:
        hdf = pd.DataFrame(heatmap_data)
        for dim in hdf["regime_dimension"].unique():
            sub = hdf[hdf["regime_dimension"] == dim]
            pivot = sub.pivot_table(index="regime_value", columns="policy",
                                     values="hit_rate", aggfunc="first")
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, max(3, len(pivot) * 0.6 + 1)))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=0.8)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"Hit Rate Heatmap — {dim} × Policy", fontweight="bold")
            fig.colorbar(im, ax=ax, label="Hit Rate")
            plt.tight_layout()
            p = OUTPUT_DIR / f"regime_heatmap_{dim}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(p))

    # ── 3. Yearly stability ──────────────────────────────────────────
    if yearly:
        ydf = pd.DataFrame(yearly)
        unique_labels = ydf["label"].unique()
        fig, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Yearly Stability — Regime-Switching Variants", fontsize=12, fontweight="bold")
        cmap = plt.cm.tab10
        for idx2, lbl in enumerate(unique_labels):
            sub = ydf[ydf["label"] == lbl].sort_values("year")
            c = cmap(idx2 / max(len(unique_labels), 1))
            axes2[0].plot(sub["year"], sub["hit_rate"], marker="o", label=lbl, color=c, linewidth=1)
            axes2[1].plot(sub["year"], sub["avg_return_bps"], marker="s", label=lbl, color=c, linewidth=1)
        axes2[0].set_title("Hit Rate by Year")
        axes2[0].set_ylabel("Hit Rate")
        axes2[0].legend(fontsize=6, loc="lower left")
        axes2[0].axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5)
        axes2[1].set_title("Avg Return (bps) by Year")
        axes2[1].set_ylabel("bps")
        axes2[1].legend(fontsize=6, loc="lower left")
        axes2[1].axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p = OUTPUT_DIR / "regime_yearly_stability.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


# ── Markdown report ──────────────────────────────────────────────────

def _generate_markdown(
    search_results: list[dict],
    yearly: list[dict],
    heatmap_data: list[dict],
    best_variant: dict | None,
    charts: list[str],
) -> str:
    lines: list[str] = [
        "# Regime-Switching Policy — Research Evaluation Report",
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
        "The regime-switching layer selects which decision policy to apply to each",
        "signal based on the prevailing market regime (gamma, volatility, macro).",
        "A small, interpretable search space of regime-to-policy mappings is evaluated",
        "against static baselines.",
        "",
        "---",
        "",
        "## Search Results",
        "",
    ]
    if search_results:
        hdr = "| Variant | N | Retention | Hit Rate | Avg Ret 60m | Cum Ret | Max DD | Sharpe |"
        sep = "|---|---|---|---|---|---|---|---|"
        lines.append(hdr)
        lines.append(sep)
        for r in search_results:
            lines.append(
                f"| {r.get('label', '')} | {r.get('n', '')} | {r.get('retention_pct', '')}% "
                f"| {r.get('hit_rate', '')} | {r.get('avg_return_60m_bps', '')} "
                f"| {r.get('cumulative_return_bps', '')} | {r.get('max_drawdown_bps', '')} "
                f"| {r.get('sharpe_proxy', '')} |"
            )

    if best_variant:
        lines += [
            "", "---", "",
            "## Best Variant",
            "",
            f"**{best_variant.get('label', 'N/A')}**",
            "",
            f"- Hit Rate: **{best_variant.get('hit_rate', 'N/A')}**",
            f"- Avg Return 60m: **{best_variant.get('avg_return_60m_bps', 'N/A')} bps**",
            f"- Sharpe Proxy: **{best_variant.get('sharpe_proxy', 'N/A')}**",
            f"- Cumulative Return: **{best_variant.get('cumulative_return_bps', 'N/A')} bps**",
            "",
        ]

    # Regime heatmap summary
    if heatmap_data:
        lines += ["---", "", "## Regime × Policy Heatmap", ""]
        hdf = pd.DataFrame(heatmap_data)
        for dim in hdf["regime_dimension"].unique():
            sub = hdf[hdf["regime_dimension"] == dim]
            lines.append(f"### {dim}")
            lines.append("")
            lines.append("| Regime | Policy | N | Hit Rate | Avg Return |")
            lines.append("|---|---|---|---|---|")
            for _, row in sub.iterrows():
                lines.append(
                    f"| {row['regime_value']} | {row['policy']} | {row['n']} "
                    f"| {row.get('hit_rate', '')} | {row.get('avg_return_bps', '')} |"
                )
            lines.append("")

    # Yearly stability
    if yearly:
        lines += ["---", "", "## Yearly Stability", ""]
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

    lines += ["---", "", "*RESEARCH ONLY — no production execution logic was modified.*"]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════

def run_regime_policy_evaluation(
    df_preloaded: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run the full regime-switching policy evaluation.

    Parameters
    ----------
    df_preloaded : If provided, skip loading and use this DataFrame
                   (must already have ML columns and EV columns).

    Returns
    -------
    Summary dict suitable for JSON serialization.
    """
    logger.info("=== Regime-Switching Policy Evaluation — start ===")

    # ── 1. Load / prepare ────────────────────────────────────────────
    if df_preloaded is not None:
        df_base = df_preloaded.copy()
    else:
        df_base = _load_dataset()
        df_base = _ensure_ml_columns(df_base)
    quality_summary = label_quality_summary(df_base)
    df_base = apply_quality_label_view(df_base)

    if "signal_timestamp" in df_base.columns:
        ts = pd.to_datetime(df_base["signal_timestamp"], errors="coerce")
        df_base["_year"] = ts.dt.year

    for col in [PRIMARY_HIT_COL, PRIMARY_RETURN_COL, SECONDARY_RETURN_COL,
                MFE_COL, MAE_COL, "ml_rank_score", "ml_confidence_score"]:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

    total_n = len(df_base)
    logger.info("Dataset: %d signals", total_n)

    # ── 2. Search over variants ──────────────────────────────────────
    variants = generate_regime_map_variants()
    search_results: list[dict] = []
    all_yearly: list[dict] = []

    for var_label, var_map in variants:
        logger.info("Evaluating variant: %s", var_label)
        df_v = apply_regime_policy(
            df_base, regime_map=var_map, fallback=DEFAULT_FALLBACK_POLICY,
        )
        metrics = _variant_metrics(df_v, var_label, total_n=total_n)
        search_results.append(metrics)
        all_yearly += _yearly_metrics(df_v, var_label)

    # ── 3. Identify best variant ─────────────────────────────────────
    scored = [r for r in search_results if r.get("sharpe_proxy") is not None and r.get("n", 0) > 0]
    best_variant = max(scored, key=lambda r: r["sharpe_proxy"]) if scored else None

    # ── 4. Regime-policy heatmap (using default mapping) ─────────────
    df_default = apply_regime_policy(df_base, regime_map=DEFAULT_REGIME_MAP)
    heatmap_data = _regime_policy_heatmap(df_default)

    # ── 5. Charts ────────────────────────────────────────────────────
    charts = _generate_charts(search_results, all_yearly, heatmap_data)

    # ── 6. Markdown report ───────────────────────────────────────────
    md = _generate_markdown(search_results, all_yearly, heatmap_data, best_variant, charts)
    md_path = OUTPUT_DIR / "regime_switching_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Saved → %s", md_path)

    # ── 7. CSV exports ───────────────────────────────────────────────
    _save_csv(pd.DataFrame(search_results), OUTPUT_DIR / "regime_policy_comparison.csv")
    if heatmap_data:
        _save_csv(pd.DataFrame(heatmap_data), OUTPUT_DIR / "regime_policy_heatmap.csv")

    # ── 8. JSON summary ──────────────────────────────────────────────
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_signals": total_n,
        "label_quality_summary": quality_summary,
        "n_variants_tested": len(variants),
        "search_results": search_results,
        "best_variant": best_variant,
        "regime_policy_heatmap": heatmap_data,
        "yearly_stability": all_yearly,
        "charts": [str(c) for c in charts],
    }
    _save_json(summary, OUTPUT_DIR / "regime_switching_report.json")

    logger.info("=== Regime-Switching Policy Evaluation — complete ===")
    return summary
