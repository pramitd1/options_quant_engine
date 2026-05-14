"""
Decision Policy Evaluation
============================
Computes per-policy performance metrics, regime-conditional analysis,
yearly stability, and generates comparison reports (JSON, Markdown, CSV).

Entry point
-----------
``run_decision_policy_evaluation()``
    Loads the backtest dataset, runs ML inference (if needed), applies all
    policies, computes metrics, and writes reports to ``research/ml_evaluation/``.

Author: Pramit Dutta
Organization: Quant Engines

RESEARCH ONLY — never imported by production engine paths.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.decision_policy.policy_config import (
    ALL_POLICIES,
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_DOWNGRADE,
    DECISION_POLICY_COMPARISON_CSV,
    DECISION_POLICY_COMPARISON_JSON,
    DECISION_POLICY_COMPARISON_MD,
    DECISION_POLICY_EVAL_DIR,
    DECISION_POLICY_REPORT_JSON,
    DECISION_POLICY_REPORT_MD,
    MAE_COL,
    MFE_COL,
    PRIMARY_HIT_COL,
    PRIMARY_RETURN_COL,
    REGIME_COLUMNS,
    SECONDARY_RETURN_COL,
    SESSION_RETURN_COL,
    SIZING_TIERS,
)
from research.decision_policy.policy_engine import apply_policies
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary

logger = logging.getLogger(__name__)

# ── Dataset loader (reuses ML evaluation runner's logic) ─────────────

_BACKTEST_DIR = Path(__file__).resolve().parents[1] / "signal_evaluation"
_PARQUET = _BACKTEST_DIR / "backtest_signals_dataset.parquet"
_CSV = _BACKTEST_DIR / "backtest_signals_dataset.csv"


def _load_dataset() -> pd.DataFrame:
    if _PARQUET.exists():
        return pd.read_parquet(_PARQUET)
    if _CSV.exists():
        return pd.read_csv(_CSV)
    raise FileNotFoundError(f"Backtest dataset not found at {_PARQUET} or {_CSV}")


def _ensure_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Run ML batch inference if the dataset lacks ML columns."""
    if "ml_rank_score" in df.columns and "ml_confidence_score" in df.columns:
        return df
    from research.ml_models.ml_inference import infer_batch
    logger.info("ML columns missing — running batch inference …")
    return infer_batch(df)


# ═════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════

def run_decision_policy_evaluation() -> dict[str, Any]:
    """
    Full evaluation pipeline:

    1. Load + extend dataset with ML scores
    2. Apply all decision policies
    3. Compute per-policy metrics
    4. Regime-conditional analysis
    5. Yearly stability
    6. Cross-methodology comparison (baseline, research_dual_model, each policy)
    7. Write reports (JSON, Markdown, CSV)

    Returns the master report dict.
    """
    DECISION_POLICY_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    df = _load_dataset()
    df = _ensure_ml_columns(df)
    logger.info("Dataset loaded: %d signals, %d columns", len(df), len(df.columns))

    # 2. Apply policies
    annotated = apply_policies(df)
    evaluation_frame = apply_quality_label_view(annotated)

    # 3. Per-policy metrics
    policy_metrics = _compute_all_policy_metrics(evaluation_frame)

    # 4. Regime analysis
    regime_analysis = _compute_regime_analysis(evaluation_frame)

    # 5. Yearly stability
    yearly = _compute_yearly_stability(evaluation_frame)

    # 6. Cross-methodology comparison
    comparison = _build_cross_methodology_comparison(evaluation_frame, policy_metrics)

    # 7. Master report
    report: dict[str, Any] = {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_size": len(df),
        "label_quality_summary": label_quality_summary(df),
        "policy_metrics": policy_metrics,
        "regime_analysis": regime_analysis,
        "yearly_stability": yearly,
        "cross_methodology_comparison": comparison,
    }

    # 8. Persist
    _save_json(report, DECISION_POLICY_REPORT_JSON)
    _save_json(comparison, DECISION_POLICY_COMPARISON_JSON)

    md = _render_markdown(report)
    DECISION_POLICY_REPORT_MD.write_text(md, encoding="utf-8")
    logger.info("Saved markdown report → %s", DECISION_POLICY_REPORT_MD)

    comp_md = _render_comparison_markdown(comparison)
    DECISION_POLICY_COMPARISON_MD.write_text(comp_md, encoding="utf-8")
    logger.info("Saved comparison markdown → %s", DECISION_POLICY_COMPARISON_MD)

    _save_comparison_csv(comparison)

    return report


# ═════════════════════════════════════════════════════════════════════
# Metric computation
# ═════════════════════════════════════════════════════════════════════

def _compute_all_policy_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute metrics for every policy present in the DataFrame."""
    results: dict[str, Any] = {}

    # Discover policy columns
    decision_cols = [c for c in df.columns if c.endswith("_decision")]

    for dcol in decision_cols:
        policy_name = dcol.removesuffix("_decision")
        results[policy_name] = _metrics_for_policy(df, policy_name)

    return results


def _metrics_for_policy(df: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    """Per-decision-class metrics for one policy."""
    dcol = f"{policy_name}_decision"
    mult_col = f"{policy_name}_size_mult"

    # Coerce outcome columns once
    hit = pd.to_numeric(df.get(PRIMARY_HIT_COL), errors="coerce")
    ret60 = pd.to_numeric(df.get(PRIMARY_RETURN_COL), errors="coerce")
    ret120 = pd.to_numeric(df.get(SECONDARY_RETURN_COL), errors="coerce")
    mfe = pd.to_numeric(df.get(MFE_COL), errors="coerce")
    mae = pd.to_numeric(df.get(MAE_COL), errors="coerce")
    session_ret = pd.to_numeric(df.get(SESSION_RETURN_COL), errors="coerce")

    per_class: dict[str, Any] = {}
    for cls in [DECISION_ALLOW, DECISION_BLOCK, DECISION_DOWNGRADE]:
        mask = df[dcol] == cls
        n = int(mask.sum())
        if n == 0:
            per_class[cls] = {"n": 0}
            continue
        per_class[cls] = {
            "n": n,
            "hit_rate_60m": _rnd(_safe_mean(hit[mask])),
            "avg_return_60m_bps": _rnd(_safe_mean(ret60[mask])),
            "avg_return_120m_bps": _rnd(_safe_mean(ret120[mask])),
            "avg_mfe_60m_bps": _rnd(_safe_mean(mfe[mask])),
            "avg_mae_60m_bps": _rnd(_safe_mean(mae[mask])),
            "avg_return_session_bps": _rnd(_safe_mean(session_ret[mask])),
        }

    # Allowed-only aggregate (the "tradeable" signal pool)
    allow_mask = df[dcol] == DECISION_ALLOW
    blocked_mask = df[dcol] == DECISION_BLOCK

    # Sizing P&L (if multiplier column exists)
    sizing_pnl: dict[str, Any] = {}
    if mult_col in df.columns and ret60 is not None:
        scored = df[allow_mask & ret60.notna()].copy()
        if not scored.empty:
            base_returns = ret60[scored.index].values
            sized_returns = base_returns * scored[mult_col].values
            sizing_pnl = {
                "n": len(scored),
                "baseline_avg_bps": _rnd(float(base_returns.mean())),
                "sized_avg_bps": _rnd(float(sized_returns.mean())),
                "baseline_total_bps": _rnd(float(base_returns.sum())),
                "sized_total_bps": _rnd(float(sized_returns.sum())),
                "sizing_improvement_pct": _rnd(
                    (float(sized_returns.mean()) - float(base_returns.mean()))
                    / max(abs(float(base_returns.mean())), 1e-9) * 100
                ),
            }

    # Drawdown proxy (cumulative return based)
    dd_proxy: dict[str, Any] = {}
    if allow_mask.any() and ret60 is not None:
        allowed_ret = ret60[allow_mask].dropna().values
        if len(allowed_ret) > 0:
            cum = np.cumsum(allowed_ret)
            dd_proxy = {
                "total_return_bps": _rnd(float(cum[-1])),
                "max_drawdown_bps": _rnd(_max_drawdown(cum)),
            }

    # Score-bucket performance (quintile of ml_rank_score among ALLOW)
    bucket_perf = _score_bucket_performance(df, allow_mask, hit, ret60)

    return {
        "per_decision_class": per_class,
        "sizing_simulation": sizing_pnl,
        "drawdown_proxy": dd_proxy,
        "score_bucket_performance": bucket_perf,
        "allowed_pct": _rnd(float(allow_mask.sum()) / max(len(df), 1) * 100),
        "blocked_pct": _rnd(float(blocked_mask.sum()) / max(len(df), 1) * 100),
    }


def _score_bucket_performance(
    df: pd.DataFrame,
    allow_mask: pd.Series,
    hit: pd.Series,
    ret60: pd.Series,
) -> list[dict[str, Any]]:
    """Quintile performance among ALLOW'd signals."""
    subset = df[allow_mask].copy()
    rank = pd.to_numeric(subset.get("ml_rank_score"), errors="coerce")
    if rank.dropna().empty:
        return []

    subset = subset.assign(_rank=rank)
    subset = subset[subset["_rank"].notna()]
    try:
        subset["_qbin"] = pd.qcut(subset["_rank"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    except ValueError:
        return []

    buckets: list[dict[str, Any]] = []
    hit_s = pd.to_numeric(subset.get(PRIMARY_HIT_COL), errors="coerce")
    ret_s = pd.to_numeric(subset.get(PRIMARY_RETURN_COL), errors="coerce")
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        mask = subset["_qbin"] == q
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket": q,
            "n": n,
            "hit_rate_60m": _rnd(_safe_mean(hit_s[mask])),
            "avg_return_60m_bps": _rnd(_safe_mean(ret_s[mask])),
        })
    return buckets


# ═════════════════════════════════════════════════════════════════════
# Regime-conditional analysis
# ═════════════════════════════════════════════════════════════════════

def _compute_regime_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Per-regime hit rate and return for ALLOW'd signals under each policy."""
    results: dict[str, Any] = {}
    decision_cols = [c for c in df.columns if c.endswith("_decision")]

    hit = pd.to_numeric(df.get(PRIMARY_HIT_COL), errors="coerce")
    ret60 = pd.to_numeric(df.get(PRIMARY_RETURN_COL), errors="coerce")

    for dcol in decision_cols:
        policy_name = dcol.removesuffix("_decision")
        allow_mask = df[dcol] == DECISION_ALLOW
        policy_regimes: dict[str, Any] = {}

        for regime_label, col_name in REGIME_COLUMNS.items():
            if col_name not in df.columns:
                continue
            regime_vals = df.loc[allow_mask, col_name].dropna().unique()
            regime_data: list[dict[str, Any]] = []
            for val in sorted(regime_vals, key=str):
                rmask = allow_mask & (df[col_name] == val)
                n = int(rmask.sum())
                if n < 5:
                    continue
                regime_data.append({
                    "regime_value": str(val),
                    "n": n,
                    "hit_rate_60m": _rnd(_safe_mean(hit[rmask])),
                    "avg_return_60m_bps": _rnd(_safe_mean(ret60[rmask])),
                })
            policy_regimes[regime_label] = regime_data

        results[policy_name] = policy_regimes

    return results


# ═════════════════════════════════════════════════════════════════════
# Yearly stability
# ═════════════════════════════════════════════════════════════════════

def _compute_yearly_stability(df: pd.DataFrame) -> dict[str, Any]:
    """Per-year hit rate and return for each policy's ALLOW population."""
    if "signal_timestamp" not in df.columns:
        return {}

    ts = pd.to_datetime(df["signal_timestamp"], errors="coerce")
    df = df.assign(_year=ts.dt.year)

    hit = pd.to_numeric(df.get(PRIMARY_HIT_COL), errors="coerce")
    ret60 = pd.to_numeric(df.get(PRIMARY_RETURN_COL), errors="coerce")

    results: dict[str, Any] = {}
    decision_cols = [c for c in df.columns if c.endswith("_decision")]

    for dcol in decision_cols:
        policy_name = dcol.removesuffix("_decision")
        allow = df[dcol] == DECISION_ALLOW
        years_data: list[dict[str, Any]] = []
        for yr in sorted(df.loc[allow, "_year"].dropna().unique()):
            ymask = allow & (df["_year"] == yr)
            n = int(ymask.sum())
            if n < 3:
                continue
            years_data.append({
                "year": int(yr),
                "n": n,
                "hit_rate_60m": _rnd(_safe_mean(hit[ymask])),
                "avg_return_60m_bps": _rnd(_safe_mean(ret60[ymask])),
            })
        results[policy_name] = years_data

    return results


# ═════════════════════════════════════════════════════════════════════
# Cross-methodology comparison
# ═════════════════════════════════════════════════════════════════════

def _build_cross_methodology_comparison(
    df: pd.DataFrame,
    policy_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare baseline (all signals), research_dual_model, and each policy.
    Returns a dict keyed by methodology name with standardised metrics.
    """
    hit = pd.to_numeric(df.get(PRIMARY_HIT_COL), errors="coerce")
    ret60 = pd.to_numeric(df.get(PRIMARY_RETURN_COL), errors="coerce")
    ret120 = pd.to_numeric(df.get(SECONDARY_RETURN_COL), errors="coerce")
    mfe = pd.to_numeric(df.get(MFE_COL), errors="coerce")
    mae = pd.to_numeric(df.get(MAE_COL), errors="coerce")

    methods: dict[str, Any] = {}

    # Baseline — all signals
    methods["baseline_all_signals"] = _method_row(
        label="Baseline (all signals)",
        n=len(df),
        hit=hit,
        ret60=ret60,
        ret120=ret120,
        mfe=mfe,
        mae=mae,
        mask=pd.Series(True, index=df.index),
    )

    # Baseline — TRADE-only
    trade_mask = df.get("trade_status") == "TRADE"
    if trade_mask is not None and trade_mask.any():
        methods["baseline_trade_only"] = _method_row(
            label="Baseline (TRADE only)",
            n=int(trade_mask.sum()),
            hit=hit,
            ret60=ret60,
            ret120=ret120,
            mfe=mfe,
            mae=mae,
            mask=trade_mask,
        )

    # Research dual-model: filter signals where ml_confidence_score > 0.50
    conf = pd.to_numeric(df.get("ml_confidence_score"), errors="coerce")
    if conf is not None and conf.notna().any():
        rdm_mask = conf >= 0.50
        methods["research_dual_model"] = _method_row(
            label="Research Dual-Model (conf ≥ 0.50)",
            n=int(rdm_mask.sum()),
            hit=hit,
            ret60=ret60,
            ret120=ret120,
            mfe=mfe,
            mae=mae,
            mask=rdm_mask,
        )

    # Each policy — ALLOW subset
    for policy_name, pm in policy_metrics.items():
        dcol = f"{policy_name}_decision"
        if dcol not in df.columns:
            continue
        allow_mask = df[dcol] == DECISION_ALLOW
        methods[f"policy_{policy_name}"] = _method_row(
            label=f"Policy: {policy_name}",
            n=int(allow_mask.sum()),
            hit=hit,
            ret60=ret60,
            ret120=ret120,
            mfe=mfe,
            mae=mae,
            mask=allow_mask,
        )

    return methods


def _method_row(
    *,
    label: str,
    n: int,
    hit: pd.Series,
    ret60: pd.Series,
    ret120: pd.Series,
    mfe: pd.Series,
    mae: pd.Series,
    mask: pd.Series,
) -> dict[str, Any]:
    return {
        "label": label,
        "n": n,
        "hit_rate_60m": _rnd(_safe_mean(hit[mask])),
        "avg_return_60m_bps": _rnd(_safe_mean(ret60[mask])),
        "avg_return_120m_bps": _rnd(_safe_mean(ret120[mask])),
        "avg_mfe_60m_bps": _rnd(_safe_mean(mfe[mask])),
        "avg_mae_60m_bps": _rnd(_safe_mean(mae[mask])),
    }


# ═════════════════════════════════════════════════════════════════════
# Report renderers
# ═════════════════════════════════════════════════════════════════════

def _render_markdown(report: dict[str, Any]) -> str:
    """Full Decision Policy evaluation report in Markdown."""
    lines: list[str] = []
    _h = lines.append

    _h("# Decision Policy Evaluation Report")
    _h(f"\n**Generated:** {report['evaluation_date']}")
    _h(f"**Dataset size:** {report['dataset_size']:,} signals")
    _h(f"\n**Author:** Pramit Dutta  |  **Organization:** Quant Engines")
    _h("\n---\n")

    # Per-policy metrics
    _h("## Policy Performance Summary\n")
    for policy_name, pm in report.get("policy_metrics", {}).items():
        _h(f"### {policy_name}\n")
        _h(f"- **Allowed:** {pm.get('allowed_pct', '?')}%  |  **Blocked:** {pm.get('blocked_pct', '?')}%\n")

        pdc = pm.get("per_decision_class", {})
        _h("| Decision | N | Hit Rate 60m | Avg Return 60m (bps) | Avg Return 120m (bps) | Avg MFE 60m | Avg MAE 60m |")
        _h("|----------|---|-------------|---------------------|----------------------|-------------|-------------|")
        for cls in [DECISION_ALLOW, DECISION_DOWNGRADE, DECISION_BLOCK]:
            d = pdc.get(cls, {})
            if d.get("n", 0) == 0:
                continue
            _h(f"| {cls} | {d['n']} | {d.get('hit_rate_60m', '—')} | {d.get('avg_return_60m_bps', '—')} | "
               f"{d.get('avg_return_120m_bps', '—')} | {d.get('avg_mfe_60m_bps', '—')} | {d.get('avg_mae_60m_bps', '—')} |")

        # Sizing
        ss = pm.get("sizing_simulation", {})
        if ss.get("n"):
            _h(f"\n**Sizing simulation** ({ss['n']} signals):")
            _h(f"- Baseline avg: {ss.get('baseline_avg_bps')} bps  →  Sized avg: {ss.get('sized_avg_bps')} bps")
            _h(f"- Improvement: {ss.get('sizing_improvement_pct')}%")

        # Drawdown
        dd = pm.get("drawdown_proxy", {})
        if dd:
            _h(f"\n**Drawdown proxy (ALLOW'd signals):** total {dd.get('total_return_bps')} bps, "
               f"max DD {dd.get('max_drawdown_bps')} bps")

        # Score bucket table
        sbp = pm.get("score_bucket_performance", [])
        if sbp:
            _h("\n**Rank-bucket performance (ALLOW'd):**\n")
            _h("| Bucket | N | Hit Rate 60m | Avg Return 60m (bps) |")
            _h("|--------|---|-------------|---------------------|")
            for b in sbp:
                _h(f"| {b['bucket']} | {b['n']} | {b.get('hit_rate_60m', '—')} | {b.get('avg_return_60m_bps', '—')} |")

        _h("")

    # Regime analysis
    regime = report.get("regime_analysis", {})
    if regime:
        _h("---\n")
        _h("## Regime-Conditional Analysis\n")
        for policy_name, regimes in regime.items():
            _h(f"### {policy_name}\n")
            for regime_label, data in regimes.items():
                if not data:
                    continue
                _h(f"**{regime_label.title()}:**\n")
                _h("| Regime | N | Hit Rate 60m | Avg Return 60m (bps) |")
                _h("|--------|---|-------------|---------------------|")
                for r in data:
                    _h(f"| {r['regime_value']} | {r['n']} | {r.get('hit_rate_60m', '—')} | {r.get('avg_return_60m_bps', '—')} |")
                _h("")

    # Yearly stability
    yearly = report.get("yearly_stability", {})
    if yearly:
        _h("---\n")
        _h("## Yearly Stability\n")
        for policy_name, years in yearly.items():
            if not years:
                continue
            _h(f"### {policy_name}\n")
            _h("| Year | N | Hit Rate 60m | Avg Return 60m (bps) |")
            _h("|------|---|-------------|---------------------|")
            for y in years:
                _h(f"| {y['year']} | {y['n']} | {y.get('hit_rate_60m', '—')} | {y.get('avg_return_60m_bps', '—')} |")
            _h("")

    return "\n".join(lines)


def _render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render the cross-methodology comparison table."""
    lines: list[str] = []
    _h = lines.append

    _h("# Decision Policy — Cross-Methodology Comparison")
    _h(f"\n**Generated:** {datetime.now().isoformat()}")
    _h(f"\n**Author:** Pramit Dutta  |  **Organization:** Quant Engines")
    _h("\n---\n")
    _h("| Methodology | N | Hit Rate 60m | Avg Return 60m (bps) | Avg Return 120m (bps) | Avg MFE 60m | Avg MAE 60m |")
    _h("|-------------|---|-------------|---------------------|----------------------|-------------|-------------|")

    for key, row in comparison.items():
        _h(f"| {row.get('label', key)} | {row['n']} | {row.get('hit_rate_60m', '—')} | "
           f"{row.get('avg_return_60m_bps', '—')} | {row.get('avg_return_120m_bps', '—')} | "
           f"{row.get('avg_mfe_60m_bps', '—')} | {row.get('avg_mae_60m_bps', '—')} |")

    _h("\n---\n")
    _h("*Baseline includes all signals; policy rows show only ALLOW'd signals.*")
    return "\n".join(lines)


def _save_comparison_csv(comparison: dict[str, Any]) -> None:
    """Save the comparison table as CSV for publication."""
    rows = []
    for key, row in comparison.items():
        rows.append({"methodology": key, **row})
    pd.DataFrame(rows).to_csv(DECISION_POLICY_COMPARISON_CSV, index=False)
    logger.info("Saved comparison CSV → %s", DECISION_POLICY_COMPARISON_CSV)


# ── Helpers ──────────────────────────────────────────────────────────

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON report → %s", path)


def _max_drawdown(cumulative: np.ndarray) -> float:
    if len(cumulative) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative)
    return float((cumulative - running_max).min())


def _safe_mean(series: pd.Series) -> float | None:
    valid = series.dropna()
    return float(valid.mean()) if not valid.empty else None


def _rnd(val: float | None, digits: int = 4) -> float | None:
    return round(val, digits) if val is not None else None
