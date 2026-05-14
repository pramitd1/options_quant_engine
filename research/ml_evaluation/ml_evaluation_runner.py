"""
ML Evaluation Runner
=====================
Main entry point that runs dual-model inference on the historical backtest
dataset and generates all four research reports.

Usage:
    python -m research.ml_evaluation.ml_evaluation_runner

This is a RESEARCH-ONLY module. It never modifies production logic.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research.ml_models.ml_config import (
    ML_EVALUATION_DIR,
    ML_EXTENDED_DATASET_PATH,
    ML_RESEARCH_ENABLED,
    FILTER_PERCENTILES,
    SIZING_BUCKETS,
)
from research.ml_models.ml_inference import infer_batch, compute_size_multiplier
from research.ml_evaluation.ml_ranking_report import build_ranking_report
from research.ml_evaluation.ml_calibration_report import build_calibration_report
from research.ml_evaluation.ml_comparison_report import build_comparison_report
from research.ml_evaluation.ml_filter_simulation import build_filter_simulation_report
from research.signal_evaluation.label_quality import label_quality_summary

logger = logging.getLogger(__name__)

BACKTEST_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "signal_evaluation" / "backtest_signals_dataset.csv"
)
BACKTEST_PARQUET_PATH = BACKTEST_DATASET_PATH.with_suffix(".parquet")


def _load_backtest_dataset() -> pd.DataFrame:
    """Load the historical backtest signals dataset."""
    if BACKTEST_PARQUET_PATH.exists():
        df = pd.read_parquet(BACKTEST_PARQUET_PATH)
    elif BACKTEST_DATASET_PATH.exists():
        df = pd.read_csv(BACKTEST_DATASET_PATH)
    else:
        raise FileNotFoundError(
            f"Backtest dataset not found at {BACKTEST_DATASET_PATH} or {BACKTEST_PARQUET_PATH}"
        )
    logger.info("Loaded backtest dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


def run_ml_evaluation() -> dict:
    """
    Run the complete ML evaluation pipeline:
      1. Load backtest dataset
      2. Run dual-model batch inference
      3. Save extended dataset
      4. Generate all four reports
      5. Save reports and summary

    Returns a summary dict with key metrics.
    """
    if not ML_RESEARCH_ENABLED:
        logger.warning("ML research is disabled. Set OQE_ML_RESEARCH_ENABLED=1 to enable.")
        return {"status": "disabled"}

    ML_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    df = _load_backtest_dataset()

    # 2. Run dual-model inference
    logger.info("Running dual-model batch inference on %d signals...", len(df))
    extended_df = infer_batch(df)

    # 3. Save extended dataset
    extended_df.to_csv(ML_EXTENDED_DATASET_PATH, index=False)
    logger.info("Saved ML-extended dataset to %s", ML_EXTENDED_DATASET_PATH)

    # 4. Generate reports
    ranking_report = build_ranking_report(extended_df)
    calibration_report = build_calibration_report(extended_df)
    comparison_report = build_comparison_report(extended_df)
    filter_report = build_filter_simulation_report(extended_df)

    # 5. Save all reports
    _save_report(ranking_report, "ml_ranking_report")
    _save_report(calibration_report, "ml_calibration_report")
    _save_report(comparison_report, "ml_vs_engine_comparison")
    _save_report(filter_report, "ml_filter_simulation")

    # 6. Build summary
    summary = _build_summary(
        extended_df, ranking_report, calibration_report, comparison_report, filter_report
    )
    _save_report(summary, "ml_evaluation_summary")

    # 7. Save markdown summary
    md_report = _render_markdown_summary(
        summary, ranking_report, calibration_report, comparison_report, filter_report
    )
    md_path = ML_EVALUATION_DIR / "ml_evaluation_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    logger.info("Saved markdown report to %s", md_path)

    return summary


def _save_report(report: dict, name: str) -> None:
    """Save a report dict as JSON."""
    path = ML_EVALUATION_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved report: %s", path)


def _build_summary(
    df: pd.DataFrame,
    ranking: dict,
    calibration: dict,
    comparison: dict,
    filter_sim: dict,
) -> dict:
    """Build a top-level evaluation summary."""
    n_total = len(df)
    n_with_rank = df["ml_rank_score"].notna().sum()
    n_with_conf = df["ml_confidence_score"].notna().sum()

    agreement_counts = df["ml_agreement_with_engine"].value_counts().to_dict()

    return {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_size": n_total,
        "label_quality_summary": label_quality_summary(df),
        "ml_coverage": {
            "rank_score_available": int(n_with_rank),
            "confidence_score_available": int(n_with_conf),
            "coverage_pct": round(n_with_rank / max(n_total, 1) * 100, 2),
        },
        "agreement_distribution": agreement_counts,
        "ranking_summary": {
            "top_quintile_hit_rate": ranking.get("quintile_analysis", [{}])[-1].get("hit_rate_60m")
            if ranking.get("quintile_analysis") else None,
            "bottom_quintile_hit_rate": ranking.get("quintile_analysis", [{}])[0].get("hit_rate_60m")
            if ranking.get("quintile_analysis") else None,
            "spread": ranking.get("spread"),
        },
        "calibration_summary": {
            "avg_ece": calibration.get("expected_calibration_error"),
        },
        "comparison_summary": comparison.get("summary"),
        "filter_simulation_summary": filter_sim.get("summary"),
    }


def _render_markdown_summary(
    summary: dict,
    ranking: dict,
    calibration: dict,
    comparison: dict,
    filter_sim: dict,
) -> str:
    """Render a professional markdown research report."""
    lines = []
    lines.append("# ML Research Evaluation Report")
    lines.append(f"**Generated:** {summary.get('evaluation_date', 'N/A')}")
    lines.append(f"**Dataset Size:** {summary.get('dataset_size', 0):,} signals")
    lines.append("")

    # ── Executive Summary
    lines.append("## Executive Summary")
    cov = summary.get("ml_coverage", {})
    lines.append(f"- ML inference coverage: **{cov.get('coverage_pct', 0)}%** ({cov.get('rank_score_available', 0):,} signals)")
    agree = summary.get("agreement_distribution", {})
    lines.append(f"- Agreement distribution: AGREE={agree.get('YES', 0)}, DISAGREE={agree.get('NO', 0)}, NO_ENGINE={agree.get('NO_ENGINE_SIGNAL', 0)}")
    rs = summary.get("ranking_summary", {})
    lines.append(f"- Ranking spread (Q5−Q1 hit rate): **{rs.get('spread', 'N/A')}**")
    cs = summary.get("calibration_summary", {})
    lines.append(f"- Expected Calibration Error (ECE): **{cs.get('avg_ece', 'N/A')}**")
    lines.append("")

    # ── Ranking Report
    lines.append("## 1. ML Ranking Analysis (GBT_shallow_v1)")
    lines.append("")
    lines.append("| Quintile | N | Hit Rate 60m | Avg Return BPS | Avg Rank Score |")
    lines.append("|----------|---|-------------|----------------|----------------|")
    for q in ranking.get("quintile_analysis", []):
        lines.append(
            f"| {q.get('bucket', '?')} | {q.get('n', 0)} | "
            f"{_fmt_pct(q.get('hit_rate_60m'))} | "
            f"{_fmt_num(q.get('avg_signed_return_60m_bps'))} | "
            f"{_fmt_num(q.get('avg_rank_score'))} |"
        )
    lines.append("")

    # ── Calibration Report
    lines.append("## 2. ML Calibration Analysis (LogReg_ElasticNet_v1)")
    lines.append("")
    lines.append("| Confidence Bucket | N | Predicted Prob | Actual Hit Rate | Gap |")
    lines.append("|-------------------|---|---------------|-----------------|-----|")
    for b in calibration.get("bucket_analysis", []):
        predicted = b.get("avg_confidence_score")
        actual = b.get("actual_hit_rate_60m")
        gap = abs(predicted - actual) if predicted is not None and actual is not None else None
        lines.append(
            f"| {b.get('bucket', '?')} | {b.get('n', 0)} | "
            f"{_fmt_pct(predicted)} | "
            f"{_fmt_pct(actual)} | "
            f"{_fmt_num(gap)} |"
        )
    lines.append("")

    # ── Engine vs ML Comparison
    lines.append("## 3. Engine vs ML Comparison")
    lines.append("")
    comp = comparison.get("summary", {})
    lines.append(f"- Engine-only hit rate (60m): **{_fmt_pct(comp.get('engine_hit_rate_60m'))}**")
    lines.append(f"- ML-agrees hit rate (60m): **{_fmt_pct(comp.get('ml_agree_hit_rate_60m'))}**")
    lines.append(f"- ML-disagrees hit rate (60m): **{_fmt_pct(comp.get('ml_disagree_hit_rate_60m'))}**")
    lines.append(f"- Engine-only avg return: **{_fmt_num(comp.get('engine_avg_return_bps'))} bps**")
    lines.append(f"- ML-agrees avg return: **{_fmt_num(comp.get('ml_agree_avg_return_bps'))} bps**")
    lines.append(f"- ML-disagrees avg return: **{_fmt_num(comp.get('ml_disagree_avg_return_bps'))} bps**")
    lines.append("")

    # ── Filter Simulation
    lines.append("## 4. ML Filter & Sizing Simulation")
    lines.append("")
    lines.append("### 4a. Filter Simulation (Remove Bottom N% by ml_rank_score)")
    lines.append("")
    lines.append("| Filter | Signals Kept | Hit Rate 60m | Avg Return BPS | Improvement |")
    lines.append("|--------|-------------|-------------|----------------|-------------|")
    for f in filter_sim.get("filter_results", []):
        lines.append(
            f"| Remove bottom {f.get('filter_pct', '?')}% | {f.get('n_kept', 0)} | "
            f"{_fmt_pct(f.get('hit_rate_60m'))} | "
            f"{_fmt_num(f.get('avg_return_bps'))} | "
            f"{_fmt_num(f.get('hit_rate_improvement_pct'))}% |"
        )
    lines.append("")

    lines.append("### 4b. Position Sizing Simulation")
    lines.append("")
    sizing = filter_sim.get("sizing_simulation", {})
    lines.append(f"- Baseline avg return: **{_fmt_num(sizing.get('baseline_avg_return_bps'))} bps**")
    lines.append(f"- ML-sized avg return: **{_fmt_num(sizing.get('ml_sized_avg_return_bps'))} bps**")
    lines.append(f"- Sizing improvement: **{_fmt_num(sizing.get('sizing_improvement_pct'))}%**")
    lines.append(f"- Baseline max drawdown: **{_fmt_num(sizing.get('baseline_max_dd_bps'))} bps**")
    lines.append(f"- ML-sized max drawdown: **{_fmt_num(sizing.get('ml_sized_max_dd_bps'))} bps**")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*This report is generated by the ML Research Evaluation framework.*")
    lines.append("*ML outputs are strictly observational and do NOT influence production trading.*")

    return "\n".join(lines)


def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.2%}" if abs(val) < 1 else f"{val:.2f}%"


def _fmt_num(val, decimals=2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    result = run_ml_evaluation()
    print(json.dumps(result, indent=2, default=str))
