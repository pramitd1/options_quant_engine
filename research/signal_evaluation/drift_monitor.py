"""
Quality-aware signal drift monitoring for research and operations.

The monitor compares a recent window against prior history and reports
whether signal quality, model calibration, label coverage, or policy retention
has shifted enough to deserve operator attention.  It never changes trading
decisions or execution behavior.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import pandas as pd

from config.signal_drift_policy import get_signal_drift_monitor_policy
from research.signal_evaluation.confidence import confidence_intervals_overlap, outcome_confidence_fields, sample_guardrail
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary
from research.signal_evaluation.reporting import SIGNAL_EVALUATION_REPORTS_DIR
from utils.timestamp_helpers import coerce_timestamp_series


DEFAULT_DIMENSIONS = (
    "source",
    "mode",
    "gamma_regime",
    "volatility_regime",
    "macro_regime",
    "global_risk_state",
)

PROBABILITY_FIELDS = (
    "hybrid_move_probability",
    "ml_confidence_score",
)

SCORE_FIELDS = (
    "ml_rank_score",
    "ml_confidence_score",
    "hybrid_move_probability",
    "trade_strength",
    "composite_signal_score",
    "tradeability_score",
)

_ALLOW_DECISIONS = {"ALLOW", "DOWNGRADE"}
_BLOCK_DECISIONS = {"BLOCK"}

TREND_HISTORY_FILENAME = "signal_drift_trend_history.csv"
TREND_DASHBOARD_JSON_FILENAME = "latest_signal_drift_trend.json"
TREND_DASHBOARD_MARKDOWN_FILENAME = "latest_signal_drift_trend.md"


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _with_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "signal_timestamp" in working.columns:
        working["signal_timestamp"] = coerce_timestamp_series(working["signal_timestamp"], utc=True)
    return working


def _maybe_apply_policies(frame: pd.DataFrame, *, apply_missing_policies: bool) -> pd.DataFrame:
    if not apply_missing_policies:
        return frame
    if any(col.endswith("_decision") for col in frame.columns):
        return frame
    required = {"ml_rank_score", "ml_confidence_score", "hybrid_move_probability"}
    if not required.issubset(frame.columns):
        return frame
    try:
        from research.decision_policy.policy_engine import apply_policies

        return apply_policies(frame)
    except Exception:
        return frame


def _window_masks(
    frame: pd.DataFrame,
    *,
    recent_days: int,
    baseline_days: int | None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    if frame.empty:
        empty = pd.Series(False, index=frame.index, dtype=bool)
        return empty, empty, {"mode": "empty", "recent_days": 0, "baseline_days": 0}

    if "signal_timestamp" in frame.columns:
        ts = coerce_timestamp_series(frame["signal_timestamp"], utc=True)
        valid_ts = ts.dropna()
        if not valid_ts.empty:
            day_key = ts.dt.normalize()
            unique_days = sorted(day_key.dropna().unique())
            recent_n = min(max(int(recent_days), 1), len(unique_days))
            recent_day_set = set(unique_days[-recent_n:])
            prior_days = unique_days[:-recent_n]
            if baseline_days is not None and baseline_days > 0:
                prior_days = prior_days[-int(baseline_days):]
            baseline_day_set = set(prior_days)
            recent_mask = day_key.isin(recent_day_set).fillna(False)
            baseline_mask = day_key.isin(baseline_day_set).fillna(False)
            return recent_mask, baseline_mask, {
                "mode": "calendar_days",
                "recent_days": int(len(recent_day_set)),
                "baseline_days": int(len(baseline_day_set)),
                "recent_start": min(recent_day_set).isoformat() if recent_day_set else None,
                "recent_end": max(recent_day_set).isoformat() if recent_day_set else None,
                "baseline_start": min(baseline_day_set).isoformat() if baseline_day_set else None,
                "baseline_end": max(baseline_day_set).isoformat() if baseline_day_set else None,
            }

    recent_count = max(int(len(frame) * 0.2), 1)
    recent_mask = pd.Series(False, index=frame.index, dtype=bool)
    recent_mask.iloc[-recent_count:] = True
    baseline_mask = ~recent_mask
    return recent_mask, baseline_mask, {
        "mode": "row_fallback",
        "recent_rows": int(recent_mask.sum()),
        "baseline_rows": int(baseline_mask.sum()),
    }


def _period(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "signal_timestamp" not in frame.columns:
        return {"start": None, "end": None, "trading_days": 0}
    ts = coerce_timestamp_series(frame["signal_timestamp"], utc=True).dropna()
    if ts.empty:
        return {"start": None, "end": None, "trading_days": 0}
    return {
        "start": ts.min().isoformat(),
        "end": ts.max().isoformat(),
        "trading_days": int(ts.dt.normalize().nunique()),
    }


def _performance_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    hit = _numeric(frame, "correct_60m")
    ret = _numeric(frame, "signed_return_60m_bps")
    labeled_hit = hit.dropna()
    labeled_ret = ret.dropna()
    wins = labeled_ret[labeled_ret > 0]
    losses = labeled_ret[labeled_ret <= 0]
    metrics = {
        "signal_count": int(len(frame)),
        "labeled_60m": int(labeled_hit.count()),
        "return_labeled_60m": int(labeled_ret.count()),
        "hit_rate_60m": _round_or_none(labeled_hit.mean(), 4),
        "avg_signed_return_60m_bps": _round_or_none(labeled_ret.mean(), 4),
        "median_signed_return_60m_bps": _round_or_none(labeled_ret.median(), 4),
        "win_rate_by_return": _round_or_none((labeled_ret > 0).mean(), 4) if not labeled_ret.empty else None,
        "avg_win_bps": _round_or_none(wins.mean(), 4) if not wins.empty else None,
        "avg_loss_bps": _round_or_none(losses.mean(), 4) if not losses.empty else None,
    }
    metrics.update(outcome_confidence_fields(labeled_hit, labeled_ret))
    return metrics


def _delta(recent: Any, baseline: Any, digits: int = 4) -> float | None:
    if recent is None or baseline is None:
        return None
    try:
        return round(float(recent) - float(baseline), digits)
    except Exception:
        return None


def _safe_abs(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return abs(float(value))
    except Exception:
        return None


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize read/modify/write cycles for shared ops artifacts."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _compare_performance(recent: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    recent_metrics = _performance_metrics(recent)
    baseline_metrics = _performance_metrics(baseline)
    return {
        "baseline": baseline_metrics,
        "recent": recent_metrics,
        "hit_rate_delta": _delta(recent_metrics.get("hit_rate_60m"), baseline_metrics.get("hit_rate_60m"), 4),
        "avg_return_delta_bps": _delta(
            recent_metrics.get("avg_signed_return_60m_bps"),
            baseline_metrics.get("avg_signed_return_60m_bps"),
            4,
        ),
        "label_count_delta": int(recent_metrics["labeled_60m"] - baseline_metrics["labeled_60m"]),
        "hit_rate_ci_overlap": confidence_intervals_overlap(
            recent_metrics.get("hit_rate_ci_low"),
            recent_metrics.get("hit_rate_ci_high"),
            baseline_metrics.get("hit_rate_ci_low"),
            baseline_metrics.get("hit_rate_ci_high"),
        ),
        "return_ci_overlap": confidence_intervals_overlap(
            recent_metrics.get("return_ci_low_bps"),
            recent_metrics.get("return_ci_high_bps"),
            baseline_metrics.get("return_ci_low_bps"),
            baseline_metrics.get("return_ci_high_bps"),
        ),
    }


def _apply_outcome_guardrails(
    comparison: dict[str, Any],
    *,
    min_recent_labeled: int,
    min_baseline_labeled: int,
) -> None:
    recent = comparison["recent"]
    baseline = comparison["baseline"]
    recent.update(
        sample_guardrail(
            recent.get("labeled_60m", 0),
            min_sample=min_recent_labeled,
            strong_sample=max(min_recent_labeled * 3, min_recent_labeled + 1),
        )
    )
    baseline.update(
        sample_guardrail(
            baseline.get("labeled_60m", 0),
            min_sample=min_baseline_labeled,
            strong_sample=max(min_baseline_labeled * 3, min_baseline_labeled + 1),
        )
    )
    sufficient = (
        int(recent.get("labeled_60m", 0)) >= int(min_recent_labeled)
        and int(baseline.get("labeled_60m", 0)) >= int(min_baseline_labeled)
    )
    comparison["outcome_evidence_status"] = "RELIABLE" if sufficient else "INSUFFICIENT_EVIDENCE"
    comparison["outcome_guardrail"] = {
        "sufficient_evidence": bool(sufficient),
        "recent_labeled_60m": int(recent.get("labeled_60m", 0)),
        "baseline_labeled_60m": int(baseline.get("labeled_60m", 0)),
        "min_recent_labeled": int(min_recent_labeled),
        "min_baseline_labeled": int(min_baseline_labeled),
    }


def _dimension_drift(
    frame: pd.DataFrame,
    baseline_mask: pd.Series,
    recent_mask: pd.Series,
    *,
    dimensions: tuple[str, ...],
    min_recent_labeled: int,
    min_baseline_labeled: int,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        if dimension not in frame.columns:
            continue
        values = pd.concat(
            [
                frame.loc[baseline_mask, dimension],
                frame.loc[recent_mask, dimension],
            ],
            ignore_index=True,
        ).dropna()
        rows: list[dict[str, Any]] = []
        for value in sorted(values.astype(str).unique()):
            value_mask = frame[dimension].astype(str).eq(value)
            baseline = frame.loc[baseline_mask & value_mask]
            recent = frame.loc[recent_mask & value_mask]
            if baseline.empty and recent.empty:
                continue
            comparison = _compare_performance(recent, baseline)
            _apply_outcome_guardrails(
                comparison,
                min_recent_labeled=min_recent_labeled,
                min_baseline_labeled=min_baseline_labeled,
            )
            baseline_metrics = comparison["baseline"]
            recent_metrics = comparison["recent"]
            status = "OK"
            if recent_metrics["labeled_60m"] < min_recent_labeled or baseline_metrics["labeled_60m"] < min_baseline_labeled:
                status = "INSUFFICIENT_SAMPLE"
            elif (comparison.get("hit_rate_delta") is not None and comparison["hit_rate_delta"] <= -0.10) or (
                comparison.get("avg_return_delta_bps") is not None and comparison["avg_return_delta_bps"] <= -20.0
            ):
                status = "DRIFT_DOWN"
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "baseline_signal_count": baseline_metrics["signal_count"],
                    "recent_signal_count": recent_metrics["signal_count"],
                    "baseline_labeled_60m": baseline_metrics["labeled_60m"],
                    "recent_labeled_60m": recent_metrics["labeled_60m"],
                    "baseline_hit_rate_60m": baseline_metrics["hit_rate_60m"],
                    "recent_hit_rate_60m": recent_metrics["hit_rate_60m"],
                    "hit_rate_delta": comparison["hit_rate_delta"],
                    "baseline_avg_return_60m_bps": baseline_metrics["avg_signed_return_60m_bps"],
                    "recent_avg_return_60m_bps": recent_metrics["avg_signed_return_60m_bps"],
                    "avg_return_delta_bps": comparison["avg_return_delta_bps"],
                    "hit_rate_ci_overlap": comparison.get("hit_rate_ci_overlap"),
                    "return_ci_overlap": comparison.get("return_ci_overlap"),
                    "sample_quality": comparison.get("outcome_evidence_status"),
                    "status": status,
                }
            )
        rows.sort(key=lambda row: (row["recent_signal_count"], row["baseline_signal_count"]), reverse=True)
        result[dimension] = rows[:top_n]
    return result


def _calibration_metrics(frame: pd.DataFrame, probability_col: str) -> dict[str, Any]:
    if probability_col not in frame.columns:
        return {"samples": 0}
    working = pd.DataFrame(
        {
            "p": _numeric(frame, probability_col),
            "y": _numeric(frame, "correct_60m"),
        }
    ).dropna()
    if working.empty:
        return {"samples": 0}
    working["p"] = working["p"].clip(0.0, 1.0)
    brier = ((working["p"] - working["y"]) ** 2).mean()
    gap = working["p"].mean() - working["y"].mean()

    bins = pd.cut(working["p"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.01], include_lowest=True)
    ece = 0.0
    for _, group in working.groupby(bins, observed=False):
        if group.empty:
            continue
        ece += len(group) * abs(float(group["p"].mean() - group["y"].mean()))
    ece = ece / max(len(working), 1)
    return {
        "samples": int(len(working)),
        "avg_prediction": _round_or_none(working["p"].mean(), 4),
        "avg_actual": _round_or_none(working["y"].mean(), 4),
        "calibration_gap": _round_or_none(gap, 4),
        "abs_calibration_gap": _round_or_none(abs(gap), 4),
        "brier": _round_or_none(brier, 6),
        "ece": _round_or_none(ece, 6),
    }


def _calibration_drift(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    *,
    probability_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in probability_fields:
        if field not in baseline.columns and field not in recent.columns:
            continue
        base = _calibration_metrics(baseline, field)
        rec = _calibration_metrics(recent, field)
        rows.append(
            {
                "probability_field": field,
                "baseline": base,
                "recent": rec,
                "abs_calibration_gap_delta": _delta(rec.get("abs_calibration_gap"), base.get("abs_calibration_gap"), 4),
                "brier_delta": _delta(rec.get("brier"), base.get("brier"), 6),
                "ece_delta": _delta(rec.get("ece"), base.get("ece"), 6),
            }
        )
    return rows


def _score_stats(frame: pd.DataFrame, field: str) -> dict[str, Any]:
    series = _numeric(frame, field).dropna()
    if series.empty:
        return {"count": 0, "mean": None, "std": None, "p25": None, "p75": None}
    return {
        "count": int(series.count()),
        "mean": _round_or_none(series.mean(), 4),
        "std": _round_or_none(series.std(ddof=0), 4),
        "p25": _round_or_none(series.quantile(0.25), 4),
        "p75": _round_or_none(series.quantile(0.75), 4),
    }


def _score_distribution_drift(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    *,
    score_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in score_fields:
        if field not in baseline.columns and field not in recent.columns:
            continue
        base = _score_stats(baseline, field)
        rec = _score_stats(recent, field)
        rows.append(
            {
                "score_field": field,
                "baseline": base,
                "recent": rec,
                "mean_delta": _delta(rec.get("mean"), base.get("mean"), 4),
                "p75_delta": _delta(rec.get("p75"), base.get("p75"), 4),
            }
        )
    return rows


def _decision_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in frame.columns
        if col.endswith("_decision") and not col.startswith("_") and frame[col].notna().any()
    )


def _policy_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {"signal_count": int(len(frame)), "retained": 0, "blocked": 0, "retention_ratio": None, "block_ratio": None}
    decisions = frame[column].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    total = int(len(decisions))
    retained = int(decisions.isin(_ALLOW_DECISIONS).sum())
    blocked = int(decisions.isin(_BLOCK_DECISIONS).sum())
    counts = decisions.value_counts().to_dict()
    return {
        "signal_count": total,
        "retained": retained,
        "blocked": blocked,
        "retention_ratio": _round_or_none(retained / max(total, 1), 4),
        "block_ratio": _round_or_none(blocked / max(total, 1), 4),
        "decision_counts": {str(key): int(value) for key, value in counts.items()},
    }


def _policy_retention_drift(baseline: pd.DataFrame, recent: pd.DataFrame) -> list[dict[str, Any]]:
    columns = sorted(set(_decision_columns(baseline)) | set(_decision_columns(recent)))
    rows: list[dict[str, Any]] = []
    for column in columns:
        base = _policy_metrics(baseline, column)
        rec = _policy_metrics(recent, column)
        rows.append(
            {
                "policy": column.removesuffix("_decision"),
                "decision_column": column,
                "baseline": base,
                "recent": rec,
                "retention_delta": _delta(rec.get("retention_ratio"), base.get("retention_ratio"), 4),
                "block_delta": _delta(rec.get("block_ratio"), base.get("block_ratio"), 4),
            }
        )
    return rows


def _label_quality_drift(raw_baseline: pd.DataFrame, raw_recent: pd.DataFrame) -> dict[str, Any]:
    base = label_quality_summary(raw_baseline)
    rec = label_quality_summary(raw_recent)
    return {
        "baseline": base,
        "recent": rec,
        "quality_coverage_delta": _delta(
            rec.get("quality_label_coverage_ratio"),
            base.get("quality_label_coverage_ratio"),
            4,
        ),
        "raw_coverage_delta": _delta(
            rec.get("raw_label_coverage_ratio"),
            base.get("raw_label_coverage_ratio"),
            4,
        ),
        "excluded_label_delta": int(rec.get("excluded_labeled_rows", 0) - base.get("excluded_labeled_rows", 0)),
    }


def _build_warnings(
    *,
    overall: dict[str, Any],
    dimension_drift: dict[str, list[dict[str, Any]]],
    calibration_drift: list[dict[str, Any]],
    policy_drift: list[dict[str, Any]],
    label_drift: dict[str, Any],
    min_recent_labeled: int,
    min_baseline_labeled: int,
    hit_rate_drop_warn: float,
    return_drop_bps_warn: float,
    calibration_gap_delta_warn: float,
    label_coverage_drop_warn: float,
    retention_delta_warn: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    recent = overall["recent"]
    baseline = overall["baseline"]
    if recent["labeled_60m"] < min_recent_labeled:
        warnings.append({
            "severity": "CAUTION",
            "category": "label_sample",
            "message": "Recent window has too few quality-approved 60m labels.",
            "value": recent["labeled_60m"],
            "threshold": min_recent_labeled,
        })
    if baseline["labeled_60m"] < min_baseline_labeled:
        warnings.append({
            "severity": "CAUTION",
            "category": "label_sample",
            "message": "Baseline window has too few quality-approved 60m labels.",
            "value": baseline["labeled_60m"],
            "threshold": min_baseline_labeled,
        })

    sufficient_outcome_evidence = bool(overall.get("outcome_guardrail", {}).get("sufficient_evidence"))
    hit_delta = overall.get("hit_rate_delta")
    if sufficient_outcome_evidence and hit_delta is not None and hit_delta <= -abs(hit_rate_drop_warn):
        warnings.append({
            "severity": "CAUTION",
            "category": "outcome_drift",
            "message": "Recent 60m hit rate is materially below baseline.",
            "value": hit_delta,
            "threshold": -abs(hit_rate_drop_warn),
        })
    return_delta = overall.get("avg_return_delta_bps")
    if sufficient_outcome_evidence and return_delta is not None and return_delta <= -abs(return_drop_bps_warn):
        warnings.append({
            "severity": "CAUTION",
            "category": "outcome_drift",
            "message": "Recent average signed 60m return is materially below baseline.",
            "value": return_delta,
            "threshold": -abs(return_drop_bps_warn),
        })

    coverage_delta = label_drift.get("quality_coverage_delta")
    if coverage_delta is not None and coverage_delta <= -abs(label_coverage_drop_warn):
        warnings.append({
            "severity": "WATCH",
            "category": "label_quality",
            "message": "Quality-approved label coverage fell versus baseline.",
            "value": coverage_delta,
            "threshold": -abs(label_coverage_drop_warn),
        })

    for row in calibration_drift:
        gap_delta = row.get("abs_calibration_gap_delta")
        if gap_delta is not None and gap_delta >= abs(calibration_gap_delta_warn):
            warnings.append({
                "severity": "WATCH",
                "category": "calibration_drift",
                "message": f"{row['probability_field']} calibration gap widened.",
                "value": gap_delta,
                "threshold": abs(calibration_gap_delta_warn),
            })

    for row in policy_drift:
        retention_delta = row.get("retention_delta")
        if retention_delta is not None and abs(retention_delta) >= abs(retention_delta_warn):
            warnings.append({
                "severity": "WATCH",
                "category": "policy_retention",
                "message": f"{row['policy']} retention changed materially.",
                "value": retention_delta,
                "threshold": abs(retention_delta_warn),
            })

    for dimension, rows in dimension_drift.items():
        for row in rows:
            if row.get("status") != "DRIFT_DOWN":
                continue
            warnings.append({
                "severity": "WATCH",
                "category": "segment_drift",
                "message": f"{dimension}={row['value']} recent outcomes weakened.",
                "value": {
                    "hit_rate_delta": row.get("hit_rate_delta"),
                    "avg_return_delta_bps": row.get("avg_return_delta_bps"),
                },
            })
    return warnings


def build_signal_drift_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | None = None,
    recent_days: int | None = None,
    baseline_days: int | None = None,
    min_recent_labeled: int | None = None,
    min_baseline_labeled: int | None = None,
    dimensions: tuple[str, ...] | list[str] | None = None,
    probability_fields: tuple[str, ...] | list[str] | None = None,
    score_fields: tuple[str, ...] | list[str] | None = None,
    top_n: int | None = None,
    apply_missing_policies: bool | None = None,
    hit_rate_drop_warn: float | None = None,
    return_drop_bps_warn: float | None = None,
    calibration_gap_delta_warn: float | None = None,
    label_coverage_drop_warn: float | None = None,
    retention_delta_warn: float | None = None,
    policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a quality-aware recent-vs-baseline drift report."""
    policy = get_signal_drift_monitor_policy(
        {
            **(policy_overrides or {}),
            "recent_days": recent_days,
            "baseline_days": baseline_days,
            "min_recent_labeled": min_recent_labeled,
            "min_baseline_labeled": min_baseline_labeled,
            "dimensions": dimensions,
            "probability_fields": probability_fields,
            "score_fields": score_fields,
            "top_n": top_n,
            "apply_missing_policies": apply_missing_policies,
            "hit_rate_drop_warn": hit_rate_drop_warn,
            "return_drop_bps_warn": return_drop_bps_warn,
            "calibration_gap_delta_warn": calibration_gap_delta_warn,
            "label_coverage_drop_warn": label_coverage_drop_warn,
            "retention_delta_warn": retention_delta_warn,
        }
    )
    recent_days = int(policy["recent_days"])
    baseline_days = policy["baseline_days"]
    min_recent_labeled = int(policy["min_recent_labeled"])
    min_baseline_labeled = int(policy["min_baseline_labeled"])
    dimensions = tuple(policy.get("dimensions") or DEFAULT_DIMENSIONS)
    probability_fields = tuple(policy.get("probability_fields") or PROBABILITY_FIELDS)
    score_fields = tuple(policy.get("score_fields") or SCORE_FIELDS)
    top_n = int(policy["top_n"])
    apply_missing_policies = bool(policy["apply_missing_policies"])
    hit_rate_drop_warn = float(policy["hit_rate_drop_warn"])
    return_drop_bps_warn = float(policy["return_drop_bps_warn"])
    calibration_gap_delta_warn = float(policy["calibration_gap_delta_warn"])
    label_coverage_drop_warn = float(policy["label_coverage_drop_warn"])
    retention_delta_warn = float(policy["retention_delta_warn"])

    raw = _with_timestamp(frame if frame is not None else pd.DataFrame())
    raw = _maybe_apply_policies(raw, apply_missing_policies=apply_missing_policies)
    quality = apply_quality_label_view(raw)

    recent_mask, baseline_mask, window_info = _window_masks(
        quality,
        recent_days=recent_days,
        baseline_days=baseline_days,
    )
    recent = quality.loc[recent_mask].copy()
    baseline = quality.loc[baseline_mask].copy()
    raw_recent = raw.loc[recent_mask].copy()
    raw_baseline = raw.loc[baseline_mask].copy()

    overall = _compare_performance(recent, baseline)
    _apply_outcome_guardrails(
        overall,
        min_recent_labeled=min_recent_labeled,
        min_baseline_labeled=min_baseline_labeled,
    )
    dimension_drift = _dimension_drift(
        quality,
        baseline_mask,
        recent_mask,
        dimensions=dimensions,
        min_recent_labeled=min_recent_labeled,
        min_baseline_labeled=min_baseline_labeled,
        top_n=top_n,
    )
    calibration_drift = _calibration_drift(baseline, recent, probability_fields=probability_fields)
    score_drift = _score_distribution_drift(baseline, recent, score_fields=score_fields)
    policy_drift = _policy_retention_drift(baseline, recent)
    label_drift = _label_quality_drift(raw_baseline, raw_recent)
    warnings = _build_warnings(
        overall=overall,
        dimension_drift=dimension_drift,
        calibration_drift=calibration_drift,
        policy_drift=policy_drift,
        label_drift=label_drift,
        min_recent_labeled=min_recent_labeled,
        min_baseline_labeled=min_baseline_labeled,
        hit_rate_drop_warn=hit_rate_drop_warn,
        return_drop_bps_warn=return_drop_bps_warn,
        calibration_gap_delta_warn=calibration_gap_delta_warn,
        label_coverage_drop_warn=label_coverage_drop_warn,
        retention_delta_warn=retention_delta_warn,
    )

    status = "OK"
    if any(item.get("severity") == "CAUTION" for item in warnings):
        status = "CAUTION"
    elif warnings:
        status = "WATCH"

    report = {
        "report_type": "signal_drift_monitor",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_path": dataset_path,
        "monitor_status": status,
        "window": window_info,
        "evaluation_period": _period(raw),
        "total_signal_count": int(len(raw)),
        "label_quality_summary": label_quality_summary(raw),
        "label_quality_drift": label_drift,
        "overall_outcome_drift": overall,
        "dimension_outcome_drift": dimension_drift,
        "calibration_drift": calibration_drift,
        "score_distribution_drift": score_drift,
        "policy_retention_drift": policy_drift,
        "warnings": warnings,
        "thresholds": {
            "policy_source": "config.signal_drift_policy",
            "min_recent_labeled": int(min_recent_labeled),
            "min_baseline_labeled": int(min_baseline_labeled),
            "recent_days": int(recent_days),
            "baseline_days": None if baseline_days is None else int(baseline_days),
            "top_n": int(top_n),
            "hit_rate_drop_warn": float(hit_rate_drop_warn),
            "return_drop_bps_warn": float(return_drop_bps_warn),
            "calibration_gap_delta_warn": float(calibration_gap_delta_warn),
            "label_coverage_drop_warn": float(label_coverage_drop_warn),
            "retention_delta_warn": float(retention_delta_warn),
            "apply_missing_policies": bool(apply_missing_policies),
            "dimensions": list(dimensions),
            "probability_fields": list(probability_fields),
            "score_fields": list(score_fields),
        },
    }
    return _sanitize_value(report)


def _flatten_dimension_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, entries in report.get("dimension_outcome_drift", {}).items():
        rows.extend(entries)
    return rows


def _calibration_row(report: dict[str, Any], field: str) -> dict[str, Any]:
    for row in report.get("calibration_drift", []) or []:
        if row.get("probability_field") == field:
            return row
    return {}


def _policy_retention_summary(report: dict[str, Any]) -> dict[str, Any]:
    deltas = [
        _safe_abs(row.get("retention_delta"))
        for row in report.get("policy_retention_drift", []) or []
    ]
    deltas = [value for value in deltas if value is not None]
    return {
        "policy_count": int(len(report.get("policy_retention_drift", []) or [])),
        "max_abs_policy_retention_delta": _round_or_none(max(deltas), 4) if deltas else None,
    }


def build_signal_drift_trend_row(
    report: dict[str, Any],
    *,
    json_path: str | None = None,
    markdown_path: str | None = None,
    report_name: str | None = None,
) -> dict[str, Any]:
    """Flatten one drift report into a compact history row."""
    overall = report.get("overall_outcome_drift", {}) or {}
    baseline = overall.get("baseline", {}) or {}
    recent = overall.get("recent", {}) or {}
    label_summary = report.get("label_quality_summary", {}) or {}
    label_drift = report.get("label_quality_drift", {}) or {}
    window = report.get("window", {}) or {}
    period = report.get("evaluation_period", {}) or {}
    warnings = report.get("warnings", []) or []
    policy_summary = _policy_retention_summary(report)
    hybrid_cal = _calibration_row(report, "hybrid_move_probability")
    ml_cal = _calibration_row(report, "ml_confidence_score")

    return {
        "generated_at": report.get("generated_at"),
        "report_name": report_name,
        "monitor_status": report.get("monitor_status"),
        "dataset_path": report.get("dataset_path"),
        "evaluation_start": period.get("start"),
        "evaluation_end": period.get("end"),
        "window_mode": window.get("mode"),
        "recent_start": window.get("recent_start"),
        "recent_end": window.get("recent_end"),
        "baseline_start": window.get("baseline_start"),
        "baseline_end": window.get("baseline_end"),
        "total_signal_count": report.get("total_signal_count"),
        "baseline_signal_count": baseline.get("signal_count"),
        "recent_signal_count": recent.get("signal_count"),
        "baseline_labeled_60m": baseline.get("labeled_60m"),
        "recent_labeled_60m": recent.get("labeled_60m"),
        "baseline_sample_quality": baseline.get("sample_quality"),
        "recent_sample_quality": recent.get("sample_quality"),
        "outcome_evidence_status": overall.get("outcome_evidence_status"),
        "baseline_hit_rate_60m": baseline.get("hit_rate_60m"),
        "recent_hit_rate_60m": recent.get("hit_rate_60m"),
        "hit_rate_delta": overall.get("hit_rate_delta"),
        "hit_rate_ci_overlap": overall.get("hit_rate_ci_overlap"),
        "baseline_avg_return_60m_bps": baseline.get("avg_signed_return_60m_bps"),
        "recent_avg_return_60m_bps": recent.get("avg_signed_return_60m_bps"),
        "avg_return_delta_bps": overall.get("avg_return_delta_bps"),
        "return_ci_overlap": overall.get("return_ci_overlap"),
        "quality_label_coverage_ratio": label_summary.get("quality_label_coverage_ratio"),
        "raw_label_coverage_ratio": label_summary.get("raw_label_coverage_ratio"),
        "quality_coverage_delta": label_drift.get("quality_coverage_delta"),
        "excluded_labeled_rows": label_summary.get("excluded_labeled_rows"),
        "warning_count": int(len(warnings)),
        "caution_count": int(sum(1 for item in warnings if item.get("severity") == "CAUTION")),
        "watch_count": int(sum(1 for item in warnings if item.get("severity") == "WATCH")),
        "hybrid_abs_calibration_gap_delta": hybrid_cal.get("abs_calibration_gap_delta"),
        "hybrid_brier_delta": hybrid_cal.get("brier_delta"),
        "hybrid_ece_delta": hybrid_cal.get("ece_delta"),
        "ml_confidence_abs_calibration_gap_delta": ml_cal.get("abs_calibration_gap_delta"),
        "ml_confidence_brier_delta": ml_cal.get("brier_delta"),
        "ml_confidence_ece_delta": ml_cal.get("ece_delta"),
        "policy_count": policy_summary["policy_count"],
        "max_abs_policy_retention_delta": policy_summary["max_abs_policy_retention_delta"],
        "report_json": json_path,
        "report_markdown": markdown_path,
    }


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def append_signal_drift_trend_history(
    report: dict[str, Any],
    history_path: str | Path,
    *,
    json_path: str | None = None,
    markdown_path: str | None = None,
    report_name: str | None = None,
) -> pd.DataFrame:
    """Append one drift report row to the trend history CSV and return history."""
    path = Path(history_path)
    row = build_signal_drift_trend_row(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        report_name=report_name,
    )
    incoming = pd.DataFrame([row])
    with _exclusive_file_lock(path):
        if path.exists():
            try:
                existing = pd.read_csv(path)
            except Exception:
                existing = pd.DataFrame()
            history = pd.concat([existing, incoming], ignore_index=True, sort=False)
        else:
            history = incoming
        _atomic_write_csv(history, path)
    return history


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "monitor_status" not in frame.columns:
        return {}
    counts = frame["monitor_status"].fillna("UNKNOWN").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _series_last(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[-1]


def _series_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return _round_or_none(values.mean(), 4)


def build_signal_drift_trend_dashboard(
    history: pd.DataFrame,
    *,
    lookback_runs: int = 20,
) -> dict[str, Any]:
    """Build a compact dashboard from accumulated drift history."""
    frame = history.copy() if history is not None else pd.DataFrame()
    if frame.empty:
        return {
            "report_type": "signal_drift_trend_dashboard",
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "run_count": 0,
            "lookback_runs": int(lookback_runs),
            "latest": {},
            "status_counts": {},
            "lookback_summary": {},
            "trend_assessment": "NO_HISTORY",
        }

    frame["_generated_at"] = pd.to_datetime(frame.get("generated_at"), errors="coerce", utc=True)
    frame = frame.sort_values("_generated_at", na_position="last").reset_index(drop=True)
    lookback = frame.tail(max(int(lookback_runs), 1)).copy()
    latest = frame.iloc[-1].drop(labels=["_generated_at"], errors="ignore").to_dict()

    recent_avg_hit_delta = _series_mean(lookback, "hit_rate_delta")
    recent_avg_return_delta = _series_mean(lookback, "avg_return_delta_bps")
    recent_avg_warning_count = _series_mean(lookback, "warning_count")

    trend_assessment = "STABLE"
    if latest.get("monitor_status") == "CAUTION" or (
        recent_avg_hit_delta is not None and recent_avg_hit_delta < -0.08
    ) or (
        recent_avg_return_delta is not None and recent_avg_return_delta < -15.0
    ):
        trend_assessment = "DETERIORATING"
    elif latest.get("monitor_status") == "WATCH" or (recent_avg_warning_count is not None and recent_avg_warning_count > 0):
        trend_assessment = "WATCH"

    return _sanitize_value(
        {
            "report_type": "signal_drift_trend_dashboard",
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "run_count": int(len(frame)),
            "lookback_runs": int(len(lookback)),
            "latest": latest,
            "status_counts": _status_counts(frame),
            "lookback_summary": {
                "avg_hit_rate_delta": recent_avg_hit_delta,
                "avg_return_delta_bps": recent_avg_return_delta,
                "avg_quality_coverage_delta": _series_mean(lookback, "quality_coverage_delta"),
                "avg_warning_count": recent_avg_warning_count,
                "latest_monitor_status": _series_last(frame, "monitor_status"),
                "latest_recent_hit_rate_60m": _round_or_none(_series_last(frame, "recent_hit_rate_60m"), 4),
                "latest_recent_avg_return_60m_bps": _round_or_none(_series_last(frame, "recent_avg_return_60m_bps"), 4),
            },
            "trend_assessment": trend_assessment,
        }
    )


def render_signal_drift_trend_markdown(dashboard: dict[str, Any]) -> str:
    """Render the trend dashboard as Markdown."""
    latest = dashboard.get("latest", {}) or {}
    summary = dashboard.get("lookback_summary", {}) or {}
    lines = [
        "# Signal Drift Trend Dashboard",
        "",
        f"- Generated at: {dashboard.get('generated_at')}",
        f"- Trend assessment: **{dashboard.get('trend_assessment')}**",
        f"- Runs tracked: {dashboard.get('run_count')}",
        f"- Lookback runs: {dashboard.get('lookback_runs')}",
        "",
        "## Latest Run",
        "",
        f"- Monitor status: {latest.get('monitor_status')}",
        f"- Recent 60m hit rate: {latest.get('recent_hit_rate_60m')}",
        f"- Hit-rate delta: {latest.get('hit_rate_delta')}",
        f"- Avg return delta (bps): {latest.get('avg_return_delta_bps')}",
        f"- Warning count: {latest.get('warning_count')}",
        "",
        "## Lookback Summary",
        "",
        f"- Avg hit-rate delta: {summary.get('avg_hit_rate_delta')}",
        f"- Avg return delta (bps): {summary.get('avg_return_delta_bps')}",
        f"- Avg quality-label coverage delta: {summary.get('avg_quality_coverage_delta')}",
        f"- Avg warning count: {summary.get('avg_warning_count')}",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    status_counts = dashboard.get("status_counts", {}) or {}
    if status_counts:
        for status, count in status_counts.items():
            lines.append(f"| {status} | {count} |")
    else:
        lines.append("| (none) | 0 |")
    lines.extend(["", "*Research/ops dashboard only. No execution behavior is changed.*"])
    return "\n".join(lines)


def write_signal_drift_trend_dashboard(
    history: pd.DataFrame,
    *,
    output_dir: str | Path,
    lookback_runs: int = 20,
) -> dict[str, Any]:
    """Write latest trend dashboard JSON/Markdown artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dashboard = build_signal_drift_trend_dashboard(history, lookback_runs=lookback_runs)
    json_path = output / TREND_DASHBOARD_JSON_FILENAME
    markdown_path = output / TREND_DASHBOARD_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(dashboard, indent=2, default=str))
    _atomic_write_text(markdown_path, render_signal_drift_trend_markdown(dashboard))
    return {
        "trend_dashboard_json_path": str(json_path),
        "trend_dashboard_markdown_path": str(markdown_path),
        "trend_dashboard": dashboard,
    }


def render_signal_drift_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown view of the drift report."""
    overall = report.get("overall_outcome_drift", {})
    baseline = overall.get("baseline", {})
    recent = overall.get("recent", {})
    label_drift = report.get("label_quality_drift", {})
    lines = [
        "# Signal Drift Monitor",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Monitor status: {report.get('monitor_status')}",
        f"- Evaluation period: {report.get('evaluation_period', {}).get('start')} -> {report.get('evaluation_period', {}).get('end')}",
        "",
        "## Overall Drift",
        "",
        "| Window | Signals | Quality Labels | Evidence | Hit Rate 60m | Avg Return 60m (bps) |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
        f"| Baseline | {baseline.get('signal_count')} | {baseline.get('labeled_60m')} | {baseline.get('sample_quality')} | {baseline.get('hit_rate_60m')} | {baseline.get('avg_signed_return_60m_bps')} |",
        f"| Recent | {recent.get('signal_count')} | {recent.get('labeled_60m')} | {recent.get('sample_quality')} | {recent.get('hit_rate_60m')} | {recent.get('avg_signed_return_60m_bps')} |",
        "",
        f"- Outcome evidence status: {overall.get('outcome_evidence_status')}",
        f"- Hit-rate delta: {overall.get('hit_rate_delta')}",
        f"- Hit-rate CI overlap: {overall.get('hit_rate_ci_overlap')}",
        f"- Avg return delta (bps): {overall.get('avg_return_delta_bps')}",
        f"- Return CI overlap: {overall.get('return_ci_overlap')}",
        f"- Quality label coverage delta: {label_drift.get('quality_coverage_delta')}",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings", [])
    if warnings:
        for item in warnings:
            lines.append(f"- {item.get('severity')}: {item.get('category')} - {item.get('message')}")
    else:
        lines.append("- None")

    policy_rows = report.get("policy_retention_drift", [])
    if policy_rows:
        lines.extend(
            [
                "",
                "## Policy Retention Drift",
                "",
                "| Policy | Baseline Retention | Recent Retention | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in policy_rows:
            lines.append(
                f"| {row.get('policy')} | {row.get('baseline', {}).get('retention_ratio')} | "
                f"{row.get('recent', {}).get('retention_ratio')} | {row.get('retention_delta')} |"
            )

    calibration_rows = report.get("calibration_drift", [])
    if calibration_rows:
        lines.extend(
            [
                "",
                "## Calibration Drift",
                "",
                "| Probability Field | Baseline Gap | Recent Gap | Gap Delta | Brier Delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in calibration_rows:
            lines.append(
                f"| {row.get('probability_field')} | {row.get('baseline', {}).get('abs_calibration_gap')} | "
                f"{row.get('recent', {}).get('abs_calibration_gap')} | {row.get('abs_calibration_gap_delta')} | {row.get('brier_delta')} |"
            )

    dimension_rows = _flatten_dimension_rows(report)
    if dimension_rows:
        lines.extend(
            [
                "",
                "## Segment Outcome Drift",
                "",
                "| Dimension | Value | Recent Labels | Hit Delta | Return Delta (bps) | Status |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in dimension_rows[:30]:
            lines.append(
                f"| {row.get('dimension')} | {row.get('value')} | {row.get('recent_labeled_60m')} | "
                f"{row.get('hit_rate_delta')} | {row.get('avg_return_delta_bps')} | {row.get('status')} |"
            )

    lines.extend(["", "*Research/ops monitor only. No execution behavior is changed.*"])
    return "\n".join(lines)


def write_signal_drift_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    write_trend_history: bool = True,
    trend_lookback_runs: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write JSON, Markdown, and CSV drift artifacts."""
    report = build_signal_drift_report(frame, dataset_path=dataset_path, **kwargs)
    output = Path(output_dir) if output_dir is not None else SIGNAL_EVALUATION_REPORTS_DIR / "drift_monitoring"
    output.mkdir(parents=True, exist_ok=True)
    name = report_name or f"signal_drift_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"

    json_path = output / f"{name}.json"
    markdown_path = output / f"{name}.md"
    json_text = json.dumps(report, indent=2, default=str)
    markdown_text = render_signal_drift_markdown(report)
    _atomic_write_text(json_path, json_text)
    _atomic_write_text(markdown_path, markdown_text)

    latest_json_path: Path | None = None
    latest_markdown_path: Path | None = None
    if write_latest:
        latest_json_path = output / "latest_signal_drift.json"
        latest_markdown_path = output / "latest_signal_drift.md"
        _atomic_write_text(latest_json_path, json_text)
        _atomic_write_text(latest_markdown_path, markdown_text)

    csv_paths: dict[str, str] = {}
    tables = {
        "dimension_outcome_drift": _flatten_dimension_rows(report),
        "policy_retention_drift": report.get("policy_retention_drift", []),
        "calibration_drift": report.get("calibration_drift", []),
        "score_distribution_drift": report.get("score_distribution_drift", []),
        "warnings": report.get("warnings", []),
    }
    for table_name, rows in tables.items():
        path = output / f"{name}_{table_name}.csv"
        pd.json_normalize(rows).to_csv(path, index=False)
        csv_paths[table_name] = str(path)

    trend_history_path: Path | None = None
    trend_dashboard_json_path: str | None = None
    trend_dashboard_markdown_path: str | None = None
    trend_dashboard: dict[str, Any] | None = None
    if write_trend_history:
        trend_history_path = output / TREND_HISTORY_FILENAME
        history = append_signal_drift_trend_history(
            report,
            trend_history_path,
            json_path=str(json_path),
            markdown_path=str(markdown_path),
            report_name=name,
        )
        dashboard_artifact = write_signal_drift_trend_dashboard(
            history,
            output_dir=output,
            lookback_runs=trend_lookback_runs,
        )
        trend_dashboard_json_path = dashboard_artifact["trend_dashboard_json_path"]
        trend_dashboard_markdown_path = dashboard_artifact["trend_dashboard_markdown_path"]
        trend_dashboard = dashboard_artifact["trend_dashboard"]

    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "latest_json_path": str(latest_json_path) if latest_json_path else None,
        "latest_markdown_path": str(latest_markdown_path) if latest_markdown_path else None,
        "trend_history_path": str(trend_history_path) if trend_history_path else None,
        "trend_dashboard_json_path": trend_dashboard_json_path,
        "trend_dashboard_markdown_path": trend_dashboard_markdown_path,
        "trend_dashboard": trend_dashboard,
        "csv_paths": csv_paths,
        "report": report,
    }
