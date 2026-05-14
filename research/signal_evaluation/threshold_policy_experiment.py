"""Research-only policy experiment sandbox for governed threshold candidates.

This module compares a governed threshold candidate against the current
baseline signal population. It writes advisory artifacts only and never writes
runtime parameter packs or live configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_governance import (
    CONFIG_HINTS,
    PROMOTE_TO_REVIEW,
    THRESHOLD_GOVERNANCE_JSON_FILENAME,
)
from research.signal_evaluation.threshold_replay import (
    DEFAULT_REGIME_FIELDS,
    DEFAULT_WALK_FORWARD_HOLDOUT_DAYS,
    DEFAULT_WALK_FORWARD_STEP_DAYS,
    DEFAULT_WALK_FORWARD_TRAIN_DAYS,
    _metrics_for_subset,
    _prepare_frame,
    _round_or_none,
    _select_threshold_subset,
    _window_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_policy_experiments"
)
DEFAULT_GOVERNANCE_REPORT_PATH = (
    PROJECT_ROOT
    / "research"
    / "signal_evaluation"
    / "reports"
    / "threshold_governance"
    / THRESHOLD_GOVERNANCE_JSON_FILENAME
)

THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME = "latest_threshold_policy_experiment.json"
THRESHOLD_POLICY_EXPERIMENT_MARKDOWN_FILENAME = "latest_threshold_policy_experiment.md"
THRESHOLD_POLICY_EXPERIMENT_POLICY_PACK_FILENAME = "latest_candidate_threshold_policy_pack.json"
THRESHOLD_POLICY_EXPERIMENT_SPLITS_FILENAME = "latest_threshold_policy_experiment_splits.csv"
THRESHOLD_POLICY_EXPERIMENT_REGIMES_FILENAME = "latest_threshold_policy_experiment_regimes.csv"
THRESHOLD_POLICY_EXPERIMENT_QUALITY_BUCKETS_FILENAME = "latest_threshold_policy_experiment_quality_buckets.csv"

APPROVED_FOR_POLICY_EXPERIMENT = "APPROVED_FOR_POLICY_EXPERIMENT"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
REJECTED_FOR_POLICY_EXPERIMENT = "REJECTED_FOR_POLICY_EXPERIMENT"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
SKIPPED_NO_PROMOTED_CANDIDATE = "SKIPPED_NO_PROMOTED_CANDIDATE"

DEFAULT_EXPERIMENT_POLICY: dict[str, Any] = {
    "min_candidate_labels": 30,
    "min_retention_ratio": 0.10,
    "min_avg_return_delta_bps": 0.0,
    "min_hit_rate_delta": 0.0,
    "max_drawdown_worsening_bps": 50.0,
    "min_walk_forward_splits": 3,
    "min_positive_split_delta_rate": 0.60,
    "min_avg_split_return_delta_bps": 0.0,
    "min_regime_labels": 10,
    "max_bad_regime_count": 0,
    "top_n_regime_rows": 20,
    "min_holdout_labels": 10,
}

REGIME_COMPARISON_FIELDS = ("signal_regime", *DEFAULT_REGIME_FIELDS)
CONFIDENCE_BUCKET_FIELDS = (
    "ml_confidence_score",
    "ml_rank_score",
    "move_probability",
    "hybrid_move_probability",
)


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


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


def _candidate_key(candidate: dict[str, Any]) -> str:
    field = candidate.get("threshold_field") or "unknown"
    value = candidate.get("threshold_value")
    return f"{field}>={value}" if value is not None else str(field)


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    candidate_value = _safe_float(candidate.get(key))
    baseline_value = _safe_float(baseline.get(key))
    if candidate_value is None or baseline_value is None:
        return None
    return _round_or_none(candidate_value - baseline_value, 4)


def _comparison_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_count_delta": _safe_int(candidate.get("signal_count")) - _safe_int(baseline.get("signal_count")),
        "label_count_delta": _safe_int(candidate.get("label_count_60m")) - _safe_int(baseline.get("label_count_60m")),
        "hit_rate_delta": _metric_delta(candidate, baseline, "hit_rate_60m"),
        "avg_return_delta_bps": _metric_delta(candidate, baseline, "avg_signed_return_60m_bps"),
        "median_return_delta_bps": _metric_delta(candidate, baseline, "median_signed_return_60m_bps"),
        "sum_return_delta_bps": _metric_delta(candidate, baseline, "sum_signed_return_60m_bps"),
        "max_drawdown_delta_bps": _metric_delta(candidate, baseline, "max_drawdown_bps"),
        "objective_delta": _metric_delta(candidate, baseline, "objective_score"),
    }


def _metrics_pair(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float | None,
    min_label_sample: int,
    strong_label_sample: int,
) -> dict[str, Any]:
    baseline_selected, baseline_eligible_count = _select_threshold_subset(
        frame,
        threshold_field="ALL_SIGNALS",
        threshold_value=None,
    )
    candidate_selected, candidate_eligible_count = _select_threshold_subset(
        frame,
        threshold_field=threshold_field,
        threshold_value=threshold_value,
    )
    baseline = _metrics_for_subset(
        baseline_selected,
        eligible_count=max(baseline_eligible_count, 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    candidate = _metrics_for_subset(
        candidate_selected,
        eligible_count=max(candidate_eligible_count, 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": _comparison_delta(candidate, baseline),
    }


def _candidate_review_from_governance(
    governance_report: dict[str, Any] | None,
    *,
    candidate_key: str | None = None,
) -> dict[str, Any]:
    report = governance_report or {}
    reviews = list(report.get("candidate_reviews", []) or [])
    if candidate_key:
        for row in reviews:
            if str(row.get("candidate_key")) == str(candidate_key):
                return row
    top = report.get("top_candidate_review", {}) or {}
    if top:
        return top
    return {}


def load_candidate_review_from_governance(
    governance_report_path: str | Path,
    *,
    candidate_key: str | None = None,
) -> dict[str, Any]:
    """Load a governed threshold candidate from a governance JSON artifact."""
    report = json.loads(Path(governance_report_path).read_text(encoding="utf-8"))
    return _candidate_review_from_governance(report, candidate_key=candidate_key)


def build_candidate_policy_pack(candidate_review: dict[str, Any]) -> dict[str, Any]:
    """Build a research-only candidate policy pack artifact without writing runtime config."""
    threshold_field = candidate_review.get("threshold_field")
    threshold_value = candidate_review.get("threshold_value")
    config_hint = candidate_review.get("config_hint") or CONFIG_HINTS.get(
        str(threshold_field),
        "research_only.unmapped_threshold",
    )
    name = f"candidate_threshold_{str(threshold_field or 'unknown').lower()}_{str(threshold_value).replace('.', '_')}"
    overrides = {}
    if threshold_field and threshold_value is not None and str(threshold_field) != "ALL_SIGNALS":
        overrides[str(config_hint)] = threshold_value
    return _sanitize_value(
        {
            "report_type": "candidate_threshold_policy_pack",
            "name": name,
            "generated_at": _utc_now(),
            "source_candidate_key": candidate_review.get("candidate_key") or _candidate_key(candidate_review),
            "source_governance_status": candidate_review.get("governance_status"),
            "threshold_rule": {
                "field": threshold_field,
                "operator": ">=",
                "value": threshold_value,
            },
            "config_hint": config_hint,
            "overrides": overrides,
            "research_only": True,
            "runtime_config_changed": False,
        }
    )


def _walk_forward_comparison(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float | None,
    min_label_sample: int,
    strong_label_sample: int,
    train_window_days: int,
    holdout_window_days: int,
    step_days: int,
    min_holdout_labels: int,
) -> dict[str, Any]:
    splits = _window_splits(
        frame,
        train_window_days=train_window_days,
        holdout_window_days=holdout_window_days,
        step_days=step_days,
    )
    rows: list[dict[str, Any]] = []
    for split in splits:
        pair = _metrics_pair(
            split["holdout"],
            threshold_field=threshold_field,
            threshold_value=threshold_value,
            min_label_sample=min_label_sample,
            strong_label_sample=strong_label_sample,
        )
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        delta = pair["delta"]
        candidate_label_count = _safe_int(candidate.get("label_count_60m"))
        baseline_label_count = _safe_int(baseline.get("label_count_60m"))
        avg_return_delta = _safe_float(delta.get("avg_return_delta_bps"))
        candidate_avg_return = _safe_float(candidate.get("avg_signed_return_60m_bps"))
        candidate_hit_rate = _safe_float(candidate.get("hit_rate_60m"))
        if candidate_label_count < int(min_holdout_labels) or baseline_label_count < int(min_holdout_labels):
            split_status = "INSUFFICIENT_HOLDOUT"
        elif (
            avg_return_delta is not None
            and avg_return_delta >= 0
            and (candidate_avg_return is None or candidate_avg_return >= 0)
            and (candidate_hit_rate is None or candidate_hit_rate >= 0.5)
        ):
            split_status = "PASS"
        elif (avg_return_delta is not None and avg_return_delta < 0) or (
            candidate_avg_return is not None and candidate_avg_return < 0
        ):
            split_status = "FAIL"
        else:
            split_status = "MIXED"
        rows.append(
            {
                "split_id": split["split_id"],
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "holdout_start": split["holdout_start"],
                "holdout_end": split["holdout_end"],
                "split_status": split_status,
                "baseline_signal_count": baseline.get("signal_count"),
                "candidate_signal_count": candidate.get("signal_count"),
                "baseline_label_count_60m": baseline.get("label_count_60m"),
                "candidate_label_count_60m": candidate.get("label_count_60m"),
                "baseline_hit_rate_60m": baseline.get("hit_rate_60m"),
                "candidate_hit_rate_60m": candidate.get("hit_rate_60m"),
                "hit_rate_delta": delta.get("hit_rate_delta"),
                "baseline_avg_return_60m_bps": baseline.get("avg_signed_return_60m_bps"),
                "candidate_avg_return_60m_bps": candidate.get("avg_signed_return_60m_bps"),
                "avg_return_delta_bps": delta.get("avg_return_delta_bps"),
                "candidate_retention_ratio": candidate.get("retention_ratio"),
                "candidate_sample_quality": candidate.get("sample_quality"),
            }
        )

    evaluated = [row for row in rows if row.get("split_status") != "INSUFFICIENT_HOLDOUT"]
    delta_returns = pd.to_numeric(pd.Series([row.get("avg_return_delta_bps") for row in evaluated]), errors="coerce").dropna()
    candidate_returns = pd.to_numeric(
        pd.Series([row.get("candidate_avg_return_60m_bps") for row in evaluated]),
        errors="coerce",
    ).dropna()
    positive_delta_rate = (
        _round_or_none(float((delta_returns >= 0).mean()), 4)
        if not delta_returns.empty
        else None
    )
    candidate_positive_return_rate = (
        _round_or_none(float((candidate_returns >= 0).mean()), 4)
        if not candidate_returns.empty
        else None
    )
    if not splits:
        robustness_status = "INSUFFICIENT_HISTORY"
    elif not evaluated:
        robustness_status = "INSUFFICIENT_HOLDOUT"
    elif (
        positive_delta_rate is not None
        and positive_delta_rate >= 0.60
        and _safe_float(delta_returns.mean()) is not None
        and float(delta_returns.mean()) >= 0
        and (candidate_positive_return_rate is None or candidate_positive_return_rate >= 0.60)
    ):
        robustness_status = "ROBUST"
    elif positive_delta_rate is not None and positive_delta_rate >= 0.50:
        robustness_status = "MIXED"
    else:
        robustness_status = "UNSTABLE"

    return {
        "summary": {
            "split_count": int(len(splits)),
            "evaluated_split_count": int(len(evaluated)),
            "pass_count": int(sum(1 for row in rows if row.get("split_status") == "PASS")),
            "fail_count": int(sum(1 for row in rows if row.get("split_status") == "FAIL")),
            "positive_delta_rate": positive_delta_rate,
            "candidate_positive_return_rate": candidate_positive_return_rate,
            "avg_holdout_return_delta_bps": _round_or_none(delta_returns.mean(), 4) if not delta_returns.empty else None,
            "avg_candidate_holdout_return_60m_bps": _round_or_none(candidate_returns.mean(), 4) if not candidate_returns.empty else None,
            "worst_candidate_holdout_return_60m_bps": _round_or_none(candidate_returns.min(), 4) if not candidate_returns.empty else None,
            "robustness_status": robustness_status,
        },
        "splits": rows,
        "config": {
            "train_window_days": int(train_window_days),
            "holdout_window_days": int(holdout_window_days),
            "step_days": int(step_days),
            "min_holdout_labels": int(min_holdout_labels),
        },
    }


def _regime_comparison(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float | None,
    min_label_sample: int,
    strong_label_sample: int,
    min_regime_labels: int,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in REGIME_COMPARISON_FIELDS:
        if field not in frame.columns:
            continue
        counts = frame[field].astype("object").where(frame[field].notna(), "UNKNOWN").value_counts()
        for value in counts.index:
            group = frame.loc[frame[field].astype("object").where(frame[field].notna(), "UNKNOWN") == value].copy()
            pair = _metrics_pair(
                group,
                threshold_field=threshold_field,
                threshold_value=threshold_value,
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
            baseline = pair["baseline"]
            candidate = pair["candidate"]
            delta = pair["delta"]
            candidate_labels = _safe_int(candidate.get("label_count_60m"))
            avg_delta = _safe_float(delta.get("avg_return_delta_bps"))
            hit_delta = _safe_float(delta.get("hit_rate_delta"))
            if candidate_labels < int(min_regime_labels):
                status = "INSUFFICIENT_REGIME_EVIDENCE"
            elif (avg_delta is not None and avg_delta < 0) or (hit_delta is not None and hit_delta < -0.05):
                status = "REGIME_DETERIORATION"
            else:
                status = "PASS"
            rows.append(
                {
                    "regime_field": field,
                    "regime_value": str(value),
                    "regime_status": status,
                    "baseline_signal_count": baseline.get("signal_count"),
                    "candidate_signal_count": candidate.get("signal_count"),
                    "baseline_label_count_60m": baseline.get("label_count_60m"),
                    "candidate_label_count_60m": candidate.get("label_count_60m"),
                    "baseline_hit_rate_60m": baseline.get("hit_rate_60m"),
                    "candidate_hit_rate_60m": candidate.get("hit_rate_60m"),
                    "hit_rate_delta": delta.get("hit_rate_delta"),
                    "baseline_avg_return_60m_bps": baseline.get("avg_signed_return_60m_bps"),
                    "candidate_avg_return_60m_bps": candidate.get("avg_signed_return_60m_bps"),
                    "avg_return_delta_bps": delta.get("avg_return_delta_bps"),
                    "candidate_retention_ratio": candidate.get("retention_ratio"),
                    "candidate_sample_quality": candidate.get("sample_quality"),
                }
            )
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("regime_status") != "REGIME_DETERIORATION",
            -_safe_int(row.get("candidate_label_count_60m")),
            str(row.get("regime_field")),
            str(row.get("regime_value")),
        ),
    )
    return rows[: int(top_n)]


def _bucket_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(["UNKNOWN"] * len(series), index=series.index)
    if float(values.max()) > 1.5:
        bins = [-float("inf"), 50.0, 65.0, 80.0, float("inf")]
        labels = ["<50", "50_65", "65_80", "80+"]
    else:
        bins = [-float("inf"), 0.50, 0.65, 0.80, float("inf")]
        labels = ["<0.50", "0.50_0.65", "0.65_0.80", "0.80+"]
    bucketed = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    return bucketed.astype("object").where(bucketed.notna(), "UNKNOWN")


def _quality_bucket_comparison(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float | None,
    min_label_sample: int,
    strong_label_sample: int,
) -> list[dict[str, Any]]:
    buckets: list[tuple[str, pd.Series]] = []
    if "label_quality_status" in frame.columns:
        buckets.append(("label_quality_status", frame["label_quality_status"].astype("object").where(frame["label_quality_status"].notna(), "UNKNOWN")))
    for field in CONFIDENCE_BUCKET_FIELDS:
        if field in frame.columns:
            buckets.append((f"{field}_bucket", _bucket_label(frame[field])))

    rows: list[dict[str, Any]] = []
    for bucket_name, bucket_values in buckets:
        for value in sorted(set(str(item) for item in bucket_values.dropna().unique())):
            group = frame.loc[bucket_values.astype(str) == value].copy()
            if group.empty:
                continue
            pair = _metrics_pair(
                group,
                threshold_field=threshold_field,
                threshold_value=threshold_value,
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
            baseline = pair["baseline"]
            candidate = pair["candidate"]
            delta = pair["delta"]
            rows.append(
                {
                    "bucket_name": bucket_name,
                    "bucket_value": value,
                    "baseline_signal_count": baseline.get("signal_count"),
                    "candidate_signal_count": candidate.get("signal_count"),
                    "baseline_label_count_60m": baseline.get("label_count_60m"),
                    "candidate_label_count_60m": candidate.get("label_count_60m"),
                    "baseline_hit_rate_60m": baseline.get("hit_rate_60m"),
                    "candidate_hit_rate_60m": candidate.get("hit_rate_60m"),
                    "hit_rate_delta": delta.get("hit_rate_delta"),
                    "baseline_avg_return_60m_bps": baseline.get("avg_signed_return_60m_bps"),
                    "candidate_avg_return_60m_bps": candidate.get("avg_signed_return_60m_bps"),
                    "avg_return_delta_bps": delta.get("avg_return_delta_bps"),
                    "candidate_sample_quality": candidate.get("sample_quality"),
                }
            )
    return rows


def _classify_experiment(
    *,
    candidate_review: dict[str, Any],
    full_sample: dict[str, Any],
    walk_forward: dict[str, Any],
    regime_rows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    threshold_field = candidate_review.get("threshold_field")
    threshold_value = candidate_review.get("threshold_value")
    governance_status = candidate_review.get("governance_status")
    candidate = full_sample.get("candidate", {}) or {}
    delta = full_sample.get("delta", {}) or {}
    walk_summary = walk_forward.get("summary", {}) or {}

    if not threshold_field or threshold_value is None or str(threshold_field) == "ALL_SIGNALS":
        reasons.append("No concrete threshold candidate was supplied.")
        return INSUFFICIENT_EVIDENCE, reasons

    candidate_labels = _safe_int(candidate.get("label_count_60m"))
    candidate_retention = _safe_float(candidate.get("retention_ratio"))
    candidate_quality = str(candidate.get("sample_quality") or "NO_EVIDENCE")
    avg_return_delta = _safe_float(delta.get("avg_return_delta_bps"))
    hit_rate_delta = _safe_float(delta.get("hit_rate_delta"))
    drawdown_delta = _safe_float(delta.get("max_drawdown_delta_bps"))

    if candidate_labels < int(policy["min_candidate_labels"]) or candidate_quality in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"}:
        reasons.append(
            f"Candidate labels {candidate_labels} below minimum {int(policy['min_candidate_labels'])} or evidence is weak."
        )
        return INSUFFICIENT_EVIDENCE, reasons
    if candidate_retention is not None and candidate_retention < float(policy["min_retention_ratio"]):
        reasons.append(
            f"Candidate retention {candidate_retention} below minimum {float(policy['min_retention_ratio'])}."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons
    if avg_return_delta is not None and avg_return_delta < float(policy["min_avg_return_delta_bps"]):
        reasons.append(
            f"Full-sample return delta {avg_return_delta} bps below minimum {float(policy['min_avg_return_delta_bps'])} bps."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons
    if hit_rate_delta is not None and hit_rate_delta < float(policy["min_hit_rate_delta"]):
        reasons.append(
            f"Full-sample hit-rate delta {hit_rate_delta} below minimum {float(policy['min_hit_rate_delta'])}."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons
    if drawdown_delta is not None and drawdown_delta < -float(policy["max_drawdown_worsening_bps"]):
        reasons.append(
            f"Drawdown worsened by {abs(drawdown_delta)} bps, exceeding {float(policy['max_drawdown_worsening_bps'])} bps."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons

    evaluated_splits = _safe_int(walk_summary.get("evaluated_split_count"))
    positive_delta_rate = _safe_float(walk_summary.get("positive_delta_rate"))
    avg_split_delta = _safe_float(walk_summary.get("avg_holdout_return_delta_bps"))
    if evaluated_splits < int(policy["min_walk_forward_splits"]):
        reasons.append(
            f"Only {evaluated_splits} walk-forward split(s) evaluated; minimum is {int(policy['min_walk_forward_splits'])}."
        )
        return REVIEW_REQUIRED, reasons
    if positive_delta_rate is not None and positive_delta_rate < float(policy["min_positive_split_delta_rate"]):
        reasons.append(
            f"Walk-forward positive delta rate {positive_delta_rate} below minimum {float(policy['min_positive_split_delta_rate'])}."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons
    if avg_split_delta is not None and avg_split_delta < float(policy["min_avg_split_return_delta_bps"]):
        reasons.append(
            f"Average walk-forward return delta {avg_split_delta} bps below minimum {float(policy['min_avg_split_return_delta_bps'])} bps."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons

    bad_regime_count = sum(1 for row in regime_rows if row.get("regime_status") == "REGIME_DETERIORATION")
    if bad_regime_count > int(policy["max_bad_regime_count"]):
        reasons.append(
            f"{bad_regime_count} regime bucket(s) deteriorated, exceeding limit {int(policy['max_bad_regime_count'])}."
        )
        return REJECTED_FOR_POLICY_EXPERIMENT, reasons

    if governance_status != PROMOTE_TO_REVIEW:
        reasons.append(f"Governance status is {governance_status}; human review is still required before experiment approval.")
        return REVIEW_REQUIRED, reasons

    reasons.append("Candidate outperformed baseline under full-sample, walk-forward, and regime guardrails.")
    return APPROVED_FOR_POLICY_EXPERIMENT, reasons


def build_threshold_policy_experiment_report(
    frame: pd.DataFrame,
    *,
    governance_report: dict[str, Any] | None = None,
    candidate_review: dict[str, Any] | None = None,
    candidate_key: str | None = None,
    dataset_path: str | Path | None = None,
    governance_report_path: str | Path | None = None,
    policy: dict[str, Any] | None = None,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    walk_forward_train_days: int = DEFAULT_WALK_FORWARD_TRAIN_DAYS,
    walk_forward_holdout_days: int = DEFAULT_WALK_FORWARD_HOLDOUT_DAYS,
    walk_forward_step_days: int = DEFAULT_WALK_FORWARD_STEP_DAYS,
) -> dict[str, Any]:
    """Compare a governed threshold candidate against baseline signal history."""
    rules = {**DEFAULT_EXPERIMENT_POLICY, **(policy or {})}
    review = candidate_review or _candidate_review_from_governance(governance_report, candidate_key=candidate_key)
    working = _prepare_frame(frame)
    threshold_field = str(review.get("threshold_field") or "")
    threshold_value = _safe_float(review.get("threshold_value"))
    policy_pack = build_candidate_policy_pack(review)
    if working.empty:
        full_sample = {"baseline": {}, "candidate": {}, "delta": {}}
        walk_forward = {"summary": {"robustness_status": "INSUFFICIENT_HISTORY", "split_count": 0, "evaluated_split_count": 0}, "splits": [], "config": {}}
        regime_rows: list[dict[str, Any]] = []
        quality_rows: list[dict[str, Any]] = []
    else:
        full_sample = _metrics_pair(
            working,
            threshold_field=threshold_field,
            threshold_value=threshold_value,
            min_label_sample=min_label_sample,
            strong_label_sample=strong_label_sample,
        )
        walk_forward = _walk_forward_comparison(
            working,
            threshold_field=threshold_field,
            threshold_value=threshold_value,
            min_label_sample=min_label_sample,
            strong_label_sample=strong_label_sample,
            train_window_days=walk_forward_train_days,
            holdout_window_days=walk_forward_holdout_days,
            step_days=walk_forward_step_days,
            min_holdout_labels=int(rules["min_holdout_labels"]),
        )
        regime_rows = _regime_comparison(
            working,
            threshold_field=threshold_field,
            threshold_value=threshold_value,
            min_label_sample=min_label_sample,
            strong_label_sample=strong_label_sample,
            min_regime_labels=int(rules["min_regime_labels"]),
            top_n=int(rules["top_n_regime_rows"]),
        )
        quality_rows = _quality_bucket_comparison(
            working,
            threshold_field=threshold_field,
            threshold_value=threshold_value,
            min_label_sample=min_label_sample,
            strong_label_sample=strong_label_sample,
        )

    decision, reasons = _classify_experiment(
        candidate_review=review,
        full_sample=full_sample,
        walk_forward=walk_forward,
        regime_rows=regime_rows,
        policy=rules,
    )
    report = {
        "report_type": "threshold_policy_experiment",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "governance_report_path": str(governance_report_path) if governance_report_path is not None else None,
        "experiment_status": decision,
        "experiment_reasons": reasons,
        "runtime_config_changed": False,
        "policy": rules,
        "candidate_review": review,
        "candidate_policy_pack": policy_pack,
        "full_sample_comparison": full_sample,
        "walk_forward_comparison": walk_forward,
        "regime_comparison": regime_rows,
        "quality_bucket_comparison": quality_rows,
    }
    return _sanitize_value(report)


def render_threshold_policy_experiment_markdown(report: dict[str, Any]) -> str:
    """Render a threshold policy experiment report as Markdown."""
    pack = report.get("candidate_policy_pack", {}) or {}
    full_sample = report.get("full_sample_comparison", {}) or {}
    baseline = full_sample.get("baseline", {}) or {}
    candidate = full_sample.get("candidate", {}) or {}
    delta = full_sample.get("delta", {}) or {}
    walk = (report.get("walk_forward_comparison", {}) or {}).get("summary", {}) or {}
    lines = [
        "# Threshold Policy Experiment",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Governance report: {report.get('governance_report_path') or 'not supplied'}",
        f"- Experiment status: **{report.get('experiment_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        "",
        "## Candidate Policy Pack",
        "",
        f"- Name: `{pack.get('name')}`",
        f"- Candidate: `{pack.get('source_candidate_key')}`",
        f"- Governance status: {pack.get('source_governance_status')}",
        f"- Config hint: `{pack.get('config_hint')}`",
        f"- Research only: {pack.get('research_only')}",
        "",
        "## Decision Reasons",
        "",
    ]
    for reason in report.get("experiment_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Full-Sample Comparison",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Signals | {baseline.get('signal_count')} | {candidate.get('signal_count')} | {delta.get('signal_count_delta')} |",
            f"| 60m labels | {baseline.get('label_count_60m')} | {candidate.get('label_count_60m')} | {delta.get('label_count_delta')} |",
            f"| 60m hit rate | {baseline.get('hit_rate_60m')} | {candidate.get('hit_rate_60m')} | {delta.get('hit_rate_delta')} |",
            f"| Avg 60m return (bps) | {baseline.get('avg_signed_return_60m_bps')} | {candidate.get('avg_signed_return_60m_bps')} | {delta.get('avg_return_delta_bps')} |",
            f"| Max drawdown (bps) | {baseline.get('max_drawdown_bps')} | {candidate.get('max_drawdown_bps')} | {delta.get('max_drawdown_delta_bps')} |",
            f"| Objective | {baseline.get('objective_score')} | {candidate.get('objective_score')} | {delta.get('objective_delta')} |",
            "",
            "## Walk-Forward Comparison",
            "",
            f"- Robustness status: {walk.get('robustness_status')}",
            f"- Evaluated splits: {walk.get('evaluated_split_count')} / {walk.get('split_count')}",
            f"- Positive delta rate: {walk.get('positive_delta_rate')}",
            f"- Avg holdout return delta (bps): {walk.get('avg_holdout_return_delta_bps')}",
            "",
            "## Regime Guardrails",
            "",
            "| Regime | Value | Status | Candidate Labels | Return Delta (bps) | Hit-Rate Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("regime_comparison", [])[:10]:
        lines.append(
            f"| {row.get('regime_field')} | {row.get('regime_value')} | {row.get('regime_status')} | "
            f"{row.get('candidate_label_count_60m')} | {row.get('avg_return_delta_bps')} | {row.get('hit_rate_delta')} |"
        )
    lines.extend(
        [
            "",
            "*This artifact is advisory. Human approval is required before any policy-pack or runtime threshold change.*",
        ]
    )
    return "\n".join(lines)


def write_threshold_policy_experiment_report(
    frame: pd.DataFrame,
    *,
    governance_report: dict[str, Any] | None = None,
    candidate_review: dict[str, Any] | None = None,
    candidate_key: str | None = None,
    dataset_path: str | Path | None = None,
    governance_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write policy experiment artifacts for a governed threshold candidate."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_policy_experiment"
    report = build_threshold_policy_experiment_report(
        frame,
        governance_report=governance_report,
        candidate_review=candidate_review,
        candidate_key=candidate_key,
        dataset_path=dataset_path,
        governance_report_path=governance_report_path,
        policy=policy,
    )
    markdown = render_threshold_policy_experiment_markdown(report)
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    policy_pack_path = output / f"{stem}_candidate_policy_pack.json"
    splits_path = output / f"{stem}_splits.csv"
    regimes_path = output / f"{stem}_regimes.csv"
    quality_path = output / f"{stem}_quality_buckets.csv"

    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, markdown)
    _atomic_write_text(policy_pack_path, json.dumps(report.get("candidate_policy_pack", {}), indent=2, sort_keys=True, default=str))
    _atomic_write_csv(pd.DataFrame(report.get("walk_forward_comparison", {}).get("splits", []) or []), splits_path)
    _atomic_write_csv(pd.DataFrame(report.get("regime_comparison", []) or []), regimes_path)
    _atomic_write_csv(pd.DataFrame(report.get("quality_bucket_comparison", []) or []), quality_path)

    latest_json_path = output / THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME
    latest_markdown_path = output / THRESHOLD_POLICY_EXPERIMENT_MARKDOWN_FILENAME
    latest_policy_pack_path = output / THRESHOLD_POLICY_EXPERIMENT_POLICY_PACK_FILENAME
    latest_splits_path = output / THRESHOLD_POLICY_EXPERIMENT_SPLITS_FILENAME
    latest_regimes_path = output / THRESHOLD_POLICY_EXPERIMENT_REGIMES_FILENAME
    latest_quality_path = output / THRESHOLD_POLICY_EXPERIMENT_QUALITY_BUCKETS_FILENAME
    if write_latest:
        _atomic_write_text(latest_json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown_path, markdown)
        _atomic_write_text(
            latest_policy_pack_path,
            json.dumps(report.get("candidate_policy_pack", {}), indent=2, sort_keys=True, default=str),
        )
        _atomic_write_csv(pd.DataFrame(report.get("walk_forward_comparison", {}).get("splits", []) or []), latest_splits_path)
        _atomic_write_csv(pd.DataFrame(report.get("regime_comparison", []) or []), latest_regimes_path)
        _atomic_write_csv(pd.DataFrame(report.get("quality_bucket_comparison", []) or []), latest_quality_path)

    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "candidate_policy_pack_path": str(policy_pack_path),
        "splits_csv_path": str(splits_path),
        "regimes_csv_path": str(regimes_path),
        "quality_buckets_csv_path": str(quality_path),
        "latest_json_path": str(latest_json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "latest_candidate_policy_pack_path": str(latest_policy_pack_path),
        "latest_splits_csv_path": str(latest_splits_path),
        "latest_regimes_csv_path": str(latest_regimes_path),
        "latest_quality_buckets_csv_path": str(latest_quality_path),
    }


def write_threshold_policy_experiment_skip(
    *,
    reason: str,
    candidate_review: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    governance_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Write an explicit skipped experiment artifact to prevent stale latest files."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_policy_experiment"
    review = candidate_review or {}
    report = _sanitize_value(
        {
            "report_type": "threshold_policy_experiment",
            "generated_at": _utc_now(),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "governance_report_path": str(governance_report_path) if governance_report_path is not None else None,
            "experiment_status": SKIPPED_NO_PROMOTED_CANDIDATE,
            "experiment_reasons": [reason],
            "runtime_config_changed": False,
            "policy": DEFAULT_EXPERIMENT_POLICY,
            "candidate_review": review,
            "candidate_policy_pack": build_candidate_policy_pack(review) if review else {},
            "full_sample_comparison": {"baseline": {}, "candidate": {}, "delta": {}},
            "walk_forward_comparison": {
                "summary": {
                    "robustness_status": "SKIPPED",
                    "split_count": 0,
                    "evaluated_split_count": 0,
                },
                "splits": [],
                "config": {},
            },
            "regime_comparison": [],
            "quality_bucket_comparison": [],
        }
    )
    markdown = render_threshold_policy_experiment_markdown(report)
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    policy_pack_path = output / f"{stem}_candidate_policy_pack.json"
    splits_path = output / f"{stem}_splits.csv"
    regimes_path = output / f"{stem}_regimes.csv"
    quality_path = output / f"{stem}_quality_buckets.csv"

    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, markdown)
    _atomic_write_text(policy_pack_path, json.dumps(report.get("candidate_policy_pack", {}), indent=2, sort_keys=True, default=str))
    _atomic_write_csv(pd.DataFrame(), splits_path)
    _atomic_write_csv(pd.DataFrame(), regimes_path)
    _atomic_write_csv(pd.DataFrame(), quality_path)

    latest_json_path = output / THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME
    latest_markdown_path = output / THRESHOLD_POLICY_EXPERIMENT_MARKDOWN_FILENAME
    latest_policy_pack_path = output / THRESHOLD_POLICY_EXPERIMENT_POLICY_PACK_FILENAME
    latest_splits_path = output / THRESHOLD_POLICY_EXPERIMENT_SPLITS_FILENAME
    latest_regimes_path = output / THRESHOLD_POLICY_EXPERIMENT_REGIMES_FILENAME
    latest_quality_path = output / THRESHOLD_POLICY_EXPERIMENT_QUALITY_BUCKETS_FILENAME
    if write_latest:
        _atomic_write_text(latest_json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown_path, markdown)
        _atomic_write_text(
            latest_policy_pack_path,
            json.dumps(report.get("candidate_policy_pack", {}), indent=2, sort_keys=True, default=str),
        )
        _atomic_write_csv(pd.DataFrame(), latest_splits_path)
        _atomic_write_csv(pd.DataFrame(), latest_regimes_path)
        _atomic_write_csv(pd.DataFrame(), latest_quality_path)

    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "candidate_policy_pack_path": str(policy_pack_path),
        "splits_csv_path": str(splits_path),
        "regimes_csv_path": str(regimes_path),
        "quality_buckets_csv_path": str(quality_path),
        "latest_json_path": str(latest_json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "latest_candidate_policy_pack_path": str(latest_policy_pack_path),
        "latest_splits_csv_path": str(latest_splits_path),
        "latest_regimes_csv_path": str(latest_regimes_path),
        "latest_quality_buckets_csv_path": str(latest_quality_path),
    }
