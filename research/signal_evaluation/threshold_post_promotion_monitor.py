"""Post-promotion monitoring for manually approved threshold candidates.

This module watches approved promotion-review ledger entries against newer
signal outcomes. It is advisory only and never applies or reverts runtime
thresholds, parameter packs, or execution settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_promotion_review import (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME,
    THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME,
    PROMOTION_REVIEW_READY,
)
from research.signal_evaluation.threshold_replay import (
    DEFAULT_REGIME_FIELDS,
    _metrics_for_subset,
    _prepare_frame,
    _round_or_none,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_post_promotion_monitoring"
)
DEFAULT_PROMOTION_REVIEW_REPORT_PATH = (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR / THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME
)
DEFAULT_PROMOTION_REVIEW_LEDGER_PATH = (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR / THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME
)

THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME = "latest_threshold_post_promotion_monitor.json"
THRESHOLD_POST_PROMOTION_MONITOR_MARKDOWN_FILENAME = "latest_threshold_post_promotion_monitor.md"
THRESHOLD_POST_PROMOTION_MONITOR_SEGMENTS_FILENAME = "latest_threshold_post_promotion_monitor_segments.csv"

POST_PROMOTION_HEALTHY = "POST_PROMOTION_HEALTHY"
POST_PROMOTION_WATCH = "POST_PROMOTION_WATCH"
POST_PROMOTION_DETERIORATING = "POST_PROMOTION_DETERIORATING"
POST_PROMOTION_INSUFFICIENT_DATA = "POST_PROMOTION_INSUFFICIENT_DATA"
POST_PROMOTION_SKIPPED_NO_APPROVAL = "POST_PROMOTION_SKIPPED_NO_APPROVAL"

REGIME_FIELDS = ("signal_regime", *DEFAULT_REGIME_FIELDS)
BUCKET_FIELDS = (
    "label_quality_status",
    "ml_confidence_score",
    "ml_rank_score",
    "move_probability",
    "hybrid_move_probability",
)

DEFAULT_POST_PROMOTION_POLICY: dict[str, Any] = {
    "min_post_approval_labels": 20,
    "min_post_approval_signal_days": 5,
    "min_suppressed_labels": 5,
    "watch_retained_avg_return_drop_bps": 5.0,
    "max_retained_avg_return_drop_bps": 10.0,
    "watch_hit_rate_drop": 0.05,
    "max_hit_rate_drop": 0.10,
    "watch_true_positive_lost_count": 1,
    "max_true_positive_lost_count": 2,
    "max_true_positive_loss_rate": 0.20,
    "watch_true_positive_lost_increase_count": 1,
    "max_true_positive_lost_increase_count": 2,
    "max_true_positive_loss_rate_increase": 0.05,
    "min_false_positive_removal_rate": 0.40,
    "min_avoided_suppressed_return_bps": 0.0,
    "min_segment_labels": 10,
    "min_segment_retained_avg_return_bps": 0.0,
    "min_segment_hit_rate": 0.45,
    "max_segment_true_positive_lost_count": 1,
    "max_deteriorating_segment_count": 0,
    "max_watch_segment_count": 2,
}


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


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_approved_decision(
    ledger_path: str | Path | None,
    *,
    candidate_key: str | None = None,
) -> dict[str, Any] | None:
    if ledger_path is None:
        return None
    path = Path(ledger_path)
    if not path.exists():
        return None
    try:
        ledger = pd.read_csv(path)
    except Exception:
        return None
    if ledger.empty or "review_action" not in ledger.columns:
        return None
    rows = ledger.loc[ledger["review_action"].astype(str).str.upper() == "APPROVED"].copy()
    if candidate_key and "candidate_key" in rows.columns:
        rows = rows.loc[rows["candidate_key"].astype(str) == str(candidate_key)].copy()
    if rows.empty:
        return None
    if "reviewed_at" in rows.columns:
        rows["_reviewed_at_ts"] = pd.to_datetime(rows["reviewed_at"], errors="coerce", utc=True)
        rows = rows.sort_values("_reviewed_at_ts", na_position="first")
    return _sanitize_value(rows.iloc[-1].drop(labels=["_reviewed_at_ts"], errors="ignore").to_dict())


def _threshold_rule(promotion_package: dict[str, Any]) -> dict[str, Any]:
    candidate = promotion_package.get("promotion_candidate", {}) or {}
    return candidate.get("threshold_rule", {}) or {}


def _select_post_promotion_sets(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if threshold_field not in frame.columns:
        return frame.iloc[0:0].copy(), frame.iloc[0:0].copy(), 0
    values = pd.to_numeric(frame[threshold_field], errors="coerce")
    eligible = frame.loc[values.notna()].copy()
    eligible_values = pd.to_numeric(eligible[threshold_field], errors="coerce")
    retained = eligible.loc[eligible_values >= float(threshold_value)].copy()
    suppressed = eligible.loc[eligible_values < float(threshold_value)].copy()
    return retained, suppressed, int(len(eligible))


def _filter_after_approval(frame: pd.DataFrame, approved_at: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "signal_timestamp" not in frame.columns:
        return frame.iloc[0:0].copy(), {
            "approval_timestamp": str(approved_at) if approved_at is not None else None,
            "post_approval_signal_count": 0,
            "post_approval_signal_dates": 0,
            "first_post_approval_signal_timestamp": None,
            "last_post_approval_signal_timestamp": None,
            "timestamp_warning": "signal_timestamp column is unavailable.",
        }
    signal_ts = pd.to_datetime(frame["signal_timestamp"], errors="coerce", utc=True)
    approval_ts = pd.to_datetime(approved_at, errors="coerce", utc=True)
    if pd.isna(approval_ts):
        filtered = frame.loc[signal_ts.notna()].copy()
    else:
        filtered = frame.loc[signal_ts.notna() & (signal_ts > approval_ts)].copy()
    filtered_ts = pd.to_datetime(filtered.get("signal_timestamp", pd.Series(dtype=object)), errors="coerce", utc=True).dropna()
    return filtered, {
        "approval_timestamp": approval_ts.isoformat() if not pd.isna(approval_ts) else None,
        "post_approval_signal_count": int(len(filtered)),
        "post_approval_signal_dates": int(filtered_ts.dt.date.nunique()) if not filtered_ts.empty else 0,
        "first_post_approval_signal_timestamp": filtered_ts.min().isoformat() if not filtered_ts.empty else None,
        "last_post_approval_signal_timestamp": filtered_ts.max().isoformat() if not filtered_ts.empty else None,
        "timestamp_warning": None if not pd.isna(approval_ts) else "approval reviewed_at timestamp is unavailable or unparsable.",
    }


def _suppression_summary(retained: pd.DataFrame, suppressed: pd.DataFrame, *, eligible_count: int) -> dict[str, Any]:
    suppressed_hit = pd.to_numeric(suppressed.get("correct_60m", pd.Series(dtype=float)), errors="coerce").dropna()
    retained_hit = pd.to_numeric(retained.get("correct_60m", pd.Series(dtype=float)), errors="coerce").dropna()
    suppressed_ret = pd.to_numeric(suppressed.get("signed_return_60m_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    false_positive_removed = int((suppressed_hit < 0.5).sum())
    true_positive_lost = int((suppressed_hit >= 0.5).sum())
    negative_return_removed = int((suppressed_ret <= 0).sum())
    positive_return_lost = int((suppressed_ret > 0).sum())
    suppressed_label_count = int(suppressed_hit.count())
    return {
        "eligible_signal_count": int(eligible_count),
        "retained_signal_count": int(len(retained)),
        "suppressed_signal_count": int(len(suppressed)),
        "retention_ratio": _round_or_none(len(retained) / max(int(eligible_count), 1), 4),
        "suppression_ratio": _round_or_none(len(suppressed) / max(int(eligible_count), 1), 4),
        "retained_label_count_60m": int(retained_hit.count()),
        "suppressed_label_count_60m": suppressed_label_count,
        "false_positive_removed_count": false_positive_removed,
        "true_positive_lost_count": true_positive_lost,
        "false_positive_retained_count": int((retained_hit < 0.5).sum()),
        "true_positive_retained_count": int((retained_hit >= 0.5).sum()),
        "negative_return_removed_count": negative_return_removed,
        "positive_return_lost_count": positive_return_lost,
        "false_positive_removal_rate": _round_or_none(false_positive_removed / max(suppressed_label_count, 1), 4)
        if suppressed_label_count
        else None,
        "true_positive_loss_rate": _round_or_none(true_positive_lost / max(suppressed_label_count, 1), 4)
        if suppressed_label_count
        else None,
        "avoided_suppressed_return_bps": _round_or_none(-suppressed_ret.sum(), 4) if not suppressed_ret.empty else None,
        "avg_suppressed_return_60m_bps": _round_or_none(suppressed_ret.mean(), 4) if not suppressed_ret.empty else None,
    }


def _bucket_label(series: pd.Series) -> pd.Series:
    if str(series.name) == "label_quality_status":
        return series.astype("object").where(series.notna(), "UNKNOWN")
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        raw = series.astype("object").where(series.notna(), "UNKNOWN")
        return raw if not raw.dropna().empty else pd.Series(["UNKNOWN"] * len(series), index=series.index)
    if float(values.max()) > 1.5:
        bins = [-float("inf"), 50.0, 65.0, 80.0, float("inf")]
        labels = ["<50", "50_65", "65_80", "80+"]
    else:
        bins = [-float("inf"), 0.50, 0.65, 0.80, float("inf")]
        labels = ["<0.50", "0.50_0.65", "0.65_0.80", "0.80+"]
    bucketed = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    return bucketed.astype("object").where(bucketed.notna(), "UNKNOWN")


def _segment_rows(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float,
    segment_fields: tuple[str, ...],
    segment_kind: str,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in segment_fields:
        if field not in frame.columns:
            continue
        values = _bucket_label(frame[field])
        for value in sorted(set(str(item) for item in values.dropna().unique())):
            group = frame.loc[values.astype(str) == value].copy()
            if group.empty:
                continue
            retained, suppressed, eligible_count = _select_post_promotion_sets(
                group,
                threshold_field=threshold_field,
                threshold_value=threshold_value,
            )
            retained_metrics = _metrics_for_subset(
                retained,
                eligible_count=max(eligible_count, 1),
                min_label_sample=1,
                strong_label_sample=max(int(policy["min_segment_labels"]), 1),
            )
            summary = _suppression_summary(retained, suppressed, eligible_count=eligible_count)
            retained_labels = _safe_int(retained_metrics.get("label_count_60m"))
            suppressed_labels = _safe_int(summary.get("suppressed_label_count_60m"))
            retained_avg_return = _safe_float(retained_metrics.get("avg_signed_return_60m_bps"))
            retained_hit_rate = _safe_float(retained_metrics.get("hit_rate_60m"))
            false_positive_removal_rate = _safe_float(summary.get("false_positive_removal_rate"))
            true_positive_lost = _safe_int(summary.get("true_positive_lost_count"))
            suppressed_avg_return = _safe_float(summary.get("avg_suppressed_return_60m_bps"))
            retained_return_bad = (
                retained_avg_return is not None
                and retained_avg_return < float(policy["min_segment_retained_avg_return_bps"])
            )
            retained_hit_bad = (
                retained_hit_rate is not None
                and retained_hit_rate < float(policy["min_segment_hit_rate"])
            )
            weak_false_positive_filter = (
                suppressed_labels >= int(policy["min_segment_labels"])
                and false_positive_removal_rate is not None
                and false_positive_removal_rate < float(policy["min_false_positive_removal_rate"])
            )
            harmful_segment_suppression = (
                true_positive_lost > int(policy["max_segment_true_positive_lost_count"])
                and suppressed_avg_return is not None
                and suppressed_avg_return > 0
                and (retained_avg_return is None or retained_avg_return < float(policy["min_segment_retained_avg_return_bps"]))
            )
            if retained_labels < int(policy["min_segment_labels"]) and suppressed_labels < int(policy["min_segment_labels"]):
                segment_status = "INSUFFICIENT_SEGMENT_DATA"
            elif retained_labels < int(policy["min_segment_labels"]):
                segment_status = (
                    "WATCH"
                    if weak_false_positive_filter
                    or (suppressed_avg_return is not None and suppressed_avg_return > 0)
                    else "INSUFFICIENT_SEGMENT_DATA"
                )
            elif retained_return_bad or (retained_hit_bad and (retained_avg_return is None or retained_return_bad)):
                segment_status = "DETERIORATING"
            elif weak_false_positive_filter or harmful_segment_suppression:
                segment_status = "WATCH"
            else:
                segment_status = "PASS"
            rows.append(
                {
                    "segment_kind": segment_kind,
                    "segment_field": field,
                    "segment_value": value,
                    "segment_status": segment_status,
                    **summary,
                    "retained_hit_rate_60m": retained_metrics.get("hit_rate_60m"),
                    "retained_avg_return_60m_bps": retained_metrics.get("avg_signed_return_60m_bps"),
                    "retained_sample_quality": retained_metrics.get("sample_quality"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row.get("segment_status") != "DETERIORATING",
            row.get("segment_status") != "WATCH",
            -_safe_int(row.get("suppressed_signal_count")),
            str(row.get("segment_field")),
            str(row.get("segment_value")),
        ),
    )


def _recommended_action(status: str) -> str:
    if status == POST_PROMOTION_HEALTHY:
        return "Reaffirm the approved candidate; continue routine post-promotion monitoring."
    if status == POST_PROMOTION_WATCH:
        return "Keep the candidate under watch and review again after more post-approval labels."
    if status == POST_PROMOTION_DETERIORATING:
        return "Open a manual rollback or threshold-reconsideration review; this monitor does not revert config."
    if status == POST_PROMOTION_INSUFFICIENT_DATA:
        return "Keep collecting post-approval outcomes before making a post-promotion judgment."
    return "No approved promotion decision is available; record APPROVED in the promotion ledger before monitoring."


def _skip_report(
    *,
    reason: str,
    dataset_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    promotion_package_report_path: str | Path | None = None,
) -> dict[str, Any]:
    return _sanitize_value(
        {
            "report_type": "threshold_post_promotion_monitor",
            "generated_at": _utc_now(),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "promotion_review_ledger_path": str(ledger_path) if ledger_path is not None else None,
            "promotion_package_report_path": str(promotion_package_report_path) if promotion_package_report_path is not None else None,
            "monitor_status": POST_PROMOTION_SKIPPED_NO_APPROVAL,
            "monitor_reasons": [reason],
            "recommended_next_action": _recommended_action(POST_PROMOTION_SKIPPED_NO_APPROVAL),
            "runtime_config_changed": False,
            "approval_decision": {},
            "promotion_candidate": {},
            "threshold_rule": {},
            "post_approval_window": {},
            "post_approval_impact": {},
            "post_approval_baseline_metrics": {},
            "post_approval_retained_metrics": {},
            "post_approval_suppressed_metrics": {},
            "post_approval_retained_vs_baseline_delta": {},
            "guardrail_summary": {},
            "shadow_expectation": {},
            "drift_from_shadow_expectation": {},
            "segment_monitoring": [],
            "segment_summary": {},
        }
    )


def build_threshold_post_promotion_monitor_report(
    frame: pd.DataFrame,
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    approval_decision: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    candidate_key: str | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an advisory monitor for the latest approved threshold promotion."""
    rules = {**DEFAULT_POST_PROMOTION_POLICY, **(policy or {})}
    ledger = Path(ledger_path) if ledger_path is not None else DEFAULT_PROMOTION_REVIEW_LEDGER_PATH
    approval = approval_decision or _latest_approved_decision(ledger, candidate_key=candidate_key)
    if not approval:
        return _skip_report(
            reason="No APPROVED threshold promotion review decision is available.",
            dataset_path=dataset_path,
            ledger_path=ledger,
            promotion_package_report_path=promotion_package_report_path,
        )

    package_path = (
        promotion_package_report_path
        or approval.get("report_json")
        or DEFAULT_PROMOTION_REVIEW_REPORT_PATH
    )
    promotion_package = promotion_package_report or _load_json_if_exists(package_path)
    if promotion_package.get("promotion_review_status") != PROMOTION_REVIEW_READY:
        return _skip_report(
            reason=(
                "Approved ledger row does not point to a PROMOTION_REVIEW_READY package; "
                f"package status is {promotion_package.get('promotion_review_status') or 'UNKNOWN'}."
            ),
            dataset_path=dataset_path,
            ledger_path=ledger,
            promotion_package_report_path=package_path,
        )

    rule = _threshold_rule(promotion_package)
    threshold_field = str(rule.get("field") or "")
    threshold_value = _safe_float(rule.get("value"))
    if not threshold_field or threshold_value is None:
        return _skip_report(
            reason="Approved promotion package does not contain a concrete threshold rule.",
            dataset_path=dataset_path,
            ledger_path=ledger,
            promotion_package_report_path=package_path,
        )

    working = _prepare_frame(frame)
    post_frame, post_window = _filter_after_approval(working, approval.get("reviewed_at"))
    retained, suppressed, eligible_count = _select_post_promotion_sets(
        post_frame,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
    )
    eligible = post_frame.loc[pd.to_numeric(post_frame.get(threshold_field, pd.Series(index=post_frame.index)), errors="coerce").notna()].copy()
    baseline_metrics = _metrics_for_subset(
        eligible,
        eligible_count=max(eligible_count, 1),
        min_label_sample=int(rules["min_post_approval_labels"]),
        strong_label_sample=max(int(rules["min_post_approval_labels"]) * 2, 1),
    )
    retained_metrics = _metrics_for_subset(
        retained,
        eligible_count=max(eligible_count, 1),
        min_label_sample=int(rules["min_post_approval_labels"]),
        strong_label_sample=max(int(rules["min_post_approval_labels"]) * 2, 1),
    )
    suppressed_metrics = _metrics_for_subset(
        suppressed,
        eligible_count=max(eligible_count, 1),
        min_label_sample=int(rules["min_suppressed_labels"]),
        strong_label_sample=max(int(rules["min_post_approval_labels"]), 1),
    )
    impact = _suppression_summary(retained, suppressed, eligible_count=eligible_count)
    retained_vs_baseline = _comparison_delta(retained_metrics, baseline_metrics)

    shadow_retained = promotion_package.get("retained_metrics", {}) or {}
    shadow_impact = promotion_package.get("impact_summary", {}) or {}
    drift = {
        "label_count_delta_vs_shadow": _safe_int(retained_metrics.get("label_count_60m"))
        - _safe_int(shadow_retained.get("label_count_60m")),
        "retained_hit_rate_delta_vs_shadow": _metric_delta(retained_metrics, shadow_retained, "hit_rate_60m"),
        "retained_avg_return_delta_bps_vs_shadow": _metric_delta(
            retained_metrics,
            shadow_retained,
            "avg_signed_return_60m_bps",
        ),
        "false_positive_removed_delta_vs_shadow": _safe_int(impact.get("false_positive_removed_count"))
        - _safe_int(shadow_impact.get("false_positive_removed_count")),
        "true_positive_lost_delta_vs_shadow": _safe_int(impact.get("true_positive_lost_count"))
        - _safe_int(shadow_impact.get("true_positive_lost_count")),
        "retention_ratio_delta_vs_shadow": _metric_delta(impact, shadow_impact, "retention_ratio"),
        "false_positive_removal_rate_delta_vs_shadow": _metric_delta(
            impact,
            shadow_impact,
            "false_positive_removal_rate",
        ),
        "true_positive_loss_rate_delta_vs_shadow": _metric_delta(
            impact,
            shadow_impact,
            "true_positive_loss_rate",
        ),
        "avoided_suppressed_return_delta_bps_vs_shadow": _metric_delta(
            impact,
            shadow_impact,
            "avoided_suppressed_return_bps",
        ),
        "avg_suppressed_return_delta_bps_vs_shadow": _metric_delta(
            impact,
            shadow_impact,
            "avg_suppressed_return_60m_bps",
        ),
    }
    segment_rows = _segment_rows(
        post_frame,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
        segment_fields=REGIME_FIELDS,
        segment_kind="regime",
        policy=rules,
    ) + _segment_rows(
        post_frame,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
        segment_fields=BUCKET_FIELDS,
        segment_kind="bucket",
        policy=rules,
    )
    deteriorating_segments = sum(1 for row in segment_rows if row.get("segment_status") == "DETERIORATING")
    watch_segments = sum(1 for row in segment_rows if row.get("segment_status") == "WATCH")

    reasons: list[str] = []
    retained_labels = _safe_int(retained_metrics.get("label_count_60m"))
    post_days = _safe_int(post_window.get("post_approval_signal_dates"))
    suppressed_labels = _safe_int(impact.get("suppressed_label_count_60m"))
    avg_return_drop = _safe_float(drift.get("retained_avg_return_delta_bps_vs_shadow"))
    hit_rate_drop = _safe_float(drift.get("retained_hit_rate_delta_vs_shadow"))
    true_positive_lost = _safe_int(impact.get("true_positive_lost_count"))
    true_positive_loss_rate = _safe_float(impact.get("true_positive_loss_rate"))
    true_positive_lost_delta = _safe_int(drift.get("true_positive_lost_delta_vs_shadow"))
    true_positive_loss_rate_delta = _safe_float(drift.get("true_positive_loss_rate_delta_vs_shadow"))
    false_positive_removal_rate = _safe_float(impact.get("false_positive_removal_rate"))
    avoided_return = _safe_float(impact.get("avoided_suppressed_return_bps"))
    avg_suppressed_return = _safe_float(impact.get("avg_suppressed_return_60m_bps"))
    true_positive_loss_is_harmful = (
        (avoided_return is None or avoided_return < float(rules["min_avoided_suppressed_return_bps"]))
        or (avg_suppressed_return is not None and avg_suppressed_return > 0)
    )
    true_positive_loss_deteriorated = true_positive_loss_is_harmful and (
        true_positive_lost_delta > int(rules["max_true_positive_lost_increase_count"])
        or (
            true_positive_loss_rate_delta is not None
            and true_positive_loss_rate_delta > float(rules["max_true_positive_loss_rate_increase"])
        )
    )
    true_positive_loss_watch = true_positive_loss_is_harmful and (
        true_positive_lost_delta >= int(rules["watch_true_positive_lost_increase_count"])
        or (
            true_positive_loss_rate_delta is not None
            and true_positive_loss_rate_delta > 0
        )
    )

    if retained_labels < int(rules["min_post_approval_labels"]) or post_days < int(rules["min_post_approval_signal_days"]):
        status = POST_PROMOTION_INSUFFICIENT_DATA
        reasons.append(
            f"Post-approval evidence is still small: retained labels {retained_labels}, signal days {post_days}."
        )
    elif (
        true_positive_loss_deteriorated
        or (avg_return_drop is not None and avg_return_drop < -float(rules["max_retained_avg_return_drop_bps"]))
        or (hit_rate_drop is not None and hit_rate_drop < -float(rules["max_hit_rate_drop"]))
        or deteriorating_segments > int(rules["max_deteriorating_segment_count"])
    ):
        status = POST_PROMOTION_DETERIORATING
        reasons.append("Post-approval evidence breached deterioration guardrails versus the promotion package.")
    elif (
        true_positive_loss_watch
        or (avg_return_drop is not None and avg_return_drop < -float(rules["watch_retained_avg_return_drop_bps"]))
        or (hit_rate_drop is not None and hit_rate_drop < -float(rules["watch_hit_rate_drop"]))
        or (suppressed_labels >= int(rules["min_suppressed_labels"]) and false_positive_removal_rate is not None and false_positive_removal_rate < float(rules["min_false_positive_removal_rate"]))
    ):
        status = POST_PROMOTION_WATCH
        reasons.append("Post-approval evidence is usable but weaker than the original shadow evidence.")
    else:
        status = POST_PROMOTION_HEALTHY
        reasons.append("Post-approval evidence remains consistent with the approved shadow promotion package.")

    guardrails = {
        "retained_label_count_60m": retained_labels,
        "suppressed_label_count_60m": suppressed_labels,
        "true_positive_lost_count": true_positive_lost,
        "true_positive_lost_delta_vs_shadow": true_positive_lost_delta,
        "true_positive_loss_rate": true_positive_loss_rate,
        "true_positive_loss_rate_delta_vs_shadow": true_positive_loss_rate_delta,
        "false_positive_removal_rate": false_positive_removal_rate,
        "avoided_suppressed_return_bps": avoided_return,
        "avg_suppressed_return_60m_bps": avg_suppressed_return,
        "true_positive_loss_interpretation": "economic_opportunity_cost"
        if not true_positive_loss_is_harmful
        else "harmful_suppression",
        "deteriorating_segment_count": deteriorating_segments,
        "watch_segment_count": watch_segments,
    }
    report = {
        "report_type": "threshold_post_promotion_monitor",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "promotion_review_ledger_path": str(ledger),
        "promotion_package_report_path": str(package_path) if package_path is not None else None,
        "monitor_status": status,
        "monitor_reasons": reasons,
        "recommended_next_action": _recommended_action(status),
        "runtime_config_changed": False,
        "approval_decision": approval,
        "promotion_candidate": promotion_package.get("promotion_candidate", {}),
        "threshold_rule": {
            "field": threshold_field,
            "operator": ">=",
            "value": threshold_value,
        },
        "policy": rules,
        "post_approval_window": post_window,
        "post_approval_impact": impact,
        "post_approval_baseline_metrics": baseline_metrics,
        "post_approval_retained_metrics": retained_metrics,
        "post_approval_suppressed_metrics": suppressed_metrics,
        "post_approval_retained_vs_baseline_delta": retained_vs_baseline,
        "guardrail_summary": guardrails,
        "shadow_expectation": {
            "impact_summary": shadow_impact,
            "retained_metrics": shadow_retained,
            "retained_vs_baseline_delta": promotion_package.get("retained_vs_baseline_delta", {}),
        },
        "drift_from_shadow_expectation": drift,
        "segment_monitoring": segment_rows,
        "segment_summary": {
            "segment_count": int(len(segment_rows)),
            "deteriorating_segment_count": int(deteriorating_segments),
            "watch_segment_count": int(watch_segments),
        },
    }
    return _sanitize_value(report)


def render_threshold_post_promotion_monitor_markdown(report: dict[str, Any]) -> str:
    """Render post-promotion monitoring as Markdown."""
    rule = report.get("threshold_rule", {}) or {}
    window = report.get("post_approval_window", {}) or {}
    impact = report.get("post_approval_impact", {}) or {}
    retained = report.get("post_approval_retained_metrics", {}) or {}
    shadow = (report.get("shadow_expectation", {}) or {}).get("retained_metrics", {}) or {}
    drift = report.get("drift_from_shadow_expectation", {}) or {}
    guardrails = report.get("guardrail_summary", {}) or {}
    segment_summary = report.get("segment_summary", {}) or {}
    lines = [
        "# Threshold Post-Promotion Monitor",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Promotion ledger: {report.get('promotion_review_ledger_path') or 'not supplied'}",
        f"- Promotion package: {report.get('promotion_package_report_path') or 'not supplied'}",
        f"- Monitor status: **{report.get('monitor_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Rule: `{rule.get('field')} {rule.get('operator', '>=')} {rule.get('value')}`",
        f"- Next action: {report.get('recommended_next_action')}",
        "",
        "## Monitor Reasons",
        "",
    ]
    for reason in report.get("monitor_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Post-Approval Window",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Approval timestamp | {window.get('approval_timestamp')} |",
            f"| Post-approval signals | {window.get('post_approval_signal_count')} |",
            f"| Post-approval signal dates | {window.get('post_approval_signal_dates')} |",
            f"| First signal | {window.get('first_post_approval_signal_timestamp')} |",
            f"| Last signal | {window.get('last_post_approval_signal_timestamp')} |",
            "",
            "## Evidence Drift",
            "",
            "| Metric | Shadow Expectation | Post Approval | Drift |",
            "| --- | ---: | ---: | ---: |",
            f"| Retained labels | {shadow.get('label_count_60m')} | {retained.get('label_count_60m')} | {drift.get('label_count_delta_vs_shadow')} |",
            f"| Retained hit rate | {shadow.get('hit_rate_60m')} | {retained.get('hit_rate_60m')} | {drift.get('retained_hit_rate_delta_vs_shadow')} |",
            f"| Retained avg return (bps) | {shadow.get('avg_signed_return_60m_bps')} | {retained.get('avg_signed_return_60m_bps')} | {drift.get('retained_avg_return_delta_bps_vs_shadow')} |",
            f"| False positives removed | {(report.get('shadow_expectation', {}) or {}).get('impact_summary', {}).get('false_positive_removed_count')} | {impact.get('false_positive_removed_count')} | {drift.get('false_positive_removed_delta_vs_shadow')} |",
            f"| True positives lost | {(report.get('shadow_expectation', {}) or {}).get('impact_summary', {}).get('true_positive_lost_count')} | {impact.get('true_positive_lost_count')} | {drift.get('true_positive_lost_delta_vs_shadow')} |",
            f"| True-positive loss rate | {(report.get('shadow_expectation', {}) or {}).get('impact_summary', {}).get('true_positive_loss_rate')} | {impact.get('true_positive_loss_rate')} | {drift.get('true_positive_loss_rate_delta_vs_shadow')} |",
            f"| Avoided suppressed return (bps) | {(report.get('shadow_expectation', {}) or {}).get('impact_summary', {}).get('avoided_suppressed_return_bps')} | {impact.get('avoided_suppressed_return_bps')} | {drift.get('avoided_suppressed_return_delta_bps_vs_shadow')} |",
            "",
            "## Guardrails",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| True-positive interpretation | {guardrails.get('true_positive_loss_interpretation')} |",
            f"| False-positive removal rate | {guardrails.get('false_positive_removal_rate')} |",
            f"| Avg suppressed return (bps) | {guardrails.get('avg_suppressed_return_60m_bps')} |",
            f"| Deteriorating segments | {guardrails.get('deteriorating_segment_count')} |",
            f"| Watch segments | {guardrails.get('watch_segment_count')} |",
            "",
            "## Segment Summary",
            "",
            f"- Segments monitored: {segment_summary.get('segment_count')}",
            f"- Deteriorating segments: {segment_summary.get('deteriorating_segment_count')}",
            f"- Watch segments: {segment_summary.get('watch_segment_count')}",
            "",
            "*This monitor is advisory. It can recommend review, but it does not apply or revert runtime configuration.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "segments_csv_path": output / f"{stem}_segments.csv",
        "latest_json_path": output / THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_POST_PROMOTION_MONITOR_MARKDOWN_FILENAME,
        "latest_segments_csv_path": output / THRESHOLD_POST_PROMOTION_MONITOR_SEGMENTS_FILENAME,
    }


def _write_monitor_bundle(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_post_promotion_monitor"
    paths = _artifact_paths(output, stem)
    markdown = render_threshold_post_promotion_monitor_markdown(report)
    segments = pd.DataFrame(report.get("segment_monitoring", []) or [])
    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(segments, paths["segments_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(segments, paths["latest_segments_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_threshold_post_promotion_monitor_report(
    frame: pd.DataFrame,
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    approval_decision: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    candidate_key: str | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write post-promotion monitoring artifacts."""
    report = build_threshold_post_promotion_monitor_report(
        frame,
        promotion_package_report=promotion_package_report,
        promotion_package_report_path=promotion_package_report_path,
        ledger_path=ledger_path,
        approval_decision=approval_decision,
        dataset_path=dataset_path,
        candidate_key=candidate_key,
        policy=policy,
    )
    return _write_monitor_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )


def write_threshold_post_promotion_monitor_skip(
    *,
    reason: str,
    dataset_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    promotion_package_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Write an explicit skipped monitor artifact."""
    report = _skip_report(
        reason=reason,
        dataset_path=dataset_path,
        ledger_path=ledger_path,
        promotion_package_report_path=promotion_package_report_path,
    )
    return _write_monitor_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )
