"""Promotion-readiness review for threshold shadow simulation output.

This module turns research-only shadow simulation evidence into an advisory
promotion-readiness decision. It never writes runtime thresholds, parameter
packs, or execution settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_shadow_simulation import (
    DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR,
    SHADOW_SIMULATION_READY,
    THRESHOLD_SHADOW_SIMULATION_JSON_FILENAME,
)
from research.signal_evaluation.threshold_replay import _round_or_none


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_shadow_review"
)
DEFAULT_SHADOW_SIMULATION_REPORT_PATH = (
    DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR / THRESHOLD_SHADOW_SIMULATION_JSON_FILENAME
)

THRESHOLD_SHADOW_REVIEW_JSON_FILENAME = "latest_threshold_shadow_review.json"
THRESHOLD_SHADOW_REVIEW_MARKDOWN_FILENAME = "latest_threshold_shadow_review.md"
THRESHOLD_SHADOW_REVIEW_SEGMENTS_FILENAME = "latest_threshold_shadow_review_segments.csv"

PROMOTION_READY = "PROMOTION_READY"
NEEDS_MORE_SHADOW_DATA = "NEEDS_MORE_SHADOW_DATA"
REJECTED_SHADOW_REGRESSION = "REJECTED_SHADOW_REGRESSION"
REJECTED_TRUE_POSITIVE_LOSS = "REJECTED_TRUE_POSITIVE_LOSS"
REJECTED_REGIME_DEGRADATION = "REJECTED_REGIME_DEGRADATION"
SKIPPED_SHADOW_NOT_READY = "SKIPPED_SHADOW_NOT_READY"

DEFAULT_SHADOW_REVIEW_POLICY: dict[str, Any] = {
    "min_eligible_signal_count": 50,
    "min_shadow_observation_days": 30,
    "min_retained_labels": 30,
    "min_suppressed_labels": 10,
    "min_false_positive_removed_count": 5,
    "min_false_positive_removal_rate": 0.50,
    "max_true_positive_lost_count": 0,
    "max_true_positive_loss_rate": 0.05,
    "min_retained_avg_return_bps": 0.0,
    "min_retained_avg_return_delta_bps": 0.0,
    "min_retained_hit_rate_delta": 0.0,
    "min_avoided_suppressed_return_bps": 0.0,
    "min_segment_signal_count": 10,
    "min_segment_avg_return_delta_bps": -5.0,
    "min_segment_hit_rate_delta": -0.05,
    "max_segment_true_positive_lost_count": 0,
    "max_bad_regime_count": 0,
    "max_bad_bucket_count": 2,
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


def _observation_summary(shadow_report: dict[str, Any]) -> dict[str, Any]:
    retained = list(shadow_report.get("retained_signal_records", []) or [])
    suppressed = list(shadow_report.get("suppressed_signal_records", []) or [])
    records = retained + suppressed
    timestamps = pd.to_datetime(
        pd.Series([row.get("signal_timestamp") for row in records], dtype="object"),
        errors="coerce",
        utc=True,
    ).dropna()
    if timestamps.empty:
        return {
            "record_count": int(len(records)),
            "retained_record_count": int(len(retained)),
            "suppressed_record_count": int(len(suppressed)),
            "first_signal_timestamp": None,
            "last_signal_timestamp": None,
            "calendar_span_days": 0,
            "distinct_signal_dates": 0,
        }
    first_ts = timestamps.min()
    last_ts = timestamps.max()
    return {
        "record_count": int(len(records)),
        "retained_record_count": int(len(retained)),
        "suppressed_record_count": int(len(suppressed)),
        "first_signal_timestamp": first_ts.isoformat(),
        "last_signal_timestamp": last_ts.isoformat(),
        "calendar_span_days": int((last_ts.normalize() - first_ts.normalize()).days) + 1,
        "distinct_signal_dates": int(timestamps.dt.date.nunique()),
    }


def _segment_failure_rows(
    shadow_report: dict[str, Any],
    *,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = list(shadow_report.get("regime_shadow", []) or []) + list(shadow_report.get("bucket_shadow", []) or [])
    failures: list[dict[str, Any]] = []
    for row in rows:
        retained_count = _safe_int(row.get("retained_signal_count"))
        suppressed_count = _safe_int(row.get("suppressed_signal_count"))
        if max(retained_count, suppressed_count) < int(policy["min_segment_signal_count"]):
            continue

        reasons: list[str] = []
        avg_delta = _safe_float(row.get("avg_return_delta_bps"))
        hit_delta = _safe_float(row.get("hit_rate_delta"))
        true_positive_lost = _safe_int(row.get("true_positive_lost_count"))
        if avg_delta is not None and avg_delta < float(policy["min_segment_avg_return_delta_bps"]):
            reasons.append(
                f"Return delta {avg_delta} bps below segment floor {float(policy['min_segment_avg_return_delta_bps'])} bps."
            )
        if hit_delta is not None and hit_delta < float(policy["min_segment_hit_rate_delta"]):
            reasons.append(
                f"Hit-rate delta {hit_delta} below segment floor {float(policy['min_segment_hit_rate_delta'])}."
            )
        if true_positive_lost > int(policy["max_segment_true_positive_lost_count"]):
            reasons.append(
                f"True positives lost {true_positive_lost} exceeds segment limit "
                f"{int(policy['max_segment_true_positive_lost_count'])}."
            )
        if reasons:
            failures.append(
                {
                    "segment_kind": row.get("segment_kind"),
                    "segment_field": row.get("segment_field"),
                    "segment_value": row.get("segment_value"),
                    "retained_signal_count": retained_count,
                    "suppressed_signal_count": suppressed_count,
                    "false_positive_removed_count": row.get("false_positive_removed_count"),
                    "true_positive_lost_count": true_positive_lost,
                    "hit_rate_delta": hit_delta,
                    "avg_return_delta_bps": avg_delta,
                    "review_reasons": reasons,
                }
            )
    return sorted(
        failures,
        key=lambda item: (
            str(item.get("segment_kind")),
            -_safe_int(item.get("true_positive_lost_count")),
            _safe_float(item.get("avg_return_delta_bps")) or 0.0,
            str(item.get("segment_field")),
            str(item.get("segment_value")),
        ),
    )


def _recommended_next_action(status: str) -> str:
    if status == PROMOTION_READY:
        return "Open a human promotion review using the shadow evidence bundle; do not auto-apply runtime config."
    if status == NEEDS_MORE_SHADOW_DATA:
        return "Keep running automated shadow mode until sample-size and benefit guardrails are met."
    if status == REJECTED_TRUE_POSITIVE_LOSS:
        return "Reject this candidate for now because shadow suppression removed too many historically correct signals."
    if status == REJECTED_REGIME_DEGRADATION:
        return "Reject this candidate for now or redesign it with regime-conditional thresholds."
    if status == REJECTED_SHADOW_REGRESSION:
        return "Reject this candidate for now because retained shadow performance regressed versus baseline."
    return "Run an approved threshold policy experiment and shadow simulation before promotion review."


def build_threshold_shadow_review_report(
    shadow_report: dict[str, Any] | None,
    *,
    shadow_simulation_report_path: str | Path | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an advisory promotion-readiness review from shadow simulation output."""
    shadow = shadow_report or {}
    rules = {**DEFAULT_SHADOW_REVIEW_POLICY, **(policy or {})}
    observation = _observation_summary(shadow)
    impact = shadow.get("impact_summary", {}) or {}
    retained = shadow.get("retained_metrics", {}) or {}
    delta = shadow.get("retained_vs_baseline_delta", {}) or {}
    segment_failures = _segment_failure_rows(shadow, policy=rules)
    bad_regime_count = sum(1 for row in segment_failures if row.get("segment_kind") == "regime")
    bad_bucket_count = sum(1 for row in segment_failures if row.get("segment_kind") == "bucket")

    suppressed_labels = _safe_int(impact.get("suppressed_label_count_60m"))
    true_positive_lost = _safe_int(impact.get("true_positive_lost_count"))
    true_positive_loss_rate = (
        _round_or_none(true_positive_lost / max(suppressed_labels, 1), 4)
        if suppressed_labels
        else None
    )
    false_positive_removed = _safe_int(impact.get("false_positive_removed_count"))
    false_positive_removal_rate = _safe_float(impact.get("false_positive_removal_rate"))
    retained_label_count = _safe_int(retained.get("label_count_60m"))
    retained_avg_return = _safe_float(retained.get("avg_signed_return_60m_bps"))
    retained_avg_delta = _safe_float(delta.get("avg_return_delta_bps"))
    retained_hit_delta = _safe_float(delta.get("hit_rate_delta"))
    avoided_return = _safe_float(impact.get("avoided_suppressed_return_bps"))

    reasons: list[str] = []
    if shadow.get("shadow_status") != SHADOW_SIMULATION_READY:
        status = SKIPPED_SHADOW_NOT_READY
        reasons.append(f"Shadow simulation status is {shadow.get('shadow_status') or 'UNKNOWN'}.")
    elif _safe_int(impact.get("eligible_signal_count")) < int(rules["min_eligible_signal_count"]):
        status = NEEDS_MORE_SHADOW_DATA
        reasons.append(
            f"Eligible shadow signals {_safe_int(impact.get('eligible_signal_count'))} below minimum "
            f"{int(rules['min_eligible_signal_count'])}."
        )
    elif observation["distinct_signal_dates"] < int(rules["min_shadow_observation_days"]):
        status = NEEDS_MORE_SHADOW_DATA
        reasons.append(
            f"Shadow observation dates {observation['distinct_signal_dates']} below minimum "
            f"{int(rules['min_shadow_observation_days'])}."
        )
    elif retained_label_count < int(rules["min_retained_labels"]):
        status = NEEDS_MORE_SHADOW_DATA
        reasons.append(
            f"Retained 60m labels {retained_label_count} below minimum {int(rules['min_retained_labels'])}."
        )
    elif suppressed_labels < int(rules["min_suppressed_labels"]):
        status = NEEDS_MORE_SHADOW_DATA
        reasons.append(
            f"Suppressed 60m labels {suppressed_labels} below minimum {int(rules['min_suppressed_labels'])}."
        )
    elif true_positive_lost > int(rules["max_true_positive_lost_count"]) or (
        true_positive_loss_rate is not None
        and true_positive_loss_rate > float(rules["max_true_positive_loss_rate"])
    ):
        status = REJECTED_TRUE_POSITIVE_LOSS
        reasons.append(
            f"True positives lost {true_positive_lost} with loss rate {true_positive_loss_rate}; "
            f"limits are {int(rules['max_true_positive_lost_count'])} and "
            f"{float(rules['max_true_positive_loss_rate'])}."
        )
    elif retained_avg_return is not None and retained_avg_return < float(rules["min_retained_avg_return_bps"]):
        status = REJECTED_SHADOW_REGRESSION
        reasons.append(
            f"Retained average return {retained_avg_return} bps below minimum "
            f"{float(rules['min_retained_avg_return_bps'])} bps."
        )
    elif retained_avg_delta is not None and retained_avg_delta < float(rules["min_retained_avg_return_delta_bps"]):
        status = REJECTED_SHADOW_REGRESSION
        reasons.append(
            f"Retained average return delta {retained_avg_delta} bps below minimum "
            f"{float(rules['min_retained_avg_return_delta_bps'])} bps."
        )
    elif retained_hit_delta is not None and retained_hit_delta < float(rules["min_retained_hit_rate_delta"]):
        status = REJECTED_SHADOW_REGRESSION
        reasons.append(
            f"Retained hit-rate delta {retained_hit_delta} below minimum "
            f"{float(rules['min_retained_hit_rate_delta'])}."
        )
    elif bad_regime_count > int(rules["max_bad_regime_count"]):
        status = REJECTED_REGIME_DEGRADATION
        reasons.append(
            f"{bad_regime_count} regime segment(s) breached shadow review guardrails; "
            f"limit is {int(rules['max_bad_regime_count'])}."
        )
    elif bad_bucket_count > int(rules["max_bad_bucket_count"]):
        status = REJECTED_REGIME_DEGRADATION
        reasons.append(
            f"{bad_bucket_count} quality/confidence bucket(s) breached shadow review guardrails; "
            f"limit is {int(rules['max_bad_bucket_count'])}."
        )
    elif false_positive_removed < int(rules["min_false_positive_removed_count"]) or (
        false_positive_removal_rate is not None
        and false_positive_removal_rate < float(rules["min_false_positive_removal_rate"])
    ):
        status = NEEDS_MORE_SHADOW_DATA
        reasons.append(
            f"False positives removed {false_positive_removed} with removal rate {false_positive_removal_rate}; "
            f"minimums are {int(rules['min_false_positive_removed_count'])} and "
            f"{float(rules['min_false_positive_removal_rate'])}."
        )
    elif avoided_return is not None and avoided_return < float(rules["min_avoided_suppressed_return_bps"]):
        status = NEEDS_MORE_SHADOW_DATA
        reasons.append(
            f"Avoided suppressed return {avoided_return} bps below minimum "
            f"{float(rules['min_avoided_suppressed_return_bps'])} bps."
        )
    else:
        status = PROMOTION_READY
        reasons.append(
            "Shadow evidence meets sample-size, false-positive removal, true-positive preservation, "
            "performance, and segment guardrails."
        )

    guardrails = {
        "true_positive_loss_rate": true_positive_loss_rate,
        "false_positive_removed_count": false_positive_removed,
        "false_positive_removal_rate": false_positive_removal_rate,
        "bad_regime_count": bad_regime_count,
        "bad_bucket_count": bad_bucket_count,
        "retained_label_count_60m": retained_label_count,
        "suppressed_label_count_60m": suppressed_labels,
    }
    report = {
        "report_type": "threshold_shadow_review",
        "generated_at": _utc_now(),
        "shadow_simulation_report_path": str(shadow_simulation_report_path) if shadow_simulation_report_path is not None else None,
        "dataset_path": shadow.get("dataset_path"),
        "review_status": status,
        "review_reasons": reasons,
        "recommended_next_action": _recommended_next_action(status),
        "requires_manual_promotion_review": status == PROMOTION_READY,
        "runtime_config_changed": False,
        "policy": rules,
        "shadow_status": shadow.get("shadow_status"),
        "policy_experiment_status": shadow.get("policy_experiment_status"),
        "threshold_rule": shadow.get("threshold_rule", {}),
        "candidate_policy_pack": shadow.get("candidate_policy_pack", {}),
        "observation_summary": observation,
        "impact_summary": impact,
        "baseline_metrics": shadow.get("baseline_metrics", {}),
        "retained_metrics": retained,
        "suppressed_metrics": shadow.get("suppressed_metrics", {}),
        "retained_vs_baseline_delta": delta,
        "guardrail_summary": guardrails,
        "segment_failures": segment_failures,
    }
    return _sanitize_value(report)


def build_threshold_shadow_review_skip(
    *,
    reason: str,
    shadow_report: dict[str, Any] | None = None,
    shadow_simulation_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an explicit skipped shadow review artifact."""
    shadow = shadow_report or {}
    report = {
        "report_type": "threshold_shadow_review",
        "generated_at": _utc_now(),
        "shadow_simulation_report_path": str(shadow_simulation_report_path) if shadow_simulation_report_path is not None else None,
        "dataset_path": shadow.get("dataset_path"),
        "review_status": SKIPPED_SHADOW_NOT_READY,
        "review_reasons": [reason],
        "recommended_next_action": _recommended_next_action(SKIPPED_SHADOW_NOT_READY),
        "requires_manual_promotion_review": False,
        "runtime_config_changed": False,
        "policy": DEFAULT_SHADOW_REVIEW_POLICY,
        "shadow_status": shadow.get("shadow_status"),
        "policy_experiment_status": shadow.get("policy_experiment_status"),
        "threshold_rule": shadow.get("threshold_rule", {}),
        "candidate_policy_pack": shadow.get("candidate_policy_pack", {}),
        "observation_summary": _observation_summary(shadow),
        "impact_summary": shadow.get("impact_summary", {}),
        "baseline_metrics": shadow.get("baseline_metrics", {}),
        "retained_metrics": shadow.get("retained_metrics", {}),
        "suppressed_metrics": shadow.get("suppressed_metrics", {}),
        "retained_vs_baseline_delta": shadow.get("retained_vs_baseline_delta", {}),
        "guardrail_summary": {},
        "segment_failures": [],
    }
    return _sanitize_value(report)


def render_threshold_shadow_review_markdown(report: dict[str, Any]) -> str:
    """Render a threshold shadow review as Markdown."""
    rule = report.get("threshold_rule", {}) or {}
    observation = report.get("observation_summary", {}) or {}
    impact = report.get("impact_summary", {}) or {}
    guardrails = report.get("guardrail_summary", {}) or {}
    delta = report.get("retained_vs_baseline_delta", {}) or {}
    lines = [
        "# Threshold Shadow Review",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Shadow simulation: {report.get('shadow_simulation_report_path') or 'not supplied'}",
        f"- Review status: **{report.get('review_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Manual promotion review required: {report.get('requires_manual_promotion_review')}",
        f"- Rule: `{rule.get('field')} {rule.get('operator', '>=')} {rule.get('value')}`",
        f"- Next action: {report.get('recommended_next_action')}",
        "",
        "## Review Reasons",
        "",
    ]
    for reason in report.get("review_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Shadow Evidence",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Observation dates | {observation.get('distinct_signal_dates')} |",
            f"| Calendar span days | {observation.get('calendar_span_days')} |",
            f"| Eligible signals | {impact.get('eligible_signal_count')} |",
            f"| Retained signals | {impact.get('retained_signal_count')} |",
            f"| Suppressed signals | {impact.get('suppressed_signal_count')} |",
            f"| False positives removed | {impact.get('false_positive_removed_count')} |",
            f"| True positives lost | {impact.get('true_positive_lost_count')} |",
            f"| True-positive loss rate | {guardrails.get('true_positive_loss_rate')} |",
            f"| False-positive removal rate | {guardrails.get('false_positive_removal_rate')} |",
            f"| Retained avg-return delta (bps) | {delta.get('avg_return_delta_bps')} |",
            f"| Retained hit-rate delta | {delta.get('hit_rate_delta')} |",
            "",
            "## Segment Guardrails",
            "",
            "| Kind | Field | Value | Suppressed | True Positives Lost | Return Delta (bps) | Reason |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    segment_failures = report.get("segment_failures", []) or []
    if segment_failures:
        for row in segment_failures[:20]:
            lines.append(
                f"| {row.get('segment_kind')} | {row.get('segment_field')} | {row.get('segment_value')} | "
                f"{row.get('suppressed_signal_count')} | {row.get('true_positive_lost_count')} | "
                f"{row.get('avg_return_delta_bps')} | {'; '.join(row.get('review_reasons', []) or [])} |"
            )
    else:
        lines.append("| none | none | none | 0 | 0 |  | No segment guardrail failures. |")
    lines.extend(
        [
            "",
            "*This artifact is advisory. It can recommend human promotion review, but it does not change live thresholds or execution.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "segments_csv_path": output / f"{stem}_segments.csv",
        "latest_json_path": output / THRESHOLD_SHADOW_REVIEW_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_SHADOW_REVIEW_MARKDOWN_FILENAME,
        "latest_segments_csv_path": output / THRESHOLD_SHADOW_REVIEW_SEGMENTS_FILENAME,
    }


def _write_shadow_review_bundle(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_shadow_review"
    paths = _artifact_paths(output, stem)
    markdown = render_threshold_shadow_review_markdown(report)
    segments = pd.DataFrame(report.get("segment_failures", []) or [])

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


def write_threshold_shadow_review_report(
    shadow_report: dict[str, Any] | None,
    *,
    shadow_simulation_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write a threshold shadow review artifact bundle."""
    report = build_threshold_shadow_review_report(
        shadow_report,
        shadow_simulation_report_path=shadow_simulation_report_path,
        policy=policy,
    )
    return _write_shadow_review_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )


def write_threshold_shadow_review_skip(
    *,
    reason: str,
    shadow_report: dict[str, Any] | None = None,
    shadow_simulation_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Write an explicit skipped shadow review artifact."""
    report = build_threshold_shadow_review_skip(
        reason=reason,
        shadow_report=shadow_report,
        shadow_simulation_report_path=shadow_simulation_report_path,
    )
    return _write_shadow_review_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )
