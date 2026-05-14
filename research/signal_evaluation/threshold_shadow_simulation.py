"""Research-only shadow simulation for approved threshold policy experiments.

This module answers: if an approved threshold candidate had been used, which
historical signals would have been retained or suppressed? It never changes
runtime thresholds, parameter packs, or execution behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_policy_experiment import (
    APPROVED_FOR_POLICY_EXPERIMENT,
    DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR,
    THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME,
)
from research.signal_evaluation.threshold_replay import (
    DEFAULT_REGIME_FIELDS,
    _metrics_for_subset,
    _prepare_frame,
    _round_or_none,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_shadow_simulation"
)
DEFAULT_POLICY_EXPERIMENT_REPORT_PATH = (
    DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR / THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME
)

THRESHOLD_SHADOW_SIMULATION_JSON_FILENAME = "latest_threshold_shadow_simulation.json"
THRESHOLD_SHADOW_SIMULATION_MARKDOWN_FILENAME = "latest_threshold_shadow_simulation.md"
THRESHOLD_SHADOW_SIMULATION_RETAINED_FILENAME = "latest_threshold_shadow_simulation_retained_signals.csv"
THRESHOLD_SHADOW_SIMULATION_SUPPRESSED_FILENAME = "latest_threshold_shadow_simulation_suppressed_signals.csv"
THRESHOLD_SHADOW_SIMULATION_REGIMES_FILENAME = "latest_threshold_shadow_simulation_regimes.csv"
THRESHOLD_SHADOW_SIMULATION_BUCKETS_FILENAME = "latest_threshold_shadow_simulation_buckets.csv"

SHADOW_SIMULATION_READY = "SHADOW_SIMULATION_READY"
SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED = "SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED"
INSUFFICIENT_SHADOW_EVIDENCE = "INSUFFICIENT_SHADOW_EVIDENCE"

REGIME_FIELDS = ("signal_regime", *DEFAULT_REGIME_FIELDS)
BUCKET_FIELDS = (
    "label_quality_status",
    "ml_confidence_score",
    "ml_rank_score",
    "move_probability",
    "hybrid_move_probability",
)
DETAIL_COLUMNS = (
    "signal_id",
    "signal_timestamp",
    "symbol",
    "source",
    "mode",
    "direction",
    "trade_status",
    "signal_regime",
    "macro_regime",
    "gamma_regime",
    "volatility_regime",
    "global_risk_state",
    "correct_60m",
    "signed_return_60m_bps",
    "label_quality_status",
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


def _threshold_rule_from_policy_experiment(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    pack = report.get("candidate_policy_pack", {}) or {}
    return pack.get("threshold_rule", {}) or {}


def _select_shadow_sets(frame: pd.DataFrame, *, threshold_field: str, threshold_value: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if threshold_field not in frame.columns:
        return frame.iloc[0:0].copy(), frame.iloc[0:0].copy(), 0
    values = pd.to_numeric(frame[threshold_field], errors="coerce")
    eligible = frame.loc[values.notna()].copy()
    eligible_values = pd.to_numeric(eligible[threshold_field], errors="coerce")
    retained = eligible.loc[eligible_values >= float(threshold_value)].copy()
    suppressed = eligible.loc[eligible_values < float(threshold_value)].copy()
    return retained, suppressed, int(len(eligible))


def _outcome_classification(row: pd.Series, *, retained: bool) -> str:
    hit = _safe_float(row.get("correct_60m"))
    ret = _safe_float(row.get("signed_return_60m_bps"))
    if hit is not None:
        if retained:
            return "TRUE_POSITIVE_RETAINED" if hit >= 0.5 else "FALSE_POSITIVE_RETAINED"
        return "TRUE_POSITIVE_LOST" if hit >= 0.5 else "FALSE_POSITIVE_REMOVED"
    if ret is not None:
        if retained:
            return "POSITIVE_RETURN_RETAINED" if ret > 0 else "NEGATIVE_RETURN_RETAINED"
        return "POSITIVE_RETURN_LOST" if ret > 0 else "NEGATIVE_RETURN_REMOVED"
    return "UNLABELED_RETAINED" if retained else "UNLABELED_SUPPRESSED"


def _detail_frame(frame: pd.DataFrame, *, retained: bool, threshold_field: str, threshold_value: float) -> pd.DataFrame:
    columns = [column for column in DETAIL_COLUMNS if column in frame.columns]
    detail = frame[columns].copy() if columns else pd.DataFrame(index=frame.index)
    detail.insert(0, "shadow_decision", "RETAINED" if retained else "SUPPRESSED")
    detail["threshold_field"] = threshold_field
    detail["threshold_value"] = threshold_value
    detail["threshold_observed_value"] = pd.to_numeric(frame.get(threshold_field, pd.Series(index=frame.index)), errors="coerce")
    detail["shadow_outcome_classification"] = [
        _outcome_classification(row, retained=retained)
        for _, row in frame.iterrows()
    ]
    return detail.reset_index(drop=True)


def _suppression_summary(retained: pd.DataFrame, suppressed: pd.DataFrame, *, eligible_count: int) -> dict[str, Any]:
    suppressed_hit = pd.to_numeric(suppressed.get("correct_60m", pd.Series(dtype=float)), errors="coerce").dropna()
    suppressed_ret = pd.to_numeric(suppressed.get("signed_return_60m_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    retained_hit = pd.to_numeric(retained.get("correct_60m", pd.Series(dtype=float)), errors="coerce").dropna()
    false_positive_removed = int((suppressed_hit < 0.5).sum())
    true_positive_lost = int((suppressed_hit >= 0.5).sum())
    false_positive_retained = int((retained_hit < 0.5).sum())
    true_positive_retained = int((retained_hit >= 0.5).sum())
    negative_return_removed = int((suppressed_ret <= 0).sum())
    positive_return_lost = int((suppressed_ret > 0).sum())
    suppressed_label_count = int(suppressed_hit.count())
    return {
        "eligible_signal_count": int(eligible_count),
        "retained_signal_count": int(len(retained)),
        "suppressed_signal_count": int(len(suppressed)),
        "retention_ratio": _round_or_none(len(retained) / max(int(eligible_count), 1), 4),
        "suppression_ratio": _round_or_none(len(suppressed) / max(int(eligible_count), 1), 4),
        "suppressed_label_count_60m": suppressed_label_count,
        "false_positive_removed_count": false_positive_removed,
        "true_positive_lost_count": true_positive_lost,
        "false_positive_retained_count": false_positive_retained,
        "true_positive_retained_count": true_positive_retained,
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


def _segment_shadow_rows(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float,
    segment_fields: tuple[str, ...],
    min_label_sample: int,
    strong_label_sample: int,
    segment_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in segment_fields:
        if field not in frame.columns:
            continue
        bucket_values = _bucket_label(frame[field])
        for value in sorted(set(str(item) for item in bucket_values.dropna().unique())):
            group = frame.loc[bucket_values.astype(str) == value].copy()
            if group.empty:
                continue
            retained, suppressed, eligible_count = _select_shadow_sets(
                group,
                threshold_field=threshold_field,
                threshold_value=threshold_value,
            )
            baseline_metrics = _metrics_for_subset(
                group,
                eligible_count=max(len(group), 1),
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
            retained_metrics = _metrics_for_subset(
                retained,
                eligible_count=max(eligible_count, 1),
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
            suppressed_metrics = _metrics_for_subset(
                suppressed,
                eligible_count=max(eligible_count, 1),
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
            )
            summary = _suppression_summary(retained, suppressed, eligible_count=eligible_count)
            rows.append(
                {
                    "segment_kind": segment_kind,
                    "segment_field": field,
                    "segment_value": value,
                    **summary,
                    "baseline_hit_rate_60m": baseline_metrics.get("hit_rate_60m"),
                    "retained_hit_rate_60m": retained_metrics.get("hit_rate_60m"),
                    "hit_rate_delta": _comparison_delta(retained_metrics, baseline_metrics).get("hit_rate_delta"),
                    "baseline_avg_return_60m_bps": baseline_metrics.get("avg_signed_return_60m_bps"),
                    "retained_avg_return_60m_bps": retained_metrics.get("avg_signed_return_60m_bps"),
                    "avg_return_delta_bps": _comparison_delta(retained_metrics, baseline_metrics).get("avg_return_delta_bps"),
                    "suppressed_avg_return_60m_bps": suppressed_metrics.get("avg_signed_return_60m_bps"),
                    "retained_sample_quality": retained_metrics.get("sample_quality"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -_safe_int(row.get("false_positive_removed_count")),
            _safe_int(row.get("true_positive_lost_count")),
            -_safe_int(row.get("suppressed_signal_count")),
            str(row.get("segment_field")),
            str(row.get("segment_value")),
        ),
    )


def build_threshold_shadow_simulation_report(
    frame: pd.DataFrame,
    *,
    policy_experiment_report: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    policy_experiment_report_path: str | Path | None = None,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
) -> dict[str, Any]:
    """Build a research-only shadow simulation for an approved threshold experiment."""
    experiment = policy_experiment_report or {}
    experiment_status = experiment.get("experiment_status")
    threshold_rule = _threshold_rule_from_policy_experiment(experiment)
    threshold_field = str(threshold_rule.get("field") or "")
    threshold_value = _safe_float(threshold_rule.get("value"))

    if experiment_status != APPROVED_FOR_POLICY_EXPERIMENT or not threshold_field or threshold_value is None:
        return _skip_report(
            reason=(
                "Policy experiment is not approved for shadow simulation "
                f"or has no concrete threshold rule; status is {experiment_status or 'UNKNOWN'}."
            ),
            policy_experiment_report=experiment,
            dataset_path=dataset_path,
            policy_experiment_report_path=policy_experiment_report_path,
        )

    working = _prepare_frame(frame)
    retained, suppressed, eligible_count = _select_shadow_sets(
        working,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
    )
    baseline_metrics = _metrics_for_subset(
        working.loc[pd.to_numeric(working.get(threshold_field, pd.Series(index=working.index)), errors="coerce").notna()].copy(),
        eligible_count=max(eligible_count, 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    retained_metrics = _metrics_for_subset(
        retained,
        eligible_count=max(eligible_count, 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    suppressed_metrics = _metrics_for_subset(
        suppressed,
        eligible_count=max(eligible_count, 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    impact = _suppression_summary(retained, suppressed, eligible_count=eligible_count)
    if eligible_count <= 0:
        shadow_status = INSUFFICIENT_SHADOW_EVIDENCE
        reasons = [f"Threshold field {threshold_field!r} is unavailable or entirely missing in the dataset."]
    else:
        shadow_status = SHADOW_SIMULATION_READY
        reasons = [
            "Approved threshold policy experiment was replayed as a research-only shadow simulation.",
            (
                f"Suppressed {impact['suppressed_signal_count']} of {impact['eligible_signal_count']} eligible signal(s), "
                f"removing {impact['false_positive_removed_count']} false positive(s) and losing "
                f"{impact['true_positive_lost_count']} true positive(s)."
            ),
        ]

    retained_detail = _detail_frame(
        retained,
        retained=True,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
    )
    suppressed_detail = _detail_frame(
        suppressed,
        retained=False,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
    )
    regime_rows = _segment_shadow_rows(
        working,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
        segment_fields=REGIME_FIELDS,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
        segment_kind="regime",
    )
    bucket_rows = _segment_shadow_rows(
        working,
        threshold_field=threshold_field,
        threshold_value=float(threshold_value),
        segment_fields=BUCKET_FIELDS,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
        segment_kind="bucket",
    )
    report = {
        "report_type": "threshold_shadow_simulation",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "policy_experiment_report_path": str(policy_experiment_report_path) if policy_experiment_report_path is not None else None,
        "shadow_status": shadow_status,
        "shadow_reasons": reasons,
        "runtime_config_changed": False,
        "policy_experiment_status": experiment_status,
        "candidate_policy_pack": experiment.get("candidate_policy_pack", {}),
        "threshold_rule": {
            "field": threshold_field,
            "operator": ">=",
            "value": threshold_value,
        },
        "impact_summary": impact,
        "baseline_metrics": baseline_metrics,
        "retained_metrics": retained_metrics,
        "suppressed_metrics": suppressed_metrics,
        "retained_vs_baseline_delta": _comparison_delta(retained_metrics, baseline_metrics),
        "regime_shadow": regime_rows,
        "bucket_shadow": bucket_rows,
        "retained_signal_records": retained_detail.to_dict(orient="records"),
        "suppressed_signal_records": suppressed_detail.to_dict(orient="records"),
    }
    return _sanitize_value(report)


def _skip_report(
    *,
    reason: str,
    policy_experiment_report: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    policy_experiment_report_path: str | Path | None = None,
) -> dict[str, Any]:
    experiment = policy_experiment_report or {}
    return _sanitize_value(
        {
            "report_type": "threshold_shadow_simulation",
            "generated_at": _utc_now(),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "policy_experiment_report_path": str(policy_experiment_report_path) if policy_experiment_report_path is not None else None,
            "shadow_status": SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED,
            "shadow_reasons": [reason],
            "runtime_config_changed": False,
            "policy_experiment_status": experiment.get("experiment_status"),
            "candidate_policy_pack": experiment.get("candidate_policy_pack", {}),
            "threshold_rule": _threshold_rule_from_policy_experiment(experiment),
            "impact_summary": {},
            "baseline_metrics": {},
            "retained_metrics": {},
            "suppressed_metrics": {},
            "retained_vs_baseline_delta": {},
            "regime_shadow": [],
            "bucket_shadow": [],
            "retained_signal_records": [],
            "suppressed_signal_records": [],
        }
    )


def render_threshold_shadow_simulation_markdown(report: dict[str, Any]) -> str:
    """Render a threshold shadow simulation report as Markdown."""
    rule = report.get("threshold_rule", {}) or {}
    impact = report.get("impact_summary", {}) or {}
    baseline = report.get("baseline_metrics", {}) or {}
    retained = report.get("retained_metrics", {}) or {}
    suppressed = report.get("suppressed_metrics", {}) or {}
    delta = report.get("retained_vs_baseline_delta", {}) or {}
    lines = [
        "# Threshold Shadow Simulation",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Policy experiment: {report.get('policy_experiment_report_path') or 'not supplied'}",
        f"- Shadow status: **{report.get('shadow_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Rule: `{rule.get('field')} {rule.get('operator', '>=')} {rule.get('value')}`",
        "",
        "## Decision Reasons",
        "",
    ]
    for reason in report.get("shadow_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Signal Stream Impact",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Eligible signals | {impact.get('eligible_signal_count')} |",
            f"| Retained signals | {impact.get('retained_signal_count')} |",
            f"| Suppressed signals | {impact.get('suppressed_signal_count')} |",
            f"| Retention ratio | {impact.get('retention_ratio')} |",
            f"| False positives removed | {impact.get('false_positive_removed_count')} |",
            f"| True positives lost | {impact.get('true_positive_lost_count')} |",
            f"| Avoided suppressed return (bps) | {impact.get('avoided_suppressed_return_bps')} |",
            "",
            "## Baseline vs Shadow-Retained",
            "",
            "| Metric | Baseline | Retained | Delta | Suppressed |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Signals | {baseline.get('signal_count')} | {retained.get('signal_count')} | {delta.get('signal_count_delta')} | {suppressed.get('signal_count')} |",
            f"| 60m labels | {baseline.get('label_count_60m')} | {retained.get('label_count_60m')} | {delta.get('label_count_delta')} | {suppressed.get('label_count_60m')} |",
            f"| 60m hit rate | {baseline.get('hit_rate_60m')} | {retained.get('hit_rate_60m')} | {delta.get('hit_rate_delta')} | {suppressed.get('hit_rate_60m')} |",
            f"| Avg 60m return (bps) | {baseline.get('avg_signed_return_60m_bps')} | {retained.get('avg_signed_return_60m_bps')} | {delta.get('avg_return_delta_bps')} | {suppressed.get('avg_signed_return_60m_bps')} |",
            f"| Max drawdown (bps) | {baseline.get('max_drawdown_bps')} | {retained.get('max_drawdown_bps')} | {delta.get('max_drawdown_delta_bps')} | {suppressed.get('max_drawdown_bps')} |",
            "",
            "## Largest Regime Impacts",
            "",
            "| Segment | Value | Suppressed | False Positives Removed | True Positives Lost | Return Delta (bps) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("regime_shadow", [])[:10]:
        lines.append(
            f"| {row.get('segment_field')} | {row.get('segment_value')} | {row.get('suppressed_signal_count')} | "
            f"{row.get('false_positive_removed_count')} | {row.get('true_positive_lost_count')} | {row.get('avg_return_delta_bps')} |"
        )
    lines.extend(
        [
            "",
            "*This artifact is advisory. It simulates signal visibility only and does not alter live generation or execution.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "retained_signals_csv_path": output / f"{stem}_retained_signals.csv",
        "suppressed_signals_csv_path": output / f"{stem}_suppressed_signals.csv",
        "regimes_csv_path": output / f"{stem}_regimes.csv",
        "buckets_csv_path": output / f"{stem}_buckets.csv",
        "latest_json_path": output / THRESHOLD_SHADOW_SIMULATION_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_SHADOW_SIMULATION_MARKDOWN_FILENAME,
        "latest_retained_signals_csv_path": output / THRESHOLD_SHADOW_SIMULATION_RETAINED_FILENAME,
        "latest_suppressed_signals_csv_path": output / THRESHOLD_SHADOW_SIMULATION_SUPPRESSED_FILENAME,
        "latest_regimes_csv_path": output / THRESHOLD_SHADOW_SIMULATION_REGIMES_FILENAME,
        "latest_buckets_csv_path": output / THRESHOLD_SHADOW_SIMULATION_BUCKETS_FILENAME,
    }


def _write_shadow_artifact_bundle(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_shadow_simulation"
    paths = _artifact_paths(output, stem)
    markdown = render_threshold_shadow_simulation_markdown(report)
    retained = pd.DataFrame(report.get("retained_signal_records", []) or [])
    suppressed = pd.DataFrame(report.get("suppressed_signal_records", []) or [])
    regimes = pd.DataFrame(report.get("regime_shadow", []) or [])
    buckets = pd.DataFrame(report.get("bucket_shadow", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(retained, paths["retained_signals_csv_path"])
    _atomic_write_csv(suppressed, paths["suppressed_signals_csv_path"])
    _atomic_write_csv(regimes, paths["regimes_csv_path"])
    _atomic_write_csv(buckets, paths["buckets_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(retained, paths["latest_retained_signals_csv_path"])
        _atomic_write_csv(suppressed, paths["latest_suppressed_signals_csv_path"])
        _atomic_write_csv(regimes, paths["latest_regimes_csv_path"])
        _atomic_write_csv(buckets, paths["latest_buckets_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_threshold_shadow_simulation_report(
    frame: pd.DataFrame,
    *,
    policy_experiment_report: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    policy_experiment_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write a threshold shadow simulation artifact bundle."""
    report = build_threshold_shadow_simulation_report(
        frame,
        policy_experiment_report=policy_experiment_report,
        dataset_path=dataset_path,
        policy_experiment_report_path=policy_experiment_report_path,
    )
    return _write_shadow_artifact_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )


def write_threshold_shadow_simulation_skip(
    *,
    reason: str,
    policy_experiment_report: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    policy_experiment_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Write an explicit skipped shadow simulation artifact."""
    report = _skip_report(
        reason=reason,
        policy_experiment_report=policy_experiment_report,
        dataset_path=dataset_path,
        policy_experiment_report_path=policy_experiment_report_path,
    )
    return _write_shadow_artifact_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )
