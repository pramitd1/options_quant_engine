"""
Candidate-vs-production comparison helpers for governed tuning workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tuning.packs import resolve_parameter_pack
from tuning.registry import get_parameter_registry


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    coerced = _safe_float(value, None)
    if coerced is None or pd.isna(coerced):
        return None
    return round(float(coerced), digits)


def _resolved_parameter_values(pack_name: str) -> dict[str, Any]:
    registry = get_parameter_registry()
    resolved_pack = resolve_parameter_pack(pack_name)
    values = {}
    for key, definition in registry.items():
        values[key] = resolved_pack.overrides.get(key, definition.default_value)
    return values


def _absolute_change(current_value: Any, candidate_value: Any) -> Any:
    current_numeric = _safe_float(current_value, None)
    candidate_numeric = _safe_float(candidate_value, None)
    if current_numeric is None or candidate_numeric is None:
        return None
    return round(candidate_numeric - current_numeric, 6)


def _relative_change_pct(current_value: Any, candidate_value: Any) -> float | None:
    current_numeric = _safe_float(current_value, None)
    candidate_numeric = _safe_float(candidate_value, None)
    if current_numeric in (None, 0) or candidate_numeric is None:
        return None
    return round(((candidate_numeric - current_numeric) / abs(current_numeric)) * 100.0, 4)


def build_parameter_change_table(
    production_pack_name: str,
    candidate_pack_name: str,
    *,
    parameter_evidence: dict[str, dict[str, Any]] | None = None,
    changed_only: bool = True,
) -> list[dict[str, Any]]:
    registry = get_parameter_registry()
    current_values = _resolved_parameter_values(production_pack_name)
    candidate_values = _resolved_parameter_values(candidate_pack_name)
    parameter_evidence = dict(parameter_evidence or {})

    rows = []
    for key, definition in registry.items():
        current_value = current_values.get(key)
        candidate_value = candidate_values.get(key)
        if changed_only and current_value == candidate_value:
            continue

        evidence = dict(parameter_evidence.get(key) or {})
        rows.append(
            {
                "parameter_key": key,
                "parameter_name": definition.name,
                "parameter_group": definition.group,
                "parameter_category": definition.category,
                "live_safe": bool(definition.live_safe),
                "current_production_value": current_value,
                "suggested_value": candidate_value,
                "absolute_change": _absolute_change(current_value, candidate_value),
                "relative_change_pct": _relative_change_pct(current_value, candidate_value),
                "recommendation_reason": evidence.get("reason"),
                "supporting_tuning_evidence": evidence.get("supporting_tuning_evidence"),
            }
        )
    return rows


def _metric_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result or {})
    objective_metrics = dict(result.get("objective_metrics", {}))
    core_metrics = dict(objective_metrics.get("metrics", {}))
    validation = dict(result.get("validation_results", {}))
    aggregate_validation = dict(validation.get("aggregate_out_of_sample_metrics", {}))
    robustness = dict(result.get("robustness_metrics", {}))

    return {
        "parameter_pack_name": result.get("parameter_pack_name"),
        "evaluation_window": dict(result.get("evaluation_date_range", {})),
        "sample_count": int(result.get("sample_count", 0)),
        "objective_score": _round_or_none(result.get("objective_score"), 6),
        "direction_hit_rate": _round_or_none(core_metrics.get("direction_hit_rate"), 6),
        "composite_signal_score": _round_or_none(core_metrics.get("average_composite_signal_score"), 6),
        "signal_frequency": _round_or_none(core_metrics.get("signal_frequency"), 6),
        "drawdown_proxy": _round_or_none(core_metrics.get("drawdown_proxy"), 6),
        "validation_out_of_sample_score": _round_or_none(validation.get("aggregate_out_of_sample_score"), 6),
        "validation_direction_hit_rate": _round_or_none(aggregate_validation.get("direction_hit_rate"), 6),
        "validation_composite_signal_score": _round_or_none(aggregate_validation.get("average_composite_signal_score"), 6),
        "validation_signal_frequency": _round_or_none(aggregate_validation.get("signal_frequency"), 6),
        "validation_drawdown_proxy": _round_or_none(aggregate_validation.get("drawdown_proxy"), 6),
        "robustness_metrics": robustness,
        "regime_summary": dict(validation.get("regime_summary", {})),
    }


def _delta_row(current_value: Any, candidate_value: Any) -> dict[str, Any]:
    return {
        "current": current_value,
        "candidate": candidate_value,
        "delta": (
            round(float(candidate_value) - float(current_value), 6)
            if _safe_float(current_value, None) is not None and _safe_float(candidate_value, None) is not None
            else None
        ),
    }


def build_candidate_vs_production_report(
    *,
    production_pack_name: str,
    candidate_pack_name: str,
    production_result: dict[str, Any],
    candidate_result: dict[str, Any],
    parameter_evidence: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    production_snapshot = _metric_snapshot(production_result)
    candidate_snapshot = _metric_snapshot(candidate_result)
    comparison_summary = dict(candidate_result.get("comparison_summary", {}))

    report = {
        "report_type": "tuning_recommendation_report",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "current_production_pack_name": production_pack_name,
        "candidate_parameter_pack_name": candidate_pack_name,
        "evaluation_window": candidate_snapshot.get("evaluation_window") or production_snapshot.get("evaluation_window"),
        "sample_count": candidate_snapshot.get("sample_count") or production_snapshot.get("sample_count"),
        "parameter_change_table": build_parameter_change_table(
            production_pack_name,
            candidate_pack_name,
            parameter_evidence=parameter_evidence,
            changed_only=True,
        ),
        "experiment_comparison": {
            "objective_score": _delta_row(
                production_snapshot.get("objective_score"),
                candidate_snapshot.get("objective_score"),
            ),
            "hit_rate": _delta_row(
                production_snapshot.get("direction_hit_rate"),
                candidate_snapshot.get("direction_hit_rate"),
            ),
            "composite_signal_score": _delta_row(
                production_snapshot.get("composite_signal_score"),
                candidate_snapshot.get("composite_signal_score"),
            ),
            "signal_frequency": _delta_row(
                production_snapshot.get("signal_frequency"),
                candidate_snapshot.get("signal_frequency"),
            ),
            "drawdown_proxy": _delta_row(
                production_snapshot.get("drawdown_proxy"),
                candidate_snapshot.get("drawdown_proxy"),
            ),
            "validation_out_of_sample_score": _delta_row(
                production_snapshot.get("validation_out_of_sample_score"),
                candidate_snapshot.get("validation_out_of_sample_score"),
            ),
            "validation_hit_rate": _delta_row(
                production_snapshot.get("validation_direction_hit_rate"),
                candidate_snapshot.get("validation_direction_hit_rate"),
            ),
            "validation_composite_signal_score": _delta_row(
                production_snapshot.get("validation_composite_signal_score"),
                candidate_snapshot.get("validation_composite_signal_score"),
            ),
            "validation_signal_frequency": _delta_row(
                production_snapshot.get("validation_signal_frequency"),
                candidate_snapshot.get("validation_signal_frequency"),
            ),
            "validation_drawdown_proxy": _delta_row(
                production_snapshot.get("validation_drawdown_proxy"),
                candidate_snapshot.get("validation_drawdown_proxy"),
            ),
            "robustness_metrics": {
                "current": production_snapshot.get("robustness_metrics", {}),
                "candidate": candidate_snapshot.get("robustness_metrics", {}),
                "delta": dict(comparison_summary.get("robustness_comparison", {})),
            },
        },
        "regime_wise_results": {
            "current": production_snapshot.get("regime_summary", {}),
            "candidate": candidate_snapshot.get("regime_summary", {}),
            "comparison": dict(comparison_summary.get("regime_comparison", {})),
        },
        "comparison_summary": comparison_summary,
        "source_experiments": {
            "production_experiment_id": production_result.get("experiment_id"),
            "candidate_experiment_id": candidate_result.get("experiment_id"),
        },
    }

    report["expected_improvement_summary"] = {
        "objective_score_delta": report["experiment_comparison"]["objective_score"].get("delta"),
        "hit_rate_delta": report["experiment_comparison"]["hit_rate"].get("delta"),
        "composite_signal_score_delta": report["experiment_comparison"]["composite_signal_score"].get("delta"),
        "signal_frequency_delta": report["experiment_comparison"]["signal_frequency"].get("delta"),
        "drawdown_proxy_delta": report["experiment_comparison"]["drawdown_proxy"].get("delta"),
        "validation_out_of_sample_score_delta": report["experiment_comparison"]["validation_out_of_sample_score"].get("delta"),
        "robustness_score_delta": comparison_summary.get("robustness_comparison", {}).get("robustness_score_delta"),
    }
    return report


def render_candidate_vs_production_markdown(report: dict[str, Any]) -> str:
    comparison = report.get("experiment_comparison", {})
    lines = [
        "# Tuning Recommendation Report",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Current production pack: {report.get('current_production_pack_name')}",
        f"- Candidate pack: {report.get('candidate_parameter_pack_name')}",
        f"- Evaluation window: {report.get('evaluation_window', {}).get('start')} -> {report.get('evaluation_window', {}).get('end')}",
        f"- Sample count: {report.get('sample_count')}",
        "",
        "## Experiment Comparison",
        "",
        "| Metric | Current | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    metric_rows = (
        ("Objective Score", "objective_score"),
        ("Hit Rate", "hit_rate"),
        ("Composite Signal Score", "composite_signal_score"),
        ("Signal Frequency", "signal_frequency"),
        ("Drawdown Proxy", "drawdown_proxy"),
        ("Validation OOS Score", "validation_out_of_sample_score"),
        ("Validation Hit Rate", "validation_hit_rate"),
        ("Validation Composite Score", "validation_composite_signal_score"),
    )
    for label, key in metric_rows:
        row = comparison.get(key, {})
        lines.append(f"| {label} | {row.get('current')} | {row.get('candidate')} | {row.get('delta')} |")

    lines.extend(
        [
            "",
            "## Parameter Changes",
            "",
            "| Parameter | Group | Current | Suggested | Abs Change | Rel Change % | Reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report.get("parameter_change_table", []):
        lines.append(
            f"| {row.get('parameter_key')} | {row.get('parameter_group')} | {row.get('current_production_value')} | {row.get('suggested_value')} | {row.get('absolute_change')} | {row.get('relative_change_pct')} | {row.get('recommendation_reason') or ''} |"
        )
    if not report.get("parameter_change_table"):
        lines.append("| (no changes) |  |  |  |  |  |  |")

    return "\n".join(lines).strip() + "\n"


def write_candidate_vs_production_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    json_path = base_dir / "recommendation_report.json"
    markdown_path = base_dir / "recommendation_report.md"
    csv_path = base_dir / "parameter_change_table.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_candidate_vs_production_markdown(report), encoding="utf-8")
    pd.DataFrame(format_parameter_evidence_for_csv(report.get("parameter_change_table", []))).to_csv(csv_path, index=False)

    comparison_path = base_dir / "experiment_comparison.json"
    comparison_path.write_text(
        json.dumps(report.get("experiment_comparison", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "parameter_change_csv_path": str(csv_path),
        "experiment_comparison_json_path": str(comparison_path),
    }


def format_parameter_evidence_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        updated = dict(row)
        evidence = updated.get("supporting_tuning_evidence")
        if isinstance(evidence, (dict, list)):
            updated["supporting_tuning_evidence"] = json.dumps(evidence, sort_keys=True)
        formatted.append(updated)
    return formatted
