"""Segmented and recency-aware probability calibration experiment.

This research-only diagnostic searches for calibration mappings that are stable
inside regime slices or recent training windows. It writes advisory artifacts
only; runtime config, parameter packs, data sources, and execution behavior are
never changed by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import (
    DEFAULT_METHODS,
    DEFAULT_PROBABILITY_FIELD,
    _clean_probability_and_label_frame,
    _comparison_rows,
    _fit_calibrator,
    _row_by_method,
    _select_candidate,
    _split_train_holdout,
)
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_LABEL_FIELD,
    _atomic_write_csv,
    _atomic_write_text,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_calibration_experiment"
)

SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_JSON_FILENAME = (
    "latest_segmented_probability_calibration_experiment.json"
)
SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_MARKDOWN_FILENAME = (
    "latest_segmented_probability_calibration_experiment.md"
)
SEGMENTED_PROBABILITY_CALIBRATION_SEGMENTS_FILENAME = (
    "latest_segmented_probability_calibration_segments.csv"
)
SEGMENTED_PROBABILITY_CALIBRATION_RECENCY_FILENAME = (
    "latest_segmented_probability_calibration_recency_windows.csv"
)
SEGMENTED_PROBABILITY_CALIBRATION_CANDIDATE_FILENAME = (
    "latest_segmented_probability_calibration_candidate_bundle.json"
)

DEFAULT_SEGMENT_FIELDS = (
    "direction",
    "macro_regime",
    "gamma_regime",
    "volatility_regime",
    "global_risk_state",
)
DEFAULT_RECENCY_WINDOWS = (0.25, 0.50, 1.00)


def _safe_segment_value(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    token = str(value).strip()
    return token if token else "UNKNOWN"


def _fit_and_select(
    *,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    methods: tuple[str, ...],
    min_train_sample: int,
    min_holdout_sample: int,
    min_brier_improvement: float,
    max_ece_regression: float,
    max_candidate_ece: float,
    n_bins: int,
) -> dict[str, Any]:
    comparison: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    fit_reports: dict[str, dict[str, Any]] = {}

    if not train.empty and not holdout.empty:
        train_prob = train["_probability"].reset_index(drop=True)
        train_label = train["_label"].reset_index(drop=True)
        for method in methods:
            fit = _fit_calibrator(str(method), train_prob, train_label)
            rows, method_curves = _comparison_rows(fit=fit, train=train, holdout=holdout, n_bins=n_bins)
            comparison.extend(rows)
            curves.extend(method_curves)
            states[fit.method] = fit.state
            fit_reports[fit.method] = fit.fit_report

    selection = _select_candidate(
        comparison,
        min_train_sample=min_train_sample,
        min_holdout_sample=min_holdout_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_ece=max_candidate_ece,
    )
    selected = str(selection.get("selected_calibrator") or "identity")
    holdout_rows = _row_by_method(comparison, split_name="holdout")
    selected_holdout = holdout_rows.get(selected, holdout_rows.get("identity", {}))
    identity_holdout = holdout_rows.get("identity", {})
    return {
        "selection": selection,
        "selected_calibrator": selected,
        "selected_holdout": selected_holdout,
        "identity_holdout": identity_holdout,
        "comparison": comparison,
        "curves": curves,
        "state": states.get(selected, states.get("identity", {})),
        "fit_report": fit_reports.get(selected, {}),
    }


def _flatten_selection_row(
    result: dict[str, Any],
    *,
    candidate_type: str,
    segment_field: str,
    segment_value: str,
) -> dict[str, Any]:
    selection = result.get("selection", {}) or {}
    selected_holdout = result.get("selected_holdout", {}) or {}
    identity_holdout = result.get("identity_holdout", {}) or {}
    return {
        "candidate_type": candidate_type,
        "segment_field": segment_field,
        "segment_value": segment_value,
        "calibration_status": selection.get("calibration_status"),
        "selected_calibrator": result.get("selected_calibrator"),
        "candidate_ready_for_review": bool(selection.get("candidate_ready_for_review")),
        "train_count": int(selection.get("train_count") or 0),
        "holdout_count": int(selection.get("holdout_count") or 0),
        "identity_holdout_brier_score": identity_holdout.get("brier_score"),
        "selected_holdout_brier_score": selected_holdout.get("brier_score"),
        "holdout_brier_improvement": selection.get("holdout_brier_improvement"),
        "identity_holdout_expected_calibration_error": identity_holdout.get("expected_calibration_error"),
        "selected_holdout_expected_calibration_error": selected_holdout.get("expected_calibration_error"),
        "holdout_ece_change": selection.get("holdout_ece_change"),
        "selected_holdout_hit_rate": selected_holdout.get("actual_hit_rate"),
        "selected_holdout_mean_predicted_probability": selected_holdout.get("mean_predicted_probability"),
        "selected_holdout_calibration_gap": selected_holdout.get("calibration_gap"),
        "selection_reason": selection.get("selection_reason"),
    }


def _build_recency_candidates(
    *,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    recency_windows: tuple[float, ...],
    methods: tuple[str, ...],
    min_train_sample: int,
    min_holdout_sample: int,
    min_brier_improvement: float,
    max_ece_regression: float,
    max_candidate_ece: float,
    n_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidate_states: list[dict[str, Any]] = []
    for raw_window in recency_windows:
        window = min(max(float(raw_window), 0.05), 1.0)
        train_count = max(int(len(train) * window), 1) if len(train) else 0
        window_train = train.tail(train_count).copy()
        result = _fit_and_select(
            train=window_train,
            holdout=holdout,
            methods=methods,
            min_train_sample=min_train_sample,
            min_holdout_sample=min_holdout_sample,
            min_brier_improvement=min_brier_improvement,
            max_ece_regression=max_ece_regression,
            max_candidate_ece=max_candidate_ece,
            n_bins=n_bins,
        )
        value = f"last_{int(round(window * 100))}_pct_train"
        row = _flatten_selection_row(
            result,
            candidate_type="recency_window",
            segment_field="train_recency_window",
            segment_value=value,
        )
        row["train_window_fraction"] = round(window, 4)
        rows.append(row)
        if row["candidate_ready_for_review"]:
            candidate_states.append(
                {
                    "candidate_type": "recency_window",
                    "segment_field": "train_recency_window",
                    "segment_value": value,
                    "selected_calibrator": result.get("selected_calibrator"),
                    "selection": result.get("selection"),
                    "state": result.get("state"),
                    "fit_report": result.get("fit_report"),
                }
            )
    return rows, candidate_states


def _build_segment_candidates(
    *,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    segment_fields: tuple[str, ...],
    methods: tuple[str, ...],
    min_train_sample: int,
    min_holdout_sample: int,
    min_brier_improvement: float,
    max_ece_regression: float,
    max_candidate_ece: float,
    n_bins: int,
    max_segments_per_field: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidate_states: list[dict[str, Any]] = []
    for field in segment_fields:
        if field not in train.columns or field not in holdout.columns:
            continue
        train_values = train[field].map(_safe_segment_value)
        holdout_values = holdout[field].map(_safe_segment_value)
        value_counts = holdout_values.value_counts().head(max(int(max_segments_per_field), 1))
        for value in value_counts.index.tolist():
            segment_train = train.loc[train_values == value].copy()
            segment_holdout = holdout.loc[holdout_values == value].copy()
            result = _fit_and_select(
                train=segment_train,
                holdout=segment_holdout,
                methods=methods,
                min_train_sample=min_train_sample,
                min_holdout_sample=min_holdout_sample,
                min_brier_improvement=min_brier_improvement,
                max_ece_regression=max_ece_regression,
                max_candidate_ece=max_candidate_ece,
                n_bins=n_bins,
            )
            row = _flatten_selection_row(
                result,
                candidate_type="regime_segment",
                segment_field=field,
                segment_value=str(value),
            )
            rows.append(row)
            if row["candidate_ready_for_review"]:
                candidate_states.append(
                    {
                        "candidate_type": "regime_segment",
                        "segment_field": field,
                        "segment_value": str(value),
                        "selected_calibrator": result.get("selected_calibrator"),
                        "selection": result.get("selection"),
                        "state": result.get("state"),
                        "fit_report": result.get("fit_report"),
                    }
                )
    return rows, candidate_states


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("calibration_status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _best_ready_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ready = [row for row in rows if row.get("candidate_ready_for_review")]
    if not ready:
        return None
    return sorted(
        ready,
        key=lambda item: (
            -(float(item.get("holdout_brier_improvement") or 0.0)),
            float(item.get("selected_holdout_expected_calibration_error") or 1e9),
        ),
    )[0]


def _rank_candidate_states(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -(float((item.get("selection") or {}).get("holdout_brier_improvement") or 0.0)),
            float((item.get("selection") or {}).get("holdout_ece_change") or 0.0),
        ),
    )
    output: list[dict[str, Any]] = []
    for priority, candidate in enumerate(ranked, start=1):
        item = dict(candidate)
        item["candidate_priority"] = int(priority)
        output.append(item)
    return output


def _overall_status(
    *,
    segment_rows: list[dict[str, Any]],
    recency_rows: list[dict[str, Any]],
    train_count: int,
    holdout_count: int,
    min_train_sample: int,
    min_holdout_sample: int,
) -> str:
    if train_count < int(min_train_sample) or holdout_count < int(min_holdout_sample):
        return "INSUFFICIENT_EVIDENCE"
    if any(row.get("candidate_ready_for_review") for row in segment_rows + recency_rows):
        return "SEGMENTED_CALIBRATION_CANDIDATES_READY"
    if any(row.get("calibration_status") == "CALIBRATION_WATCH" for row in segment_rows + recency_rows):
        return "SEGMENTED_CALIBRATION_WATCH"
    if not segment_rows and not recency_rows:
        return "INSUFFICIENT_SEGMENT_EVIDENCE"
    return "NO_SEGMENTED_CALIBRATION_IMPROVEMENT"


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("calibration_status")
    ready_count = int(report.get("selection_summary", {}).get("review_ready_candidate_count") or 0)
    if status == "SEGMENTED_CALIBRATION_CANDIDATES_READY":
        return [
            f"Review {ready_count} segment/recency calibration candidates in shadow research before any manual adoption.",
            "Prioritize candidates that improve holdout Brier without increasing ECE and that have stable future-window evidence.",
            "Keep runtime probabilities unchanged until a human-approved calibration bundle passes promotion governance.",
        ]
    if status in {"INSUFFICIENT_EVIDENCE", "INSUFFICIENT_SEGMENT_EVIDENCE"}:
        return [
            "Collect more quality-approved labels in each regime before adopting segmented calibration.",
            "Keep runtime probabilities unchanged; sample-size guardrails blocked segment calibration selection.",
        ]
    if status == "SEGMENTED_CALIBRATION_WATCH":
        return [
            "Treat segment calibration as watch-only because at least one slice improved partially but failed guardrails.",
            "Run the experiment again after more recent labels accumulate and inspect regime hit-rate drift.",
        ]
    return [
        "Do not adopt segmented calibration yet; tested regime and recency candidates did not beat raw probabilities.",
        "Move next to probability-generation feature diagnostics and regime-conditioned threshold policy rather than post-hoc mapping.",
    ]


def build_segmented_probability_calibration_experiment_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    train_fraction: float = 0.70,
    segment_fields: tuple[str, ...] = DEFAULT_SEGMENT_FIELDS,
    recency_windows: tuple[float, ...] = DEFAULT_RECENCY_WINDOWS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    min_train_sample: int = 100,
    min_holdout_sample: int = 50,
    min_segment_train_sample: int = 120,
    min_segment_holdout_sample: int = 60,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_ece: float = 0.12,
    n_bins: int = 10,
    max_segments_per_field: int = 12,
) -> dict[str, Any]:
    """Build a segmented/recency-aware calibration experiment report."""
    raw = frame if frame is not None else pd.DataFrame()
    working = _clean_probability_and_label_frame(raw, probability_field=probability_field, label_field=label_field)
    train, holdout = _split_train_holdout(working, train_fraction=train_fraction)

    recency_rows, recency_candidates = _build_recency_candidates(
        train=train,
        holdout=holdout,
        recency_windows=recency_windows,
        methods=methods,
        min_train_sample=min_train_sample,
        min_holdout_sample=min_holdout_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_ece=max_candidate_ece,
        n_bins=n_bins,
    )
    segment_rows, segment_candidates = _build_segment_candidates(
        train=train,
        holdout=holdout,
        segment_fields=segment_fields,
        methods=methods,
        min_train_sample=min_segment_train_sample,
        min_holdout_sample=min_segment_holdout_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_ece=max_candidate_ece,
        n_bins=n_bins,
        max_segments_per_field=max_segments_per_field,
    )

    ready_candidates = _rank_candidate_states(recency_candidates + segment_candidates)
    all_rows = recency_rows + segment_rows
    best_ready = _best_ready_candidate(all_rows)
    status = _overall_status(
        segment_rows=segment_rows,
        recency_rows=recency_rows,
        train_count=len(train),
        holdout_count=len(holdout),
        min_train_sample=min_train_sample,
        min_holdout_sample=min_holdout_sample,
    )
    selection_summary = {
        "evaluated_regime_segment_count": int(len(segment_rows)),
        "evaluated_recency_window_count": int(len(recency_rows)),
        "review_ready_candidate_count": int(len(ready_candidates)),
        "watch_candidate_count": int(sum(row.get("calibration_status") == "CALIBRATION_WATCH" for row in all_rows)),
        "insufficient_evidence_count": int(
            sum(row.get("calibration_status") == "INSUFFICIENT_EVIDENCE" for row in all_rows)
        ),
        "status_counts": _status_counts(all_rows),
        "best_ready_candidate": best_ready or {},
    }
    candidate_bundle = {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": _utc_now(),
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": probability_field,
        "label_field": label_field,
        "calibration_status": status,
        "candidate_count": int(len(ready_candidates)),
        "candidates": ready_candidates,
    }
    report = {
        "report_type": "segmented_probability_calibration_experiment",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": int(len(raw)),
        "quality_labeled_row_count": int(len(working)),
        "probability_field": probability_field,
        "label_field": label_field,
        "train_fraction": float(train_fraction),
        "train_count": int(len(train)),
        "holdout_count": int(len(holdout)),
        "segment_fields": list(segment_fields),
        "recency_windows": [float(value) for value in recency_windows],
        "methods_tested": list(methods),
        "calibration_status": status,
        "selection_summary": selection_summary,
        "recency_window_results": recency_rows,
        "segment_results": segment_rows,
        "candidate_bundle": candidate_bundle,
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(report)
    return _sanitize_value(report)


def render_segmented_probability_calibration_experiment_markdown(report: dict[str, Any]) -> str:
    """Render segmented probability calibration experiment output as Markdown."""
    summary = report.get("selection_summary", {}) or {}
    lines = [
        "# Segmented Probability Calibration Experiment",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Rows: {report.get('row_count')}",
        f"- Quality-labeled rows: {report.get('quality_labeled_row_count')}",
        f"- Train / holdout: {report.get('train_count')} / {report.get('holdout_count')}",
        f"- Calibration status: `{report.get('calibration_status')}`",
        f"- Review-ready candidates: {summary.get('review_ready_candidate_count')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Recency Windows",
        "",
        "| Window | Status | Selected | Brier Improvement | ECE Change | Holdout Hit Rate |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report.get("recency_window_results", []) or []:
        lines.append(
            f"| `{row.get('segment_value')}` | `{row.get('calibration_status')}` | "
            f"`{row.get('selected_calibrator')}` | {row.get('holdout_brier_improvement')} | "
            f"{row.get('holdout_ece_change')} | {row.get('selected_holdout_hit_rate')} |"
        )

    lines.extend(
        [
            "",
            "## Best Regime Segments",
            "",
            "| Field | Value | Status | Selected | Brier Improvement | ECE Change | Holdout Count |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    segment_rows = sorted(
        report.get("segment_results", []) or [],
        key=lambda item: (
            not bool(item.get("candidate_ready_for_review")),
            -(float(item.get("holdout_brier_improvement") or -1e9)),
        ),
    )
    for row in segment_rows[:15]:
        lines.append(
            f"| `{row.get('segment_field')}` | `{row.get('segment_value')}` | `{row.get('calibration_status')}` | "
            f"`{row.get('selected_calibrator')}` | {row.get('holdout_brier_improvement')} | "
            f"{row.get('holdout_ece_change')} | {row.get('holdout_count')} |"
        )

    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("*Research-only artifact. Runtime config, parameter packs, data sources, and execution behavior are unchanged.*")
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "segments_csv_path": output / f"{stem}_segments.csv",
        "recency_csv_path": output / f"{stem}_recency_windows.csv",
        "candidate_bundle_json_path": output / f"{stem}_candidate_bundle.json",
        "latest_json_path": output / SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_JSON_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_MARKDOWN_FILENAME,
        "latest_segments_csv_path": output / SEGMENTED_PROBABILITY_CALIBRATION_SEGMENTS_FILENAME,
        "latest_recency_csv_path": output / SEGMENTED_PROBABILITY_CALIBRATION_RECENCY_FILENAME,
        "latest_candidate_bundle_json_path": output / SEGMENTED_PROBABILITY_CALIBRATION_CANDIDATE_FILENAME,
    }


def write_segmented_probability_calibration_experiment_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    train_fraction: float = 0.70,
    segment_fields: tuple[str, ...] = DEFAULT_SEGMENT_FIELDS,
    recency_windows: tuple[float, ...] = DEFAULT_RECENCY_WINDOWS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    min_train_sample: int = 100,
    min_holdout_sample: int = 50,
    min_segment_train_sample: int = 120,
    min_segment_holdout_sample: int = 60,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_ece: float = 0.12,
    n_bins: int = 10,
    max_segments_per_field: int = 12,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write segmented probability calibration experiment artifacts."""
    report = build_segmented_probability_calibration_experiment_report(
        frame,
        dataset_path=dataset_path,
        probability_field=probability_field,
        label_field=label_field,
        train_fraction=train_fraction,
        segment_fields=segment_fields,
        recency_windows=recency_windows,
        methods=methods,
        min_train_sample=min_train_sample,
        min_holdout_sample=min_holdout_sample,
        min_segment_train_sample=min_segment_train_sample,
        min_segment_holdout_sample=min_segment_holdout_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_ece=max_candidate_ece,
        n_bins=n_bins,
        max_segments_per_field=max_segments_per_field,
    )
    assert_artifact_schema(report, "segmented_probability_calibration_experiment")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_calibration_experiment"
    paths = _artifact_paths(output, stem)
    markdown = render_segmented_probability_calibration_experiment_markdown(report)
    segments = pd.DataFrame(report.get("segment_results", []) or [])
    recency = pd.DataFrame(report.get("recency_window_results", []) or [])
    candidate_bundle = report.get("candidate_bundle", {}) or {}

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(segments, paths["segments_csv_path"])
    _atomic_write_csv(recency, paths["recency_csv_path"])
    _atomic_write_text(
        paths["candidate_bundle_json_path"],
        json.dumps(candidate_bundle, indent=2, sort_keys=True, default=str),
    )
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(segments, paths["latest_segments_csv_path"])
        _atomic_write_csv(recency, paths["latest_recency_csv_path"])
        _atomic_write_text(
            paths["latest_candidate_bundle_json_path"],
            json.dumps(candidate_bundle, indent=2, sort_keys=True, default=str),
        )
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_calibration_experiment_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset and write segmented calibration experiment artifacts."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_segmented_probability_calibration_experiment_report(frame, dataset_path=path, **kwargs)
