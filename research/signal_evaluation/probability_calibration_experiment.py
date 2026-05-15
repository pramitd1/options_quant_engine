"""Research-only probability calibration experiment for signal probabilities.

This module fits and compares post-hoc calibration mappings on quality-approved
signal labels. It writes review artifacts only; it never changes runtime config,
parameter packs, data-source choices, or execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_LABEL_FIELD,
    DEFAULT_PROBABILITY_FIELD,
    _atomic_write_csv,
    _atomic_write_text,
    _prepare_labeled_frame,
    _probability_series,
    _round_or_none,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)
from strategy.score_calibration import ScoreCalibrator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBABILITY_CALIBRATION_EXPERIMENT_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "probability_calibration_experiment"
)

PROBABILITY_CALIBRATION_EXPERIMENT_JSON_FILENAME = "latest_probability_calibration_experiment.json"
PROBABILITY_CALIBRATION_EXPERIMENT_MARKDOWN_FILENAME = "latest_probability_calibration_experiment.md"
PROBABILITY_CALIBRATION_EXPERIMENT_COMPARISON_FILENAME = "latest_probability_calibration_experiment_comparison.csv"
PROBABILITY_CALIBRATION_EXPERIMENT_CURVE_FILENAME = "latest_probability_calibration_experiment_curve.csv"
PROBABILITY_CALIBRATION_EXPERIMENT_CANDIDATE_FILENAME = "latest_probability_calibration_candidate.json"

DEFAULT_METHODS = ("identity", "linear_shrink", "temperature_score", "isotonic_score")


@dataclass(frozen=True)
class CalibrationFit:
    method: str
    state: dict[str, Any]
    fit_report: dict[str, Any]
    predict: Callable[[pd.Series], pd.Series]


def _clean_probability_and_label_frame(
    frame: pd.DataFrame,
    *,
    probability_field: str,
    label_field: str,
) -> pd.DataFrame:
    labeled = _prepare_labeled_frame(frame if frame is not None else pd.DataFrame())
    if labeled.empty:
        return labeled.assign(_probability=pd.Series(dtype="float64"), _label=pd.Series(dtype="float64"))

    labeled["_probability"] = _probability_series(labeled, probability_field)
    labeled["_label"] = pd.to_numeric(labeled.get(label_field, pd.Series(index=labeled.index)), errors="coerce")
    working = labeled.dropna(subset=["_probability", "_label"]).copy()
    working["_label"] = working["_label"].clip(lower=0.0, upper=1.0)
    if "signal_timestamp" in working.columns:
        working = working.sort_values("signal_timestamp", kind="mergesort")
    return working.reset_index(drop=True)


def _split_train_holdout(frame: pd.DataFrame, *, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    fraction = min(max(float(train_fraction), 0.10), 0.90)
    split_at = int(len(frame) * fraction)
    split_at = max(min(split_at, len(frame) - 1), 1) if len(frame) > 1 else len(frame)
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def _clip_probability(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype="float64").clip(lower=0.0, upper=1.0)


def _brier_score(predicted: pd.Series, labels: pd.Series) -> float | None:
    valid = predicted.notna() & labels.notna()
    if not bool(valid.any()):
        return None
    return float(((predicted.loc[valid] - labels.loc[valid]) ** 2).mean())


def _calibration_curve(
    predicted: pd.Series,
    labels: pd.Series,
    *,
    method: str,
    split_name: str,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    n_bins = max(int(n_bins), 2)
    predicted = _clip_probability(predicted).reset_index(drop=True)
    labels = pd.to_numeric(labels.reset_index(drop=True), errors="coerce")
    valid = predicted.notna() & labels.notna()
    if not bool(valid.any()):
        return []

    working = pd.DataFrame({"_predicted": predicted.loc[valid], "_label": labels.loc[valid]})
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges[-1] = 1.000001
    working["_bucket"] = pd.cut(
        working["_predicted"],
        bins=edges,
        include_lowest=True,
        right=False,
        labels=False,
    )
    rows: list[dict[str, Any]] = []
    for bucket, group in working.groupby("_bucket", dropna=False):
        if pd.isna(bucket) or group.empty:
            continue
        bucket_index = int(bucket)
        prob_min = float(edges[bucket_index])
        prob_max = float(min(edges[bucket_index + 1], 1.0))
        avg_pred = float(group["_predicted"].mean())
        hit_rate = float(group["_label"].mean())
        gap = avg_pred - hit_rate
        rows.append(
            {
                "method": method,
                "split": split_name,
                "bucket_index": bucket_index,
                "probability_min": _round_or_none(prob_min, 4),
                "probability_max": _round_or_none(prob_max, 4),
                "signal_count": int(len(group)),
                "avg_predicted_probability": _round_or_none(avg_pred, 6),
                "actual_hit_rate": _round_or_none(hit_rate, 6),
                "calibration_gap": _round_or_none(gap, 6),
                "abs_calibration_gap": _round_or_none(abs(gap), 6),
            }
        )
    return rows


def _metrics(
    predicted: pd.Series,
    labels: pd.Series,
    *,
    method: str,
    split_name: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    predicted = _clip_probability(predicted).reset_index(drop=True)
    labels = pd.to_numeric(labels.reset_index(drop=True), errors="coerce")
    valid = predicted.notna() & labels.notna()
    sample_count = int(valid.sum())
    if sample_count <= 0:
        return {
            "method": method,
            "split": split_name,
            "sample_count": 0,
            "brier_score": None,
            "expected_calibration_error": None,
            "max_calibration_error": None,
            "mean_predicted_probability": None,
            "actual_hit_rate": None,
            "calibration_gap": None,
        }

    p = predicted.loc[valid]
    y = labels.loc[valid]
    curve = _calibration_curve(p, y, method=method, split_name=split_name, n_bins=n_bins)
    total = max(float(sample_count), 1.0)
    ece = None
    mce = None
    if curve:
        ece = sum(float(row["signal_count"]) / total * float(row["abs_calibration_gap"]) for row in curve)
        mce = max(float(row["abs_calibration_gap"]) for row in curve)
    mean_pred = float(p.mean())
    hit_rate = float(y.mean())
    return {
        "method": method,
        "split": split_name,
        "sample_count": sample_count,
        "brier_score": _round_or_none(_brier_score(p, y), 8),
        "expected_calibration_error": _round_or_none(ece, 8),
        "max_calibration_error": _round_or_none(mce, 8),
        "mean_predicted_probability": _round_or_none(mean_pred, 6),
        "actual_hit_rate": _round_or_none(hit_rate, 6),
        "calibration_gap": _round_or_none(mean_pred - hit_rate, 6),
    }


def _fit_identity(train_prob: pd.Series, train_label: pd.Series) -> CalibrationFit:
    state = {
        "method": "identity",
        "description": "No probability mapping; returns the raw model probability.",
    }
    return CalibrationFit(
        method="identity",
        state=state,
        fit_report={"status": "reference", "sample_count": int(len(train_prob))},
        predict=lambda values: _clip_probability(values).reset_index(drop=True),
    )


def _fit_linear_shrink(train_prob: pd.Series, train_label: pd.Series) -> CalibrationFit:
    base_rate = float(train_label.mean()) if len(train_label) else 0.5
    best_alpha = 1.0
    best_brier = float("inf")
    losses: list[dict[str, float]] = []
    for alpha in np.linspace(0.0, 1.0, 101):
        pred = _clip_probability((float(alpha) * train_prob) + ((1.0 - float(alpha)) * base_rate))
        brier = _brier_score(pred, train_label)
        if brier is None:
            continue
        losses.append({"alpha": float(alpha), "brier_score": float(brier)})
        if brier < best_brier:
            best_brier = float(brier)
            best_alpha = float(alpha)

    state = {
        "method": "linear_shrink",
        "alpha": float(best_alpha),
        "base_rate": float(base_rate),
        "formula": "alpha * raw_probability + (1 - alpha) * train_base_rate",
    }

    def predict(values: pd.Series) -> pd.Series:
        raw = _clip_probability(values).reset_index(drop=True)
        return _clip_probability((best_alpha * raw) + ((1.0 - best_alpha) * base_rate))

    return CalibrationFit(
        method="linear_shrink",
        state=state,
        fit_report={
            "status": "fitted",
            "sample_count": int(len(train_prob)),
            "best_alpha": _round_or_none(best_alpha, 4),
            "base_rate": _round_or_none(base_rate, 6),
            "best_train_brier_score": _round_or_none(best_brier, 8),
            "search_steps": len(losses),
        },
        predict=predict,
    )


def _fit_score_calibrator(method: str, train_prob: pd.Series, train_label: pd.Series) -> CalibrationFit:
    score_method = "temperature" if method == "temperature_score" else "isotonic"
    calibrator = ScoreCalibrator(method=score_method, n_bins=10)
    fit_report = calibrator.fit((train_prob * 100.0).tolist(), train_label.tolist())
    state = calibrator.to_state()
    state.update(
        {
            "method": method,
            "score_calibrator_method": score_method,
            "input_probability_scale": "0_to_1",
            "score_scale_used_for_fit": "probability * 100",
        }
    )

    def predict(values: pd.Series) -> pd.Series:
        raw_scores = (_clip_probability(values).reset_index(drop=True) * 100.0).tolist()
        calibrated = pd.Series(calibrator.calibrate_batch(raw_scores), dtype="float64") / 100.0
        return _clip_probability(calibrated)

    return CalibrationFit(
        method=method,
        state=state,
        fit_report=_sanitize_value(fit_report),
        predict=predict,
    )


def _fit_calibrator(method: str, train_prob: pd.Series, train_label: pd.Series) -> CalibrationFit:
    if method == "identity":
        return _fit_identity(train_prob, train_label)
    if method == "linear_shrink":
        return _fit_linear_shrink(train_prob, train_label)
    if method in {"temperature_score", "isotonic_score"}:
        return _fit_score_calibrator(method, train_prob, train_label)
    raise ValueError(f"Unsupported calibration method: {method}")


def _comparison_rows(
    *,
    fit: CalibrationFit,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    n_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for split_name, split_frame in (("train", train), ("holdout", holdout)):
        predicted = fit.predict(split_frame["_probability"])
        labels = split_frame["_label"].reset_index(drop=True)
        row = _metrics(predicted, labels, method=fit.method, split_name=split_name, n_bins=n_bins)
        rows.append(row)
        curves.extend(_calibration_curve(predicted, labels, method=fit.method, split_name=split_name, n_bins=n_bins))
    return rows, curves


def _row_by_method(rows: list[dict[str, Any]], *, split_name: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("method")): row for row in rows if row.get("split") == split_name}


def _select_candidate(
    comparison_rows: list[dict[str, Any]],
    *,
    min_train_sample: int,
    min_holdout_sample: int,
    min_brier_improvement: float,
    max_ece_regression: float,
    max_candidate_ece: float,
) -> dict[str, Any]:
    holdout = _row_by_method(comparison_rows, split_name="holdout")
    train = _row_by_method(comparison_rows, split_name="train")
    identity = holdout.get("identity", {})
    identity_brier = identity.get("brier_score")
    identity_ece = identity.get("expected_calibration_error")
    train_count = int(next(iter(train.values())).get("sample_count", 0)) if train else 0
    holdout_count = int(identity.get("sample_count", 0) or 0)

    if train_count < int(min_train_sample) or holdout_count < int(min_holdout_sample):
        return {
            "calibration_status": "INSUFFICIENT_EVIDENCE",
            "selected_calibrator": "identity",
            "selection_reason": "sample_size_guardrail_failed",
            "train_count": train_count,
            "holdout_count": holdout_count,
            "min_train_sample": int(min_train_sample),
            "min_holdout_sample": int(min_holdout_sample),
            "holdout_brier_improvement": None,
            "holdout_ece_change": None,
            "candidate_ready_for_review": False,
        }

    candidates = []
    for method, row in holdout.items():
        if method == "identity":
            continue
        brier = row.get("brier_score")
        ece = row.get("expected_calibration_error")
        if brier is None or identity_brier is None:
            continue
        ece_change = None if ece is None or identity_ece is None else float(ece) - float(identity_ece)
        candidates.append(
            {
                "method": method,
                "brier_score": float(brier),
                "expected_calibration_error": None if ece is None else float(ece),
                "holdout_brier_improvement": float(identity_brier) - float(brier),
                "holdout_ece_change": ece_change,
            }
        )
    if not candidates:
        return {
            "calibration_status": "CALIBRATION_WATCH",
            "selected_calibrator": "identity",
            "selection_reason": "no_candidate_metrics_available",
            "train_count": train_count,
            "holdout_count": holdout_count,
            "holdout_brier_improvement": None,
            "holdout_ece_change": None,
            "candidate_ready_for_review": False,
        }

    best = sorted(candidates, key=lambda item: (item["brier_score"], item["expected_calibration_error"] or 1e9))[0]
    improvement = float(best["holdout_brier_improvement"])
    ece_change = best.get("holdout_ece_change")
    candidate_ece = best.get("expected_calibration_error")
    ece_guard_passed = ece_change is None or float(ece_change) <= float(max_ece_regression)
    candidate_ece_passed = candidate_ece is None or float(candidate_ece) <= float(max_candidate_ece)
    improvement_passed = improvement >= float(min_brier_improvement)

    if improvement_passed and ece_guard_passed and candidate_ece_passed:
        status = "CALIBRATION_CANDIDATE_READY"
        reason = "holdout_brier_improved_without_ece_regression"
        selected = best["method"]
        ready = True
    elif improvement > 0:
        status = "CALIBRATION_WATCH"
        reason = "candidate_improved_brier_but_failed_guardrails"
        selected = best["method"]
        ready = False
    else:
        status = "NO_CALIBRATION_IMPROVEMENT"
        reason = "identity_mapping_remains_best_on_holdout"
        selected = "identity"
        ready = False

    return {
        "calibration_status": status,
        "selected_calibrator": selected,
        "selection_reason": reason,
        "train_count": train_count,
        "holdout_count": holdout_count,
        "min_train_sample": int(min_train_sample),
        "min_holdout_sample": int(min_holdout_sample),
        "min_brier_improvement": float(min_brier_improvement),
        "max_ece_regression": float(max_ece_regression),
        "max_candidate_ece": float(max_candidate_ece),
        "identity_holdout_brier_score": identity_brier,
        "identity_holdout_expected_calibration_error": identity_ece,
        "holdout_brier_improvement": _round_or_none(improvement, 8),
        "holdout_ece_change": _round_or_none(ece_change, 8),
        "candidate_ready_for_review": bool(ready),
    }


def _recommend_actions(selection: dict[str, Any]) -> list[str]:
    status = selection.get("calibration_status")
    selected = selection.get("selected_calibrator")
    if status == "CALIBRATION_CANDIDATE_READY":
        return [
            f"Review `{selected}` candidate artifact against live-signal expectations before any manual runtime adoption.",
            "Keep the candidate in research/shadow review until regime slices and future holdout windows stay stable.",
            "Do not change runtime config or parameter packs from this experiment output automatically.",
        ]
    if status == "INSUFFICIENT_EVIDENCE":
        return [
            "Collect more quality-approved labels before fitting or adopting probability calibration.",
            "Keep runtime probabilities unchanged; sample-size guardrails blocked calibration selection.",
        ]
    if status == "NO_CALIBRATION_IMPROVEMENT":
        return [
            "Keep the raw probability mapping for now; tested calibrators did not improve holdout Brier score.",
            "Investigate feature-level probability generation and regime-conditioned thresholds before runtime adoption.",
        ]
    return [
        "Treat calibration as watch-only: evidence improved one metric but did not pass all guardrails.",
        "Expand holdout evidence and evaluate regime-conditioned calibration before proposing manual adoption.",
    ]


def build_probability_calibration_experiment_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    train_fraction: float = 0.70,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    min_train_sample: int = 100,
    min_holdout_sample: int = 50,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_ece: float = 0.12,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit and compare calibration mappings on a chronological holdout split."""
    raw = frame if frame is not None else pd.DataFrame()
    working = _clean_probability_and_label_frame(raw, probability_field=probability_field, label_field=label_field)
    train, holdout = _split_train_holdout(working, train_fraction=train_fraction)

    comparison: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    candidate_states: dict[str, dict[str, Any]] = {}
    fit_reports: dict[str, dict[str, Any]] = {}

    if not train.empty and not holdout.empty:
        train_prob = train["_probability"].reset_index(drop=True)
        train_label = train["_label"].reset_index(drop=True)
        for method in methods:
            fit = _fit_calibrator(str(method), train_prob, train_label)
            rows, method_curves = _comparison_rows(fit=fit, train=train, holdout=holdout, n_bins=n_bins)
            comparison.extend(rows)
            curves.extend(method_curves)
            candidate_states[fit.method] = fit.state
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
    holdout_metrics = _row_by_method(comparison, split_name="holdout").get(selected, {})
    candidate_state = candidate_states.get(selected, candidate_states.get("identity", {}))
    candidate_artifact = {
        "artifact_type": "probability_calibration_candidate",
        "generated_at": _utc_now(),
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": probability_field,
        "label_field": label_field,
        "selected_calibrator": selected,
        "calibration_status": selection.get("calibration_status"),
        "selection": selection,
        "state": candidate_state,
        "fit_report": fit_reports.get(selected, {}),
    }

    report = {
        "report_type": "probability_calibration_experiment",
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
        "methods_tested": list(methods),
        "calibration_status": selection.get("calibration_status"),
        "selected_calibrator": selected,
        "selection": selection,
        "holdout_metrics": holdout_metrics,
        "calibrator_comparison": comparison,
        "calibration_curve": curves,
        "candidate_calibrator": candidate_artifact,
        "recommended_next_actions": _recommend_actions(selection),
    }
    return _sanitize_value(report)


def render_probability_calibration_experiment_markdown(report: dict[str, Any]) -> str:
    """Render probability calibration experiment output as Markdown."""
    selection = report.get("selection", {}) or {}
    holdout = report.get("holdout_metrics", {}) or {}
    lines = [
        "# Probability Calibration Experiment",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Rows: {report.get('row_count')}",
        f"- Quality-labeled rows: {report.get('quality_labeled_row_count')}",
        f"- Train / holdout: {report.get('train_count')} / {report.get('holdout_count')}",
        f"- Probability field: `{report.get('probability_field')}`",
        f"- Calibration status: `{report.get('calibration_status')}`",
        f"- Selected calibrator: `{report.get('selected_calibrator')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Selection",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Selection reason | `{selection.get('selection_reason')}` |",
        f"| Holdout Brier improvement | {selection.get('holdout_brier_improvement')} |",
        f"| Holdout ECE change | {selection.get('holdout_ece_change')} |",
        f"| Candidate ready for review | {selection.get('candidate_ready_for_review')} |",
        f"| Selected holdout Brier score | {holdout.get('brier_score')} |",
        f"| Selected holdout ECE | {holdout.get('expected_calibration_error')} |",
        "",
        "## Holdout Comparison",
        "",
        "| Method | Brier | ECE | Mean Pred | Hit Rate | Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    holdout_rows = [row for row in report.get("calibrator_comparison", []) or [] if row.get("split") == "holdout"]
    for row in sorted(holdout_rows, key=lambda item: item.get("brier_score") if item.get("brier_score") is not None else 1e9):
        lines.append(
            f"| `{row.get('method')}` | {row.get('brier_score')} | {row.get('expected_calibration_error')} | "
            f"{row.get('mean_predicted_probability')} | {row.get('actual_hit_rate')} | {row.get('calibration_gap')} |"
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
        "comparison_csv_path": output / f"{stem}_comparison.csv",
        "curve_csv_path": output / f"{stem}_curve.csv",
        "candidate_json_path": output / f"{stem}_candidate_calibrator.json",
        "latest_json_path": output / PROBABILITY_CALIBRATION_EXPERIMENT_JSON_FILENAME,
        "latest_markdown_path": output / PROBABILITY_CALIBRATION_EXPERIMENT_MARKDOWN_FILENAME,
        "latest_comparison_csv_path": output / PROBABILITY_CALIBRATION_EXPERIMENT_COMPARISON_FILENAME,
        "latest_curve_csv_path": output / PROBABILITY_CALIBRATION_EXPERIMENT_CURVE_FILENAME,
        "latest_candidate_json_path": output / PROBABILITY_CALIBRATION_EXPERIMENT_CANDIDATE_FILENAME,
    }


def write_probability_calibration_experiment_report(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    train_fraction: float = 0.70,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    min_train_sample: int = 100,
    min_holdout_sample: int = 50,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_ece: float = 0.12,
    n_bins: int = 10,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write probability calibration experiment artifacts."""
    report = build_probability_calibration_experiment_report(
        frame,
        dataset_path=dataset_path,
        probability_field=probability_field,
        label_field=label_field,
        train_fraction=train_fraction,
        methods=methods,
        min_train_sample=min_train_sample,
        min_holdout_sample=min_holdout_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_ece=max_candidate_ece,
        n_bins=n_bins,
    )
    assert_artifact_schema(report, "probability_calibration_experiment")
    output = Path(output_dir) if output_dir is not None else DEFAULT_PROBABILITY_CALIBRATION_EXPERIMENT_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "probability_calibration_experiment"
    paths = _artifact_paths(output, stem)
    markdown = render_probability_calibration_experiment_markdown(report)
    comparison = pd.DataFrame(report.get("calibrator_comparison", []) or [])
    curve = pd.DataFrame(report.get("calibration_curve", []) or [])
    candidate = report.get("candidate_calibrator", {}) or {}

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(comparison, paths["comparison_csv_path"])
    _atomic_write_csv(curve, paths["curve_csv_path"])
    _atomic_write_text(paths["candidate_json_path"], json.dumps(candidate, indent=2, sort_keys=True, default=str))
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(comparison, paths["latest_comparison_csv_path"])
        _atomic_write_csv(curve, paths["latest_curve_csv_path"])
        _atomic_write_text(
            paths["latest_candidate_json_path"],
            json.dumps(candidate, indent=2, sort_keys=True, default=str),
        )
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_probability_calibration_experiment_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset from disk and write calibration experiment artifacts."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_probability_calibration_experiment_report(frame, dataset_path=path, **kwargs)
