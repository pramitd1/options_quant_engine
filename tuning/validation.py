"""
Module: validation.py

Purpose:
    Run walk-forward and regime-aware validation for candidate parameter packs.

Role in the System:
    Part of the tuning layer that tests whether research candidates remain robust outside their fitting sample.

Key Outputs:
    Walk-forward split summaries, robustness metrics, and baseline-vs-candidate validation comparisons.

Downstream Usage:
    Consumed by governed tuning campaigns, candidate-vs-production reports, and promotion decisions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from tuning.objectives import apply_selection_policy, compute_frame_metrics, compute_objective_score
from tuning.regimes import REGIME_COLUMNS, label_validation_regimes
from tuning.walk_forward import DEFAULT_WALK_FORWARD_CONFIG, apply_walk_forward_split, build_walk_forward_splits


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `validation` module. The module sits in the tuning layer that searches, validates, and promotes parameter packs.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (float): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _selection_penalty(metrics: dict[str, Any]) -> float:
    """
    Purpose:
        Penalize parameter packs that produce too few signals to be operationally useful.

    Context:
        Tuning should not reward a candidate solely for looking good on a tiny number of trades. This helper converts low signal frequency into an explicit objective penalty.

    Inputs:
        metrics (dict[str, Any]): Metric summary for the selected trades.

    Returns:
        float: Frequency penalty added to validation components.

    Notes:
        The penalty rises only when signal frequency falls below the minimum operating band implied by the selection policy.
    """
    return max(0.0, 0.12 - _safe_float(metrics.get("signal_frequency"), 0.0)) * 4.0


def _validation_components(
    *,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    parameter_count: int,
) -> dict[str, Any]:
    """
    Purpose:
        Build the penalty-aware metric bundle used by the validation objective.

    Context:
        The tuning objective weighs raw out-of-sample quality alongside stability, selectivity, and parameter-count discipline. This helper assembles those components in one place.

    Inputs:
        train_metrics (dict[str, Any]): Metrics computed on the training slice.
        validation_metrics (dict[str, Any]): Metrics computed on the out-of-sample validation slice.
        parameter_count (int): Number of tuned parameters in the candidate pack.

    Returns:
        dict[str, Any]: Validation metrics augmented with penalty terms.

    Notes:
        The penalties are heuristic rather than statistical; they are governance guardrails intended to discourage overfit or operationally thin candidates.
    """
    return {
        **validation_metrics,
        "selectivity_penalty": _selection_penalty(validation_metrics),
        "stability_penalty": abs(
            _safe_float(train_metrics.get("direction_hit_rate"), 0.0)
            - _safe_float(validation_metrics.get("direction_hit_rate"), 0.0)
        ),
        "parsimony_penalty": min(parameter_count, 25) / 25.0,
        "validation_gap_penalty": max(
            0.0,
            _safe_float(train_metrics.get("average_composite_signal_score"), 0.0)
            - _safe_float(validation_metrics.get("average_composite_signal_score"), 0.0),
        )
        / 100.0,
    }


def summarize_metrics_by_regime(
    frame: pd.DataFrame,
    *,
    regime_columns: list[str] | None = None,
    minimum_regime_sample_count: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Purpose:
        Slice selected signals by regime and summarize performance within each slice.

    Context:
        Regime-aware validation matters because a candidate can look strong in aggregate while collapsing in a specific volatility, flow, or macro environment.

    Inputs:
        frame (pd.DataFrame): Evaluation frame containing signal outcomes.
        regime_columns (list[str] | None): Regime columns to summarize.
        minimum_regime_sample_count (int): Minimum sample count required before a regime slice is treated as sufficiently populated.

    Returns:
        dict[str, list[dict[str, Any]]]: Per-regime summaries including sample counts and computed metrics.

    Notes:
        The summaries stay fully serializable so they can be embedded into experiment artifacts and governance reports.
    """
    labeled = label_validation_regimes(frame)
    regime_columns = list(regime_columns or REGIME_COLUMNS)
    summaries: dict[str, list[dict[str, Any]]] = {}

    for regime_column in regime_columns:
        if regime_column not in labeled.columns:
            summaries[regime_column] = []
            continue

        regime_rows = []
        for regime_label, regime_frame in labeled.groupby(regime_column, dropna=False):
            metrics = compute_frame_metrics(regime_frame, len(regime_frame))
            regime_rows.append(
                {
                    "regime_label": str(regime_label),
                    "sample_count": int(len(regime_frame)),
                    "insufficient_sample": int(len(regime_frame)) < int(minimum_regime_sample_count),
                    "metrics": metrics,
                }
            )
        summaries[regime_column] = sorted(regime_rows, key=lambda row: (row["regime_label"], row["sample_count"]))

    return summaries


def compute_robustness_metrics(
    *,
    split_results: list[dict[str, Any]],
    regime_summary: dict[str, list[dict[str, Any]]],
    minimum_regime_sample_count: int = 5,
) -> dict[str, Any]:
    """
    Purpose:
        Quantify how stable validation quality remains across time splits and regime slices.

    Context:
        Governed tuning cares about more than the mean validation score. This helper measures dispersion, weak regime pockets, and low-frequency behavior so promotion decisions can penalize fragile candidates.

    Inputs:
        split_results (list[dict[str, Any]]): Per-split walk-forward validation records.
        regime_summary (dict[str, list[dict[str, Any]]]): Per-regime validation summaries.
        minimum_regime_sample_count (int): Minimum regime sample count required before collapse checks are applied.

    Returns:
        dict[str, Any]: Robustness diagnostics and a bounded robustness score.

    Notes:
        The resulting score is heuristic. It is designed as a governance filter, not a statistically pure confidence interval.
    """
    validation_scores = [_safe_float(item.get("validation_objective_score"), 0.0) for item in split_results]
    validation_frequencies = [
        _safe_float(item.get("validation_metrics", {}).get("signal_frequency"), 0.0)
        for item in split_results
    ]
    split_score_dispersion = float(pd.Series(validation_scores).std(ddof=0)) if validation_scores else 0.0
    split_frequency_dispersion = float(pd.Series(validation_frequencies).std(ddof=0)) if validation_frequencies else 0.0

    regime_objective_scores = []
    insufficient_regimes = 0
    minimum_regime_sample = None
    for regime_rows in regime_summary.values():
        for row in regime_rows:
            minimum_regime_sample = row["sample_count"] if minimum_regime_sample is None else min(minimum_regime_sample, row["sample_count"])
            if row.get("insufficient_sample"):
                insufficient_regimes += 1
            regime_objective_scores.append(_safe_float(row.get("metrics", {}).get("direction_hit_rate"), 0.0))
    regime_score_dispersion = float(pd.Series(regime_objective_scores).std(ddof=0)) if regime_objective_scores else 0.0

    collapse_penalty = 0.0
    for regime_rows in regime_summary.values():
        for row in regime_rows:
            if row["sample_count"] >= minimum_regime_sample_count and _safe_float(row["metrics"].get("direction_hit_rate"), 0.0) < 0.35:
                collapse_penalty += 0.1

    low_frequency_penalty = max(0.0, 0.08 - min(validation_frequencies or [0.0])) * 4.0
    robustness_score = max(
        0.0,
        1.0
        - split_score_dispersion
        - split_frequency_dispersion
        - regime_score_dispersion
        - collapse_penalty
        - low_frequency_penalty
        - min(insufficient_regimes, 10) * 0.02,
    )

    return {
        "split_score_dispersion": round(split_score_dispersion, 6),
        "split_frequency_dispersion": round(split_frequency_dispersion, 6),
        "regime_score_dispersion": round(regime_score_dispersion, 6),
        "minimum_regime_sample_count": int(minimum_regime_sample or 0),
        "insufficient_regime_count": int(insufficient_regimes),
        "collapse_penalty": round(collapse_penalty, 6),
        "low_signal_frequency_penalty": round(low_frequency_penalty, 6),
        "robustness_score": round(robustness_score, 6),
    }


def run_walk_forward_validation(
    frame: pd.DataFrame,
    *,
    selection_thresholds: dict[str, Any] | None = None,
    objective_weights: dict[str, float] | None = None,
    parameter_count: int = 0,
    walk_forward_config: dict[str, Any] | None = None,
    regime_columns: list[str] | None = None,
    minimum_regime_sample_count: int = 5,
) -> dict[str, Any]:
    """
    Purpose:
        Run the repo's standard walk-forward validation workflow on a signal dataset.

    Context:
        This is the main validation entry point used by tuning experiments. It applies time-based splits, re-runs the selection policy on each split, and returns both aggregate and regime-aware out-of-sample diagnostics.

    Inputs:
        frame (pd.DataFrame): Signal-evaluation dataset to validate.
        selection_thresholds (dict[str, Any] | None): Selection-policy thresholds used to filter trades inside each split.
        objective_weights (dict[str, float] | None): Weights for the objective-score calculation.
        parameter_count (int): Number of tuned parameters in the candidate pack.
        walk_forward_config (dict[str, Any] | None): Split configuration for the walk-forward process.
        regime_columns (list[str] | None): Regime columns to summarize in the out-of-sample slice.
        minimum_regime_sample_count (int): Minimum regime sample count required for regime-level collapse checks.

    Returns:
        dict[str, Any]: Full walk-forward validation payload with split details, aggregate scores, and robustness diagnostics.

    Notes:
        The workflow is deliberately time-based only. It avoids random shuffles so the validation path better matches how the strategy would have evolved in production.
    """
    ordered = frame.copy() if frame is not None else pd.DataFrame()
    config = dict(DEFAULT_WALK_FORWARD_CONFIG)
    if walk_forward_config:
        config.update(walk_forward_config)

    splits = build_walk_forward_splits(
        ordered,
        split_type=config["split_type"],
        train_window_days=config["train_window_days"],
        validation_window_days=config["validation_window_days"],
        step_size_days=config.get("step_size_days"),
        minimum_train_rows=config["minimum_train_rows"],
        minimum_validation_rows=config["minimum_validation_rows"],
    )

    split_results = []
    all_train_selected = []
    all_validation_selected = []

    for split in splits:
        split_frames = apply_walk_forward_split(ordered, split)
        train_selected = apply_selection_policy(split_frames.train, thresholds=selection_thresholds)
        validation_selected = apply_selection_policy(split_frames.validation, thresholds=selection_thresholds)

        train_metrics = compute_frame_metrics(train_selected, len(split_frames.train))
        validation_metrics = compute_frame_metrics(validation_selected, len(split_frames.validation))
        components = _validation_components(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            parameter_count=parameter_count,
        )
        validation_objective_score = compute_objective_score(
            components,
            objective_weights=objective_weights,
        )

        train_regimes = summarize_metrics_by_regime(
            train_selected,
            regime_columns=regime_columns,
            minimum_regime_sample_count=minimum_regime_sample_count,
        )
        validation_regimes = summarize_metrics_by_regime(
            validation_selected,
            regime_columns=regime_columns,
            minimum_regime_sample_count=minimum_regime_sample_count,
        )

        split_results.append(
            {
                **split.to_dict(),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "validation_objective_score": validation_objective_score,
                "validation_components": components,
                "train_regime_summary": train_regimes,
                "validation_regime_summary": validation_regimes,
            }
        )
        all_train_selected.append(train_selected)
        all_validation_selected.append(validation_selected)

    train_frame = pd.concat(all_train_selected, ignore_index=True) if all_train_selected else pd.DataFrame()
    validation_frame = pd.concat(all_validation_selected, ignore_index=True) if all_validation_selected else pd.DataFrame()

    aggregate_train_metrics = compute_frame_metrics(train_frame, len(train_frame))
    aggregate_validation_metrics = compute_frame_metrics(validation_frame, len(validation_frame))
    aggregate_components = _validation_components(
        train_metrics=aggregate_train_metrics,
        validation_metrics=aggregate_validation_metrics,
        parameter_count=parameter_count,
    )
    aggregate_validation_score = compute_objective_score(
        aggregate_components,
        objective_weights=objective_weights,
    )

    regime_summary = summarize_metrics_by_regime(
        validation_frame,
        regime_columns=regime_columns,
        minimum_regime_sample_count=minimum_regime_sample_count,
    )
    robustness_metrics = compute_robustness_metrics(
        split_results=split_results,
        regime_summary=regime_summary,
        minimum_regime_sample_count=minimum_regime_sample_count,
    )

    safeguards = {
        "time_based_only": True,
        "split_count": int(len(split_results)),
        "minimum_split_count_ok": len(split_results) >= 2,
        "minimum_regime_sample_count": int(minimum_regime_sample_count),
        "small_regime_slices_present": bool(robustness_metrics["insufficient_regime_count"] > 0),
        "out_of_sample_emphasis": True,
    }

    return {
        "validation_type": "walk_forward_regime_aware",
        "config": config,
        "split_results": split_results,
        "aggregate_train_metrics": aggregate_train_metrics,
        "aggregate_out_of_sample_metrics": aggregate_validation_metrics,
        "aggregate_out_of_sample_components": aggregate_components,
        "aggregate_out_of_sample_score": aggregate_validation_score,
        "regime_summary": regime_summary,
        "robustness_metrics": robustness_metrics,
        "safeguards": safeguards,
    }


def compare_validation_results(
    baseline_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    *,
    baseline_pack_name: str = "baseline_v1",
    candidate_pack_name: str = "candidate_v1",
) -> dict[str, Any]:
    """
    Purpose:
        Compare validation results for governance or diagnostic purposes.
    
    Context:
        Public function in the `validation` module. It forms part of the tuning workflow exposed by this module.
    
    Inputs:
        baseline_validation (dict[str, Any]): Structured mapping for baseline validation.
        candidate_validation (dict[str, Any]): Structured mapping for candidate validation.
        baseline_pack_name (str): Production or baseline parameter pack used as the comparison reference.
        candidate_pack_name (str): Human-readable or stable name for candidate pack.
    
    Returns:
        dict[str, Any]: Structured mapping returned by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    baseline_validation = dict(baseline_validation or {})
    candidate_validation = dict(candidate_validation or {})

    baseline_aggregate = dict(baseline_validation.get("aggregate_out_of_sample_metrics", {}))
    candidate_aggregate = dict(candidate_validation.get("aggregate_out_of_sample_metrics", {}))
    baseline_robustness = dict(baseline_validation.get("robustness_metrics", {}))
    candidate_robustness = dict(candidate_validation.get("robustness_metrics", {}))

    aggregate_delta = {
        metric: round(_safe_float(candidate_aggregate.get(metric), 0.0) - _safe_float(baseline_aggregate.get(metric), 0.0), 6)
        for metric in sorted(set(baseline_aggregate) | set(candidate_aggregate))
    }

    split_map = defaultdict(dict)
    for row in baseline_validation.get("split_results", []):
        split_map[row["split_id"]]["baseline"] = row
    for row in candidate_validation.get("split_results", []):
        split_map[row["split_id"]]["candidate"] = row

    split_comparison = []
    for split_id, payload in sorted(split_map.items()):
        baseline_score = _safe_float(payload.get("baseline", {}).get("validation_objective_score"), 0.0)
        candidate_score = _safe_float(payload.get("candidate", {}).get("validation_objective_score"), 0.0)
        split_comparison.append(
            {
                "split_id": split_id,
                "baseline_validation_objective_score": round(baseline_score, 6),
                "candidate_validation_objective_score": round(candidate_score, 6),
                "delta": round(candidate_score - baseline_score, 6),
            }
        )

    regime_comparison = {}
    regime_columns = sorted(
        set(baseline_validation.get("regime_summary", {}).keys())
        | set(candidate_validation.get("regime_summary", {}).keys())
    )
    for regime_column in regime_columns:
        row_map = defaultdict(dict)
        for row in baseline_validation.get("regime_summary", {}).get(regime_column, []):
            row_map[row["regime_label"]]["baseline"] = row
        for row in candidate_validation.get("regime_summary", {}).get(regime_column, []):
            row_map[row["regime_label"]]["candidate"] = row

        comparison_rows = []
        for regime_label, payload in sorted(row_map.items()):
            baseline_rate = _safe_float(payload.get("baseline", {}).get("metrics", {}).get("direction_hit_rate"), 0.0)
            candidate_rate = _safe_float(payload.get("candidate", {}).get("metrics", {}).get("direction_hit_rate"), 0.0)
            comparison_rows.append(
                {
                    "regime_label": regime_label,
                    "baseline_sample_count": int(payload.get("baseline", {}).get("sample_count", 0)),
                    "candidate_sample_count": int(payload.get("candidate", {}).get("sample_count", 0)),
                    "baseline_direction_hit_rate": round(baseline_rate, 6),
                    "candidate_direction_hit_rate": round(candidate_rate, 6),
                    "direction_hit_rate_delta": round(candidate_rate - baseline_rate, 6),
                }
            )
        regime_comparison[regime_column] = comparison_rows

    return {
        "baseline_pack_name": baseline_pack_name,
        "candidate_pack_name": candidate_pack_name,
        "aggregate_comparison": {
            "baseline_out_of_sample_score": round(_safe_float(baseline_validation.get("aggregate_out_of_sample_score"), 0.0), 6),
            "candidate_out_of_sample_score": round(_safe_float(candidate_validation.get("aggregate_out_of_sample_score"), 0.0), 6),
            "out_of_sample_score_delta": round(
                _safe_float(candidate_validation.get("aggregate_out_of_sample_score"), 0.0)
                - _safe_float(baseline_validation.get("aggregate_out_of_sample_score"), 0.0),
                6,
            ),
            "metric_deltas": aggregate_delta,
        },
        "robustness_comparison": {
            "baseline_robustness_score": round(_safe_float(baseline_robustness.get("robustness_score"), 0.0), 6),
            "candidate_robustness_score": round(_safe_float(candidate_robustness.get("robustness_score"), 0.0), 6),
            "robustness_score_delta": round(
                _safe_float(candidate_robustness.get("robustness_score"), 0.0)
                - _safe_float(baseline_robustness.get("robustness_score"), 0.0),
                6,
            ),
            "baseline": baseline_robustness,
            "candidate": candidate_robustness,
        },
        "split_comparison": split_comparison,
        "regime_comparison": regime_comparison,
    }
