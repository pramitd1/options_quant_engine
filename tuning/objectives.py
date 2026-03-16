"""
Module: objectives.py

Purpose:
    Implement objectives utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tuning.models import ObjectiveResult


DEFAULT_OBJECTIVE_WEIGHTS = {
    "direction_hit_rate": 0.24,
    "average_composite_signal_score": 0.18,
    "average_tradeability_score": 0.14,
    "average_target_reachability_score": 0.12,
    "drawdown_proxy": -0.10,
    "regime_stability": 0.08,
    "signal_frequency": 0.08,
    "selectivity_penalty": -0.04,
    "stability_penalty": -0.02,
    "parsimony_penalty": -0.02,
    "validation_gap_penalty": -0.02,
}


@dataclass(frozen=True)
class SplitFrames:
    """
    Purpose:
        Represent SplitFrames within the repository.
    
    Context:
        Used within the `objectives` module. The class standardizes records that move through search, validation, shadow mode, and promotion.
    
    Attributes:
        train (pd.DataFrame): DataFrame containing train.
        validation (pd.DataFrame): DataFrame containing validation.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, and promoted without accidental mutation.
    """
    train: pd.DataFrame
    validation: pd.DataFrame


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `objectives` module. The module sits in the tuning layer that searches, validates, and promotes parameter packs.

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


def _mean_or_default(series: pd.Series, default: float = 0.0) -> float:
    """
    Purpose:
        Process mean or default for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        series (pd.Series): Input associated with series.
        default (float): Input associated with default.
    
    Returns:
        float: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return default
    return float(numeric.mean())


def time_train_validation_split(frame: pd.DataFrame, validation_fraction: float = 0.30) -> SplitFrames:
    """
    Purpose:
        Process time train validation split for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        validation_fraction (float): Input associated with validation fraction.
    
    Returns:
        SplitFrames: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty:
        return SplitFrames(train=frame.copy(), validation=frame.copy())

    ordered = frame.copy()
    ordered["signal_timestamp"] = pd.to_datetime(ordered["signal_timestamp"], errors="coerce")
    ordered = ordered.sort_values("signal_timestamp", kind="stable").reset_index(drop=True)
    split_idx = max(int(len(ordered) * (1.0 - validation_fraction)), 1)
    split_idx = min(split_idx, len(ordered))
    return SplitFrames(
        train=ordered.iloc[:split_idx].copy(),
        validation=ordered.iloc[split_idx:].copy(),
    )


def apply_selection_policy(frame: pd.DataFrame, *, thresholds: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Purpose:
        Process apply selection policy for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        thresholds (dict[str, Any] | None): Input associated with thresholds.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty:
        return frame.copy()

    thresholds = thresholds or {}
    selected = frame.copy()
    trade_strength_floor = _safe_float(thresholds.get("trade_strength_floor"), 0.0)
    composite_floor = _safe_float(thresholds.get("composite_signal_score_floor"), 0.0)
    tradeability_floor = _safe_float(thresholds.get("tradeability_score_floor"), 0.0)
    move_probability_floor = _safe_float(thresholds.get("move_probability_floor"), 0.0)
    option_efficiency_floor = _safe_float(thresholds.get("option_efficiency_score_floor"), 0.0)
    global_risk_cap = _safe_float(thresholds.get("global_risk_score_cap"), 100.0)
    require_overnight_allowed = bool(thresholds.get("require_overnight_hold_allowed", False))

    selected = selected[pd.to_numeric(selected.get("trade_strength"), errors="coerce").fillna(-1e9) >= trade_strength_floor]
    selected = selected[pd.to_numeric(selected.get("composite_signal_score"), errors="coerce").fillna(-1e9) >= composite_floor]
    selected = selected[pd.to_numeric(selected.get("tradeability_score"), errors="coerce").fillna(-1e9) >= tradeability_floor]
    selected = selected[pd.to_numeric(selected.get("hybrid_move_probability"), errors="coerce").fillna(-1e9) >= move_probability_floor]
    if "option_efficiency_score" in selected.columns:
        selected = selected[pd.to_numeric(selected.get("option_efficiency_score"), errors="coerce").fillna(50.0) >= option_efficiency_floor]
    if "global_risk_score" in selected.columns:
        selected = selected[pd.to_numeric(selected.get("global_risk_score"), errors="coerce").fillna(0.0) <= global_risk_cap]
    if require_overnight_allowed and "overnight_hold_allowed" in selected.columns:
        selected = selected[selected["overnight_hold_allowed"].astype(str).str.lower().isin({"true", "1"})]

    return selected.reset_index(drop=True)


def _regime_stability(frame: pd.DataFrame) -> float:
    """
    Purpose:
        Process regime stability for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        float: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or "signal_regime" not in frame.columns:
        return 0.0
    counts = frame["signal_regime"].fillna("UNKNOWN").value_counts(normalize=True)
    if counts.empty:
        return 0.0
    concentration = float(counts.max())
    return max(0.0, 1.0 - concentration)


def compute_frame_metrics(selected: pd.DataFrame, total_sample_count: int) -> dict[str, Any]:
    """
    Purpose:
        Compute frame metrics from the supplied inputs.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        selected (pd.DataFrame): Input associated with selected.
        total_sample_count (int): Input associated with total sample count.
    
    Returns:
        dict[str, Any]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    selected_count = int(len(selected))
    signal_frequency = 0.0 if total_sample_count <= 0 else selected_count / float(total_sample_count)
    direction_hit_rate = _mean_or_default(selected.get("correct_60m", pd.Series(dtype=float)))
    average_composite_signal_score = _mean_or_default(selected.get("composite_signal_score", pd.Series(dtype=float)))
    average_tradeability_score = _mean_or_default(selected.get("tradeability_score", pd.Series(dtype=float)))
    average_target_reachability_score = _mean_or_default(selected.get("target_reachability_score", pd.Series(dtype=float)), 50.0)
    drawdown_proxy = abs(_mean_or_default(selected.get("mae_60m_bps", pd.Series(dtype=float))))
    regime_stability = _regime_stability(selected)
    average_realized_return_60m = _mean_or_default(selected.get("signed_return_60m_bps", pd.Series(dtype=float)))
    average_realized_return_session_close = _mean_or_default(
        selected.get("signed_return_session_close_bps", pd.Series(dtype=float))
    )

    return {
        "selected_count": selected_count,
        "signal_frequency": round(signal_frequency, 4),
        "direction_hit_rate": round(direction_hit_rate, 4),
        "average_composite_signal_score": round(average_composite_signal_score, 4),
        "average_tradeability_score": round(average_tradeability_score, 4),
        "average_target_reachability_score": round(average_target_reachability_score, 4),
        "average_realized_return_60m_bps": round(average_realized_return_60m, 4),
        "average_realized_return_session_close_bps": round(average_realized_return_session_close, 4),
        "drawdown_proxy": round(drawdown_proxy, 4),
        "regime_stability": round(regime_stability, 4),
    }


def compute_objective_score(components: dict[str, Any], *, objective_weights: dict[str, float] | None = None) -> float:
    """
    Purpose:
        Compute objective score from the supplied inputs.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        components (dict[str, Any]): Input associated with components.
        objective_weights (dict[str, float] | None): Input associated with objective weights.
    
    Returns:
        float: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    weights = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if objective_weights:
        weights.update(objective_weights)

    score = 0.0
    for name, weight in weights.items():
        score += weight * _safe_float(components.get(name), 0.0)
    return round(score, 6)


def compute_objective(
    frame: pd.DataFrame,
    *,
    thresholds: dict[str, Any] | None = None,
    objective_weights: dict[str, float] | None = None,
    parameter_count: int = 0,
) -> ObjectiveResult:
    """
    Purpose:
        Compute objective from the supplied inputs.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        thresholds (dict[str, Any] | None): Input associated with thresholds.
        objective_weights (dict[str, float] | None): Input associated with objective weights.
        parameter_count (int): Input associated with parameter count.
    
    Returns:
        ObjectiveResult: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    frame = frame.copy() if frame is not None else pd.DataFrame()
    split = time_train_validation_split(frame)
    train_selected = apply_selection_policy(split.train, thresholds=thresholds)
    validation_selected = apply_selection_policy(split.validation, thresholds=thresholds)

    train_metrics = compute_frame_metrics(train_selected, len(split.train))
    validation_metrics = compute_frame_metrics(validation_selected, len(split.validation))

    total_selected = pd.concat([train_selected, validation_selected], ignore_index=True)
    metrics = compute_frame_metrics(total_selected, len(frame))

    selectivity_penalty = max(0.0, 0.12 - metrics["signal_frequency"]) * 4.0
    stability_penalty = abs(train_metrics["direction_hit_rate"] - validation_metrics["direction_hit_rate"])
    parsimony_penalty = min(parameter_count, 25) / 25.0
    validation_gap_penalty = max(
        0.0,
        train_metrics["average_composite_signal_score"] - validation_metrics["average_composite_signal_score"],
    ) / 100.0

    safeguards = {
        "minimum_sample_ok": metrics["selected_count"] >= 10,
        "frequency_ok": metrics["signal_frequency"] >= 0.05,
        "stability_gap": round(stability_penalty, 4),
        "parsimony_penalty": round(parsimony_penalty, 4),
        "selectivity_penalty": round(selectivity_penalty, 4),
        "validation_gap_penalty": round(validation_gap_penalty, 4),
    }

    components = dict(metrics)
    components.update(
        {
            "selectivity_penalty": selectivity_penalty,
            "stability_penalty": stability_penalty,
            "parsimony_penalty": parsimony_penalty,
            "validation_gap_penalty": validation_gap_penalty,
        }
    )

    return ObjectiveResult(
        objective_score=compute_objective_score(components, objective_weights=objective_weights),
        metrics=metrics,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        safeguards=safeguards,
    )
