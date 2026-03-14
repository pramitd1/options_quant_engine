"""
Automated group-level tuning campaigns.

The campaign runner is intentionally conservative:
- group-wise rather than fully global optimization
- walk-forward validation aware
- robust-score ranking instead of raw objective only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.signal_evaluation.dataset import SIGNAL_DATASET_PATH
from tuning.experiments import TUNING_RESEARCH_DIR
from tuning.models import TuningGroupPlan
from tuning.registry import get_parameter_registry
from tuning.search import (
    run_coordinate_descent_search,
    run_latin_hypercube_search,
)


TUNING_CAMPAIGN_LEDGER_PATH = TUNING_RESEARCH_DIR / "tuning_campaign_ledger.jsonl"


DEFAULT_GROUP_MAX_TRIALS = {
    "trade_strength": 24,
    "confirmation_filter": 18,
    "macro_news": 28,
    "global_risk": 28,
    "gamma_vol_acceleration": 24,
    "dealer_pressure": 24,
    "option_efficiency": 20,
    "strike_selection": 24,
    "large_move_probability": 18,
    "event_windows": 16,
    "keyword_category": 20,
    "evaluation_thresholds": 18,
}


def _group_parameter_keys(group: str, *, allow_live_unsafe: bool) -> list[str]:
    keys = []
    for key, definition in get_parameter_registry().items():
        if definition.group != group or not definition.tunable:
            continue
        if not allow_live_unsafe and not definition.live_safe:
            continue
        keys.append((definition.tuning_priority, key))
    return [key for _, key in sorted(keys)]


def default_group_tuning_plans(*, allow_live_unsafe: bool = False) -> list[TuningGroupPlan]:
    groups = sorted({definition.group for _, definition in get_parameter_registry().items()})
    plans = []
    for group in groups:
        keys = _group_parameter_keys(group, allow_live_unsafe=allow_live_unsafe)
        if not keys:
            continue
        first = get_parameter_registry().get(keys[0])
        plans.append(
            TuningGroupPlan(
                group=group,
                description=f"Automated tuning campaign for {group}",
                search_strategy=first.search_strategy,
                validation_mode=first.validation_mode,
                parameter_keys=tuple(keys),
                max_trials=DEFAULT_GROUP_MAX_TRIALS.get(group, 18),
                overfit_risk=first.overfit_risk,
                live_safe_only=not allow_live_unsafe,
                notes="Registry-derived plan",
            )
        )
    return sorted(plans, key=lambda plan: min(
        get_parameter_registry().get(key).tuning_priority for key in plan.parameter_keys
    ))


def _score_result(result: dict[str, Any]) -> float:
    validation = dict(result.get("validation_results", {}))
    robustness = dict(result.get("robustness_metrics", {}))
    out_of_sample_score = float(validation.get("aggregate_out_of_sample_score", result.get("objective_score", 0.0)))
    robustness_score = float(robustness.get("robustness_score", 0.0))
    comparison = dict(result.get("comparison_summary", {}))
    aggregate_delta = dict(comparison.get("aggregate_delta", {}))
    direction_delta = float(aggregate_delta.get("direction_hit_rate", 0.0))
    return round(out_of_sample_score + (0.15 * robustness_score) + (0.10 * direction_delta), 6)


def _append_campaign_result(payload: dict[str, Any], path: str | Path = TUNING_CAMPAIGN_LEDGER_PATH) -> Path:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
    return ledger_path


def run_group_tuning_campaign(
    parameter_pack_name: str,
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    groups: list[str] | None = None,
    allow_live_unsafe: bool = False,
    walk_forward_config: dict[str, Any] | None = None,
    objective_weights: dict[str, float] | None = None,
    selection_thresholds: dict[str, Any] | None = None,
    comparison_baseline_pack: str | None = "baseline_v1",
    seed: int = 19,
    persist: bool = True,
) -> dict[str, Any]:
    plans = default_group_tuning_plans(allow_live_unsafe=allow_live_unsafe)
    if groups:
        allowed = set(groups)
        plans = [plan for plan in plans if plan.group in allowed]

    campaign_steps = []
    incumbent_overrides: dict[str, Any] = {}
    best_score = float("-inf")
    best_result: dict[str, Any] | None = None

    for step_idx, plan in enumerate(plans):
        step_seed = seed + (step_idx * 17)
        lhs_iterations = max(plan.max_trials // 2, 6)
        lhs_results = run_latin_hypercube_search(
            parameter_pack_name,
            parameter_keys=list(plan.parameter_keys),
            dataset_path=str(dataset_path),
            iterations=lhs_iterations,
            seed=step_seed,
            allow_live_unsafe=allow_live_unsafe,
            selection_thresholds=selection_thresholds,
            objective_weights=objective_weights,
            walk_forward_config=walk_forward_config,
            base_overrides=incumbent_overrides,
            persist=persist,
        )
        lhs_best = max(lhs_results, key=_score_result) if lhs_results else None
        coord_seed_overrides = (
            dict(lhs_best.get("parameter_overrides", {}))
            if lhs_best is not None
            else dict(incumbent_overrides)
        )
        coord_results = run_coordinate_descent_search(
            parameter_pack_name,
            parameter_keys=list(plan.parameter_keys),
            dataset_path=str(dataset_path),
            initial_overrides=coord_seed_overrides,
            passes=1,
            allow_live_unsafe=allow_live_unsafe,
            selection_thresholds=selection_thresholds,
            objective_weights=objective_weights,
            walk_forward_config=walk_forward_config,
            persist=persist,
        )
        combined = list(lhs_results) + list(coord_results)
        step_best = max(combined, key=_score_result) if combined else None
        if step_best is not None and _score_result(step_best) > best_score:
            best_result = step_best
            best_score = _score_result(step_best)
            incumbent_overrides = dict(step_best.get("parameter_overrides", {}))

        campaign_steps.append(
            {
                "group": plan.group,
                "plan": plan.to_dict(),
                "lhs_trial_count": len(lhs_results),
                "coordinate_trial_count": len(coord_results),
                "best_result": step_best,
                "best_score": _score_result(step_best) if step_best else None,
            }
        )

    payload = {
        "parameter_pack_name": parameter_pack_name,
        "dataset_path": str(dataset_path),
        "walk_forward_config": dict(walk_forward_config or {}),
        "selection_thresholds": dict(selection_thresholds or {}),
        "objective_weights": dict(objective_weights or {}),
        "comparison_baseline_pack": comparison_baseline_pack,
        "allow_live_unsafe": bool(allow_live_unsafe),
        "seed": int(seed),
        "steps": campaign_steps,
        "best_result": best_result,
        "best_score": best_score if best_result is not None else None,
        "final_overrides": dict(incumbent_overrides),
    }

    if persist:
        _append_campaign_result(payload)

    return payload
