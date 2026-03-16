"""
Module: search.py

Purpose:
    Implement search utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from research.signal_evaluation.dataset import SIGNAL_DATASET_PATH
from tuning.experiments import run_parameter_experiment
from tuning.registry import get_parameter_registry


def _allowed_key(key: str, *, allow_live_unsafe: bool) -> bool:
    """
    Purpose:
        Determine whether a registry key may participate in the current search run.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        key (str): Registry key for the parameter under consideration.
        allow_live_unsafe (bool): Whether parameters marked as unsafe for live trading may still be included in the search space.
    
    Returns:
        bool: `True` when the parameter is eligible for inclusion in the search space.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    definition = get_parameter_registry().get(key)
    if not definition.tunable:
        return False
    if not allow_live_unsafe and not definition.live_safe:
        return False
    return True


def _experiment_score(result: dict[str, Any]) -> float:
    """
    Purpose:
        Compute the ranking score used to compare tuning experiment outputs.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        result (dict[str, Any]): Structured result payload produced by an earlier computation or evaluation step.
    
    Returns:
        float: Composite score used to rank experiments before promotion or comparison.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    validation = dict(result.get("validation_results", {}))
    robustness = dict(result.get("robustness_metrics", {}))
    if validation:
        return float(validation.get("aggregate_out_of_sample_score", result.get("objective_score", 0.0))) + (
            0.10 * float(robustness.get("robustness_score", 0.0))
        )
    return float(result.get("objective_score", 0.0))


def _coerce_numeric(value: Any, *, value_type: str):
    """
    Purpose:
        Coerce a sampled candidate into the declared parameter type.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        value (Any): Raw value supplied by the caller.
        value_type (str): Declared parameter type used when coercing or sampling numeric values.
    
    Returns:
        Any: Candidate value converted into the declared numeric type when appropriate.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    if value_type == "int":
        return int(round(float(value)))
    if value_type == "float":
        return round(float(value), 6)
    return value


def _numeric_bounds(definition, current_value=None) -> tuple[float, float] | None:
    """
    Purpose:
        Derive numeric search bounds for a tunable parameter definition.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        definition (Any): Parameter definition record describing bounds, allowed values, and governance flags.
        current_value (Any): Current value associated with the definition when serializing or searching.
    
    Returns:
        tuple[float, float] | None: Lower and upper bounds for numeric search, or `None` for non-numeric parameters.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    if definition.value_type not in {"int", "float"}:
        return None
    lo = definition.min_value
    hi = definition.max_value
    if lo is not None and hi is not None:
        return float(lo), float(hi)

    value = definition.default_value if current_value is None else current_value
    try:
        current = float(value)
    except Exception:
        return None

    radius = max(abs(current) * 0.25, 1.0)
    return current - radius, current + radius


def _sample_parameter_value(definition, rng: random.Random):
    """
    Purpose:
        Sample one candidate value from a parameter definition.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        definition (Any): Parameter definition record describing bounds, allowed values, and governance flags.
        rng (random.Random): Random generator used to keep candidate sampling reproducible.
    
    Returns:
        Any: One sampled candidate value drawn from allowed values or numeric bounds.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    if definition.allowed_values:
        return rng.choice(list(definition.allowed_values))
    bounds = _numeric_bounds(definition)
    if bounds is None:
        return definition.default_value
    lo, hi = bounds
    if definition.value_type == "int":
        return rng.randint(int(round(lo)), int(round(hi)))
    return round(rng.uniform(lo, hi), 6)


def _latin_hypercube_values(definition, *, iterations: int, rng: random.Random, base_value=None) -> list[Any]:
    """
    Purpose:
        Generate stratified candidate values for one parameter using a Latin-hypercube style draw.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        definition (Any): Parameter definition record describing bounds, allowed values, and governance flags.
        iterations (int): Number of candidate samples or search iterations to generate.
        rng (random.Random): Random generator used to keep candidate sampling reproducible.
        base_value (Any): Reference value around which new candidate values are generated.
    
    Returns:
        list[Any]: Stratified candidate values spanning the parameter's search range.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    if definition.allowed_values:
        choices = list(definition.allowed_values)
        if not choices:
            return [definition.default_value] * iterations
        samples = [choices[idx % len(choices)] for idx in range(iterations)]
        rng.shuffle(samples)
        return samples

    bounds = _numeric_bounds(definition, current_value=base_value)
    if bounds is None:
        return [definition.default_value] * iterations

    lo, hi = bounds
    if iterations <= 0:
        return []

    samples = []
    for idx in range(iterations):
        lower = idx / iterations
        upper = (idx + 1) / iterations
        draw = rng.uniform(lower, upper)
        value = lo + (hi - lo) * draw
        samples.append(_coerce_numeric(value, value_type=definition.value_type))
    rng.shuffle(samples)
    return samples


def _coordinate_neighbors(definition, center_value: Any) -> list[Any]:
    """
    Purpose:
        Construct nearby candidate values around the current parameter setting.
    
    Context:
        Internal helper in the `search` module. It keeps parameter-search and governance logic separate from experiment execution.
    
    Inputs:
        definition (Any): Parameter definition record describing bounds, allowed values, and governance flags.
        center_value (Any): Current center value used to propose nearby alternatives.
    
    Returns:
        list[Any]: Nearby candidate values that form a local neighborhood around the center point.
    
    Notes:
        The helper exists so search-space construction and candidate generation stay deterministic and easy to test.
    """
    if definition.allowed_values:
        return [value for value in definition.allowed_values if value != center_value]

    bounds = _numeric_bounds(definition, current_value=center_value)
    if bounds is None:
        return []

    lo, hi = bounds
    center = definition.default_value if center_value is None else center_value
    try:
        center = float(center)
    except Exception:
        center = float(definition.default_value)

    width = max(hi - lo, 1e-6)
    if definition.value_type == "int":
        step = max(int(round(width * 0.10)), 1)
        candidates = {int(round(center - step)), int(round(center + step))}
    else:
        step = max(width * 0.10, 0.01)
        candidates = {round(center - step, 6), round(center + step, 6)}

    clipped = []
    for candidate in candidates:
        candidate = max(lo, min(hi, candidate))
        clipped.append(_coerce_numeric(candidate, value_type=definition.value_type))
    return [value for value in clipped if value != center_value]


def run_grid_search(
    parameter_pack_name: str,
    *,
    grid: dict[str, list[Any]],
    dataset_path: str | None = None,
    allow_live_unsafe: bool = False,
    selection_thresholds: dict | None = None,
    objective_weights: dict | None = None,
    walk_forward_config: dict | None = None,
    comparison_baseline_pack: str | None = None,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """
    Purpose:
        Process run grid search for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        parameter_pack_name (str): Human-readable name for parameter pack.
        grid (dict[str, list[Any]]): Input associated with grid.
        dataset_path (str | None): Input associated with dataset path.
        allow_live_unsafe (bool): Boolean flag associated with allow_live_unsafe.
        selection_thresholds (dict | None): Input associated with selection thresholds.
        objective_weights (dict | None): Input associated with objective weights.
        walk_forward_config (dict | None): Input associated with walk forward config.
        comparison_baseline_pack (str | None): Input associated with comparison baseline pack.
        persist (bool): Boolean flag associated with persist.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    valid_grid = {
        key: list(values)
        for key, values in grid.items()
        if _allowed_key(key, allow_live_unsafe=allow_live_unsafe)
    }
    keys = list(valid_grid)
    if not keys:
        return []

    results = []
    for combination in itertools.product(*(valid_grid[key] for key in keys)):
        overrides = dict(zip(keys, combination))
        experiment = run_parameter_experiment(
            parameter_pack_name,
            dataset_path=dataset_path or SIGNAL_DATASET_PATH,
            pack_overrides=overrides,
            selection_thresholds=selection_thresholds,
            objective_weights=objective_weights,
            walk_forward_config=walk_forward_config,
            comparison_baseline_pack=comparison_baseline_pack,
            search_metadata={"strategy": "grid_search"},
            persist=persist,
        )
        results.append(experiment.to_dict())
    return results


def run_random_search(
    parameter_pack_name: str,
    *,
    parameter_keys: list[str],
    dataset_path: str | None = None,
    iterations: int = 10,
    seed: int = 7,
    allow_live_unsafe: bool = False,
    selection_thresholds: dict | None = None,
    objective_weights: dict | None = None,
    walk_forward_config: dict | None = None,
    comparison_baseline_pack: str | None = None,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """
    Purpose:
        Process run random search for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        parameter_pack_name (str): Human-readable name for parameter pack.
        parameter_keys (list[str]): Input associated with parameter keys.
        dataset_path (str | None): Input associated with dataset path.
        iterations (int): Input associated with iterations.
        seed (int): Input associated with seed.
        allow_live_unsafe (bool): Boolean flag associated with allow_live_unsafe.
        selection_thresholds (dict | None): Input associated with selection thresholds.
        objective_weights (dict | None): Input associated with objective weights.
        walk_forward_config (dict | None): Input associated with walk forward config.
        comparison_baseline_pack (str | None): Input associated with comparison baseline pack.
        persist (bool): Boolean flag associated with persist.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    rng = random.Random(seed)
    registry = get_parameter_registry()
    eligible = [
        registry.get(key)
        for key in parameter_keys
        if key in registry.keys() and _allowed_key(key, allow_live_unsafe=allow_live_unsafe)
    ]

    results = []
    for _ in range(max(iterations, 0)):
        overrides = {
            definition.key: _sample_parameter_value(definition, rng)
            for definition in eligible
        }
        experiment = run_parameter_experiment(
            parameter_pack_name,
            dataset_path=dataset_path or SIGNAL_DATASET_PATH,
            pack_overrides=overrides,
            selection_thresholds=selection_thresholds,
            objective_weights=objective_weights,
            walk_forward_config=walk_forward_config,
            comparison_baseline_pack=comparison_baseline_pack,
            search_metadata={"strategy": "random_search", "seed": seed},
            persist=persist,
        )
        results.append(experiment.to_dict())
    return results


def run_latin_hypercube_search(
    parameter_pack_name: str,
    *,
    parameter_keys: list[str],
    dataset_path: str | None = None,
    iterations: int = 12,
    seed: int = 17,
    allow_live_unsafe: bool = False,
    selection_thresholds: dict | None = None,
    objective_weights: dict | None = None,
    walk_forward_config: dict | None = None,
    base_overrides: dict[str, Any] | None = None,
    comparison_baseline_pack: str | None = None,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """
    Purpose:
        Process run latin hypercube search for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        parameter_pack_name (str): Human-readable name for parameter pack.
        parameter_keys (list[str]): Input associated with parameter keys.
        dataset_path (str | None): Input associated with dataset path.
        iterations (int): Input associated with iterations.
        seed (int): Input associated with seed.
        allow_live_unsafe (bool): Boolean flag associated with allow_live_unsafe.
        selection_thresholds (dict | None): Input associated with selection thresholds.
        objective_weights (dict | None): Input associated with objective weights.
        walk_forward_config (dict | None): Input associated with walk forward config.
        base_overrides (dict[str, Any] | None): Input associated with base overrides.
        comparison_baseline_pack (str | None): Input associated with comparison baseline pack.
        persist (bool): Boolean flag associated with persist.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    rng = random.Random(seed)
    registry = get_parameter_registry()
    eligible = [
        registry.get(key)
        for key in parameter_keys
        if key in registry.keys() and _allowed_key(key, allow_live_unsafe=allow_live_unsafe)
    ]
    if not eligible or iterations <= 0:
        return []

    base_overrides = dict(base_overrides or {})
    sample_matrix = {
        definition.key: _latin_hypercube_values(
            definition,
            iterations=iterations,
            rng=rng,
            base_value=base_overrides.get(definition.key, definition.default_value),
        )
        for definition in eligible
    }

    results = []
    for row_idx in range(iterations):
        overrides = dict(base_overrides)
        for definition in eligible:
            overrides[definition.key] = sample_matrix[definition.key][row_idx]
        experiment = run_parameter_experiment(
            parameter_pack_name,
            dataset_path=dataset_path or SIGNAL_DATASET_PATH,
            pack_overrides=overrides,
            selection_thresholds=selection_thresholds,
            objective_weights=objective_weights,
            walk_forward_config=walk_forward_config,
            comparison_baseline_pack=comparison_baseline_pack,
            search_metadata={"strategy": "latin_hypercube_search", "seed": seed},
            persist=persist,
        )
        results.append(experiment.to_dict())
    return results


def run_coordinate_descent_search(
    parameter_pack_name: str,
    *,
    parameter_keys: list[str],
    dataset_path: str | None = None,
    initial_overrides: dict[str, Any] | None = None,
    passes: int = 1,
    allow_live_unsafe: bool = False,
    selection_thresholds: dict | None = None,
    objective_weights: dict | None = None,
    walk_forward_config: dict | None = None,
    comparison_baseline_pack: str | None = None,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """
    Purpose:
        Process run coordinate descent search for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        parameter_pack_name (str): Human-readable name for parameter pack.
        parameter_keys (list[str]): Input associated with parameter keys.
        dataset_path (str | None): Input associated with dataset path.
        initial_overrides (dict[str, Any] | None): Input associated with initial overrides.
        passes (int): Input associated with passes.
        allow_live_unsafe (bool): Boolean flag associated with allow_live_unsafe.
        selection_thresholds (dict | None): Input associated with selection thresholds.
        objective_weights (dict | None): Input associated with objective weights.
        walk_forward_config (dict | None): Input associated with walk forward config.
        comparison_baseline_pack (str | None): Input associated with comparison baseline pack.
        persist (bool): Boolean flag associated with persist.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    registry = get_parameter_registry()
    eligible = [
        registry.get(key)
        for key in parameter_keys
        if key in registry.keys() and _allowed_key(key, allow_live_unsafe=allow_live_unsafe)
    ]
    if not eligible:
        return []

    current_overrides = dict(initial_overrides or {})
    baseline = run_parameter_experiment(
        parameter_pack_name,
        dataset_path=dataset_path or SIGNAL_DATASET_PATH,
        pack_overrides=current_overrides,
        selection_thresholds=selection_thresholds,
        objective_weights=objective_weights,
        walk_forward_config=walk_forward_config,
        comparison_baseline_pack=comparison_baseline_pack,
        search_metadata={"strategy": "coordinate_descent", "phase": "baseline"},
        persist=persist,
    ).to_dict()
    results = [baseline]
    best_result = baseline

    for pass_idx in range(max(passes, 0)):
        for definition in eligible:
            center = current_overrides.get(definition.key, definition.default_value)
            for candidate_value in _coordinate_neighbors(definition, center):
                candidate_overrides = dict(current_overrides)
                candidate_overrides[definition.key] = candidate_value
                experiment = run_parameter_experiment(
                    parameter_pack_name,
                    dataset_path=dataset_path or SIGNAL_DATASET_PATH,
                    pack_overrides=candidate_overrides,
                    selection_thresholds=selection_thresholds,
                    objective_weights=objective_weights,
                    walk_forward_config=walk_forward_config,
                    comparison_baseline_pack=comparison_baseline_pack,
                    search_metadata={
                        "strategy": "coordinate_descent",
                        "phase": "neighbor",
                        "pass_index": pass_idx,
                        "parameter_key": definition.key,
                    },
                    persist=persist,
                ).to_dict()
                results.append(experiment)
                if _experiment_score(experiment) > _experiment_score(best_result):
                    best_result = experiment
                    current_overrides = dict(best_result.get("parameter_overrides", {}))

    return results
