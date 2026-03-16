"""
Module: experiments.py

Purpose:
    Implement experiments utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd

from research.signal_evaluation.dataset import SIGNAL_DATASET_PATH, load_signals_dataset
from config.signal_evaluation_scoring import get_signal_evaluation_selection_policy
from tuning.artifacts import append_jsonl_record
from tuning.models import ExperimentResult
from tuning.objectives import compute_objective
from tuning.packs import resolve_parameter_pack
from tuning.runtime import temporary_parameter_pack
from tuning.validation import compare_validation_results, run_walk_forward_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_RESEARCH_DIR = PROJECT_ROOT / "research" / "parameter_tuning"
EXPERIMENT_LEDGER_PATH = TUNING_RESEARCH_DIR / "experiment_ledger.jsonl"


def _ensure_tuning_dir() -> None:
    """
    Purpose:
        Ensure tuning dir exists and is ready for use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    TUNING_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def _evaluation_date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    """
    Purpose:
        Process evaluation date range for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        dict[str, str | None]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame.empty or "signal_timestamp" not in frame.columns:
        return {"start": None, "end": None}
    ts = pd.to_datetime(frame["signal_timestamp"], errors="coerce").dropna()
    if ts.empty:
        return {"start": None, "end": None}
    return {"start": ts.min().isoformat(), "end": ts.max().isoformat()}


def append_experiment_result(result: ExperimentResult, path: str | Path = EXPERIMENT_LEDGER_PATH) -> Path:
    """
    Purpose:
        Process append experiment result for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        result (ExperimentResult): Input associated with result.
        path (str | Path): Input associated with path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return append_jsonl_record(result.to_dict(), path)


def run_parameter_experiment(
    parameter_pack_name: str,
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    pack_overrides: dict | None = None,
    selection_thresholds: dict | None = None,
    objective_weights: dict | None = None,
    walk_forward_config: dict | None = None,
    comparison_baseline_pack: str | None = None,
    notes: str | None = None,
    assumptions: list[str] | None = None,
    search_metadata: dict | None = None,
    persist: bool = True,
) -> ExperimentResult:
    """
    Purpose:
        Process run parameter experiment for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        parameter_pack_name (str): Human-readable name for parameter pack.
        dataset_path (str | Path): Input associated with dataset path.
        pack_overrides (dict | None): Input associated with pack overrides.
        selection_thresholds (dict | None): Input associated with selection thresholds.
        objective_weights (dict | None): Input associated with objective weights.
        walk_forward_config (dict | None): Input associated with walk forward config.
        comparison_baseline_pack (str | None): Input associated with comparison baseline pack.
        notes (str | None): Input associated with notes.
        assumptions (list[str] | None): Input associated with assumptions.
        search_metadata (dict | None): Input associated with search metadata.
        persist (bool): Boolean flag associated with persist.
    
    Returns:
        ExperimentResult: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    _ensure_tuning_dir()
    resolved_pack = resolve_parameter_pack(parameter_pack_name)
    effective_overrides = dict(resolved_pack.overrides)
    if pack_overrides:
        effective_overrides.update(pack_overrides)

    frame = load_signals_dataset(dataset_path)
    parameter_count = len(effective_overrides)

    validation_results = {}
    robustness_metrics = {}
    comparison_summary = {}

    with temporary_parameter_pack(parameter_pack_name, overrides=pack_overrides):
        effective_selection_thresholds = (
            dict(get_signal_evaluation_selection_policy())
            if selection_thresholds is None
            else dict(selection_thresholds)
        )
        objective = compute_objective(
            frame,
            thresholds=effective_selection_thresholds,
            objective_weights=objective_weights,
            parameter_count=parameter_count,
        )
        if walk_forward_config:
            validation_results = run_walk_forward_validation(
                frame,
                selection_thresholds=effective_selection_thresholds,
                objective_weights=objective_weights,
                parameter_count=parameter_count,
                walk_forward_config=walk_forward_config,
            )
            robustness_metrics = dict(validation_results.get("robustness_metrics", {}))

    if comparison_baseline_pack and walk_forward_config:
        baseline_pack = resolve_parameter_pack(comparison_baseline_pack)
        with temporary_parameter_pack(comparison_baseline_pack):
            baseline_thresholds = (
                dict(get_signal_evaluation_selection_policy())
                if selection_thresholds is None
                else dict(selection_thresholds)
            )
            baseline_validation = run_walk_forward_validation(
                frame,
                selection_thresholds=baseline_thresholds,
                objective_weights=objective_weights,
                parameter_count=len(baseline_pack.overrides),
                walk_forward_config=walk_forward_config,
            )
        comparison_summary = compare_validation_results(
            baseline_validation,
            validation_results,
            baseline_pack_name=comparison_baseline_pack,
            candidate_pack_name=parameter_pack_name,
        )

    result = ExperimentResult(
        experiment_id=f"exp_{uuid.uuid4().hex[:12]}",
        parameter_pack_name=parameter_pack_name,
        timestamp=pd.Timestamp.utcnow().isoformat(),
        evaluation_date_range=_evaluation_date_range(frame),
        sample_count=int(len(frame)),
        objective_metrics={
            "metrics": objective.metrics,
            "train_metrics": objective.train_metrics,
            "validation_metrics": objective.validation_metrics,
            "safeguards": objective.safeguards,
        },
        objective_score=objective.objective_score,
        notes=notes,
        assumptions=list(assumptions or []),
        parameter_overrides=effective_overrides,
        dataset_path=str(Path(dataset_path)),
        search_metadata=dict(search_metadata or {}),
        validation_results=validation_results,
        robustness_metrics=robustness_metrics,
        comparison_summary=comparison_summary,
    )

    if persist:
        append_experiment_result(result)

    return result
