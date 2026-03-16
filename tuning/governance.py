"""
Module: governance.py

Purpose:
    Orchestrate governed candidate generation, evidence capture, and production-vs-candidate review workflows.

Role in the System:
    Part of the tuning layer that connects research tuning results to the parameter registry and promotion workflow.

Key Outputs:
    Candidate parameter packs, supporting evidence maps, comparison reports, and review context.

Downstream Usage:
    Consumed by promotion tooling, governance reviews, and operator approval workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation import (
    SIGNAL_DATASET_PATH,
    SIGNAL_EVALUATION_REPORTS_DIR,
    load_signals_dataset,
    write_signal_evaluation_report,
)
from tuning.campaigns import run_group_tuning_campaign
from tuning.comparison import (
    build_candidate_vs_production_report,
    write_candidate_vs_production_report,
)
from tuning.experiments import run_parameter_experiment
from tuning.packs import RESEARCH_PARAMETER_PACKS_DIR, resolve_parameter_pack
from tuning.promotion import (
    PROMOTION_LEDGER_PATH,
    PROMOTION_STATE_PATH,
    append_promotion_event,
    get_active_live_pack,
    load_promotion_state,
    update_pack_state,
)
from tuning.models import PromotionLedgerEvent
from tuning.registry import get_parameter_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_REPORTS_DIR = PROJECT_ROOT / "research" / "parameter_tuning" / "reports"


def _coerce_walk_forward_config(walk_forward_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Purpose:
        Normalize walk-forward configuration inputs into the governance-ready schema.
    
    Context:
        Internal helper in the tuning layer. It keeps candidate generation and governance transformations deterministic and easy to test.
    
    Inputs:
        walk_forward_config (dict[str, Any] | None): Configuration mapping for walk forward.
    
    Returns:
        dict[str, Any]: Walk-forward configuration normalized into the schema expected by governance checks.
    
    Notes:
        This helper exists to keep surrounding orchestration readable without hiding important assumptions.
    """
    config = {
        "split_type": "rolling",
        "train_window_days": 180,
        "validation_window_days": 60,
        "step_size_days": 30,
        "minimum_train_rows": 50,
        "minimum_validation_rows": 20,
    }
    if walk_forward_config:
        config.update(dict(walk_forward_config))
    return config


def _pack_values(pack_name: str) -> dict[str, Any]:
    """
    Purpose:
        Resolve a parameter pack into a full key-value mapping across the registry.

    Context:
        Governance comparisons need effective values, not just the sparse override set stored in a pack file.

    Inputs:
        pack_name (str): Parameter-pack name to resolve.

    Returns:
        dict[str, Any]: Effective parameter values for every registry key.

    Notes:
        Missing overrides fall back to registry defaults so the returned mapping is complete.
    """
    registry = get_parameter_registry()
    resolved_pack = resolve_parameter_pack(pack_name)
    values = {}
    for key, definition in registry.items():
        values[key] = resolved_pack.overrides.get(key, definition.default_value)
    return values


def _delta_overrides(base_pack_name: str, effective_overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Purpose:
        Reduce a candidate's effective values down to the true differences versus the baseline pack.

    Context:
        Candidate packs should only store meaningful deltas from production so review artifacts stay concise and auditable.

    Inputs:
        base_pack_name (str): Baseline or production pack used as the reference.
        effective_overrides (dict[str, Any]): Candidate values after tuning.

    Returns:
        dict[str, Any]: Sparse override mapping containing only changed keys.

    Notes:
        This keeps generated candidate packs readable and prevents no-op overrides from polluting governance history.
    """
    baseline_values = _pack_values(base_pack_name)
    deltas = {}
    for key, candidate_value in dict(effective_overrides or {}).items():
        if baseline_values.get(key) != candidate_value:
            deltas[key] = candidate_value
    return deltas


def _candidate_pack_name(production_pack_name: str, explicit_name: str | None = None) -> str:
    """
    Purpose:
        Generate the stable name used for a newly materialized candidate pack.

    Context:
        Governed tuning needs traceable candidate names so reports, ledgers, and approval records can all refer to the same pack.

    Inputs:
        production_pack_name (str): Current production or baseline pack name.
        explicit_name (str | None): Optional caller-supplied candidate name.

    Returns:
        str: Candidate pack name.

    Notes:
        When no explicit name is supplied, the helper appends a UTC timestamp to the production pack name.
    """
    if explicit_name:
        return explicit_name
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ").lower()
    return f"{production_pack_name}__candidate__{stamp}"


def _recommendation_report_name(candidate_pack_name: str) -> str:
    """
    Purpose:
        Normalize a candidate pack name into a filesystem-friendly report stem.

    Context:
        Governance reports are persisted to disk, so pack names need a simple normalization step before they become directory or file names.

    Inputs:
        candidate_pack_name (str): Candidate pack name.

    Returns:
        str: Filesystem-safe report name stem.

    Notes:
        The helper is intentionally minimal because the pack naming scheme is already controlled elsewhere.
    """
    return candidate_pack_name.replace(".", "_")


def build_parameter_evidence_map(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Purpose:
        Build a per-parameter evidence map from a governed tuning campaign.

    Context:
        Reviewers need to know not just which parameters changed, but which campaign step produced the recommendation and what supporting metrics came with it.

    Inputs:
        campaign (dict[str, Any]): Governed tuning campaign payload.

    Returns:
        dict[str, dict[str, Any]]: Evidence keyed by parameter name.

    Notes:
        The evidence is intentionally compact so it can be embedded into comparison reports and approval records.
    """
    evidence_map: dict[str, dict[str, Any]] = {}
    for step in campaign.get("steps", []):
        plan = dict(step.get("plan", {}))
        best_result = dict(step.get("best_result") or {})
        if not best_result:
            continue
        parameter_overrides = dict(best_result.get("parameter_overrides") or {})
        supporting_tuning_evidence = {
            "group": step.get("group"),
            "search_strategy": plan.get("search_strategy"),
            "validation_mode": plan.get("validation_mode"),
            "best_score": step.get("best_score"),
            "objective_score": best_result.get("objective_score"),
            "validation_out_of_sample_score": (best_result.get("validation_results") or {}).get(
                "aggregate_out_of_sample_score"
            ),
            "robustness_score": (best_result.get("robustness_metrics") or {}).get("robustness_score"),
            "experiment_id": best_result.get("experiment_id"),
        }
        for key in plan.get("parameter_keys", []):
            if key not in parameter_overrides:
                continue
            evidence_map[key] = {
                "reason": f"Selected by governed tuning campaign for group '{step.get('group')}'.",
                "supporting_tuning_evidence": supporting_tuning_evidence,
            }
    return evidence_map


def materialize_candidate_parameter_pack(
    *,
    candidate_pack_name: str,
    parent_pack_name: str,
    overrides: dict[str, Any],
    description: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path = RESEARCH_PARAMETER_PACKS_DIR,
    overwrite: bool = False,
) -> Path:
    """
    Purpose:
        Materialize a candidate parameter pack on disk.

    Context:
        Governed tuning produces candidate overrides in memory; this function turns that result into a versioned pack artifact that can enter the promotion workflow.

    Inputs:
        candidate_pack_name (str): Name for the new candidate pack.
        parent_pack_name (str): Baseline pack from which the candidate was derived.
        overrides (dict[str, Any]): Sparse override mapping to persist.
        description (str | None): Optional human-readable description.
        notes (str | None): Optional review notes.
        metadata (dict[str, Any] | None): Additional metadata to persist with the pack.
        output_dir (str | Path): Destination directory for candidate packs.
        overwrite (bool): Whether an existing file may be replaced.

    Returns:
        Path: Path to the written candidate pack.

    Notes:
        The pack is stored as JSON so it can be inspected and reviewed outside Python tooling.
    """
    candidate_dir = Path(output_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{candidate_pack_name}.json"
    if candidate_path.exists() and not overwrite:
        raise FileExistsError(f"Candidate parameter pack already exists: {candidate_path}")

    payload = {
        "name": candidate_pack_name,
        "version": "1.0.0",
        "description": description or f"Candidate pack derived from {parent_pack_name}",
        "parent": parent_pack_name,
        "notes": notes,
        "tags": ["candidate", "research_generated"],
        "metadata": {
            "state": "candidate",
            "parent_pack": parent_pack_name,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            **dict(metadata or {}),
        },
        "overrides": dict(overrides),
    }
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return candidate_path


def evaluate_current_production_signal_quality(
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    production_pack_name: str | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    top_n: int = 10,
    state_path: str | Path = PROMOTION_STATE_PATH,
) -> dict[str, Any]:
    """
    Purpose:
        Generate a fresh signal-evaluation report for the currently active production pack.

    Context:
        Governance decisions compare candidate packs against the latest production baseline, so the workflow begins by refreshing the production-quality report from the signal-evaluation dataset.

    Inputs:
        dataset_path (str | Path): Signal-evaluation dataset to analyze.
        production_pack_name (str | None): Production pack to evaluate. Falls back to the active live pack.
        output_dir (str | Path | None): Output directory for the generated report.
        report_name (str | None): Optional report name override.
        top_n (int): Number of top slices or regimes to surface in the report.
        state_path (str | Path): Promotion state path used to resolve the live pack when needed.

    Returns:
        dict[str, Any]: Written report metadata and summary information.

    Notes:
        This function does not tune anything; it snapshots the current production baseline for comparison.
    """
    production_pack_name = production_pack_name or get_active_live_pack(state_path)
    frame = load_signals_dataset(dataset_path)
    return write_signal_evaluation_report(
        frame,
        production_pack_name=production_pack_name,
        dataset_path=str(Path(dataset_path)),
        output_dir=output_dir or SIGNAL_EVALUATION_REPORTS_DIR,
        report_name=report_name,
        top_n=top_n,
    )


def run_controlled_tuning_workflow(
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    production_pack_name: str | None = None,
    candidate_pack_name: str | None = None,
    groups: list[str] | None = None,
    allow_live_unsafe: bool = False,
    walk_forward_config: dict[str, Any] | None = None,
    objective_weights: dict[str, float] | None = None,
    selection_thresholds: dict[str, Any] | None = None,
    seed: int = 19,
    created_by: str | None = None,
    notes: str | None = None,
    persist: bool = True,
    state_path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
    reports_dir: str | Path = TUNING_REPORTS_DIR,
    candidate_packs_dir: str | Path = RESEARCH_PARAMETER_PACKS_DIR,
) -> dict[str, Any]:
    """
    Purpose:
        Run the governed end-to-end tuning workflow from baseline evaluation to candidate artifact creation.

    Context:
        This is the highest-level orchestration entry point in the tuning governance layer. It refreshes production diagnostics, runs grouped tuning, materializes a candidate pack, records workflow state, and writes the candidate-vs-production report bundle.

    Inputs:
        dataset_path (str | Path): Signal-evaluation dataset used for tuning and validation.
        production_pack_name (str | None): Baseline production pack. Falls back to the active live pack.
        candidate_pack_name (str | None): Optional explicit name for the generated candidate pack.
        groups (list[str] | None): Parameter groups to include in the governed campaign.
        allow_live_unsafe (bool): Whether to allow tuning changes marked unsafe for live deployment.
        walk_forward_config (dict[str, Any] | None): Walk-forward validation configuration.
        objective_weights (dict[str, float] | None): Objective-score weights for candidate selection.
        selection_thresholds (dict[str, Any] | None): Trade-selection thresholds applied during experiments.
        seed (int): Random seed for reproducible search behavior.
        created_by (str | None): Operator or workflow identifier recorded in metadata.
        notes (str | None): Optional notes attached to the candidate workflow.
        persist (bool): Whether experiment artifacts should be persisted.
        state_path (str | Path): Promotion state path.
        ledger_path (str | Path): Promotion ledger path.
        reports_dir (str | Path): Output directory for candidate-vs-production reports.
        candidate_packs_dir (str | Path): Output directory for generated candidate packs.

    Returns:
        dict[str, Any]: Full workflow payload containing the candidate pack, reports, and experiment results.

    Notes:
        The workflow intentionally stops at candidate creation and evidence capture. Manual review and promotion remain separate steps.
    """
    production_pack_name = production_pack_name or get_active_live_pack(state_path)
    walk_forward_config = _coerce_walk_forward_config(walk_forward_config)

    evaluation_report = evaluate_current_production_signal_quality(
        dataset_path=dataset_path,
        production_pack_name=production_pack_name,
        report_name=f"{_recommendation_report_name(production_pack_name)}__current_signal_quality",
        state_path=state_path,
    )

    campaign = run_group_tuning_campaign(
        production_pack_name,
        dataset_path=dataset_path,
        groups=groups,
        allow_live_unsafe=allow_live_unsafe,
        walk_forward_config=walk_forward_config,
        objective_weights=objective_weights,
        selection_thresholds=selection_thresholds,
        comparison_baseline_pack=production_pack_name,
        seed=seed,
        persist=persist,
    )
    best_result = dict(campaign.get("best_result") or {})
    if not best_result:
        raise ValueError("Tuning campaign did not produce a candidate result")

    effective_overrides = dict(campaign.get("final_overrides") or best_result.get("parameter_overrides") or {})
    candidate_overrides = _delta_overrides(production_pack_name, effective_overrides)
    if not candidate_overrides:
        raise ValueError("Tuning campaign did not produce any parameter changes relative to production")

    candidate_pack_name = _candidate_pack_name(production_pack_name, explicit_name=candidate_pack_name)
    candidate_pack_path = materialize_candidate_parameter_pack(
        candidate_pack_name=candidate_pack_name,
        parent_pack_name=production_pack_name,
        overrides=candidate_overrides,
        notes=notes,
        metadata={
            "created_by": created_by,
            "seed": seed,
            "groups": list(groups or []),
            "source_best_experiment_id": best_result.get("experiment_id"),
        },
        output_dir=candidate_packs_dir,
    )

    update_pack_state(
        state_name="candidate",
        pack_name=candidate_pack_name,
        reason="candidate_generated_from_tuning",
        assigned_by=created_by,
        notes=notes,
        source_experiment_id=best_result.get("experiment_id"),
        path=state_path,
        ledger_path=ledger_path,
    )

    production_result = run_parameter_experiment(
        production_pack_name,
        dataset_path=dataset_path,
        selection_thresholds=selection_thresholds,
        objective_weights=objective_weights,
        walk_forward_config=walk_forward_config,
        persist=persist,
    ).to_dict()
    candidate_result = run_parameter_experiment(
        candidate_pack_name,
        dataset_path=dataset_path,
        selection_thresholds=selection_thresholds,
        objective_weights=objective_weights,
        walk_forward_config=walk_forward_config,
        comparison_baseline_pack=production_pack_name,
        notes=notes,
        persist=persist,
    ).to_dict()

    parameter_evidence = build_parameter_evidence_map(campaign)
    report = build_candidate_vs_production_report(
        production_pack_name=production_pack_name,
        candidate_pack_name=candidate_pack_name,
        production_result=production_result,
        candidate_result=candidate_result,
        parameter_evidence=parameter_evidence,
    )

    report_dir = Path(reports_dir) / _recommendation_report_name(candidate_pack_name)
    report_paths = write_candidate_vs_production_report(report, output_dir=report_dir)

    append_promotion_event(
        PromotionLedgerEvent(
            timestamp=pd.Timestamp.utcnow().isoformat(),
            event_type="candidate_recommendation_created",
            parameter_pack_name=candidate_pack_name,
            previous_state=None,
            new_state="candidate",
            reason="controlled_tuning_workflow_completed",
            experiment_references={
                "best_experiment_id": best_result.get("experiment_id"),
                "candidate_experiment_id": candidate_result.get("experiment_id"),
                "production_experiment_id": production_result.get("experiment_id"),
            },
            metadata={
                "production_pack_name": production_pack_name,
                "candidate_pack_path": str(candidate_pack_path),
                "report_dir": str(report_dir),
                "expected_improvement_summary": report.get("expected_improvement_summary", {}),
            },
        ),
        path=ledger_path,
    )

    return {
        "production_pack_name": production_pack_name,
        "candidate_pack_name": candidate_pack_name,
        "candidate_pack_path": str(candidate_pack_path),
        "signal_evaluation_report": evaluation_report,
        "tuning_campaign": campaign,
        "candidate_report": report,
        "candidate_report_paths": report_paths,
        "production_result": production_result,
        "candidate_result": candidate_result,
    }


def get_candidate_review_context(
    *,
    state_path: str | Path = PROMOTION_STATE_PATH,
) -> dict[str, Any]:
    """
    Purpose:
        Return the current production/candidate pairing needed for a review decision.

    Context:
        Approval UIs and governance scripts need a small, stable snapshot of the promotion state rather than the full state document.

    Inputs:
        state_path (str | Path): Promotion state file to read.

    Returns:
        dict[str, Any]: Current production pack, candidate pack, and any recorded manual approval.

    Notes:
        This is a read-only convenience helper for review tooling.
    """
    state = load_promotion_state(state_path)
    return {
        "production_pack_name": state.get("live"),
        "candidate_pack_name": state.get("candidate"),
        "manual_approval": dict((state.get("manual_approvals") or {}).get(state.get("candidate"), {})),
    }
