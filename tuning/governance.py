"""
Governed research-to-production workflow orchestration.
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
    registry = get_parameter_registry()
    resolved_pack = resolve_parameter_pack(pack_name)
    values = {}
    for key, definition in registry.items():
        values[key] = resolved_pack.overrides.get(key, definition.default_value)
    return values


def _delta_overrides(base_pack_name: str, effective_overrides: dict[str, Any]) -> dict[str, Any]:
    baseline_values = _pack_values(base_pack_name)
    deltas = {}
    for key, candidate_value in dict(effective_overrides or {}).items():
        if baseline_values.get(key) != candidate_value:
            deltas[key] = candidate_value
    return deltas


def _candidate_pack_name(production_pack_name: str, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ").lower()
    return f"{production_pack_name}__candidate__{stamp}"


def _recommendation_report_name(candidate_pack_name: str) -> str:
    return candidate_pack_name.replace(".", "_")


def build_parameter_evidence_map(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
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
    state = load_promotion_state(state_path)
    return {
        "production_pack_name": state.get("live"),
        "candidate_pack_name": state.get("candidate"),
        "manual_approval": dict((state.get("manual_approvals") or {}).get(state.get("candidate"), {})),
    }
