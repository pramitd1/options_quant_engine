"""
Core models for the parameter registry and tuning framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterDefinition:
    key: str
    name: str
    module: str
    group: str
    category: str
    default_value: Any
    value_type: str
    description: str
    tunable: bool = True
    live_safe: bool = True
    min_value: float | int | None = None
    max_value: float | int | None = None
    allowed_values: tuple[Any, ...] | None = None
    search_strategy: str = "group_random_search"
    validation_mode: str = "walk_forward_regime_aware"
    overfit_risk: str = "medium"
    tuning_priority: int = 50
    tune_as_group: bool = True

    def to_dict(self, current_value: Any | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_values"] = list(self.allowed_values) if self.allowed_values is not None else None
        payload["current_value"] = self.default_value if current_value is None else current_value
        return payload


@dataclass(frozen=True)
class ParameterPack:
    name: str
    version: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "parent": self.parent,
            "notes": self.notes,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "overrides": dict(self.overrides),
        }


@dataclass(frozen=True)
class ObjectiveResult:
    objective_score: float
    metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    safeguards: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_score": float(self.objective_score),
            "metrics": dict(self.metrics),
            "train_metrics": dict(self.train_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "safeguards": dict(self.safeguards),
        }


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    parameter_pack_name: str
    timestamp: str
    evaluation_date_range: dict[str, Any]
    sample_count: int
    objective_metrics: dict[str, Any]
    objective_score: float
    notes: str | None = None
    assumptions: list[str] = field(default_factory=list)
    parameter_overrides: dict[str, Any] = field(default_factory=dict)
    dataset_path: str | None = None
    search_metadata: dict[str, Any] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)
    robustness_metrics: dict[str, Any] = field(default_factory=dict)
    comparison_summary: dict[str, Any] = field(default_factory=dict)
    tuning_campaign: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "parameter_pack_name": self.parameter_pack_name,
            "timestamp": self.timestamp,
            "evaluation_date_range": dict(self.evaluation_date_range),
            "sample_count": int(self.sample_count),
            "objective_metrics": dict(self.objective_metrics),
            "objective_score": float(self.objective_score),
            "notes": self.notes,
            "assumptions": list(self.assumptions),
            "parameter_overrides": dict(self.parameter_overrides),
            "dataset_path": self.dataset_path,
            "search_metadata": dict(self.search_metadata),
            "validation_results": dict(self.validation_results),
            "robustness_metrics": dict(self.robustness_metrics),
            "comparison_summary": dict(self.comparison_summary),
            "tuning_campaign": dict(self.tuning_campaign),
        }


@dataclass(frozen=True)
class TuningGroupPlan:
    group: str
    description: str
    search_strategy: str
    validation_mode: str
    parameter_keys: tuple[str, ...] = ()
    max_trials: int = 24
    overfit_risk: str = "medium"
    live_safe_only: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "description": self.description,
            "search_strategy": self.search_strategy,
            "validation_mode": self.validation_mode,
            "parameter_keys": list(self.parameter_keys),
            "max_trials": int(self.max_trials),
            "overfit_risk": self.overfit_risk,
            "live_safe_only": bool(self.live_safe_only),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    current_live_pack: str
    candidate_pack: str
    baseline_pack: str
    reason: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": bool(self.approved),
            "current_live_pack": self.current_live_pack,
            "candidate_pack": self.candidate_pack,
            "baseline_pack": self.baseline_pack,
            "reason": self.reason,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class WalkForwardSplit:
    split_id: str
    split_type: str
    train_start: str | None
    train_end: str | None
    validation_start: str | None
    validation_end: str | None
    train_count: int
    validation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "split_type": self.split_type,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "train_count": int(self.train_count),
            "validation_count": int(self.validation_count),
        }


@dataclass(frozen=True)
class PackStateAssignment:
    pack_name: str | None
    state: str
    assigned_at: str | None = None
    assigned_by: str | None = None
    notes: str | None = None
    source_experiment_id: str | None = None
    source_validation_experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "state": self.state,
            "assigned_at": self.assigned_at,
            "assigned_by": self.assigned_by,
            "notes": self.notes,
            "source_experiment_id": self.source_experiment_id,
            "source_validation_experiment_id": self.source_validation_experiment_id,
        }


@dataclass(frozen=True)
class ManualApprovalRecord:
    required: bool
    approved: bool
    reviewer: str | None = None
    timestamp: str | None = None
    notes: str | None = None
    approval_type: str = "PROMOTION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": bool(self.required),
            "approved": bool(self.approved),
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "approval_type": self.approval_type,
        }


@dataclass(frozen=True)
class PromotionLedgerEvent:
    timestamp: str
    event_type: str
    parameter_pack_name: str | None
    previous_state: str | None
    new_state: str | None
    reason: str
    criteria_summary: dict[str, Any] = field(default_factory=dict)
    experiment_references: dict[str, Any] = field(default_factory=dict)
    human_approval: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "parameter_pack_name": self.parameter_pack_name,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "reason": self.reason,
            "criteria_summary": dict(self.criteria_summary),
            "experiment_references": dict(self.experiment_references),
            "human_approval": dict(self.human_approval),
            "metadata": dict(self.metadata),
        }
