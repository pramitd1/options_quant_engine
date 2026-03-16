"""
Module: models.py

Purpose:
    Define the core tuning records used for parameter registry entries, parameter packs, objective summaries, and experiment artifacts.

Role in the System:
    Part of the tuning layer that standardizes how candidate and production parameter packs move through search, validation, shadow mode, and promotion.

Key Outputs:
    Immutable dataclass records used by the parameter registry, experiment persistence, and promotion workflow.

Downstream Usage:
    Consumed by tuning search, governance checks, shadow-mode comparisons, and parameter-pack serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterDefinition:
    """
    Purpose:
        Dataclass representing ParameterDefinition within the repository.
    
    Context:
        Used within the tuning layer that searches, validates, and governs parameter packs. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        key (str): Value supplied for key.
        name (str): Value supplied for name.
        module (str): Value supplied for module.
        group (str): Value supplied for group.
        category (str): Value supplied for category.
        default_value (Any): Continuous value for default.
        value_type (str): Value supplied for value type.
        description (str): Value supplied for description.
        tunable (bool): Boolean flag controlling whether tunable is active.
        live_safe (bool): Boolean flag controlling whether live safe is active.
        min_value (float | int | None): Continuous value for min.
        max_value (float | int | None): Continuous value for max.
        allowed_values (tuple[Any, ...] | None): Collection of values for allowed.
        search_strategy (str): Value supplied for search strategy.
        validation_mode (str): Value supplied for validation mode.
        overfit_risk (str): Value supplied for overfit risk.
        tuning_priority (int): Value supplied for tuning priority.
        tune_as_group (bool): Boolean flag controlling whether tune as group is active.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
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
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `ParameterDefinition` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            current_value (Any | None): Current value to expose alongside the registry default when serializing.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        payload = asdict(self)
        payload["allowed_values"] = list(self.allowed_values) if self.allowed_values is not None else None
        payload["current_value"] = self.default_value if current_value is None else current_value
        return payload


@dataclass(frozen=True)
class ParameterPack:
    """
    Purpose:
        Dataclass representing ParameterPack within the repository.
    
    Context:
        Used within the tuning layer that searches, validates, and governs parameter packs. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        name (str): Value supplied for name.
        version (str): Value supplied for version.
        description (str): Value supplied for description.
        overrides (dict[str, Any]): Structured mapping for overrides.
        parent (str | None): Value supplied for parent.
        notes (str | None): Value supplied for notes.
        tags (tuple[str, ...]): Value supplied for tags.
        metadata (dict[str, Any]): Structured mapping for metadata.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    name: str
    version: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `ParameterPack` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Immutable container for the composite tuning objective and the metric bundles that support it.
    
    Context:
        Used within the `models` module. The class standardizes records that move through search, validation, shadow mode, governance, and promotion.
    
    Attributes:
        objective_score (float): Composite score used to compare experiment results.
        metrics (dict[str, Any]): Metric bundle associated with the result.
        train_metrics (dict[str, Any]): Metrics measured on the training segment.
        validation_metrics (dict[str, Any]): Metrics measured on the validation segment.
        safeguards (dict[str, Any]): Safeguard diagnostics used during governance review.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, promoted, and audited without accidental mutation.
    """
    objective_score: float
    metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    safeguards: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `ObjectiveResult` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return {
            "objective_score": float(self.objective_score),
            "metrics": dict(self.metrics),
            "train_metrics": dict(self.train_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "safeguards": dict(self.safeguards),
        }


@dataclass(frozen=True)
class ExperimentResult:
    """
    Purpose:
        Immutable record for one tuning experiment, including overrides, validation results, and promotion metadata.
    
    Context:
        Used within the `models` module. The class standardizes records that move through search, validation, shadow mode, governance, and promotion.
    
    Attributes:
        experiment_id (str): Stable identifier for the experiment record.
        parameter_pack_name (str): Name of the parameter pack associated with the record.
        timestamp (str): Timestamp recorded for the object.
        evaluation_date_range (dict[str, Any]): Date range covered by the evaluation.
        sample_count (int): Number of observations included in the evaluation.
        objective_metrics (dict[str, Any]): Metrics that feed the composite objective score.
        objective_score (float): Composite score used to compare experiment results.
        notes (str | None): Optional researcher or operator notes.
        assumptions (list[str]): Documented assumptions captured for the experiment.
        parameter_overrides (dict[str, Any]): Concrete parameter overrides evaluated in the experiment.
        dataset_path (str | None): Filesystem path to the signal-evaluation dataset used for tuning or validation.
        search_metadata (dict[str, Any]): Metadata describing the search configuration that produced the record.
        validation_results (dict[str, Any]): Structured walk-forward or out-of-sample validation results.
        robustness_metrics (dict[str, Any]): Metrics that estimate stability across slices or regimes.
        comparison_summary (dict[str, Any]): Summary of performance relative to a baseline parameter pack.
        tuning_campaign (dict[str, Any]): Campaign metadata grouping the record with related tuning work.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, promoted, and audited without accidental mutation.
    """
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
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `ExperimentResult` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Dataclass representing TuningGroupPlan within the repository.
    
    Context:
        Used within the tuning layer that searches, validates, and governs parameter packs. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        group (str): Value supplied for group.
        description (str): Value supplied for description.
        search_strategy (str): Value supplied for search strategy.
        validation_mode (str): Value supplied for validation mode.
        parameter_keys (tuple[str, ...]): Value supplied for parameter keys.
        max_trials (int): Value supplied for max trials.
        overfit_risk (str): Value supplied for overfit risk.
        live_safe_only (bool): Boolean flag controlling whether live safe only is active.
        notes (str | None): Value supplied for notes.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
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
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `TuningGroupPlan` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Immutable record capturing the outcome of a promotion check for a candidate parameter pack.
    
    Context:
        Used within the `models` module. The class standardizes records that move through search, validation, shadow mode, governance, and promotion.
    
    Attributes:
        approved (bool): Whether the manual approval requirement has been satisfied.
        current_live_pack (str): Currently assigned live parameter pack at the time of the decision.
        candidate_pack (str): Candidate parameter pack under review.
        baseline_pack (str): Baseline parameter pack used as the comparison anchor.
        reason (str): Human-readable explanation for the decision or ledger event.
        diagnostics (dict[str, Any]): Supporting diagnostics that explain why the decision was reached.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, promoted, and audited without accidental mutation.
    """
    approved: bool
    current_live_pack: str
    candidate_pack: str
    baseline_pack: str
    reason: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `PromotionDecision` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Dataclass representing WalkForwardSplit within the repository.
    
    Context:
        Used within the tuning layer that searches, validates, and governs parameter packs. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        split_id (str): Value supplied for split id.
        split_type (str): Value supplied for split type.
        train_start (str | None): Value supplied for train start.
        train_end (str | None): Value supplied for train end.
        validation_start (str | None): Value supplied for validation start.
        validation_end (str | None): Value supplied for validation end.
        train_count (int): Count recorded for train.
        validation_count (int): Count recorded for validation.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    split_id: str
    split_type: str
    train_start: str | None
    train_end: str | None
    validation_start: str | None
    validation_end: str | None
    train_count: int
    validation_count: int

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `WalkForwardSplit` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Immutable record capturing which governance state a parameter pack currently occupies.
    
    Context:
        Used within the `models` module. The class standardizes records that move through search, validation, shadow mode, governance, and promotion.
    
    Attributes:
        pack_name (str | None): Parameter-pack name associated with the governance-state assignment.
        state (str): Governance state currently assigned to the parameter pack.
        assigned_at (str | None): Timestamp when the governance-state assignment was recorded.
        assigned_by (str | None): Reviewer or workflow that recorded the governance-state assignment.
        notes (str | None): Optional researcher or operator notes.
        source_experiment_id (str | None): Experiment identifier that produced the referenced pack state.
        source_validation_experiment_id (str | None): Validation experiment identifier associated with the referenced pack state.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, promoted, and audited without accidental mutation.
    """
    pack_name: str | None
    state: str
    assigned_at: str | None = None
    assigned_by: str | None = None
    notes: str | None = None
    source_experiment_id: str | None = None
    source_validation_experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `PackStateAssignment` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Immutable record of the manual approval state attached to a promotion or governance action.
    
    Context:
        Used within the `models` module. The class standardizes records that move through search, validation, shadow mode, governance, and promotion.
    
    Attributes:
        required (bool): Whether manual approval is required before the workflow can proceed.
        approved (bool): Whether the manual approval requirement has been satisfied.
        reviewer (str | None): Reviewer who approved or rejected the action, when applicable.
        timestamp (str | None): Timestamp recorded for the object.
        notes (str | None): Optional researcher or operator notes.
        approval_type (str): Type of approval workflow being recorded, such as promotion or rollback.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, promoted, and audited without accidental mutation.
    """
    required: bool
    approved: bool
    reviewer: str | None = None
    timestamp: str | None = None
    notes: str | None = None
    approval_type: str = "PROMOTION"

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `ManualApprovalRecord` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
    """
    Purpose:
        Immutable ledger event describing a state transition in the promotion workflow.
    
    Context:
        Used within the `models` module. The class standardizes records that move through search, validation, shadow mode, governance, and promotion.
    
    Attributes:
        timestamp (str): Timestamp recorded for the object.
        event_type (str): Type of promotion-ledger event being recorded.
        parameter_pack_name (str | None): Name of the parameter pack associated with the record.
        previous_state (str | None): Previous governance state before the ledger event was recorded.
        new_state (str | None): New governance state after the ledger event was recorded.
        reason (str): Human-readable explanation for the decision or ledger event.
        criteria_summary (dict[str, Any]): Summary of the quantitative checks used to justify the state transition.
        experiment_references (dict[str, Any]): References to experiments or validation runs associated with the event.
        human_approval (dict[str, Any]): Serialized manual-approval record associated with the event.
        metadata (dict[str, Any]): Supplemental metadata captured alongside the object.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, promoted, and audited without accidental mutation.
    """
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
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `PromotionLedgerEvent` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
