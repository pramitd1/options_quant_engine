"""
Module: promotion.py

Purpose:
    Implement promotion utilities for parameter search, validation, governance, or promotion workflows.

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
from typing import Any

import pandas as pd

from tuning.artifacts import append_jsonl_record
from tuning.models import ManualApprovalRecord, PackStateAssignment, PromotionDecision, PromotionLedgerEvent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_RESEARCH_DIR = PROJECT_ROOT / "research" / "parameter_tuning"
PROMOTION_STATE_PATH = TUNING_RESEARCH_DIR / "promotion_state.json"
PROMOTION_LEDGER_PATH = TUNING_RESEARCH_DIR / "promotion_ledger.jsonl"

PACK_STATES = ("baseline", "candidate", "shadow", "live")

DEFAULT_PROMOTION_CRITERIA = {
    "minimum_sample_count": 25,
    "minimum_improvement": 0.01,
    "maximum_stability_gap": 0.08,
    "minimum_frequency_ratio": 0.60,
    "minimum_out_of_sample_improvement": 0.0,
    "minimum_robustness_score": 0.35,
    "maximum_drawdown_proxy_ratio": 1.25,
    "require_manual_approval": False,
    "important_regime_max_collapse": -0.08,
}


def _utc_now_iso() -> str:
    """
    Purpose:
        Process utc now iso for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return pd.Timestamp.utcnow().isoformat()


def _default_assignment(pack_name: str | None, state: str) -> dict[str, Any]:
    """
    Purpose:
        Process default assignment for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        pack_name (str | None): Human-readable name for pack.
        state (str): Input associated with state.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return PackStateAssignment(
        pack_name=pack_name,
        state=state,
        assigned_at=None,
        assigned_by=None,
        notes=None,
        source_experiment_id=None,
        source_validation_experiment_id=None,
    ).to_dict()


def _default_state() -> dict[str, Any]:
    """
    Purpose:
        Process default state for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return {
        "baseline": "baseline_v1",
        "candidate": "candidate_v1",
        "shadow": None,
        "live": "baseline_v1",
        "previous_live": None,
        "baseline_assignment": _default_assignment("baseline_v1", "baseline"),
        "candidate_assignment": _default_assignment("candidate_v1", "candidate"),
        "shadow_assignment": _default_assignment(None, "shadow"),
        "live_assignment": _default_assignment("baseline_v1", "live"),
        "manual_approvals": {},
        "criteria": dict(DEFAULT_PROMOTION_CRITERIA),
    }


def _normalize_state(state: dict | None) -> dict[str, Any]:
    """
    Purpose:
        Normalize state into the repository-standard form.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        state (dict | None): Input associated with state.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    normalized = _default_state()
    incoming = dict(state or {})

    for key in ("baseline", "candidate", "shadow", "live", "previous_live"):
        if key in incoming:
            normalized[key] = incoming.get(key)

    for key in ("baseline_assignment", "candidate_assignment", "shadow_assignment", "live_assignment"):
        payload = dict(incoming.get(key) or normalized[key])
        normalized[key] = payload

    if normalized["baseline_assignment"].get("pack_name") != normalized["baseline"]:
        normalized["baseline_assignment"]["pack_name"] = normalized["baseline"]
    if normalized["candidate_assignment"].get("pack_name") != normalized["candidate"]:
        normalized["candidate_assignment"]["pack_name"] = normalized["candidate"]
    if normalized["shadow_assignment"].get("pack_name") != normalized["shadow"]:
        normalized["shadow_assignment"]["pack_name"] = normalized["shadow"]
    if normalized["live_assignment"].get("pack_name") != normalized["live"]:
        normalized["live_assignment"]["pack_name"] = normalized["live"]

    normalized["manual_approvals"] = dict(incoming.get("manual_approvals") or {})
    normalized["criteria"] = dict(DEFAULT_PROMOTION_CRITERIA)
    normalized["criteria"].update(dict(incoming.get("criteria") or {}))
    return normalized


def load_promotion_state(path: str | Path = PROMOTION_STATE_PATH) -> dict:
    """
    Purpose:
        Process load promotion state for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        dict: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state_path = Path(path)
    if not state_path.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = _default_state()
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        return state
    return _normalize_state(json.loads(state_path.read_text()))


def write_promotion_state(state: dict, path: str | Path = PROMOTION_STATE_PATH) -> Path:
    """
    Purpose:
        Write promotion state to the appropriate output artifact.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        state (dict): Input associated with state.
        path (str | Path): Input associated with path.
    
    Returns:
        None: The function communicates through side effects such as terminal output or persisted artifacts.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_state(state)
    state_path.write_text(json.dumps(normalized, indent=2, sort_keys=True))
    return state_path


def append_promotion_event(event: PromotionLedgerEvent | dict[str, Any], path: str | Path = PROMOTION_LEDGER_PATH) -> Path:
    """
    Purpose:
        Process append promotion event for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        event (PromotionLedgerEvent | dict[str, Any]): Input associated with event.
        path (str | Path): Input associated with path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    event_payload = event.to_dict() if hasattr(event, "to_dict") else dict(event or {})
    return append_jsonl_record(event_payload, path)


def _state_key(state_name: str) -> str:
    """
    Purpose:
        Process state key for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        state_name (str): Human-readable name for state.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    normalized = str(state_name).strip().lower()
    if normalized not in PACK_STATES:
        raise ValueError(f"Unsupported pack state: {state_name}")
    return f"{normalized}_assignment"


def get_active_live_pack(path: str | Path = PROMOTION_STATE_PATH) -> str:
    """
    Purpose:
        Return active live pack for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return str(load_promotion_state(path).get("live") or "baseline_v1")


def get_active_shadow_pack(path: str | Path = PROMOTION_STATE_PATH) -> str | None:
    """
    Purpose:
        Return active shadow pack for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        str | None: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    shadow = load_promotion_state(path).get("shadow")
    return str(shadow) if shadow else None


def get_promotion_runtime_context(path: str | Path = PROMOTION_STATE_PATH) -> dict[str, Any]:
    """
    Purpose:
        Return promotion runtime context for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state = load_promotion_state(path)
    return {
        "baseline_pack": state.get("baseline"),
        "candidate_pack": state.get("candidate"),
        "shadow_pack": state.get("shadow"),
        "live_pack": state.get("live"),
        "previous_live_pack": state.get("previous_live"),
        "criteria": dict(state.get("criteria") or {}),
        "manual_approvals": dict(state.get("manual_approvals") or {}),
    }


def update_pack_state(
    *,
    state_name: str,
    pack_name: str | None,
    reason: str,
    assigned_by: str | None = None,
    notes: str | None = None,
    source_experiment_id: str | None = None,
    source_validation_experiment_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
) -> dict:
    """
    Purpose:
        Update pack state using the supplied inputs.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        state_name (str): Human-readable name for state.
        pack_name (str | None): Human-readable name for pack.
        reason (str): Input associated with reason.
        assigned_by (str | None): Input associated with assigned by.
        notes (str | None): Input associated with notes.
        source_experiment_id (str | None): Input associated with source experiment identifier.
        source_validation_experiment_id (str | None): Input associated with source validation experiment identifier.
        metadata (dict[str, Any] | None): Input associated with metadata.
        path (str | Path): Input associated with path.
        ledger_path (str | Path): Input associated with ledger path.
    
    Returns:
        dict: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state = load_promotion_state(path)
    state_name = str(state_name).strip().lower()
    assignment_key = _state_key(state_name)
    previous_assignment = dict(state.get(assignment_key) or {})
    previous_pack = state.get(state_name)

    assignment = PackStateAssignment(
        pack_name=pack_name,
        state=state_name,
        assigned_at=_utc_now_iso(),
        assigned_by=assigned_by,
        notes=notes,
        source_experiment_id=source_experiment_id,
        source_validation_experiment_id=source_validation_experiment_id,
    ).to_dict()

    if state_name == "live":
        state["previous_live"] = previous_pack
    state[state_name] = pack_name
    state[assignment_key] = assignment
    write_promotion_state(state, path)

    append_promotion_event(
        PromotionLedgerEvent(
            timestamp=_utc_now_iso(),
            event_type=f"{state_name}_state_updated",
            parameter_pack_name=pack_name,
            previous_state=previous_assignment.get("state"),
            new_state=state_name,
            reason=reason,
            experiment_references={
                "source_experiment_id": source_experiment_id,
                "source_validation_experiment_id": source_validation_experiment_id,
            },
            metadata={
                "previous_pack_name": previous_pack,
                **dict(metadata or {}),
            },
        ),
        path=ledger_path,
    )
    return load_promotion_state(path)


def record_manual_approval(
    *,
    pack_name: str,
    approved: bool,
    reviewer: str,
    notes: str | None = None,
    approval_type: str = "PROMOTION",
    required: bool = True,
    path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
) -> dict:
    """
    Purpose:
        Process record manual approval for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        pack_name (str): Human-readable name for pack.
        approved (bool): Boolean flag associated with approved.
        reviewer (str): Input associated with reviewer.
        notes (str | None): Input associated with notes.
        approval_type (str): Input associated with approval type.
        required (bool): Boolean flag associated with required.
        path (str | Path): Input associated with path.
        ledger_path (str | Path): Input associated with ledger path.
    
    Returns:
        dict: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state = load_promotion_state(path)
    record = ManualApprovalRecord(
        required=required,
        approved=approved,
        reviewer=reviewer,
        timestamp=_utc_now_iso(),
        notes=notes,
        approval_type=approval_type,
    ).to_dict()
    state.setdefault("manual_approvals", {})
    state["manual_approvals"][pack_name] = record
    write_promotion_state(state, path)

    append_promotion_event(
        PromotionLedgerEvent(
            timestamp=record["timestamp"],
            event_type="manual_approval_recorded",
            parameter_pack_name=pack_name,
            previous_state=None,
            new_state=None,
            reason="manual_approval_updated",
            human_approval=record,
        ),
        path=ledger_path,
    )
    return state


def get_manual_approval_record(
    pack_name: str,
    *,
    path: str | Path = PROMOTION_STATE_PATH,
) -> dict[str, Any]:
    """
    Purpose:
        Return manual approval record for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        pack_name (str): Human-readable name for pack.
        path (str | Path): Input associated with path.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state = load_promotion_state(path)
    return dict((state.get("manual_approvals") or {}).get(pack_name, {}))


def evaluate_promotion(
    *,
    baseline_result: dict,
    candidate_result: dict,
    minimum_sample_count: int = DEFAULT_PROMOTION_CRITERIA["minimum_sample_count"],
    minimum_improvement: float = DEFAULT_PROMOTION_CRITERIA["minimum_improvement"],
    maximum_stability_gap: float = DEFAULT_PROMOTION_CRITERIA["maximum_stability_gap"],
    minimum_frequency_ratio: float = DEFAULT_PROMOTION_CRITERIA["minimum_frequency_ratio"],
    minimum_out_of_sample_improvement: float = DEFAULT_PROMOTION_CRITERIA["minimum_out_of_sample_improvement"],
    minimum_robustness_score: float = DEFAULT_PROMOTION_CRITERIA["minimum_robustness_score"],
    maximum_drawdown_proxy_ratio: float = DEFAULT_PROMOTION_CRITERIA["maximum_drawdown_proxy_ratio"],
    important_regime_max_collapse: float = DEFAULT_PROMOTION_CRITERIA["important_regime_max_collapse"],
    require_manual_approval: bool = DEFAULT_PROMOTION_CRITERIA["require_manual_approval"],
    manual_approval: dict[str, Any] | None = None,
) -> PromotionDecision:
    """
    Purpose:
        Process evaluate promotion for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        baseline_result (dict): Input associated with baseline result.
        candidate_result (dict): Input associated with candidate result.
        minimum_sample_count (int): Input associated with minimum sample count.
        minimum_improvement (float): Input associated with minimum improvement.
        maximum_stability_gap (float): Input associated with maximum stability gap.
        minimum_frequency_ratio (float): Input associated with minimum frequency ratio.
        minimum_out_of_sample_improvement (float): Input associated with minimum out of sample improvement.
        minimum_robustness_score (float): Score value for minimum robustness.
        maximum_drawdown_proxy_ratio (float): Input associated with maximum drawdown proxy ratio.
        important_regime_max_collapse (float): Input associated with important regime max collapse.
        require_manual_approval (bool): Boolean flag associated with require_manual_approval.
        manual_approval (dict[str, Any] | None): Input associated with manual approval.
    
    Returns:
        PromotionDecision: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    baseline_metrics = dict(baseline_result.get("objective_metrics", {}))
    candidate_metrics = dict(candidate_result.get("objective_metrics", {}))
    baseline_safeguards = dict(baseline_metrics.get("safeguards", {}))
    candidate_safeguards = dict(candidate_metrics.get("safeguards", {}))
    baseline_core = dict(baseline_metrics.get("metrics", {}))
    candidate_core = dict(candidate_metrics.get("metrics", {}))

    baseline_score = float(baseline_result.get("objective_score", 0.0))
    candidate_score = float(candidate_result.get("objective_score", 0.0))
    baseline_freq = float(baseline_core.get("signal_frequency", 0.0))
    candidate_freq = float(candidate_core.get("signal_frequency", 0.0))
    candidate_sample_count = int(candidate_result.get("sample_count", 0))
    stability_gap = float(candidate_safeguards.get("stability_gap", 1.0))
    frequency_ratio = 1.0 if baseline_freq <= 0 else candidate_freq / baseline_freq
    baseline_validation = dict(baseline_result.get("validation_results", {}))
    candidate_validation = dict(candidate_result.get("validation_results", {}))
    baseline_robustness = dict(baseline_result.get("robustness_metrics", {}))
    candidate_robustness = dict(candidate_result.get("robustness_metrics", {}))
    baseline_oos_score = float(baseline_validation.get("aggregate_out_of_sample_score", baseline_score))
    candidate_oos_score = float(candidate_validation.get("aggregate_out_of_sample_score", candidate_score))
    candidate_robustness_score = float(candidate_robustness.get("robustness_score", 1.0))
    baseline_drawdown = float(baseline_validation.get("aggregate_out_of_sample_metrics", {}).get("drawdown_proxy", baseline_core.get("drawdown_proxy", 0.0)) or 0.0)
    candidate_drawdown = float(candidate_validation.get("aggregate_out_of_sample_metrics", {}).get("drawdown_proxy", candidate_core.get("drawdown_proxy", 0.0)) or 0.0)
    drawdown_ratio = 1.0 if baseline_drawdown <= 0 else candidate_drawdown / baseline_drawdown
    validation_hooks_present = bool(candidate_validation)

    comparison_summary = dict(candidate_result.get("comparison_summary") or {})
    worst_regime_delta = 0.0
    for regime_rows in comparison_summary.get("regime_comparison", {}).values():
        for row in regime_rows:
            worst_regime_delta = min(worst_regime_delta, float(row.get("direction_hit_rate_delta", 0.0)))

    approval_payload = dict(manual_approval or {})
    approval_required = bool(require_manual_approval)
    approval_granted = bool(approval_payload.get("approved", False)) if approval_required else True

    approved = (
        candidate_sample_count >= minimum_sample_count
        and candidate_score >= baseline_score + minimum_improvement
        and stability_gap <= maximum_stability_gap
        and frequency_ratio >= minimum_frequency_ratio
        and bool(candidate_safeguards.get("minimum_sample_ok", False))
        and (not validation_hooks_present or candidate_oos_score >= baseline_oos_score + minimum_out_of_sample_improvement)
        and (not validation_hooks_present or candidate_robustness_score >= minimum_robustness_score)
        and (not validation_hooks_present or drawdown_ratio <= maximum_drawdown_proxy_ratio)
        and (not comparison_summary or worst_regime_delta >= important_regime_max_collapse)
        and approval_granted
    )

    reason = "candidate_rejected"
    if approved:
        reason = "candidate_meets_promotion_thresholds"
    elif candidate_sample_count < minimum_sample_count:
        reason = "candidate_sample_count_too_small"
    elif candidate_score < baseline_score + minimum_improvement:
        reason = "candidate_improvement_insufficient"
    elif stability_gap > maximum_stability_gap:
        reason = "candidate_stability_gap_too_large"
    elif frequency_ratio < minimum_frequency_ratio:
        reason = "candidate_signal_frequency_collapsed"
    elif validation_hooks_present and candidate_oos_score < baseline_oos_score + minimum_out_of_sample_improvement:
        reason = "candidate_out_of_sample_improvement_insufficient"
    elif validation_hooks_present and candidate_robustness_score < minimum_robustness_score:
        reason = "candidate_robustness_too_low"
    elif validation_hooks_present and drawdown_ratio > maximum_drawdown_proxy_ratio:
        reason = "candidate_drawdown_proxy_worse_than_allowed"
    elif comparison_summary and worst_regime_delta < important_regime_max_collapse:
        reason = "candidate_regime_collapse_exceeds_limit"
    elif approval_required and not approval_granted:
        reason = "manual_approval_required"

    return PromotionDecision(
        approved=approved,
        current_live_pack=str(load_promotion_state().get("live", "baseline_v1")),
        candidate_pack=str(candidate_result.get("parameter_pack_name", "candidate_v1")),
        baseline_pack=str(baseline_result.get("parameter_pack_name", "baseline_v1")),
        reason=reason,
        diagnostics={
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_improvement": round(candidate_score - baseline_score, 6),
            "candidate_sample_count": candidate_sample_count,
            "stability_gap": round(stability_gap, 6),
            "frequency_ratio": round(frequency_ratio, 6),
            "validation_hooks_present": validation_hooks_present,
            "baseline_out_of_sample_score": round(baseline_oos_score, 6),
            "candidate_out_of_sample_score": round(candidate_oos_score, 6),
            "out_of_sample_score_delta": round(candidate_oos_score - baseline_oos_score, 6),
            "baseline_robustness_score": round(float(baseline_robustness.get("robustness_score", 0.0)), 6),
            "candidate_robustness_score": round(candidate_robustness_score, 6),
            "baseline_drawdown_proxy": round(baseline_drawdown, 6),
            "candidate_drawdown_proxy": round(candidate_drawdown, 6),
            "drawdown_proxy_ratio": round(drawdown_ratio, 6),
            "worst_regime_direction_hit_rate_delta": round(worst_regime_delta, 6),
            "manual_approval_required": approval_required,
            "manual_approval": approval_payload,
            "criteria": {
                "minimum_sample_count": minimum_sample_count,
                "minimum_improvement": minimum_improvement,
                "maximum_stability_gap": maximum_stability_gap,
                "minimum_frequency_ratio": minimum_frequency_ratio,
                "minimum_out_of_sample_improvement": minimum_out_of_sample_improvement,
                "minimum_robustness_score": minimum_robustness_score,
                "maximum_drawdown_proxy_ratio": maximum_drawdown_proxy_ratio,
                "important_regime_max_collapse": important_regime_max_collapse,
                "require_manual_approval": approval_required,
            },
            "baseline_safeguards": baseline_safeguards,
            "candidate_safeguards": candidate_safeguards,
        },
    )


def promote_candidate(
    candidate_pack_name: str,
    *,
    baseline_pack_name: str | None = None,
    approved_by: str | None = None,
    reason: str = "candidate_promoted_to_live",
    source_experiment_id: str | None = None,
    source_validation_experiment_id: str | None = None,
    require_manual_approval: bool = True,
    expected_improvement_summary: dict[str, Any] | None = None,
    path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
) -> Path:
    """
    Purpose:
        Process promote candidate for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        candidate_pack_name (str): Human-readable name for candidate pack.
        baseline_pack_name (str | None): Human-readable name for baseline pack.
        approved_by (str | None): Input associated with approved by.
        reason (str): Input associated with reason.
        source_experiment_id (str | None): Input associated with source experiment identifier.
        source_validation_experiment_id (str | None): Input associated with source validation experiment identifier.
        require_manual_approval (bool): Boolean flag associated with require_manual_approval.
        expected_improvement_summary (dict[str, Any] | None): Input associated with expected improvement summary.
        path (str | Path): Input associated with path.
        ledger_path (str | Path): Input associated with ledger path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not approved_by:
        raise ValueError("Promotion requires an explicit approver via approved_by")

    state = load_promotion_state(path)
    baseline_pack_name = baseline_pack_name or str(state.get("live") or "baseline_v1")
    if state.get("candidate") != candidate_pack_name:
        raise ValueError(
            f"Candidate promotion requires the active candidate pack. "
            f"Current state candidate={state.get('candidate')}, requested={candidate_pack_name}"
        )

    approval_record = dict((state.get("manual_approvals") or {}).get(candidate_pack_name, {}))
    if require_manual_approval and not approval_record.get("approved", False):
        raise PermissionError(
            f"Candidate pack '{candidate_pack_name}' does not have recorded manual approval"
        )

    previous_live = state.get("live")
    state = update_pack_state(
        state_name="baseline",
        pack_name=baseline_pack_name,
        reason="baseline_reference_updated",
        assigned_by=approved_by,
        path=path,
        ledger_path=ledger_path,
    )
    state = update_pack_state(
        state_name="live",
        pack_name=candidate_pack_name,
        reason=reason,
        assigned_by=approved_by,
        source_experiment_id=source_experiment_id,
        source_validation_experiment_id=source_validation_experiment_id,
        metadata={
            "previous_live_pack": previous_live,
            "expected_improvement_summary": dict(expected_improvement_summary or {}),
        },
        path=path,
        ledger_path=ledger_path,
    )
    if state.get("shadow") == candidate_pack_name:
        state = update_pack_state(
            state_name="shadow",
            pack_name=None,
            reason="promoted_shadow_pack_cleared",
            assigned_by=approved_by,
            path=path,
            ledger_path=ledger_path,
        )
    append_promotion_event(
        PromotionLedgerEvent(
            timestamp=_utc_now_iso(),
            event_type="candidate_promoted_to_live",
            parameter_pack_name=candidate_pack_name,
            previous_state="candidate",
            new_state="live",
            reason=reason,
            experiment_references={
                "source_experiment_id": source_experiment_id,
                "source_validation_experiment_id": source_validation_experiment_id,
            },
            human_approval=approval_record,
            metadata={
                "approved_by": approved_by,
                "previous_live_pack": previous_live,
                "baseline_pack_name": baseline_pack_name,
                "expected_improvement_summary": dict(expected_improvement_summary or {}),
            },
        ),
        path=ledger_path,
    )
    return write_promotion_state(state, path)


def move_candidate_to_shadow(
    candidate_pack_name: str,
    *,
    assigned_by: str | None = None,
    reason: str = "candidate_moved_to_shadow",
    source_experiment_id: str | None = None,
    source_validation_experiment_id: str | None = None,
    path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
) -> dict:
    """
    Purpose:
        Process move candidate to shadow for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        candidate_pack_name (str): Human-readable name for candidate pack.
        assigned_by (str | None): Input associated with assigned by.
        reason (str): Input associated with reason.
        source_experiment_id (str | None): Input associated with source experiment identifier.
        source_validation_experiment_id (str | None): Input associated with source validation experiment identifier.
        path (str | Path): Input associated with path.
        ledger_path (str | Path): Input associated with ledger path.
    
    Returns:
        dict: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return update_pack_state(
        state_name="shadow",
        pack_name=candidate_pack_name,
        reason=reason,
        assigned_by=assigned_by,
        source_experiment_id=source_experiment_id,
        source_validation_experiment_id=source_validation_experiment_id,
        path=path,
        ledger_path=ledger_path,
    )


def rollback_live_pack(
    *,
    rollback_to_pack: str | None = None,
    reason: str = "rollback_to_previous_live",
    reviewer: str | None = None,
    path: str | Path = PROMOTION_STATE_PATH,
    ledger_path: str | Path = PROMOTION_LEDGER_PATH,
) -> dict:
    """
    Purpose:
        Process rollback live pack for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        rollback_to_pack (str | None): Input associated with rollback to pack.
        reason (str): Input associated with reason.
        reviewer (str | None): Input associated with reviewer.
        path (str | Path): Input associated with path.
        ledger_path (str | Path): Input associated with ledger path.
    
    Returns:
        dict: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    state = load_promotion_state(path)
    target_pack = rollback_to_pack or state.get("previous_live") or state.get("baseline")
    current_live = state.get("live")
    state = update_pack_state(
        state_name="live",
        pack_name=target_pack,
        reason=reason,
        assigned_by=reviewer,
        metadata={"rolled_back_from": current_live},
        path=path,
        ledger_path=ledger_path,
    )
    append_promotion_event(
        PromotionLedgerEvent(
            timestamp=_utc_now_iso(),
            event_type="rollback_executed",
            parameter_pack_name=target_pack,
            previous_state="live",
            new_state="live",
            reason=reason,
            metadata={"rolled_back_from": current_live},
        ),
        path=ledger_path,
    )
    return state
