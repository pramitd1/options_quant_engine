"""Lightweight schema contracts for generated signal-evaluation artifacts.

The contracts in this module are intentionally small. They guard the fields
that downstream ops workflows depend on, without trying to turn every report
into a heavyweight serialization model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ArtifactSchemaValidationError(ValueError):
    """Raised when a generated artifact violates its schema contract."""


@dataclass(frozen=True)
class FieldContract:
    path: str
    expected_types: tuple[str, ...] = ("any",)
    allow_none: bool = False


@dataclass(frozen=True)
class ArtifactContract:
    name: str
    required_fields: tuple[FieldContract, ...]


MISSING = object()


def _field(path: str, *expected_types: str, allow_none: bool = False) -> FieldContract:
    return FieldContract(path=path, expected_types=expected_types or ("any",), allow_none=allow_none)


ARTIFACT_CONTRACTS: dict[str, ArtifactContract] = {
    "threshold_runtime_activation_marker": ArtifactContract(
        name="threshold_runtime_activation_marker",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("candidate_pack_name", "str"),
            _field("activated_at", "str"),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
        ),
    ),
    "threshold_signal_rollout_monitor": ArtifactContract(
        name="threshold_signal_rollout_monitor",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("rollout_status", "str"),
            _field("rollout_reasons", "list"),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("baseline_pack_name", "str"),
            _field("candidate_pack_name", "str"),
            _field("post_adoption_traceability", "dict"),
            _field("post_adoption_traceability.traceability_status", "str"),
            _field("post_adoption_traceability.post_adoption_signal_count", "int"),
            _field("post_adoption_traceability.candidate_pack_signal_count", "int"),
            _field("post_adoption_traceability.non_candidate_pack_signal_count", "int"),
            _field("post_adoption_traceability.missing_parameter_pack_count", "int"),
            _field("candidate_label_readiness", "dict"),
            _field("candidate_label_readiness.label_count_60m", "int"),
            _field("execution_side_effects", "dict"),
            _field("execution_side_effects.execution_side_effect_check_passed", "bool"),
            _field("execution_side_effects.orders_submitted", "bool"),
        ),
    ),
    "threshold_adoption_history_dashboard": ArtifactContract(
        name="threshold_adoption_history_dashboard",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("run_count", "int"),
            _field("lookback_runs", "int"),
            _field("trend_assessment", "str"),
            _field("latest", "dict"),
            _field("status_counts", "dict"),
            _field("runtime_signal_status_counts", "dict"),
            _field("rollout_status_counts", "dict"),
            _field("lookback_summary", "dict"),
            _field("operator_message", "str"),
        ),
    ),
    "threshold_post_activation_verification": ArtifactContract(
        name="threshold_post_activation_verification",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("verification_status", "str"),
            _field("verification_reasons", "list"),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("candidate_pack_name", "str"),
            _field("active_parameter_pack", "dict"),
            _field("active_pack_matches_marker", "bool"),
            _field("checked_conditions", "dict"),
            _field("checked_conditions.active_pack_matches_marker", "bool"),
            _field("checked_conditions.threshold_matches", "bool"),
            _field("checked_conditions.non_candidate_pack_signal_count", "int"),
            _field("checked_conditions.missing_parameter_pack_count", "int"),
            _field("checked_conditions.candidate_label_count_60m", "int"),
            _field("checked_conditions.total_nonempty_side_effect_fields", "int"),
            _field("rollout_monitor", "dict"),
            _field("adoption_history", "dict"),
        ),
    ),
    "signal_quality_model_audit": ArtifactContract(
        name="signal_quality_model_audit",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("row_count", "int"),
            _field("quality_labeled_row_count", "int"),
            _field("probability_field", "str"),
            _field("label_quality", "dict"),
            _field("calibration_summary", "dict"),
            _field("calibration_summary.label_count", "int"),
            _field("calibration_summary.calibration_status", "str"),
            _field("calibration_bins", "list"),
            _field("regime_calibration", "list"),
            _field("feature_stability", "list"),
            _field("ranking_feature_audit", "list"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "probability_calibration_experiment": ArtifactContract(
        name="probability_calibration_experiment",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("row_count", "int"),
            _field("quality_labeled_row_count", "int"),
            _field("probability_field", "str"),
            _field("label_field", "str"),
            _field("train_count", "int"),
            _field("holdout_count", "int"),
            _field("methods_tested", "list"),
            _field("calibration_status", "str"),
            _field("selected_calibrator", "str"),
            _field("selection", "dict"),
            _field("selection.candidate_ready_for_review", "bool"),
            _field("holdout_metrics", "dict"),
            _field("calibrator_comparison", "list"),
            _field("calibration_curve", "list"),
            _field("candidate_calibrator", "dict"),
            _field("candidate_calibrator.research_only", "bool"),
            _field("candidate_calibrator.approval_required_for_runtime_use", "bool"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_calibration_experiment": ArtifactContract(
        name="segmented_probability_calibration_experiment",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("row_count", "int"),
            _field("quality_labeled_row_count", "int"),
            _field("probability_field", "str"),
            _field("label_field", "str"),
            _field("train_count", "int"),
            _field("holdout_count", "int"),
            _field("segment_fields", "list"),
            _field("recency_windows", "list"),
            _field("methods_tested", "list"),
            _field("calibration_status", "str"),
            _field("selection_summary", "dict"),
            _field("selection_summary.review_ready_candidate_count", "int"),
            _field("selection_summary.evaluated_regime_segment_count", "int"),
            _field("selection_summary.evaluated_recency_window_count", "int"),
            _field("recency_window_results", "list"),
            _field("segment_results", "list"),
            _field("candidate_bundle", "dict"),
            _field("candidate_bundle.research_only", "bool"),
            _field("candidate_bundle.approval_required_for_runtime_use", "bool"),
            _field("candidate_bundle.candidate_count", "int"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_forward_shadow": ArtifactContract(
        name="segmented_probability_forward_shadow",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("candidate_bundle_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("row_count", "int"),
            _field("quality_labeled_row_count", "int"),
            _field("probability_field", "str"),
            _field("label_field", "str"),
            _field("candidate_count", "int"),
            _field("validation_window", "dict"),
            _field("validation_window.validation_mode_used", "str"),
            _field("validation_window.strict_forward_row_count", "int"),
            _field("validation_window.holdout_replay_row_count", "int"),
            _field("shadow_validation_status", "str"),
            _field("selection_summary", "dict"),
            _field("selection_summary.evaluated_routing_policy_count", "int"),
            _field("routing_policy_results", "list"),
            _field("candidate_route_results", "list"),
            _field("calibration_curve", "list"),
            _field("route_decision_count", "int"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_ev_shadow_evaluation": ArtifactContract(
        name="segmented_probability_ev_shadow_evaluation",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("candidate_bundle_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("row_count", "int"),
            _field("quality_labeled_row_count", "int"),
            _field("probability_field", "str"),
            _field("label_field", "str"),
            _field("return_field", "str"),
            _field("candidate_count", "int"),
            _field("validation_window", "dict"),
            _field("validation_window.validation_mode_used", "str"),
            _field("validation_window.strict_forward_row_count", "int"),
            _field("validation_window.holdout_replay_row_count", "int"),
            _field("top_fraction", "number"),
            _field("ev_shadow_status", "str"),
            _field("selection_summary", "dict"),
            _field("selection_summary.evaluated_routing_policy_count", "int"),
            _field("policy_results", "list"),
            _field("candidate_route_results", "list"),
            _field("regime_payoff_results", "list"),
            _field("route_decision_count", "int"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_ev_rejection_attribution": ArtifactContract(
        name="segmented_probability_ev_rejection_attribution",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("ev_shadow_report_path", "str", allow_none=True),
            _field("ev_shadow_routes_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("ev_shadow_status", "str"),
            _field("attribution_status", "str"),
            _field("attribution_reasons", "list"),
            _field("analysis_policy", "str"),
            _field("top_fraction", "number"),
            _field("return_field", "str"),
            _field("route_decision_count", "int"),
            _field("analysis_route_decision_count", "int"),
            _field("rejection_summary", "dict"),
            _field("rejection_summary.raw_top_sample_count", "int"),
            _field("rejection_summary.shadow_top_sample_count", "int"),
            _field("rejection_summary.top_bucket_risk_adjusted_return_delta_bps", "number", allow_none=True),
            _field("rejection_summary.top_bucket_hit_rate_delta", "number", allow_none=True),
            _field("candidate_attribution", "list"),
            _field("shadow_only_candidate_attribution", "list"),
            _field("negative_route_candidates", "list"),
            _field("regime_attribution", "list"),
            _field("policy_comparison", "list"),
            _field("routing_diagnostics", "dict"),
            _field("routing_diagnostics.likely_failure_mode", "str"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_guarded_ev_experiment": ArtifactContract(
        name="segmented_probability_guarded_ev_experiment",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str"),
            _field("candidate_bundle_path", "str"),
            _field("ev_shadow_report_path", "str", allow_none=True),
            _field("ev_rejection_attribution_path", "str", allow_none=True),
            _field("ev_shadow_routes_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("guarded_ev_status", "str"),
            _field("analysis_policy", "str"),
            _field("simulation_mode", "str"),
            _field("ev_rejection_attribution_status", "str"),
            _field("ev_rejection_is_actionable", "bool"),
            _field("probability_field", "str"),
            _field("label_field", "str"),
            _field("return_field", "str"),
            _field("top_fraction", "number"),
            _field("raw_rank_ceiling_multiplier", "number"),
            _field("quarantined_candidate_keys", "list"),
            _field("quarantined_candidate_count", "int"),
            _field("candidate_count_before_quarantine", "int"),
            _field("candidate_count_after_quarantine", "int"),
            _field("route_decision_count", "int"),
            _field("validation_window", "dict"),
            _field("variant_results", "list"),
            _field("candidate_exposure", "list"),
            _field("selection_summary", "dict"),
            _field("selection_summary.evaluated_variant_count", "int"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_guarded_candidate_bundle": ArtifactContract(
        name="segmented_probability_guarded_candidate_bundle",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("source_candidate_bundle_path", "str", allow_none=True),
            _field("guarded_ev_experiment_path", "str", allow_none=True),
            _field("guarded_candidate_bundle_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("guarded_candidate_bundle_status", "str"),
            _field("guarded_candidate_bundle_reasons", "list"),
            _field("guarded_ev_status", "str"),
            _field("recommended_guarded_variant", "str", allow_none=True),
            _field("rank_preservation_policy", "dict"),
            _field("rank_preservation_policy.governance_only", "bool"),
            _field("rank_preservation_policy.runtime_behavior_changed", "bool"),
            _field("rank_preservation_policy.requires_guard_aware_shadow_evaluation", "bool"),
            _field("source_candidate_count", "int"),
            _field("kept_candidate_count", "int"),
            _field("quarantined_candidate_count", "int"),
            _field("quarantined_candidate_keys", "list"),
            _field("missing_quarantine_keys", "list"),
            _field("required_next_validations", "list"),
            _field("guarded_candidate_bundle", "dict"),
            _field("guarded_candidate_bundle.research_only", "bool"),
            _field("guarded_candidate_bundle.approval_required_for_runtime_use", "bool"),
            _field("guarded_candidate_bundle.runtime_config_changed", "bool"),
            _field("guarded_candidate_bundle.parameter_pack_file_changed", "bool"),
            _field("guarded_candidate_bundle.execution_behavior_changed", "bool"),
            _field("guarded_candidate_bundle.candidate_count", "int"),
            _field("candidate_rows", "list"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_guarded_shadow_validation": ArtifactContract(
        name="segmented_probability_guarded_shadow_validation",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("candidate_bundle_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("row_count", "int"),
            _field("quality_labeled_row_count", "int"),
            _field("probability_field", "str"),
            _field("label_field", "str"),
            _field("return_field", "str"),
            _field("candidate_count", "int"),
            _field("quarantined_candidate_keys", "list"),
            _field("validation_window", "dict"),
            _field("validation_window.validation_mode_used", "str"),
            _field("validation_window.strict_forward_row_count", "int"),
            _field("validation_window.holdout_replay_row_count", "int"),
            _field("top_fraction", "number"),
            _field("raw_rank_ceiling_multiplier", "number"),
            _field("rank_preservation_policy", "dict"),
            _field("rank_preservation_policy_present", "bool"),
            _field("guarded_bundle_side_effect_flags_clean", "bool"),
            _field("guarded_bundle_research_only", "bool"),
            _field("guarded_bundle_approval_required_for_runtime_use", "bool"),
            _field("guarded_shadow_status", "str"),
            _field("selection_summary", "dict"),
            _field("selection_summary.evaluated_routing_policy_count", "int"),
            _field("policy_results", "list"),
            _field("candidate_route_results", "list"),
            _field("calibration_curve", "list"),
            _field("route_decision_count", "int"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_forward_shadow_accumulation": ArtifactContract(
        name="segmented_probability_forward_shadow_accumulation",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("run_count", "int"),
            _field("lookback_runs", "int"),
            _field("trend_assessment", "str"),
            _field("latest", "dict"),
            _field("status_counts", "dict"),
            _field("validation_mode_counts", "dict"),
            _field("shadow_status_counts", "dict"),
            _field("lookback_summary", "dict"),
            _field("lookback_summary.true_forward_pass_runs", "int"),
            _field("lookback_summary.holdout_replay_pending_runs", "int"),
            _field("lookback_summary.latest_strict_forward_row_count", "int"),
            _field("operator_message", "str"),
        ),
    ),
    "segmented_probability_forward_shadow_readiness": ArtifactContract(
        name="segmented_probability_forward_shadow_readiness",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("accumulation_dashboard_path", "str", allow_none=True),
            _field("forward_shadow_report_path", "str", allow_none=True),
            _field("candidate_staleness_path", "str", allow_none=True),
            _field("ev_shadow_path", "str", allow_none=True),
            _field("guarded_shadow_path", "str", allow_none=True),
            _field("history_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("readiness_status", "str"),
            _field("readiness_reasons", "list"),
            _field("checked_conditions", "dict"),
            _field("checked_conditions.validation_mode_is_true_forward", "bool"),
            _field("checked_conditions.sufficient_forward_sample", "bool"),
            _field("checked_conditions.shadow_validation_passed", "bool"),
            _field("checked_conditions.routing_policy_stable", "bool"),
            _field("checked_conditions.candidate_routes_clean", "bool"),
            _field("checked_conditions.side_effects_absent", "bool"),
            _field("checked_conditions.candidate_bundle_fresh", "bool"),
            _field("checked_conditions.candidate_staleness_schema_passed", "bool"),
            _field("checked_conditions.candidate_staleness_status_active_review", "bool"),
            _field("checked_conditions.candidate_bundle_not_superseded", "bool"),
            _field("checked_conditions.candidate_forward_label_population_stable", "bool"),
            _field("checked_conditions.candidate_staleness_routing_policy_stable", "bool"),
            _field("checked_conditions.ev_shadow_schema_passed", "bool"),
            _field("checked_conditions.ev_shadow_status_not_rejected", "bool"),
            _field("checked_conditions.ev_shadow_has_sufficient_evidence", "bool"),
            _field("checked_conditions.ev_shadow_top_bucket_risk_adjusted_return_not_regressed", "bool"),
            _field("checked_conditions.ev_shadow_top_bucket_hit_rate_not_regressed", "bool"),
            _field("checked_conditions.ev_shadow_liquidity_status_ok", "bool"),
            _field("checked_conditions.ev_shadow_key_candidate_routes_non_negative", "bool"),
            _field("checked_conditions.guarded_shadow_schema_passed", "bool"),
            _field("checked_conditions.guarded_shadow_status_passed", "bool"),
            _field("checked_conditions.guarded_shadow_rank_preservation_policy_present", "bool"),
            _field("checked_conditions.guarded_shadow_quarantined_route_top_exposure_zero", "bool"),
            _field("checked_conditions.guarded_shadow_top_bucket_risk_adjusted_return_not_regressed", "bool"),
            _field("checked_conditions.guarded_shadow_top_bucket_hit_rate_not_regressed", "bool"),
            _field("checked_conditions.guarded_shadow_candidate_bundle_research_only", "bool"),
            _field("checked_conditions.guarded_shadow_candidate_bundle_approval_required", "bool"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_shadow_soak": ArtifactContract(
        name="segmented_probability_shadow_soak",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str"),
            _field("candidate_bundle_path", "str"),
            _field("guarded_candidate_bundle_path", "str"),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("soak_status", "str"),
            _field("soak_reasons", "list"),
            _field("dataset_summary", "dict"),
            _field("dataset_summary.dataset_exists", "bool"),
            _field("dataset_summary.row_count", "int"),
            _field("dataset_summary.quality_labeled_row_count", "int"),
            _field("outcome_refresh_summary", "dict"),
            _field("outcome_refresh_summary.outcome_refresh_source", "str"),
            _field("outcome_refresh_summary.outcome_refresh_attempted", "bool"),
            _field("outcome_refresh_summary.outcome_refresh_row_count", "int"),
            _field("outcome_refresh_summary.new_quality_labeled_rows_after_candidate", "int"),
            _field("outcome_refresh_summary.new_raw_labeled_rows_after_candidate", "int"),
            _field("outcome_refresh_summary.guarded_pre_refresh", "dict"),
            _field("outcome_refresh_summary.guarded_post_refresh", "dict"),
            _field("outcome_refresh_summary.new_quality_labeled_rows_after_guarded_candidate", "int"),
            _field("outcome_refresh_summary.new_raw_labeled_rows_after_guarded_candidate", "int"),
            _field("outcome_refresh_summary.pre_refresh", "dict"),
            _field("outcome_refresh_summary.post_refresh", "dict"),
            _field("forward_sample_progress", "dict"),
            _field("forward_sample_progress.strict_forward_row_count", "int"),
            _field("forward_sample_progress.min_forward_sample", "int"),
            _field("forward_sample_progress.forward_sample_gap", "int"),
            _field("forward_sample_progress.accumulation_status", "str", allow_none=True),
            _field("guarded_forward_sample_progress", "dict"),
            _field("guarded_forward_sample_progress.rows_after_guarded_candidate", "int"),
            _field("guarded_forward_sample_progress.quality_labeled_rows_after_guarded_candidate", "int"),
            _field("guarded_forward_sample_progress.guarded_strict_forward_row_count", "int"),
            _field("guarded_forward_sample_progress.min_forward_sample", "int"),
            _field("guarded_forward_sample_progress.forward_sample_gap", "int"),
            _field("guarded_validation_summary", "dict"),
            _field("guarded_validation_summary.guarded_shadow_status", "str", allow_none=True),
            _field("guarded_validation_summary.strict_forward_row_count", "int"),
            _field("guarded_validation_summary.min_forward_sample", "int"),
            _field("guarded_validation_summary.forward_sample_gap", "int"),
            _field("guarded_validation_summary.recommended_routing_policy", "str", allow_none=True),
            _field("readiness_summary", "dict"),
            _field("readiness_summary.readiness_status", "str", allow_none=True),
            _field("candidate_staleness_summary", "dict"),
            _field("candidate_staleness_summary.staleness_status", "str", allow_none=True),
            _field("guarded_candidate_staleness_summary", "dict"),
            _field("guarded_candidate_staleness_summary.guarded_staleness_status", "str", allow_none=True),
            _field("legacy_ev_shadow_summary", "dict"),
            _field("artifact_paths", "dict"),
            _field("artifact_paths.history_path", "str", allow_none=True),
            _field("artifact_paths.readiness_json_path", "str", allow_none=True),
            _field("artifact_paths.guarded_candidate_staleness_json_path", "str", allow_none=True),
            _field("artifact_paths.guarded_shadow_soak_history_path", "str", allow_none=True),
            _field("checked_conditions", "dict"),
            _field("checked_conditions.dataset_exists", "bool"),
            _field("checked_conditions.dataset_readable", "bool"),
            _field("checked_conditions.outcome_refresh_succeeded", "bool"),
            _field("checked_conditions.outcome_refresh_added_quality_labels", "bool"),
            _field("checked_conditions.outcome_refresh_added_guarded_quality_labels", "bool"),
            _field("checked_conditions.candidate_bundle_unchanged", "bool"),
            _field("checked_conditions.guarded_candidate_bundle_unchanged", "bool"),
            _field("checked_conditions.side_effects_absent", "bool"),
            _field("checked_conditions.guarded_shadow_passed", "bool"),
            _field("checked_conditions.guarded_strict_forward_sample_met", "bool"),
            _field("checked_conditions.guarded_candidate_staleness_active_or_accumulating", "bool"),
            _field("checked_conditions.strict_forward_sample_met", "bool"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_guarded_candidate_staleness": ArtifactContract(
        name="segmented_probability_guarded_candidate_staleness",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("guarded_candidate_bundle_path", "str", allow_none=True),
            _field("guarded_candidate_bundle_search_dir", "str", allow_none=True),
            _field("guarded_history_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("guarded_staleness_status", "str"),
            _field("guarded_staleness_reasons", "list"),
            _field("guarded_candidate_summary", "dict"),
            _field("guarded_candidate_summary.candidate_count", "int"),
            _field("guarded_candidate_summary.quarantined_candidate_count", "int"),
            _field("guarded_candidate_summary.guarded_candidate_age_days", "number", allow_none=True),
            _field("guarded_candidate_summary.stale_after_days", "number"),
            _field("guarded_candidate_summary.expire_after_days", "number"),
            _field("guarded_candidate_summary.research_only", "bool"),
            _field("guarded_candidate_summary.approval_required_for_runtime_use", "bool"),
            _field("guarded_candidate_summary.rank_preservation_policy_present", "bool"),
            _field("dataset_currency", "dict"),
            _field("dataset_currency.row_count", "int"),
            _field("dataset_currency.quality_labeled_row_count", "int"),
            _field("dataset_currency.rows_after_guarded_candidate_generated", "int"),
            _field("dataset_currency.quality_labeled_rows_after_guarded_candidate_generated", "int"),
            _field("dataset_currency.dataset_currency_status", "str"),
            _field("forward_label_population_shift", "dict"),
            _field("forward_label_population_shift.shift_status", "str"),
            _field("forward_label_population_shift.shifted_materially", "bool"),
            _field("forward_label_population_shift.post_guarded_label_count", "int"),
            _field("guarded_routing_policy_stability", "dict"),
            _field("guarded_routing_policy_stability.policy_stability_status", "str"),
            _field("guarded_routing_policy_stability.guarded_routing_policy_changed", "bool"),
            _field("supersession", "dict"),
            _field("supersession.superseded", "bool"),
            _field("checked_conditions", "dict"),
            _field("checked_conditions.guarded_candidate_count_positive", "bool"),
            _field("checked_conditions.guarded_bundle_not_superseded", "bool"),
            _field("checked_conditions.guarded_bundle_side_effect_flags_clean", "bool"),
            _field("checked_conditions.rank_preservation_policy_present", "bool"),
            _field("checked_conditions.guarded_forward_sample_met", "bool"),
            _field("recommended_next_actions", "list"),
        ),
    ),
    "segmented_probability_candidate_staleness": ArtifactContract(
        name="segmented_probability_candidate_staleness",
        required_fields=(
            _field("report_type", "str"),
            _field("generated_at", "str"),
            _field("dataset_path", "str", allow_none=True),
            _field("candidate_bundle_path", "str", allow_none=True),
            _field("history_path", "str", allow_none=True),
            _field("runtime_config_changed", "bool"),
            _field("parameter_pack_file_changed", "bool"),
            _field("execution_behavior_changed", "bool"),
            _field("staleness_status", "str"),
            _field("staleness_reasons", "list"),
            _field("candidate_summary", "dict"),
            _field("candidate_summary.candidate_count", "int"),
            _field("candidate_summary.candidate_age_days", "number", allow_none=True),
            _field("candidate_summary.stale_after_days", "number"),
            _field("candidate_summary.expire_after_days", "number"),
            _field("dataset_currency", "dict"),
            _field("dataset_currency.row_count", "int"),
            _field("dataset_currency.quality_labeled_row_count", "int"),
            _field("dataset_currency.rows_after_candidate_generated", "int"),
            _field("dataset_currency.quality_labeled_rows_after_candidate_generated", "int"),
            _field("dataset_currency.dataset_currency_status", "str"),
            _field("forward_label_population_shift", "dict"),
            _field("forward_label_population_shift.shift_status", "str"),
            _field("forward_label_population_shift.shifted_materially", "bool"),
            _field("forward_label_population_shift.post_candidate_label_count", "int"),
            _field("routing_policy_stability", "dict"),
            _field("routing_policy_stability.policy_stability_status", "str"),
            _field("routing_policy_stability.routing_policy_changed", "bool"),
            _field("supersession", "dict"),
            _field("supersession.superseded", "bool"),
            _field("checked_conditions", "dict"),
            _field("checked_conditions.candidate_count_positive", "bool"),
            _field("checked_conditions.candidate_bundle_superseded", "bool"),
            _field("recommended_next_actions", "list"),
        ),
    ),
}


def _path_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "any":
        return True
    if expected == "str":
        return isinstance(value, str)
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "dict":
        return isinstance(value, dict)
    if expected == "list":
        return isinstance(value, list)
    return False


def _resolve_contract(payload: dict[str, Any], contract_name: str | None) -> ArtifactContract | None:
    name = contract_name or str(payload.get("report_type") or "").strip()
    return ARTIFACT_CONTRACTS.get(name)


def validate_artifact_schema(
    payload: dict[str, Any],
    contract_name: str | None = None,
) -> dict[str, Any]:
    """Validate an artifact payload against a registered lightweight contract."""
    if not isinstance(payload, dict):
        return {
            "schema_status": "FAIL",
            "contract_name": contract_name,
            "missing_fields": [],
            "type_mismatches": [],
            "errors": ["Artifact payload is not a dictionary."],
        }

    contract = _resolve_contract(payload, contract_name)
    if contract is None:
        resolved_name = contract_name or payload.get("report_type")
        return {
            "schema_status": "FAIL",
            "contract_name": resolved_name,
            "missing_fields": [],
            "type_mismatches": [],
            "errors": [f"No schema contract is registered for `{resolved_name}`."],
        }

    missing_fields: list[str] = []
    type_mismatches: list[dict[str, Any]] = []
    for field in contract.required_fields:
        value = _path_value(payload, field.path)
        if value is MISSING:
            missing_fields.append(field.path)
            continue
        if value is None and field.allow_none:
            continue
        if value is None and not field.allow_none:
            type_mismatches.append(
                {
                    "field": field.path,
                    "expected": list(field.expected_types),
                    "actual": "NoneType",
                }
            )
            continue
        if not any(_type_matches(value, expected) for expected in field.expected_types):
            type_mismatches.append(
                {
                    "field": field.path,
                    "expected": list(field.expected_types),
                    "actual": type(value).__name__,
                }
            )

    errors = []
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    if type_mismatches:
        errors.append(
            "Type mismatches: "
            + ", ".join(
                f"{item['field']} expected {item['expected']} got {item['actual']}"
                for item in type_mismatches
            )
        )

    return {
        "schema_status": "FAIL" if errors else "PASS",
        "contract_name": contract.name,
        "missing_fields": missing_fields,
        "type_mismatches": type_mismatches,
        "errors": errors,
    }


def assert_artifact_schema(
    payload: dict[str, Any],
    contract_name: str | None = None,
) -> dict[str, Any]:
    """Validate an artifact and raise when the contract is violated."""
    validation = validate_artifact_schema(payload, contract_name=contract_name)
    if validation.get("schema_status") != "PASS":
        contract = validation.get("contract_name") or contract_name or "unknown"
        errors = "; ".join(validation.get("errors", []) or ["unknown schema error"])
        raise ArtifactSchemaValidationError(f"{contract} schema validation failed: {errors}")
    return validation
