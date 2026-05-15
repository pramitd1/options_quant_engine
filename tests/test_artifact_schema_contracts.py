from __future__ import annotations

import pytest

from research.signal_evaluation.artifact_schema_contracts import (
    ArtifactSchemaValidationError,
    assert_artifact_schema,
    validate_artifact_schema,
)
from research.signal_evaluation.threshold_runtime_activation import build_threshold_runtime_activation_marker


def _rollout_payload() -> dict:
    return {
        "report_type": "threshold_signal_rollout_monitor",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "rollout_status": "CANDIDATE_SIGNAL_ROLLOUT_HEALTHY",
        "rollout_reasons": ["clean"],
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "baseline_pack_name": "baseline_v1",
        "candidate_pack_name": "candidate_v1",
        "post_adoption_traceability": {
            "traceability_status": "ALL_POST_ADOPTION_SIGNALS_CANDIDATE_PACK",
            "post_adoption_signal_count": 3,
            "candidate_pack_signal_count": 3,
            "non_candidate_pack_signal_count": 0,
            "missing_parameter_pack_count": 0,
        },
        "candidate_label_readiness": {
            "label_count_60m": 2,
        },
        "execution_side_effects": {
            "execution_side_effect_check_passed": True,
            "orders_submitted": False,
        },
    }


def _post_activation_payload() -> dict:
    return {
        "report_type": "threshold_post_activation_verification",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "verification_status": "POST_ACTIVATION_VERIFICATION_CLEAN",
        "verification_reasons": ["clean"],
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "candidate_pack_name": "candidate_v1",
        "active_parameter_pack": {"name": "candidate_v1", "layers": ["candidate_v1"]},
        "active_pack_matches_marker": True,
        "checked_conditions": {
            "active_pack_matches_marker": True,
            "threshold_matches": True,
            "non_candidate_pack_signal_count": 0,
            "missing_parameter_pack_count": 0,
            "candidate_label_count_60m": 2,
            "total_nonempty_side_effect_fields": 0,
        },
        "rollout_monitor": {},
        "adoption_history": {},
    }


def _signal_quality_payload() -> dict:
    return {
        "report_type": "signal_quality_model_audit",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 100,
        "quality_labeled_row_count": 80,
        "probability_field": "hybrid_move_probability",
        "label_quality": {},
        "calibration_summary": {
            "label_count": 80,
            "calibration_status": "CALIBRATED",
        },
        "calibration_bins": [],
        "regime_calibration": [],
        "feature_stability": [],
        "ranking_feature_audit": [],
        "recommended_next_actions": [],
    }


def _probability_calibration_payload() -> dict:
    return {
        "report_type": "probability_calibration_experiment",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 100,
        "quality_labeled_row_count": 80,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "train_count": 56,
        "holdout_count": 24,
        "methods_tested": ["identity", "linear_shrink"],
        "calibration_status": "CALIBRATION_CANDIDATE_READY",
        "selected_calibrator": "linear_shrink",
        "selection": {
            "candidate_ready_for_review": True,
        },
        "holdout_metrics": {},
        "calibrator_comparison": [],
        "calibration_curve": [],
        "candidate_calibrator": {
            "research_only": True,
            "approval_required_for_runtime_use": True,
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_calibration_payload() -> dict:
    return {
        "report_type": "segmented_probability_calibration_experiment",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 100,
        "quality_labeled_row_count": 80,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "train_count": 56,
        "holdout_count": 24,
        "segment_fields": ["macro_regime"],
        "recency_windows": [0.5, 1.0],
        "methods_tested": ["identity", "linear_shrink"],
        "calibration_status": "SEGMENTED_CALIBRATION_CANDIDATES_READY",
        "selection_summary": {
            "review_ready_candidate_count": 1,
            "evaluated_regime_segment_count": 2,
            "evaluated_recency_window_count": 2,
        },
        "recency_window_results": [],
        "segment_results": [],
        "candidate_bundle": {
            "research_only": True,
            "approval_required_for_runtime_use": True,
            "candidate_count": 1,
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_forward_shadow_payload() -> dict:
    return {
        "report_type": "segmented_probability_forward_shadow",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "candidate_bundle_path": "research/signal_evaluation/reports/bundle.json",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 100,
        "quality_labeled_row_count": 80,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "candidate_count": 2,
        "validation_window": {
            "validation_mode_used": "holdout_replay",
            "strict_forward_row_count": 0,
            "holdout_replay_row_count": 24,
        },
        "shadow_validation_status": "SHADOW_REPLAY_VALIDATION_PASS_FORWARD_DATA_PENDING",
        "selection_summary": {
            "evaluated_routing_policy_count": 3,
        },
        "routing_policy_results": [],
        "candidate_route_results": [],
        "calibration_curve": [],
        "route_decision_count": 72,
        "recommended_next_actions": [],
    }


def _segmented_probability_ev_shadow_payload() -> dict:
    return {
        "report_type": "segmented_probability_ev_shadow_evaluation",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "candidate_bundle_path": "research/signal_evaluation/reports/bundle.json",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 100,
        "quality_labeled_row_count": 80,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "return_field": "signed_return_60m_bps",
        "candidate_count": 2,
        "validation_window": {
            "validation_mode_used": "holdout_replay",
            "strict_forward_row_count": 0,
            "holdout_replay_row_count": 24,
        },
        "top_fraction": 0.25,
        "ev_shadow_status": "EV_SHADOW_EVALUATION_WATCH",
        "selection_summary": {
            "evaluated_routing_policy_count": 3,
        },
        "policy_results": [],
        "candidate_route_results": [],
        "regime_payoff_results": [],
        "route_decision_count": 72,
        "recommended_next_actions": [],
    }


def _segmented_probability_ev_rejection_attribution_payload() -> dict:
    return {
        "report_type": "segmented_probability_ev_rejection_attribution",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "ev_shadow_report_path": "research/signal_evaluation/reports/ev_shadow.json",
        "ev_shadow_routes_path": "research/signal_evaluation/reports/ev_shadow_routes.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "ev_shadow_status": "EV_SHADOW_EVALUATION_REJECTED",
        "attribution_status": "EV_REJECTION_ATTRIBUTION_ACTIONABLE",
        "attribution_reasons": ["shadow_top_worse_than_raw_top"],
        "analysis_policy": "regime_first",
        "top_fraction": 0.25,
        "return_field": "signed_return_60m_bps",
        "route_decision_count": 300,
        "analysis_route_decision_count": 100,
        "rejection_summary": {
            "raw_top_sample_count": 25,
            "shadow_top_sample_count": 25,
            "top_bucket_risk_adjusted_return_delta_bps": -7.5,
            "top_bucket_hit_rate_delta": -0.08,
        },
        "candidate_attribution": [],
        "shadow_only_candidate_attribution": [],
        "negative_route_candidates": [],
        "regime_attribution": [],
        "policy_comparison": [],
        "routing_diagnostics": {
            "likely_failure_mode": "CALIBRATED_TOP_BUCKET_UNDERPERFORMED_RAW_TOP_BUCKET",
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_guarded_ev_experiment_payload() -> dict:
    return {
        "report_type": "segmented_probability_guarded_ev_experiment",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "candidate_bundle_path": "research/signal_evaluation/reports/candidate_bundle.json",
        "ev_shadow_report_path": "research/signal_evaluation/reports/ev_shadow.json",
        "ev_rejection_attribution_path": "research/signal_evaluation/reports/ev_attribution.json",
        "ev_shadow_routes_path": "research/signal_evaluation/reports/ev_shadow_routes.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_ev_status": "GUARDED_EV_EXPERIMENT_PASS",
        "analysis_policy": "regime_first",
        "simulation_mode": "route_csv_fallback",
        "ev_rejection_attribution_status": "EV_REJECTION_ATTRIBUTION_ACTIONABLE",
        "ev_rejection_is_actionable": True,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "return_field": "signed_return_60m_bps",
        "top_fraction": 0.25,
        "raw_rank_ceiling_multiplier": 1.0,
        "quarantined_candidate_keys": ["regime_segment:direction=PUT"],
        "quarantined_candidate_count": 1,
        "candidate_count_before_quarantine": 3,
        "candidate_count_after_quarantine": 2,
        "route_decision_count": 100,
        "validation_window": {},
        "variant_results": [],
        "candidate_exposure": [],
        "selection_summary": {
            "evaluated_variant_count": 4,
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_guarded_candidate_bundle_payload() -> dict:
    return {
        "report_type": "segmented_probability_guarded_candidate_bundle",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "source_candidate_bundle_path": "research/signal_evaluation/reports/source_bundle.json",
        "guarded_ev_experiment_path": "research/signal_evaluation/reports/guarded_ev.json",
        "guarded_candidate_bundle_path": "research/signal_evaluation/reports/guarded_bundle.json",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_candidate_bundle_status": "GUARDED_CANDIDATE_BUNDLE_READY",
        "guarded_candidate_bundle_reasons": [],
        "guarded_ev_status": "GUARDED_EV_EXPERIMENT_PASS",
        "recommended_guarded_variant": "quarantine_plus_rank_guard",
        "rank_preservation_policy": {
            "governance_only": True,
            "runtime_behavior_changed": False,
            "requires_guard_aware_shadow_evaluation": True,
        },
        "source_candidate_count": 3,
        "kept_candidate_count": 1,
        "quarantined_candidate_count": 2,
        "quarantined_candidate_keys": [
            "regime_segment:direction=PUT",
            "regime_segment:gamma_regime=NEGATIVE_GAMMA",
        ],
        "missing_quarantine_keys": [],
        "required_next_validations": ["guard_aware_forward_shadow"],
        "guarded_candidate_bundle": {
            "research_only": True,
            "approval_required_for_runtime_use": True,
            "runtime_config_changed": False,
            "parameter_pack_file_changed": False,
            "execution_behavior_changed": False,
            "candidate_count": 1,
        },
        "candidate_rows": [],
        "recommended_next_actions": [],
    }


def _segmented_probability_guarded_shadow_validation_payload() -> dict:
    return {
        "report_type": "segmented_probability_guarded_shadow_validation",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "candidate_bundle_path": "research/signal_evaluation/reports/guarded_bundle.json",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 100,
        "quality_labeled_row_count": 80,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "return_field": "signed_return_60m_bps",
        "candidate_count": 1,
        "quarantined_candidate_keys": ["regime_segment:direction=PUT"],
        "validation_window": {
            "validation_mode_used": "holdout_replay",
            "strict_forward_row_count": 0,
            "holdout_replay_row_count": 24,
        },
        "top_fraction": 0.25,
        "raw_rank_ceiling_multiplier": 1.0,
        "rank_preservation_policy": {},
        "rank_preservation_policy_present": True,
        "guarded_bundle_side_effect_flags_clean": True,
        "guarded_bundle_research_only": True,
        "guarded_bundle_approval_required_for_runtime_use": True,
        "guarded_shadow_status": "GUARDED_SHADOW_VALIDATION_PASS",
        "selection_summary": {
            "evaluated_routing_policy_count": 3,
        },
        "policy_results": [],
        "candidate_route_results": [],
        "calibration_curve": [],
        "route_decision_count": 72,
        "recommended_next_actions": [],
    }


def _segmented_probability_forward_shadow_accumulation_payload() -> dict:
    return {
        "report_type": "segmented_probability_forward_shadow_accumulation",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "run_count": 1,
        "lookback_runs": 20,
        "trend_assessment": "WATCH",
        "latest": {},
        "status_counts": {},
        "validation_mode_counts": {},
        "shadow_status_counts": {},
        "lookback_summary": {
            "true_forward_pass_runs": 0,
            "holdout_replay_pending_runs": 1,
            "latest_strict_forward_row_count": 0,
        },
        "operator_message": "Holdout replay is pending forward labels.",
    }


def _segmented_probability_forward_shadow_readiness_payload() -> dict:
    return {
        "report_type": "segmented_probability_forward_shadow_readiness",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "accumulation_dashboard_path": "latest_accumulation.json",
        "forward_shadow_report_path": "latest_forward_shadow.json",
        "candidate_staleness_path": "latest_candidate_staleness.json",
        "ev_shadow_path": "latest_ev_shadow.json",
        "guarded_shadow_path": "latest_guarded_shadow.json",
        "history_path": "history.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "readiness_status": "FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW",
        "readiness_reasons": [],
        "checked_conditions": {
            "validation_mode_is_true_forward": True,
            "sufficient_forward_sample": True,
            "shadow_validation_passed": True,
            "routing_policy_stable": True,
            "candidate_routes_clean": True,
            "side_effects_absent": True,
            "candidate_bundle_fresh": True,
            "candidate_staleness_schema_passed": True,
            "candidate_staleness_status_active_review": True,
            "candidate_bundle_not_superseded": True,
            "candidate_forward_label_population_stable": True,
            "candidate_staleness_routing_policy_stable": True,
            "ev_shadow_schema_passed": True,
            "ev_shadow_status_not_rejected": True,
            "ev_shadow_has_sufficient_evidence": True,
            "ev_shadow_top_bucket_risk_adjusted_return_not_regressed": True,
            "ev_shadow_top_bucket_hit_rate_not_regressed": True,
            "ev_shadow_liquidity_status_ok": True,
            "ev_shadow_key_candidate_routes_non_negative": True,
            "guarded_shadow_schema_passed": True,
            "guarded_shadow_status_passed": True,
            "guarded_shadow_rank_preservation_policy_present": True,
            "guarded_shadow_quarantined_route_top_exposure_zero": True,
            "guarded_shadow_top_bucket_risk_adjusted_return_not_regressed": True,
            "guarded_shadow_top_bucket_hit_rate_not_regressed": True,
            "guarded_shadow_candidate_bundle_research_only": True,
            "guarded_shadow_candidate_bundle_approval_required": True,
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_shadow_soak_payload() -> dict:
    return {
        "report_type": "segmented_probability_shadow_soak",
        "generated_at": "2026-05-15T04:05:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "candidate_bundle_path": "latest_candidate_bundle.json",
        "guarded_candidate_bundle_path": "latest_guarded_candidate_bundle.json",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "soak_status": "SOAK_ACCUMULATING_TRUE_FORWARD_LABELS",
        "soak_reasons": ["insufficient_true_forward_sample"],
        "dataset_summary": {
            "dataset_exists": True,
            "dataset_size_bytes": 1000,
            "dataset_modified_at": "2026-05-15T04:00:00+00:00",
            "row_count": 100,
            "quality_labeled_row_count": 80,
            "latest_signal_timestamp": "2026-05-15T03:30:00+00:00",
            "dataset_read_error": None,
        },
        "outcome_refresh_summary": {
            "outcome_refresh_source": "local_spot_history",
            "outcome_refresh_attempted": True,
            "outcome_refresh_error": None,
            "outcome_refresh_row_count": 100,
            "new_quality_labeled_rows_after_candidate": 0,
            "new_raw_labeled_rows_after_candidate": 0,
            "new_quality_labeled_rows_after_guarded_candidate": 0,
            "new_raw_labeled_rows_after_guarded_candidate": 0,
            "pre_refresh": {
                "rows_after_candidate": 0,
                "quality_labeled_rows_after_candidate": 0,
                "raw_labeled_rows_after_candidate": 0,
            },
            "post_refresh": {
                "rows_after_candidate": 0,
                "quality_labeled_rows_after_candidate": 0,
                "raw_labeled_rows_after_candidate": 0,
            },
            "guarded_pre_refresh": {
                "rows_after_candidate": 0,
                "quality_labeled_rows_after_candidate": 0,
                "raw_labeled_rows_after_candidate": 0,
            },
            "guarded_post_refresh": {
                "rows_after_candidate": 0,
                "quality_labeled_rows_after_candidate": 0,
                "raw_labeled_rows_after_candidate": 0,
            },
        },
        "forward_sample_progress": {
            "validation_mode_used": "holdout_replay",
            "strict_forward_row_count": 0,
            "min_forward_sample": 100,
            "forward_sample_gap": 100,
            "forward_sample_progress_ratio": 0.0,
            "previous_strict_forward_row_count": None,
            "new_true_forward_rows_since_previous_soak": None,
            "accumulation_status": "HOLDOUT_REPLAY_PASS_PENDING_FORWARD_LABELS",
            "trend_assessment": "WATCH",
        },
        "guarded_forward_sample_progress": {
            "guarded_candidate_generated_at": "2026-05-15T04:30:00+00:00",
            "rows_after_guarded_candidate": 0,
            "quality_labeled_rows_after_guarded_candidate": 0,
            "raw_labeled_rows_after_guarded_candidate": 0,
            "pending_rows_after_guarded_candidate": 0,
            "partial_rows_after_guarded_candidate": 0,
            "complete_rows_after_guarded_candidate": 0,
            "latest_post_guarded_signal_timestamp": None,
            "guarded_validation_mode_used": "holdout_replay",
            "guarded_strict_forward_row_count": 0,
            "min_forward_sample": 100,
            "forward_sample_gap": 100,
            "forward_sample_progress_ratio": 0.0,
            "previous_guarded_strict_forward_row_count": None,
            "new_post_guarded_true_forward_rows_since_previous_soak": None,
            "history_path": "segmented_probability_shadow_soak_history.csv",
        },
        "guarded_validation_summary": {
            "guarded_shadow_status": "GUARDED_SHADOW_VALIDATION_PASS",
            "validation_mode_used": "holdout_replay",
            "strict_forward_row_count": 0,
            "min_forward_sample": 100,
            "forward_sample_gap": 100,
            "holdout_replay_row_count": 72,
            "recommended_routing_policy": "recency_first",
            "recommended_policy_status": "GUARDED_SHADOW_VALIDATION_PASS",
            "top_bucket_risk_delta_bps": 0.0,
            "top_bucket_hit_rate_delta": 0.0,
            "quarantined_route_top_count": 0,
            "rank_preservation_policy_present": True,
            "guarded_bundle_research_only": True,
            "guarded_bundle_approval_required_for_runtime_use": True,
        },
        "readiness_summary": {
            "readiness_status": "FORWARD_SHADOW_READINESS_BLOCKED",
            "readiness_reasons": ["insufficient_true_forward_sample"],
            "allow_holdout_replay_guarded_validation": False,
            "recommended_next_actions": [],
        },
        "candidate_staleness_summary": {
            "staleness_status": "ACTIVE_REVIEW",
            "staleness_reasons": [],
            "candidate_age_days": 1.0,
            "post_candidate_label_count": 0,
            "routing_policy_status": "POLICY_STABLE",
        },
        "guarded_candidate_staleness_summary": {
            "guarded_staleness_status": "GUARDED_ACCUMULATING_FORWARD_LABELS",
            "guarded_staleness_reasons": ["guarded_forward_sample_below_minimum"],
            "guarded_candidate_age_days": 0.5,
            "post_guarded_label_count": 0,
            "guarded_routing_policy_status": "INSUFFICIENT_GUARDED_FORWARD_EVIDENCE",
            "dataset_currency_status": "CURRENT_AT_GUARDED_GENERATION",
        },
        "legacy_ev_shadow_summary": {
            "refreshed": True,
            "ev_shadow_status": "EV_SHADOW_EVALUATION_PASS",
            "recommended_routing_policy": "recency_first",
            "top_bucket_risk_delta_bps": 0.0,
            "top_bucket_hit_rate_delta": 0.0,
        },
        "artifact_paths": {
            "history_path": "history.csv",
            "forward_shadow_json_path": "latest_forward_shadow.json",
            "accumulation_dashboard_json_path": "latest_accumulation.json",
            "candidate_staleness_json_path": "latest_candidate_staleness.json",
            "guarded_candidate_staleness_json_path": "latest_guarded_candidate_staleness.json",
            "ev_shadow_json_path": "latest_ev_shadow.json",
            "guarded_shadow_json_path": "latest_guarded_shadow.json",
            "readiness_json_path": "latest_readiness.json",
            "guarded_shadow_soak_history_path": "segmented_probability_shadow_soak_history.csv",
        },
        "checked_conditions": {
            "dataset_exists": True,
            "dataset_readable": True,
            "outcome_refresh_succeeded": True,
            "outcome_refresh_added_quality_labels": False,
            "outcome_refresh_added_guarded_quality_labels": False,
            "candidate_bundle_unchanged": True,
            "guarded_candidate_bundle_unchanged": True,
            "side_effects_absent": True,
            "readiness_ready_for_manual_review": False,
            "guarded_shadow_passed": True,
            "guarded_strict_forward_sample_met": False,
            "candidate_staleness_active_review": True,
            "guarded_candidate_staleness_active_or_accumulating": True,
            "strict_forward_sample_met": False,
            "true_forward_accumulation_passed": False,
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_candidate_staleness_payload() -> dict:
    return {
        "report_type": "segmented_probability_candidate_staleness",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "candidate_bundle_path": "latest_candidate_bundle.json",
        "candidate_bundle_search_dir": "research/signal_evaluation/reports/segmented_probability_calibration_experiment",
        "history_path": "history.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "staleness_status": "ACTIVE_REVIEW",
        "staleness_reasons": [],
        "candidate_summary": {
            "candidate_count": 2,
            "candidate_age_days": 0.5,
            "stale_after_days": 7.0,
            "expire_after_days": 14.0,
        },
        "dataset_currency": {
            "row_count": 100,
            "quality_labeled_row_count": 80,
            "rows_after_candidate_generated": 0,
            "quality_labeled_rows_after_candidate_generated": 0,
            "dataset_currency_status": "CURRENT_AT_CANDIDATE_GENERATION",
        },
        "forward_label_population_shift": {
            "shift_status": "INSUFFICIENT_FORWARD_LABELS",
            "shifted_materially": False,
            "post_candidate_label_count": 0,
        },
        "routing_policy_stability": {
            "policy_stability_status": "POLICY_STABLE",
            "routing_policy_changed": False,
        },
        "supersession": {
            "superseded": False,
        },
        "checked_conditions": {
            "candidate_count_positive": True,
            "candidate_bundle_superseded": False,
        },
        "recommended_next_actions": [],
    }


def _segmented_probability_guarded_candidate_staleness_payload() -> dict:
    return {
        "report_type": "segmented_probability_guarded_candidate_staleness",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "guarded_candidate_bundle_path": "latest_guarded_candidate_bundle.json",
        "guarded_candidate_bundle_search_dir": "research/signal_evaluation/reports/segmented_probability_guarded_candidate_bundle",
        "guarded_history_path": "segmented_probability_shadow_soak_history.csv",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_staleness_status": "GUARDED_ACCUMULATING_FORWARD_LABELS",
        "guarded_staleness_reasons": ["guarded_forward_sample_below_minimum"],
        "guarded_candidate_summary": {
            "candidate_count": 2,
            "quarantined_candidate_count": 1,
            "guarded_candidate_age_days": 0.5,
            "stale_after_days": 7.0,
            "expire_after_days": 14.0,
            "research_only": True,
            "approval_required_for_runtime_use": True,
            "rank_preservation_policy_present": True,
        },
        "dataset_currency": {
            "row_count": 100,
            "quality_labeled_row_count": 80,
            "rows_after_guarded_candidate_generated": 0,
            "quality_labeled_rows_after_guarded_candidate_generated": 0,
            "dataset_currency_status": "CURRENT_AT_GUARDED_GENERATION",
        },
        "forward_label_population_shift": {
            "shift_status": "INSUFFICIENT_FORWARD_LABELS",
            "shifted_materially": False,
            "post_guarded_label_count": 0,
        },
        "guarded_routing_policy_stability": {
            "policy_stability_status": "INSUFFICIENT_GUARDED_FORWARD_EVIDENCE",
            "guarded_routing_policy_changed": False,
        },
        "supersession": {
            "superseded": False,
        },
        "checked_conditions": {
            "guarded_candidate_count_positive": True,
            "guarded_bundle_not_superseded": True,
            "guarded_bundle_side_effect_flags_clean": True,
            "rank_preservation_policy_present": True,
            "guarded_forward_sample_met": False,
        },
        "recommended_next_actions": [],
    }


def test_runtime_activation_marker_schema_passes():
    marker = build_threshold_runtime_activation_marker(
        candidate_pack_name="candidate_v1",
        activated_at="2026-05-15T09:40:00+05:30",
        threshold_value=85.0,
    )

    validation = validate_artifact_schema(marker)

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "threshold_runtime_activation_marker"


def test_schema_contract_reports_missing_nested_fields():
    payload = _rollout_payload()
    del payload["candidate_label_readiness"]["label_count_60m"]

    validation = validate_artifact_schema(payload)

    assert validation["schema_status"] == "FAIL"
    assert "candidate_label_readiness.label_count_60m" in validation["missing_fields"]


def test_schema_contract_reports_type_mismatches():
    payload = _post_activation_payload()
    payload["active_pack_matches_marker"] = "true"

    validation = validate_artifact_schema(payload)

    assert validation["schema_status"] == "FAIL"
    assert {
        "field": "active_pack_matches_marker",
        "expected": ["bool"],
        "actual": "str",
    } in validation["type_mismatches"]


def test_schema_assertion_raises_actionable_error():
    payload = _rollout_payload()
    payload.pop("rollout_reasons")

    with pytest.raises(ArtifactSchemaValidationError, match="rollout_reasons"):
        assert_artifact_schema(payload)


def test_signal_quality_model_audit_schema_passes():
    validation = validate_artifact_schema(_signal_quality_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "signal_quality_model_audit"


def test_probability_calibration_experiment_schema_passes():
    validation = validate_artifact_schema(_probability_calibration_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "probability_calibration_experiment"


def test_segmented_probability_calibration_experiment_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_calibration_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_calibration_experiment"


def test_segmented_probability_forward_shadow_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_forward_shadow_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_forward_shadow"


def test_segmented_probability_ev_shadow_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_ev_shadow_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_ev_shadow_evaluation"


def test_segmented_probability_ev_rejection_attribution_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_ev_rejection_attribution_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_ev_rejection_attribution"


def test_segmented_probability_guarded_ev_experiment_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_guarded_ev_experiment_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_guarded_ev_experiment"


def test_segmented_probability_guarded_candidate_bundle_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_guarded_candidate_bundle_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_guarded_candidate_bundle"


def test_segmented_probability_guarded_shadow_validation_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_guarded_shadow_validation_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_guarded_shadow_validation"


def test_segmented_probability_forward_shadow_accumulation_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_forward_shadow_accumulation_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_forward_shadow_accumulation"


def test_segmented_probability_forward_shadow_readiness_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_forward_shadow_readiness_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_forward_shadow_readiness"


def test_segmented_probability_shadow_soak_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_shadow_soak_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_shadow_soak"


def test_segmented_probability_candidate_staleness_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_candidate_staleness_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_candidate_staleness"


def test_segmented_probability_guarded_candidate_staleness_schema_passes():
    validation = validate_artifact_schema(_segmented_probability_guarded_candidate_staleness_payload())

    assert validation["schema_status"] == "PASS"
    assert validation["contract_name"] == "segmented_probability_guarded_candidate_staleness"
