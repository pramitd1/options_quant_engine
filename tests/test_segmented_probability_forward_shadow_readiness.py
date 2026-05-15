from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_forward_shadow import (
    FORWARD_SHADOW_PASS,
    SHADOW_REPLAY_PASS,
)
from research.signal_evaluation.segmented_probability_forward_shadow_accumulator import (
    ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD,
    ACCUMULATION_TRUE_FORWARD_PASS,
)
from research.signal_evaluation.segmented_probability_forward_shadow_readiness import (
    FORWARD_SHADOW_READINESS_BLOCKED,
    FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW,
    build_segmented_probability_forward_shadow_readiness_report,
    write_segmented_probability_forward_shadow_readiness_report,
)


def _forward_shadow_payload(
    *,
    validation_mode: str = "after_candidate_generated",
    shadow_status: str = FORWARD_SHADOW_PASS,
    candidate_regressed: bool = False,
) -> dict:
    candidate_status = "CANDIDATE_ROUTE_REGRESSED_BRIER" if candidate_regressed else "CANDIDATE_ROUTE_IMPROVED"
    return {
        "report_type": "segmented_probability_forward_shadow",
        "generated_at": "2026-05-10T04:00:00+00:00",
        "dataset_path": "unit.csv",
        "candidate_bundle_path": "bundle.json",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 200,
        "quality_labeled_row_count": 160,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "candidate_count": 2,
        "train_fraction": 0.70,
        "validation_window": {
            "validation_mode_requested": "auto",
            "validation_mode_used": validation_mode,
            "fallback_reason": None if validation_mode == "after_candidate_generated" else "insufficient_rows_after_candidate_generated",
            "candidate_generated_at": "2026-05-01T04:00:00+00:00",
            "strict_forward_row_count": 120 if validation_mode == "after_candidate_generated" else 0,
            "holdout_replay_row_count": 60,
            "train_count": 100,
        },
        "shadow_validation_status": shadow_status,
        "selection_summary": {
            "recommended_routing_policy": "recency_first",
            "recommended_policy_status": shadow_status,
            "recommended_policy_brier_improvement": 0.02,
            "recommended_policy_ece_change": -0.03,
            "evaluated_routing_policy_count": 3,
            "route_policy_status_counts": {shadow_status: 1},
        },
        "routing_policy_results": [
            {
                "route_policy": "recency_first",
                "shadow_status": shadow_status,
                "sample_count": 120 if validation_mode == "after_candidate_generated" else 60,
                "candidate_regression_count": 1 if candidate_regressed else 0,
            }
        ],
        "candidate_route_results": [
            {
                "route_policy": "recency_first",
                "candidate_key": "recency_window:train_recency_window=last_25_pct_train",
                "candidate_status": candidate_status,
                "sample_count": 120,
            }
        ],
        "calibration_curve": [],
        "route_decision_count": 120,
        "recommended_next_actions": [],
    }


def _accumulation_payload(*, validation_mode: str = "after_candidate_generated") -> dict:
    true_forward = validation_mode == "after_candidate_generated"
    status = ACCUMULATION_TRUE_FORWARD_PASS if true_forward else ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD
    return {
        "report_type": "segmented_probability_forward_shadow_accumulation",
        "generated_at": "2026-05-10T04:01:00+00:00",
        "run_count": 1,
        "lookback_runs": 20,
        "trend_assessment": "READY_FOR_MANUAL_REVIEW" if true_forward else "WATCH",
        "latest": {
            "observed_at": "2026-05-10T04:01:00+00:00",
            "accumulation_status": status,
            "validation_mode_used": validation_mode,
            "strict_forward_row_count": 120 if true_forward else 0,
            "min_shadow_sample": 100,
            "shadow_validation_status": FORWARD_SHADOW_PASS if true_forward else SHADOW_REPLAY_PASS,
            "recommended_routing_policy": "recency_first",
            "recommended_policy_status": FORWARD_SHADOW_PASS if true_forward else SHADOW_REPLAY_PASS,
            "candidate_generated_at": "2026-05-01T04:00:00+00:00",
            "candidate_count": 2,
            "runtime_config_changed": False,
            "parameter_pack_file_changed": False,
            "execution_behavior_changed": False,
        },
        "status_counts": {status: 1},
        "validation_mode_counts": {validation_mode: 1},
        "shadow_status_counts": {FORWARD_SHADOW_PASS if true_forward else SHADOW_REPLAY_PASS: 1},
        "lookback_summary": {
            "true_forward_pass_runs": 1 if true_forward else 0,
            "holdout_replay_pending_runs": 0 if true_forward else 1,
            "latest_strict_forward_row_count": 120 if true_forward else 0,
        },
        "operator_message": "ready" if true_forward else "pending",
    }


def _history_frame(*, validation_mode: str = "after_candidate_generated") -> pd.DataFrame:
    true_forward = validation_mode == "after_candidate_generated"
    return pd.DataFrame(
        [
            {
                "observed_at": "2026-05-10T04:01:00+00:00",
                "validation_mode_used": validation_mode,
                "strict_forward_row_count": 120 if true_forward else 0,
                "accumulation_status": (
                    ACCUMULATION_TRUE_FORWARD_PASS
                    if true_forward
                    else ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD
                ),
                "recommended_routing_policy": "recency_first",
            }
        ]
    )


def _staleness_payload(
    *,
    status: str = "ACTIVE_REVIEW",
    superseded: bool = False,
    shifted: bool = False,
    routing_changed: bool = False,
    runtime_side_effect: bool = False,
) -> dict:
    return {
        "report_type": "segmented_probability_candidate_staleness",
        "generated_at": "2026-05-10T04:02:00+00:00",
        "dataset_path": "unit.csv",
        "candidate_bundle_path": "bundle.json",
        "candidate_bundle_search_dir": ".",
        "history_path": "history.csv",
        "runtime_config_changed": runtime_side_effect,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "staleness_status": status,
        "staleness_reasons": [] if status == "ACTIVE_REVIEW" else ["candidate_bundle_superseded"],
        "candidate_summary": {
            "candidate_count": 2,
            "candidate_generated_at": "2026-05-01T04:00:00+00:00",
            "candidate_age_days": 9.0,
            "stale_after_days": 14.0,
            "expire_after_days": 30.0,
        },
        "dataset_currency": {
            "row_count": 200,
            "quality_labeled_row_count": 160,
            "rows_after_candidate_generated": 120,
            "quality_labeled_rows_after_candidate_generated": 120,
            "dataset_currency_status": "FORWARD_ROWS_ACCUMULATING",
        },
        "forward_label_population_shift": {
            "shift_status": "MATERIAL_SHIFT_DETECTED" if shifted else "NO_MATERIAL_SHIFT",
            "shifted_materially": shifted,
            "post_candidate_label_count": 120,
        },
        "routing_policy_stability": {
            "policy_stability_status": "POLICY_CHANGED" if routing_changed else "POLICY_STABLE",
            "latest_recommended_routing_policy": "recency_first",
            "routing_policy_changed": routing_changed,
        },
        "supersession": {
            "superseded": superseded,
            "candidate_bundle_search_status": "NEWER_BUNDLE_FOUND" if superseded else "NO_NEWER_BUNDLE_FOUND",
        },
        "checked_conditions": {
            "candidate_count_positive": True,
            "candidate_bundle_superseded": superseded,
            "forward_label_population_shifted": shifted,
            "routing_policy_stable": not routing_changed,
        },
        "recommended_next_actions": [],
    }


def _ev_shadow_payload(
    *,
    status: str = "EV_SHADOW_EVALUATION_PASS",
    risk_delta: float = 3.5,
    hit_delta: float = 0.04,
    liquidity_status: str = "OK",
    negative_route: bool = False,
    runtime_side_effect: bool = False,
) -> dict:
    route_status = "EV_ROUTE_NEGATIVE" if negative_route else "EV_ROUTE_HELPFUL"
    return {
        "report_type": "segmented_probability_ev_shadow_evaluation",
        "generated_at": "2026-05-10T04:03:00+00:00",
        "dataset_path": "unit.csv",
        "candidate_bundle_path": "bundle.json",
        "runtime_config_changed": runtime_side_effect,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 200,
        "quality_labeled_row_count": 160,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "return_field": "signed_return_60m_bps",
        "candidate_count": 2,
        "validation_window": {
            "validation_mode_used": "after_candidate_generated",
            "strict_forward_row_count": 120,
            "holdout_replay_row_count": 60,
        },
        "top_fraction": 0.25,
        "ev_shadow_status": status,
        "selection_summary": {
            "recommended_routing_policy": "recency_first",
            "recommended_policy_status": status,
            "recommended_policy_score": 5.5,
            "recommended_policy_risk_adjusted_return_delta_bps": risk_delta,
            "recommended_policy_hit_rate_delta": hit_delta,
            "evaluated_routing_policy_count": 1,
        },
        "policy_results": [
            {
                "route_policy": "recency_first",
                "ev_shadow_status": status,
                "status_reason": "unit",
                "shadow_vs_raw_top_risk_adjusted_return_delta_bps": risk_delta,
                "shadow_vs_raw_top_hit_rate_delta": hit_delta,
                "liquidity_status": liquidity_status,
            }
        ],
        "candidate_route_results": [
            {
                "route_policy": "recency_first",
                "candidate_key": "recency_window:train_recency_window=last_25_pct_train",
                "ev_route_status": route_status,
                "sample_count": 120,
            }
        ],
        "regime_payoff_results": [],
        "route_decision_count": 120,
        "recommended_next_actions": [],
    }


def _guarded_shadow_payload(
    *,
    status: str = "GUARDED_SHADOW_VALIDATION_PASS",
    validation_mode: str = "after_candidate_generated",
    risk_delta: float = 3.5,
    hit_delta: float = 0.04,
    quarantined_top_count: int = 0,
    rank_policy_present: bool = True,
    research_only: bool = True,
    approval_required: bool = True,
    runtime_side_effect: bool = False,
) -> dict:
    true_forward = validation_mode == "after_candidate_generated"
    return {
        "report_type": "segmented_probability_guarded_shadow_validation",
        "generated_at": "2026-05-10T04:04:00+00:00",
        "dataset_path": "unit.csv",
        "candidate_bundle_path": "guarded_bundle.json",
        "runtime_config_changed": runtime_side_effect,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": 200,
        "quality_labeled_row_count": 160,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "return_field": "signed_return_60m_bps",
        "candidate_count": 1,
        "quarantined_candidate_keys": [
            "regime_segment:direction=PUT",
            "regime_segment:gamma_regime=NEGATIVE_GAMMA",
        ],
        "validation_window": {
            "validation_mode_used": validation_mode,
            "strict_forward_row_count": 120 if true_forward else 0,
            "holdout_replay_row_count": 60,
        },
        "top_fraction": 0.25,
        "raw_rank_ceiling_multiplier": 1.0,
        "rank_preservation_policy": {"governance_only": True} if rank_policy_present else {},
        "rank_preservation_policy_present": rank_policy_present,
        "guarded_bundle_side_effect_flags_clean": not runtime_side_effect,
        "guarded_bundle_research_only": research_only,
        "guarded_bundle_approval_required_for_runtime_use": approval_required,
        "guarded_shadow_status": status,
        "selection_summary": {
            "recommended_routing_policy": "recency_first",
            "recommended_policy_status": status,
            "recommended_policy_score": 5.5,
            "recommended_policy_risk_delta_vs_raw_bps": risk_delta,
            "recommended_policy_hit_delta_vs_raw": hit_delta,
            "evaluated_routing_policy_count": 1,
        },
        "policy_results": [
            {
                "route_policy": "recency_first",
                "guarded_policy_status": status,
                "status_reason": "unit",
                "guarded_vs_raw_top_risk_adjusted_return_delta_bps": risk_delta,
                "guarded_vs_raw_top_hit_rate_delta": hit_delta,
                "quarantined_route_top_count": quarantined_top_count,
                "quarantined_route_top_rate": 0.0 if quarantined_top_count == 0 else 0.25,
                "policy_score": 5.5,
            }
        ],
        "candidate_route_results": [],
        "calibration_curve": [],
        "route_decision_count": 120,
        "recommended_next_actions": [],
    }


def test_forward_shadow_readiness_passes_only_true_forward_clean_gate():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW
    assert report["readiness_reasons"] == []
    assert report["checked_conditions"]["validation_mode_is_true_forward"] is True
    assert report["checked_conditions"]["routing_policy_stable"] is True
    assert report["checked_conditions"]["candidate_routes_clean"] is True
    assert report["checked_conditions"]["candidate_staleness_status_active_review"] is True
    assert report["checked_conditions"]["ev_shadow_status_not_rejected"] is True


def test_forward_shadow_readiness_blocks_holdout_replay():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(validation_mode="holdout_replay"),
        forward_shadow_report=_forward_shadow_payload(
            validation_mode="holdout_replay",
            shadow_status=SHADOW_REPLAY_PASS,
        ),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(validation_mode="holdout_replay"),
        history=_history_frame(validation_mode="holdout_replay"),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "validation_mode_not_after_candidate_generated" in report["readiness_reasons"]
    assert "insufficient_true_forward_sample" in report["readiness_reasons"]


def test_forward_shadow_readiness_allows_explicit_guarded_holdout_mode():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(validation_mode="holdout_replay"),
        forward_shadow_report=_forward_shadow_payload(
            validation_mode="holdout_replay",
            shadow_status=SHADOW_REPLAY_PASS,
        ),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(validation_mode="holdout_replay"),
        history=_history_frame(validation_mode="holdout_replay"),
        min_forward_sample=50,
        max_candidate_age_days=30,
        allow_holdout_replay_guarded_validation=True,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW
    assert report["checked_conditions"]["allow_holdout_replay_guarded_validation"] is True
    assert report["checked_conditions"]["validation_mode_requirement_met"] is True
    assert report["checked_conditions"]["sufficient_forward_or_guarded_holdout_sample"] is True


def test_forward_shadow_readiness_blocks_candidate_route_regression():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(candidate_regressed=True),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "candidate_route_regression_detected" in report["readiness_reasons"]


def test_forward_shadow_readiness_blocks_missing_staleness_artifact():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "candidate_staleness_schema_failed" in report["readiness_reasons"]
    assert "candidate_staleness_status_not_active_review" in report["readiness_reasons"]


def test_forward_shadow_readiness_blocks_missing_guarded_shadow_artifact():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "guarded_shadow_schema_failed" in report["readiness_reasons"]
    assert "guarded_shadow_status_not_passed" in report["readiness_reasons"]


def test_forward_shadow_readiness_uses_guarded_shadow_over_rejected_legacy_ev_shadow():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(
            status="EV_SHADOW_EVALUATION_REJECTED",
            risk_delta=-7.5,
            hit_delta=-0.08,
            negative_route=True,
        ),
        guarded_shadow_report=_guarded_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW
    assert "ev_shadow_status_rejected" not in report["readiness_reasons"]
    assert report["checked_conditions"]["ev_shadow_status_not_rejected"] is False
    assert report["checked_conditions"]["guarded_shadow_status_passed"] is True


def test_forward_shadow_readiness_blocks_guarded_shadow_regression():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        candidate_staleness_report=_staleness_payload(),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(risk_delta=-2.0, hit_delta=-0.01),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "guarded_shadow_top_bucket_risk_adjusted_return_regressed" in report["readiness_reasons"]
    assert "guarded_shadow_top_bucket_hit_rate_regressed" in report["readiness_reasons"]


def test_forward_shadow_readiness_blocks_superseded_staleness_artifact():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        candidate_staleness_report=_staleness_payload(status="SUPERSEDED", superseded=True),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "candidate_staleness_status_not_active_review" in report["readiness_reasons"]
    assert "candidate_bundle_superseded" in report["readiness_reasons"]


def test_forward_shadow_readiness_blocks_shifted_staleness_artifact():
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=_accumulation_payload(),
        forward_shadow_report=_forward_shadow_payload(),
        candidate_staleness_report=_staleness_payload(status="STALE_WATCH", shifted=True),
        ev_shadow_report=_ev_shadow_payload(),
        guarded_shadow_report=_guarded_shadow_payload(),
        history=_history_frame(),
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert report["readiness_status"] == FORWARD_SHADOW_READINESS_BLOCKED
    assert "candidate_forward_label_population_shifted" in report["readiness_reasons"]


def test_forward_shadow_readiness_writer_outputs_artifacts(tmp_path: Path):
    accumulation_path = tmp_path / "accumulation.json"
    shadow_path = tmp_path / "forward_shadow.json"
    staleness_path = tmp_path / "candidate_staleness.json"
    ev_path = tmp_path / "ev_shadow.json"
    guarded_path = tmp_path / "guarded_shadow.json"
    history_path = tmp_path / "history.csv"
    accumulation_path.write_text(json.dumps(_accumulation_payload()), encoding="utf-8")
    shadow_path.write_text(json.dumps(_forward_shadow_payload()), encoding="utf-8")
    staleness_path.write_text(json.dumps(_staleness_payload()), encoding="utf-8")
    ev_path.write_text(json.dumps(_ev_shadow_payload()), encoding="utf-8")
    guarded_path.write_text(json.dumps(_guarded_shadow_payload()), encoding="utf-8")
    _history_frame().to_csv(history_path, index=False)

    artifact = write_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard_path=accumulation_path,
        forward_shadow_report_path=shadow_path,
        candidate_staleness_path=staleness_path,
        ev_shadow_path=ev_path,
        guarded_shadow_path=guarded_path,
        history_path=history_path,
        output_dir=tmp_path / "readiness",
        min_forward_sample=100,
        max_candidate_age_days=30,
        as_of="2026-05-10T04:00:00+00:00",
    )

    assert Path(artifact["readiness_json_path"]).exists()
    assert Path(artifact["readiness_markdown_path"]).exists()
    assert artifact["readiness_report"]["readiness_status"] == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW
