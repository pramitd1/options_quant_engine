from __future__ import annotations

from pathlib import Path

from research.signal_evaluation.segmented_probability_guarded_candidate_bundle import (
    GUARDED_CANDIDATE_BUNDLE_BLOCKED,
    GUARDED_CANDIDATE_BUNDLE_READY,
    build_segmented_probability_guarded_candidate_bundle_report,
    write_segmented_probability_guarded_candidate_bundle_report,
)


def _source_bundle() -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "calibration_status": "SEGMENTED_CALIBRATION_CANDIDATES_READY",
        "candidate_count": 3,
        "candidates": [
            {
                "candidate_priority": 1,
                "candidate_type": "regime_segment",
                "segment_field": "direction",
                "segment_value": "PUT",
                "selected_calibrator": "linear_shrink",
                "state": {"method": "linear_shrink", "alpha": 0.0, "base_rate": 0.38},
                "selection": {"selection_reason": "unit", "holdout_count": 20},
            },
            {
                "candidate_priority": 2,
                "candidate_type": "recency_window",
                "segment_field": "train_recency_window",
                "segment_value": "last_25_pct_train",
                "selected_calibrator": "isotonic_score",
                "state": {"method": "isotonic_score", "calibration_mapping": {"55": 0.63}},
                "selection": {"selection_reason": "unit", "holdout_count": 100},
            },
            {
                "candidate_priority": 3,
                "candidate_type": "regime_segment",
                "segment_field": "gamma_regime",
                "segment_value": "NEGATIVE_GAMMA",
                "selected_calibrator": "temperature_score",
                "state": {"method": "temperature_score", "temperature": 2.0},
                "selection": {"selection_reason": "unit", "holdout_count": 15},
            },
        ],
    }


def _guarded_ev_experiment(status: str = "GUARDED_EV_EXPERIMENT_PASS") -> dict:
    return {
        "report_type": "segmented_probability_guarded_ev_experiment",
        "generated_at": "2026-05-15T05:00:00+00:00",
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_ev_status": status,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "top_fraction": 0.25,
        "raw_rank_ceiling_multiplier": 1.0,
        "quarantined_candidate_keys": [
            "regime_segment:direction=PUT",
            "regime_segment:gamma_regime=NEGATIVE_GAMMA",
        ],
        "selection_summary": {
            "recommended_guarded_variant": "quarantine_plus_rank_guard",
            "recommended_guarded_variant_status": status,
        },
    }


def test_guarded_candidate_bundle_quarantines_negative_routes():
    report = build_segmented_probability_guarded_candidate_bundle_report(
        _source_bundle(),
        _guarded_ev_experiment(),
        source_candidate_bundle_path="source.json",
        guarded_ev_experiment_path="guarded_ev.json",
        guarded_candidate_bundle_path="guarded_bundle.json",
    )

    bundle = report["guarded_candidate_bundle"]
    assert report["report_type"] == "segmented_probability_guarded_candidate_bundle"
    assert report["guarded_candidate_bundle_status"] == GUARDED_CANDIDATE_BUNDLE_READY
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert bundle["research_only"] is True
    assert bundle["approval_required_for_runtime_use"] is True
    assert bundle["candidate_count"] == 1
    assert bundle["candidates"][0]["candidate_type"] == "recency_window"
    assert bundle["quarantined_candidate_keys"] == [
        "regime_segment:direction=PUT",
        "regime_segment:gamma_regime=NEGATIVE_GAMMA",
    ]
    assert bundle["rank_preservation_policy"]["governance_only"] is True
    assert bundle["rank_preservation_policy"]["requires_guard_aware_shadow_evaluation"] is True
    assert "guard_aware_ev_shadow_evaluation" in bundle["required_next_validations"]


def test_guarded_candidate_bundle_blocks_when_guarded_ev_failed():
    report = build_segmented_probability_guarded_candidate_bundle_report(
        _source_bundle(),
        _guarded_ev_experiment(status="GUARDED_EV_EXPERIMENT_REJECTED"),
    )

    assert report["guarded_candidate_bundle_status"] == GUARDED_CANDIDATE_BUNDLE_BLOCKED
    assert "guarded_ev_experiment_not_passed" in report["guarded_candidate_bundle_reasons"]


def test_guarded_candidate_bundle_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_guarded_candidate_bundle_report(
        _source_bundle(),
        _guarded_ev_experiment(),
        source_candidate_bundle_path="source.json",
        guarded_ev_experiment_path="guarded_ev.json",
        output_dir=tmp_path,
        report_name="unit_guarded_candidate_bundle",
    )

    for key in [
        "report_json_path",
        "markdown_path",
        "candidate_bundle_json_path",
        "candidates_csv_path",
        "latest_report_json_path",
        "latest_markdown_path",
        "latest_candidate_bundle_json_path",
        "latest_candidates_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["guarded_candidate_bundle_status"] == GUARDED_CANDIDATE_BUNDLE_READY
    assert artifact["candidate_bundle"]["candidate_count"] == 1
