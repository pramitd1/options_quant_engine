from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_forward_shadow import (
    NEEDS_MORE_FORWARD_DATA,
    SHADOW_REPLAY_PASS,
    build_segmented_probability_forward_shadow_report,
    write_segmented_probability_forward_shadow_report,
)


def _shadow_frame(row_count: int = 40) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    put_seen = 0
    call_seen = 0
    rows = []
    for idx in range(row_count):
        is_put = idx % 2 == 0
        if is_put:
            hit = 1 if put_seen % 2 == 0 else 0
            put_seen += 1
            direction = "PUT"
            probability = 0.90
        else:
            hit = 1 if call_seen % 2 == 0 else 0
            call_seen += 1
            direction = "CALL"
            probability = 0.50
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 15)).isoformat(),
                "direction": direction,
                "gamma_regime": "NEGATIVE_GAMMA" if idx % 4 == 0 else "POSITIVE_GAMMA",
                "hybrid_move_probability": probability,
                "correct_60m": hit,
                "calibration_label": hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 20 if hit else -20,
                "primary_outcome_return_bps": 20 if hit else -20,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def _candidate_bundle() -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "candidate_count": 2,
        "candidates": [
            {
                "candidate_priority": 1,
                "candidate_type": "regime_segment",
                "segment_field": "direction",
                "segment_value": "PUT",
                "selected_calibrator": "linear_shrink",
                "state": {
                    "method": "linear_shrink",
                    "alpha": 0.0,
                    "base_rate": 0.5,
                },
            },
            {
                "candidate_priority": 2,
                "candidate_type": "recency_window",
                "segment_field": "train_recency_window",
                "segment_value": "last_25_pct_train",
                "selected_calibrator": "isotonic_score",
                "state": {
                    "method": "isotonic_score",
                    "calibration_mapping": {
                        "50": 0.5,
                        "90": 0.6,
                    },
                },
            },
        ],
    }


def test_forward_shadow_validates_routing_precedence():
    report = build_segmented_probability_forward_shadow_report(
        _shadow_frame(),
        candidate_bundle=_candidate_bundle(),
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        min_shadow_sample=10,
        min_candidate_sample=3,
        min_brier_improvement=0.001,
        routing_policies=("candidate_priority", "regime_first", "recency_first"),
    )

    assert report["report_type"] == "segmented_probability_forward_shadow"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["shadow_validation_status"] == SHADOW_REPLAY_PASS
    assert report["selection_summary"]["recommended_routing_policy"] in {"candidate_priority", "regime_first"}
    assert report["route_decision_count"] == 36

    routes = report["_route_decisions_frame"]
    candidate_priority_put = routes.loc[
        (routes["route_policy"] == "candidate_priority") & (routes["direction"] == "PUT")
    ].iloc[0]
    recency_first_put = routes.loc[
        (routes["route_policy"] == "recency_first") & (routes["direction"] == "PUT")
    ].iloc[0]
    assert candidate_priority_put["assigned_candidate_type"] == "regime_segment"
    assert recency_first_put["assigned_candidate_type"] == "recency_window"
    assert candidate_priority_put["matched_candidate_count"] == 2


def test_forward_shadow_respects_sample_guardrails():
    report = build_segmented_probability_forward_shadow_report(
        _shadow_frame(12),
        candidate_bundle=_candidate_bundle(),
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        min_shadow_sample=50,
        min_candidate_sample=20,
    )

    assert report["shadow_validation_status"] == NEEDS_MORE_FORWARD_DATA
    assert report["routing_policy_results"][0]["shadow_status"] == NEEDS_MORE_FORWARD_DATA
    assert "Collect more quality-approved forward rows" in report["recommended_next_actions"][0]


def test_forward_shadow_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_forward_shadow_report(
        _shadow_frame(),
        candidate_bundle=_candidate_bundle(),
        candidate_bundle_path="unit_bundle.json",
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        min_shadow_sample=10,
        min_candidate_sample=3,
        min_brier_improvement=0.001,
        output_dir=tmp_path,
        report_name="unit_segmented_probability_forward_shadow",
    )

    for key in [
        "json_path",
        "markdown_path",
        "policies_csv_path",
        "candidates_csv_path",
        "routes_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_policies_csv_path",
        "latest_candidates_csv_path",
        "latest_routes_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["candidate_count"] == 2
