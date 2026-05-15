from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_calibration_experiment import (
    build_segmented_probability_calibration_experiment_report,
    write_segmented_probability_calibration_experiment_report,
)


def _segmented_frame(row_count: int = 80) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    shifted_seen = 0
    stable_seen = 0
    rows = []
    for idx in range(row_count):
        shifted = idx % 2 == 0
        if shifted:
            hit = 1 if shifted_seen % 2 == 0 else 0
            shifted_seen += 1
            probability = 0.90
            macro_regime = "SHIFTED"
        else:
            hit = 1 if stable_seen % 2 == 0 else 0
            stable_seen += 1
            probability = 0.50
            macro_regime = "STABLE"
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 15)).isoformat(),
                "direction": "CALL" if idx % 3 else "PUT",
                "macro_regime": macro_regime,
                "gamma_regime": "POSITIVE_GAMMA" if idx % 4 else "NEGATIVE_GAMMA",
                "volatility_regime": "NORMAL_VOL",
                "global_risk_state": "GLOBAL_NEUTRAL",
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


def test_segmented_probability_calibration_finds_review_ready_slice():
    report = build_segmented_probability_calibration_experiment_report(
        _segmented_frame(),
        dataset_path="unit.csv",
        segment_fields=("macro_regime",),
        recency_windows=(0.50, 1.00),
        min_train_sample=20,
        min_holdout_sample=10,
        min_segment_train_sample=10,
        min_segment_holdout_sample=5,
        min_brier_improvement=0.01,
        max_candidate_ece=0.10,
    )

    assert report["report_type"] == "segmented_probability_calibration_experiment"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["calibration_status"] == "SEGMENTED_CALIBRATION_CANDIDATES_READY"
    assert report["selection_summary"]["review_ready_candidate_count"] >= 1
    assert report["candidate_bundle"]["research_only"] is True
    assert report["candidate_bundle"]["candidate_count"] >= 1
    assert report["candidate_bundle"]["candidates"][0]["candidate_priority"] == 1
    shifted = [
        row
        for row in report["segment_results"]
        if row["segment_field"] == "macro_regime" and row["segment_value"] == "SHIFTED"
    ][0]
    assert shifted["candidate_ready_for_review"] is True
    assert shifted["holdout_brier_improvement"] > 0


def test_segmented_probability_calibration_respects_sample_guardrails():
    report = build_segmented_probability_calibration_experiment_report(
        _segmented_frame(12),
        dataset_path="unit.csv",
        segment_fields=("macro_regime",),
        min_train_sample=50,
        min_holdout_sample=20,
        min_segment_train_sample=50,
        min_segment_holdout_sample=20,
    )

    assert report["calibration_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["candidate_bundle"]["candidate_count"] == 0
    assert "sample-size guardrails" in report["recommended_next_actions"][1]


def test_segmented_probability_calibration_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_calibration_experiment_report(
        _segmented_frame(),
        dataset_path="unit.csv",
        segment_fields=("macro_regime",),
        recency_windows=(0.50, 1.00),
        min_train_sample=20,
        min_holdout_sample=10,
        min_segment_train_sample=10,
        min_segment_holdout_sample=5,
        output_dir=tmp_path,
        report_name="unit_segmented_probability_calibration_experiment",
    )

    for key in [
        "json_path",
        "markdown_path",
        "segments_csv_path",
        "recency_csv_path",
        "candidate_bundle_json_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_segments_csv_path",
        "latest_recency_csv_path",
        "latest_candidate_bundle_json_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["candidate_bundle"]["approval_required_for_runtime_use"] is True
