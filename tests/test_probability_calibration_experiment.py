from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.probability_calibration_experiment import (
    build_probability_calibration_experiment_report,
    write_probability_calibration_experiment_report,
)


def _overconfident_frame(row_count: int = 40) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    rows = []
    for idx in range(row_count):
        hit = 1 if idx % 2 == 0 else 0
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 15)).isoformat(),
                "direction": "CALL" if idx % 3 else "PUT",
                "hybrid_move_probability": 0.90,
                "correct_60m": hit,
                "calibration_label": hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 20 if hit else -20,
                "primary_outcome_return_bps": 20 if hit else -20,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_probability_calibration_experiment_selects_review_candidate():
    report = build_probability_calibration_experiment_report(
        _overconfident_frame(),
        dataset_path="unit.csv",
        min_train_sample=10,
        min_holdout_sample=5,
        min_brier_improvement=0.01,
        max_candidate_ece=0.10,
    )

    assert report["report_type"] == "probability_calibration_experiment"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["train_count"] == 28
    assert report["holdout_count"] == 12
    assert report["calibration_status"] == "CALIBRATION_CANDIDATE_READY"
    assert report["selected_calibrator"] in {"linear_shrink", "isotonic_score"}
    assert report["selection"]["candidate_ready_for_review"] is True
    assert report["selection"]["holdout_brier_improvement"] > 0
    assert report["candidate_calibrator"]["research_only"] is True
    assert any(row["method"] == "identity" for row in report["calibrator_comparison"])
    assert any(row["split"] == "holdout" for row in report["calibration_curve"])


def test_probability_calibration_experiment_respects_sample_guardrails():
    report = build_probability_calibration_experiment_report(
        _overconfident_frame(12),
        dataset_path="unit.csv",
        min_train_sample=20,
        min_holdout_sample=20,
    )

    assert report["calibration_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["selected_calibrator"] == "identity"
    assert report["selection"]["candidate_ready_for_review"] is False
    assert "sample-size guardrails" in report["recommended_next_actions"][1]


def test_probability_calibration_experiment_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_probability_calibration_experiment_report(
        _overconfident_frame(),
        dataset_path="unit.csv",
        min_train_sample=10,
        min_holdout_sample=5,
        output_dir=tmp_path,
        report_name="unit_probability_calibration_experiment",
    )

    for key in [
        "json_path",
        "markdown_path",
        "comparison_csv_path",
        "curve_csv_path",
        "candidate_json_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_comparison_csv_path",
        "latest_curve_csv_path",
        "latest_candidate_json_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["candidate_calibrator"]["approval_required_for_runtime_use"] is True
