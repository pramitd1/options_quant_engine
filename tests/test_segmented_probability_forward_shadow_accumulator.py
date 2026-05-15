from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_forward_shadow import (
    build_segmented_probability_forward_shadow_report,
)
from research.signal_evaluation.segmented_probability_forward_shadow_accumulator import (
    ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD,
    ACCUMULATION_TRUE_FORWARD_PASS,
    build_segmented_probability_forward_shadow_accumulation_dashboard,
    build_segmented_probability_forward_shadow_history_row,
    write_segmented_probability_forward_shadow_accumulation,
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


def _candidate_bundle(*, generated_at: str) -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": generated_at,
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


def test_accumulation_row_tracks_holdout_replay_pending_forward_labels():
    shadow_report = build_segmented_probability_forward_shadow_report(
        _shadow_frame(),
        candidate_bundle=_candidate_bundle(generated_at="2026-05-15T04:00:00+00:00"),
        validation_mode="auto",
        min_shadow_sample=10,
        min_candidate_sample=3,
        min_brier_improvement=0.001,
    )
    row = build_segmented_probability_forward_shadow_history_row(
        shadow_report,
        report_path="unit_forward_shadow.json",
        min_shadow_sample=10,
    )
    dashboard = build_segmented_probability_forward_shadow_accumulation_dashboard(pd.DataFrame([row]))

    assert row["accumulation_status"] == ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD
    assert row["validation_mode_used"] == "holdout_replay"
    assert row["strict_forward_row_count"] == 0
    assert row["forward_sample_gap"] == 10
    assert dashboard["trend_assessment"] == "WATCH"
    assert dashboard["lookback_summary"]["holdout_replay_pending_runs"] == 1


def test_accumulation_row_switches_to_true_forward_when_enough_rows_exist():
    shadow_report = build_segmented_probability_forward_shadow_report(
        _shadow_frame(),
        candidate_bundle=_candidate_bundle(generated_at="2026-03-31T04:00:00+00:00"),
        validation_mode="auto",
        min_shadow_sample=10,
        min_candidate_sample=3,
        min_brier_improvement=0.001,
    )
    row = build_segmented_probability_forward_shadow_history_row(
        shadow_report,
        report_path="unit_forward_shadow.json",
        min_shadow_sample=10,
    )
    dashboard = build_segmented_probability_forward_shadow_accumulation_dashboard(pd.DataFrame([row]))

    assert row["accumulation_status"] == ACCUMULATION_TRUE_FORWARD_PASS
    assert row["validation_mode_used"] == "after_candidate_generated"
    assert row["strict_forward_row_count"] == 40
    assert row["forward_sample_gap"] == 0
    assert dashboard["trend_assessment"] == "READY_FOR_MANUAL_REVIEW"


def test_accumulator_writer_runs_shadow_and_appends_history(tmp_path: Path):
    dataset_path = tmp_path / "signals.csv"
    bundle_path = tmp_path / "candidate_bundle.json"
    _shadow_frame().to_csv(dataset_path, index=False)
    bundle_path.write_text(
        json.dumps(_candidate_bundle(generated_at="2026-05-15T04:00:00+00:00")),
        encoding="utf-8",
    )

    artifact = write_segmented_probability_forward_shadow_accumulation(
        dataset_path=dataset_path,
        candidate_bundle_path=bundle_path,
        shadow_output_dir=tmp_path / "shadow",
        output_dir=tmp_path / "accumulation",
        min_shadow_sample=10,
        min_candidate_sample=3,
        min_brier_improvement=0.001,
    )

    assert Path(artifact["history_path"]).exists()
    assert Path(artifact["accumulation_dashboard_json_path"]).exists()
    assert Path(artifact["accumulation_dashboard_markdown_path"]).exists()
    assert artifact["history_row"]["accumulation_status"] == ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD
    assert artifact["accumulation_dashboard"]["run_count"] == 1
    assert artifact["forward_shadow_artifact"]["latest_json_path"]
