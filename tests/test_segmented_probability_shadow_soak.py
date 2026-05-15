from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_shadow_soak import (
    SOAK_ACCUMULATING_TRUE_FORWARD_LABELS,
    write_segmented_probability_shadow_soak_report,
)


def _shadow_frame(row_count: int = 80) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    rows = []
    for idx in range(row_count):
        hit = 1 if idx % 2 == 0 else 0
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 15)).isoformat(),
                "direction": "PUT" if idx % 2 == 0 else "CALL",
                "hybrid_move_probability": 0.90,
                "correct_60m": hit,
                "calibration_label": hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 20 if hit else -20,
                "primary_outcome_return_bps": 20 if hit else -20,
                "mae_60m_bps": -8,
                "mfe_60m_bps": 12,
                "selected_option_ba_spread_pct": 1.0,
                "selected_option_volume": 100,
                "selected_option_open_interest": 500,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def _candidate_bundle(*, generated_at: str, guarded: bool = False) -> dict:
    bundle = {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": generated_at,
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
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
                "selected_calibrator": "linear_shrink",
                "state": {
                    "method": "linear_shrink",
                    "alpha": 0.0,
                    "base_rate": 0.5,
                },
            },
        ],
    }
    if guarded:
        bundle.update(
            {
                "guarded_candidate_bundle_status": "GUARDED_CANDIDATE_BUNDLE_READY",
                "rank_preservation_policy": {
                    "governance_only": True,
                    "runtime_behavior_changed": False,
                    "requires_guard_aware_shadow_evaluation": True,
                    "top_fraction": 0.25,
                    "raw_rank_ceiling_multiplier": 1.0,
                },
                "quarantined_candidate_keys": [],
            }
        )
    return bundle


def test_shadow_soak_writer_runs_daily_evidence_loop_and_marks_no_new_rows(tmp_path: Path):
    dataset_path = tmp_path / "signals.csv"
    bundle_path = tmp_path / "candidate_bundle.json"
    guarded_bundle_path = tmp_path / "guarded_candidate_bundle.json"
    _shadow_frame().to_csv(dataset_path, index=False)
    generated_at = "2026-03-31T04:00:00+00:00"
    bundle_path.write_text(json.dumps(_candidate_bundle(generated_at=generated_at)), encoding="utf-8")
    guarded_bundle_path.write_text(
        json.dumps(_candidate_bundle(generated_at=generated_at, guarded=True)),
        encoding="utf-8",
    )

    first = write_segmented_probability_shadow_soak_report(
        dataset_path=dataset_path,
        candidate_bundle_path=bundle_path,
        guarded_candidate_bundle_path=guarded_bundle_path,
        output_dir=tmp_path / "soak",
        accumulation_output_dir=tmp_path / "accumulation",
        forward_shadow_output_dir=tmp_path / "forward_shadow",
        candidate_staleness_output_dir=tmp_path / "staleness",
        guarded_candidate_staleness_output_dir=tmp_path / "guarded_staleness",
        ev_shadow_output_dir=tmp_path / "ev_shadow",
        guarded_shadow_output_dir=tmp_path / "guarded_shadow",
        readiness_output_dir=tmp_path / "readiness",
        outcome_refresh_source="skip",
        min_shadow_sample=20,
        min_forward_sample=100,
        min_candidate_sample=3,
        min_ev_sample=20,
        min_top_sample=5,
        min_shift_sample=100,
        max_candidate_age_days=30,
        expire_after_days=30,
        as_of="2026-04-02T04:00:00+00:00",
    )

    report = first["shadow_soak_report"]
    assert Path(first["shadow_soak_json_path"]).exists()
    assert Path(first["shadow_soak_markdown_path"]).exists()
    assert Path(first["shadow_soak_history_path"]).exists()
    assert report["soak_status"] == SOAK_ACCUMULATING_TRUE_FORWARD_LABELS
    assert report["forward_sample_progress"]["strict_forward_row_count"] == 80
    assert report["forward_sample_progress"]["forward_sample_gap"] == 20
    assert report["guarded_forward_sample_progress"]["guarded_strict_forward_row_count"] == 80
    assert report["guarded_forward_sample_progress"]["forward_sample_gap"] == 20
    assert report["guarded_forward_sample_progress"]["new_post_guarded_true_forward_rows_since_previous_soak"] is None
    assert report["checked_conditions"]["candidate_bundle_unchanged"] is True
    assert report["checked_conditions"]["guarded_candidate_bundle_unchanged"] is True
    assert report["checked_conditions"]["guarded_candidate_staleness_active_or_accumulating"] is True
    assert report["guarded_candidate_staleness_summary"]["guarded_staleness_status"] in {
        "GUARDED_ACTIVE_REVIEW",
        "GUARDED_ACCUMULATING_FORWARD_LABELS",
    }
    assert report["checked_conditions"]["side_effects_absent"] is True
    assert report["outcome_refresh_summary"]["outcome_refresh_attempted"] is False
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False

    second = write_segmented_probability_shadow_soak_report(
        dataset_path=dataset_path,
        candidate_bundle_path=bundle_path,
        guarded_candidate_bundle_path=guarded_bundle_path,
        output_dir=tmp_path / "soak",
        accumulation_output_dir=tmp_path / "accumulation",
        forward_shadow_output_dir=tmp_path / "forward_shadow",
        candidate_staleness_output_dir=tmp_path / "staleness",
        guarded_candidate_staleness_output_dir=tmp_path / "guarded_staleness",
        ev_shadow_output_dir=tmp_path / "ev_shadow",
        guarded_shadow_output_dir=tmp_path / "guarded_shadow",
        readiness_output_dir=tmp_path / "readiness",
        outcome_refresh_source="skip",
        min_shadow_sample=20,
        min_forward_sample=100,
        min_candidate_sample=3,
        min_ev_sample=20,
        min_top_sample=5,
        min_shift_sample=100,
        max_candidate_age_days=30,
        expire_after_days=30,
        as_of="2026-04-02T04:00:00+00:00",
    )

    second_report = second["shadow_soak_report"]
    assert second_report["forward_sample_progress"]["new_true_forward_rows_since_previous_soak"] == 0
    assert second_report["guarded_forward_sample_progress"]["new_post_guarded_true_forward_rows_since_previous_soak"] == 0
    assert "no_new_true_forward_rows_since_previous_soak" in second_report["soak_reasons"]
    assert "no_new_post_guarded_true_forward_rows_since_previous_soak" in second_report["soak_reasons"]
    assert second["shadow_soak_history_run_count"] == 2


def test_shadow_soak_separates_original_and_guarded_forward_windows(tmp_path: Path):
    dataset_path = tmp_path / "signals.csv"
    bundle_path = tmp_path / "candidate_bundle.json"
    guarded_bundle_path = tmp_path / "guarded_candidate_bundle.json"
    _shadow_frame().to_csv(dataset_path, index=False)
    bundle_path.write_text(
        json.dumps(_candidate_bundle(generated_at="2026-03-31T04:00:00+00:00")),
        encoding="utf-8",
    )
    guarded_bundle_path.write_text(
        json.dumps(_candidate_bundle(generated_at="2026-04-02T04:00:00+00:00", guarded=True)),
        encoding="utf-8",
    )

    artifact = write_segmented_probability_shadow_soak_report(
        dataset_path=dataset_path,
        candidate_bundle_path=bundle_path,
        guarded_candidate_bundle_path=guarded_bundle_path,
        output_dir=tmp_path / "soak",
        accumulation_output_dir=tmp_path / "accumulation",
        forward_shadow_output_dir=tmp_path / "forward_shadow",
        candidate_staleness_output_dir=tmp_path / "staleness",
        guarded_candidate_staleness_output_dir=tmp_path / "guarded_staleness",
        ev_shadow_output_dir=tmp_path / "ev_shadow",
        guarded_shadow_output_dir=tmp_path / "guarded_shadow",
        readiness_output_dir=tmp_path / "readiness",
        outcome_refresh_source="skip",
        min_shadow_sample=20,
        min_forward_sample=100,
        min_candidate_sample=3,
        min_ev_sample=20,
        min_top_sample=5,
        min_shift_sample=100,
        max_candidate_age_days=30,
        expire_after_days=30,
        as_of="2026-04-02T04:00:00+00:00",
    )

    report = artifact["shadow_soak_report"]
    assert report["forward_sample_progress"]["strict_forward_row_count"] == 80
    assert report["guarded_forward_sample_progress"]["guarded_strict_forward_row_count"] == 0
    assert report["guarded_forward_sample_progress"]["rows_after_guarded_candidate"] == 0
    assert report["guarded_forward_sample_progress"]["forward_sample_gap"] == 100
    assert (
        report["guarded_candidate_staleness_summary"]["guarded_staleness_status"]
        == "GUARDED_ACCUMULATING_FORWARD_LABELS"
    )


def test_shadow_soak_prefers_guarded_staleness_for_guarded_bundle_review(tmp_path: Path):
    dataset_path = tmp_path / "signals.csv"
    bundle_path = tmp_path / "candidate_bundle.json"
    guarded_bundle_path = tmp_path / "guarded_candidate_bundle.json"
    accumulation_dir = tmp_path / "accumulation"
    accumulation_dir.mkdir()
    history_path = accumulation_dir / "segmented_probability_forward_shadow_history.csv"
    pd.DataFrame(
        [
            {
                "observed_at": "2026-04-01T04:00:00+00:00",
                "recommended_routing_policy": "recency_first",
                "accumulation_status": "HOLDOUT_REPLAY_PASS_PENDING_FORWARD_LABELS",
            },
            {
                "observed_at": "2026-04-01T05:00:00+00:00",
                "recommended_routing_policy": "regime_first",
                "accumulation_status": "HOLDOUT_REPLAY_PASS_PENDING_FORWARD_LABELS",
            },
        ]
    ).to_csv(history_path, index=False)
    _shadow_frame().to_csv(dataset_path, index=False)
    bundle_path.write_text(
        json.dumps(_candidate_bundle(generated_at="2026-03-31T04:00:00+00:00")),
        encoding="utf-8",
    )
    guarded_bundle_path.write_text(
        json.dumps(_candidate_bundle(generated_at="2026-04-02T04:00:00+00:00", guarded=True)),
        encoding="utf-8",
    )

    artifact = write_segmented_probability_shadow_soak_report(
        dataset_path=dataset_path,
        candidate_bundle_path=bundle_path,
        guarded_candidate_bundle_path=guarded_bundle_path,
        output_dir=tmp_path / "soak",
        accumulation_output_dir=accumulation_dir,
        forward_shadow_output_dir=tmp_path / "forward_shadow",
        candidate_staleness_output_dir=tmp_path / "staleness",
        guarded_candidate_staleness_output_dir=tmp_path / "guarded_staleness",
        ev_shadow_output_dir=tmp_path / "ev_shadow",
        guarded_shadow_output_dir=tmp_path / "guarded_shadow",
        readiness_output_dir=tmp_path / "readiness",
        outcome_refresh_source="skip",
        min_shadow_sample=20,
        min_forward_sample=100,
        min_candidate_sample=3,
        min_ev_sample=20,
        min_top_sample=5,
        min_shift_sample=100,
        policy_lookback_runs=10,
        max_candidate_age_days=30,
        expire_after_days=30,
        as_of="2026-04-02T04:00:00+00:00",
    )

    report = artifact["shadow_soak_report"]
    assert report["candidate_staleness_summary"]["staleness_status"] in {"STALE_WATCH", "SUPERSEDED"}
    assert (
        report["guarded_candidate_staleness_summary"]["guarded_staleness_status"]
        == "GUARDED_ACCUMULATING_FORWARD_LABELS"
    )
    assert report["soak_status"] == SOAK_ACCUMULATING_TRUE_FORWARD_LABELS
    assert "recommended_routing_policy_changed" not in report["soak_reasons"]
