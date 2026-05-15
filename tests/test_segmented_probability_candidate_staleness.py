from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_candidate_staleness import (
    ACTIVE_REVIEW,
    EXPIRED,
    STALE_WATCH,
    SUPERSEDED,
    build_segmented_probability_candidate_staleness_report,
    write_segmented_probability_candidate_staleness_report,
)


def _candidate_bundle(generated_at: str = "2026-05-10T04:00:00+00:00") -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": generated_at,
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "calibration_status": "SEGMENTED_CALIBRATION_CANDIDATES_READY",
        "candidate_count": 1,
        "candidates": [
            {
                "candidate_priority": 1,
                "candidate_type": "recency_window",
                "segment_field": "train_recency_window",
                "segment_value": "last_25_pct_train",
                "selected_calibrator": "linear_shrink",
                "state": {"method": "linear_shrink", "alpha": 0.5, "base_rate": 0.52},
            }
        ],
    }


def _history_frame(*, changed: bool = False) -> pd.DataFrame:
    policies = ["recency_first", "regime_first"] if changed else ["recency_first", "recency_first"]
    return pd.DataFrame(
        [
            {
                "observed_at": f"2026-05-10T0{idx}:00:00+00:00",
                "recommended_routing_policy": policy,
                "accumulation_status": "HOLDOUT_REPLAY_PASS_PENDING_FORWARD_LABELS",
            }
            for idx, policy in enumerate(policies)
        ]
    )


def _staleness_frame(
    *,
    before_count: int = 80,
    after_count: int = 0,
    after_hit: int = 1,
    after_direction: str = "CALL",
) -> pd.DataFrame:
    candidate_ts = pd.Timestamp("2026-05-10T04:00:00+00:00")
    rows = []
    for idx in range(before_count):
        hit = 1 if idx % 2 == 0 else 0
        signal_ts = candidate_ts - pd.Timedelta(days=3) + pd.Timedelta(minutes=idx)
        rows.append(
            {
                "signal_id": f"pre-{idx}",
                "signal_timestamp": signal_ts.isoformat(),
                "direction": "PUT" if idx % 2 == 0 else "CALL",
                "macro_regime": "RANGE",
                "gamma_regime": "POSITIVE_GAMMA",
                "hybrid_move_probability": 0.55,
                "correct_60m": hit,
                "calibration_label": hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 10 if hit else -10,
                "primary_outcome_return_bps": 10 if hit else -10,
                "label_quality_status": "CLEAN",
            }
        )
    for idx in range(after_count):
        rows.append(
            {
                "signal_id": f"post-{idx}",
                "signal_timestamp": (candidate_ts + pd.Timedelta(minutes=idx + 1)).isoformat(),
                "direction": after_direction,
                "macro_regime": "TREND",
                "gamma_regime": "NEGATIVE_GAMMA",
                "hybrid_move_probability": 0.85,
                "correct_60m": after_hit,
                "calibration_label": after_hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 20 if after_hit else -20,
                "primary_outcome_return_bps": 20 if after_hit else -20,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_candidate_staleness_active_review_for_fresh_candidate():
    report = build_segmented_probability_candidate_staleness_report(
        dataset=_staleness_frame(),
        candidate_bundle=_candidate_bundle(),
        history=_history_frame(),
        dataset_path="unit.csv",
        candidate_bundle_path="bundle.json",
        candidate_bundle_search_dir=".",
        history_path="history.csv",
        as_of="2026-05-11T04:00:00+00:00",
    )

    assert report["staleness_status"] == ACTIVE_REVIEW
    assert report["staleness_reasons"] == []
    assert report["checked_conditions"]["candidate_count_positive"] is True
    assert report["dataset_currency"]["dataset_currency_status"] == "CURRENT_AT_CANDIDATE_GENERATION"


def test_candidate_staleness_expires_old_candidate():
    report = build_segmented_probability_candidate_staleness_report(
        dataset=_staleness_frame(),
        candidate_bundle=_candidate_bundle(),
        history=_history_frame(),
        expire_after_days=14,
        as_of="2026-05-30T04:00:00+00:00",
    )

    assert report["staleness_status"] == EXPIRED
    assert "candidate_age_exceeds_expiry_window" in report["staleness_reasons"]


def test_candidate_staleness_detects_population_shift_and_material_new_labels():
    report = build_segmented_probability_candidate_staleness_report(
        dataset=_staleness_frame(after_count=25, after_hit=1, after_direction="CALL"),
        candidate_bundle=_candidate_bundle(),
        history=_history_frame(),
        min_shift_sample=20,
        max_new_labeled_rows_before_stale=20,
        as_of="2026-05-11T04:00:00+00:00",
    )

    assert report["staleness_status"] == STALE_WATCH
    assert "material_new_data_since_candidate_generation" in report["staleness_reasons"]
    assert "forward_label_population_shifted" in report["staleness_reasons"]
    assert report["forward_label_population_shift"]["shifted_materially"] is True


def test_candidate_staleness_detects_routing_policy_changes():
    report = build_segmented_probability_candidate_staleness_report(
        dataset=_staleness_frame(),
        candidate_bundle=_candidate_bundle(),
        history=_history_frame(changed=True),
        as_of="2026-05-11T04:00:00+00:00",
    )

    assert report["staleness_status"] == STALE_WATCH
    assert "recommended_routing_policy_changed" in report["staleness_reasons"]
    assert report["routing_policy_stability"]["routing_policy_changed"] is True


def test_candidate_staleness_detects_superseded_bundle(tmp_path: Path):
    old_path = tmp_path / "old_candidate_bundle.json"
    new_path = tmp_path / "new_candidate_bundle.json"
    dataset_path = tmp_path / "signals.csv"
    old_path.write_text(
        json.dumps(_candidate_bundle("2026-05-10T04:00:00+00:00")),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps(_candidate_bundle("2026-05-12T04:00:00+00:00")),
        encoding="utf-8",
    )
    _staleness_frame().to_csv(dataset_path, index=False)

    artifact = write_segmented_probability_candidate_staleness_report(
        dataset_path=dataset_path,
        candidate_bundle_path=old_path,
        candidate_bundle_search_dir=tmp_path,
        history_path=None,
        output_dir=tmp_path / "out",
        as_of="2026-05-13T04:00:00+00:00",
    )

    assert Path(artifact["staleness_json_path"]).exists()
    assert Path(artifact["staleness_markdown_path"]).exists()
    assert artifact["staleness_report"]["staleness_status"] == SUPERSEDED
    supersession = artifact["staleness_report"]["supersession"]
    assert supersession["newer_candidate_bundle_path"] == str(new_path)
