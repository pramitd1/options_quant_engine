from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_guarded_candidate_staleness import (
    GUARDED_ACCUMULATING_FORWARD_LABELS,
    GUARDED_STALE_WATCH,
    GUARDED_SUPERSEDED,
    build_segmented_probability_guarded_candidate_staleness_report,
    write_segmented_probability_guarded_candidate_staleness_report,
)


def _guarded_bundle(generated_at: str = "2026-05-10T04:00:00+00:00") -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "bundle_variant": "guarded_ev_quarantine_plus_rank_guard",
        "generated_at": generated_at,
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "calibration_status": "GUARDED_CANDIDATE_BUNDLE_READY",
        "guarded_candidate_bundle_status": "GUARDED_CANDIDATE_BUNDLE_READY",
        "candidate_count": 1,
        "source_candidate_count": 2,
        "quarantined_candidate_count": 1,
        "quarantined_candidate_keys": ["regime_segment|direction|CALL"],
        "rank_preservation_policy": {
            "policy_name": "raw_rank_preservation_guard",
            "enabled_for_research_review": True,
            "governance_only": True,
        },
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


def _guarded_history_frame(*, changed: bool = False) -> pd.DataFrame:
    policies = ["recency_first", "regime_first"] if changed else ["recency_first", "recency_first"]
    return pd.DataFrame(
        [
            {
                "observed_at": f"2026-05-11T0{idx}:00:00+00:00",
                "guarded_candidate_bundle_path": "guarded_bundle.json",
                "guarded_candidate_generated_at": "2026-05-10T04:00:00+00:00",
                "guarded_validation_mode_used": "after_candidate_generated",
                "guarded_strict_forward_row_count": 30 + idx,
                "guarded_recommended_routing_policy": policy,
                "guarded_shadow_status": "GUARDED_SHADOW_VALIDATION_PASS",
                "soak_status": "SOAK_ACCUMULATING_TRUE_FORWARD_LABELS",
            }
            for idx, policy in enumerate(policies)
        ]
    )


def _guarded_frame(*, before_count: int = 80, after_count: int = 0) -> pd.DataFrame:
    guarded_ts = pd.Timestamp("2026-05-10T04:00:00+00:00")
    rows = []
    for idx in range(before_count):
        hit = 1 if idx % 2 == 0 else 0
        signal_ts = guarded_ts - pd.Timedelta(days=2) + pd.Timedelta(minutes=idx)
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
        hit = 1 if idx % 2 == 0 else 0
        rows.append(
            {
                "signal_id": f"post-{idx}",
                "signal_timestamp": (guarded_ts + pd.Timedelta(minutes=idx + 1)).isoformat(),
                "direction": "PUT",
                "macro_regime": "RANGE",
                "gamma_regime": "POSITIVE_GAMMA",
                "hybrid_move_probability": 0.60,
                "correct_60m": hit,
                "calibration_label": hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 15 if hit else -10,
                "primary_outcome_return_bps": 15 if hit else -10,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_guarded_staleness_accumulates_when_no_post_guarded_labels():
    report = build_segmented_probability_guarded_candidate_staleness_report(
        dataset=_guarded_frame(),
        guarded_candidate_bundle=_guarded_bundle(),
        guarded_history=pd.DataFrame(),
        dataset_path="signals.csv",
        guarded_candidate_bundle_path="guarded_bundle.json",
        guarded_history_path="soak_history.csv",
        as_of="2026-05-11T04:00:00+00:00",
    )

    assert report["guarded_staleness_status"] == GUARDED_ACCUMULATING_FORWARD_LABELS
    assert "guarded_forward_sample_below_minimum" in report["guarded_staleness_reasons"]
    assert report["guarded_routing_policy_stability"]["policy_stability_status"] == "NO_GUARDED_SOAK_HISTORY"
    assert report["checked_conditions"]["guarded_bundle_side_effect_flags_clean"] is True
    assert report["runtime_config_changed"] is False


def test_guarded_staleness_detects_true_forward_routing_policy_changes():
    report = build_segmented_probability_guarded_candidate_staleness_report(
        dataset=_guarded_frame(after_count=40),
        guarded_candidate_bundle=_guarded_bundle(),
        guarded_history=_guarded_history_frame(changed=True),
        guarded_candidate_bundle_path="guarded_bundle.json",
        min_forward_sample=20,
        min_shift_sample=100,
        min_policy_observations=2,
        as_of="2026-05-11T04:00:00+00:00",
    )

    assert report["guarded_staleness_status"] == GUARDED_STALE_WATCH
    assert "guarded_recommended_routing_policy_changed" in report["guarded_staleness_reasons"]
    assert report["guarded_routing_policy_stability"]["guarded_routing_policy_changed"] is True


def test_guarded_staleness_detects_superseded_bundle(tmp_path: Path):
    old_path = tmp_path / "old_guarded_candidate_bundle.json"
    new_path = tmp_path / "new_guarded_candidate_bundle.json"
    dataset_path = tmp_path / "signals.csv"
    old_path.write_text(
        json.dumps(_guarded_bundle("2026-05-10T04:00:00+00:00")),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps(_guarded_bundle("2026-05-12T04:00:00+00:00")),
        encoding="utf-8",
    )
    _guarded_frame().to_csv(dataset_path, index=False)

    artifact = write_segmented_probability_guarded_candidate_staleness_report(
        dataset_path=dataset_path,
        guarded_candidate_bundle_path=old_path,
        guarded_candidate_bundle_search_dir=tmp_path,
        guarded_history_path=None,
        output_dir=tmp_path / "out",
        as_of="2026-05-13T04:00:00+00:00",
    )

    assert Path(artifact["guarded_staleness_json_path"]).exists()
    assert Path(artifact["guarded_staleness_markdown_path"]).exists()
    assert artifact["guarded_staleness_report"]["guarded_staleness_status"] == GUARDED_SUPERSEDED
    supersession = artifact["guarded_staleness_report"]["supersession"]
    assert supersession["newer_guarded_candidate_bundle_path"] == str(new_path)
