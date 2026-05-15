from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_guarded_shadow_validation import (
    GUARDED_SHADOW_VALIDATION_PASS,
    build_segmented_probability_guarded_shadow_validation_report,
    write_segmented_probability_guarded_shadow_validation_report,
)


def _guarded_frame(row_count: int = 40) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    rows = []
    for idx in range(row_count):
        bucket = idx % 10
        raw_top = bucket in {0, 1, 2}
        shadow_bad_top = bucket in {3, 4, 5}
        probability = 0.95 if raw_top else (0.50 if shadow_bad_top else 0.40)
        return_bps = 35.0 if raw_top else (-30.0 if shadow_bad_top else 2.0)
        mae_bps = -8.0 if raw_top else (-20.0 if shadow_bad_top else -5.0)
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 5)).isoformat(),
                "direction": "PUT" if shadow_bad_top else "CALL",
                "macro_regime": "RISK_OFF" if shadow_bad_top else "RISK_ON",
                "gamma_regime": "NEGATIVE_GAMMA" if shadow_bad_top else "POSITIVE_GAMMA",
                "volatility_regime": "NORMAL_VOL",
                "hybrid_move_probability": probability,
                "correct_60m": 1,
                "calibration_label": 1,
                "calibration_label_available": True,
                "signed_return_60m_bps": return_bps,
                "primary_outcome_return_bps": return_bps,
                "mfe_60m_bps": 40.0 if raw_top else 8.0,
                "mae_60m_bps": mae_bps,
                "selected_option_ba_spread_pct": 0.25,
                "selected_option_volume": 1000,
                "selected_option_open_interest": 5000,
                "option_chain_is_valid": True,
                "option_chain_is_stale": False,
                "option_chain_validation_status": "VALID",
                "market_data_provenance_status": "OK",
                "data_quality_score": 95,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def _guarded_bundle() -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "bundle_variant": "guarded_ev_quarantine_plus_rank_guard",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "candidate_count": 1,
        "quarantined_candidate_keys": [
            "regime_segment:direction=PUT",
            "regime_segment:gamma_regime=NEGATIVE_GAMMA",
        ],
        "rank_preservation_policy": {
            "governance_only": True,
            "runtime_behavior_changed": False,
            "requires_guard_aware_shadow_evaluation": True,
            "top_fraction": 0.25,
            "raw_rank_ceiling_multiplier": 1.0,
        },
        "candidates": [
            {
                "candidate_priority": 1,
                "candidate_type": "recency_window",
                "segment_field": "train_recency_window",
                "segment_value": "last_25_pct_train",
                "selected_calibrator": "isotonic_score",
                "state": {
                    "method": "isotonic_score",
                    "calibration_mapping": {
                        "0": 0.20,
                        "50": 0.99,
                        "95": 0.90,
                    },
                },
            }
        ],
    }


def test_guarded_shadow_validation_applies_rank_preservation_to_ev_top_bucket():
    report = build_segmented_probability_guarded_shadow_validation_report(
        _guarded_frame(),
        candidate_bundle=_guarded_bundle(),
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        routing_policies=("candidate_priority",),
        train_fraction=0.10,
        min_shadow_sample=10,
        min_ev_sample=10,
        min_top_sample=5,
        min_brier_improvement=0.001,
    )

    assert report["report_type"] == "segmented_probability_guarded_shadow_validation"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["guarded_shadow_status"] == GUARDED_SHADOW_VALIDATION_PASS
    policy = report["policy_results"][0]
    assert policy["guarded_policy_status"] == GUARDED_SHADOW_VALIDATION_PASS
    assert policy["raw_rank_guard_applied"] is True
    assert policy["raw_rank_capped_count"] > 0
    assert policy["guarded_top_raw_top_overlap_rate"] == 1.0
    assert policy["guarded_vs_raw_top_risk_adjusted_return_delta_bps"] == 0.0
    assert policy["quarantined_route_top_count"] == 0


def test_guarded_shadow_validation_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_guarded_shadow_validation_report(
        _guarded_frame(),
        candidate_bundle=_guarded_bundle(),
        candidate_bundle_path="guarded_bundle.json",
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        routing_policies=("candidate_priority",),
        train_fraction=0.10,
        min_shadow_sample=10,
        min_ev_sample=10,
        min_top_sample=5,
        min_brier_improvement=0.001,
        output_dir=tmp_path,
        report_name="unit_guarded_shadow_validation",
    )

    for key in [
        "json_path",
        "markdown_path",
        "policies_csv_path",
        "candidates_csv_path",
        "routes_csv_path",
        "calibration_curve_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_policies_csv_path",
        "latest_candidates_csv_path",
        "latest_routes_csv_path",
        "latest_calibration_curve_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["guarded_shadow_status"] == GUARDED_SHADOW_VALIDATION_PASS
