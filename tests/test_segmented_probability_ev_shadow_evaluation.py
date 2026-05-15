from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_ev_shadow_evaluation import (
    EV_SHADOW_NEEDS_MORE_DATA,
    EV_SHADOW_PASS,
    build_segmented_probability_ev_shadow_evaluation_report,
    write_segmented_probability_ev_shadow_evaluation_report,
)


def _ev_shadow_frame(row_count: int = 80) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    rows = []
    for idx in range(row_count):
        is_put = idx % 2 == 0
        hit = 1 if is_put else 0
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 15)).isoformat(),
                "direction": "PUT" if is_put else "CALL",
                "macro_regime": "RISK_OFF" if is_put else "RISK_ON",
                "gamma_regime": "NEGATIVE_GAMMA" if is_put else "POSITIVE_GAMMA",
                "volatility_regime": "ELEVATED_VOL" if is_put else "NORMAL_VOL",
                "hybrid_move_probability": 0.35 if is_put else 0.90,
                "correct_60m": hit,
                "calibration_label": hit,
                "calibration_label_available": True,
                "signed_return_60m_bps": 40.0 if is_put else -20.0,
                "primary_outcome_return_bps": 40.0 if is_put else -20.0,
                "mfe_60m_bps": 65.0 if is_put else 12.0,
                "mae_60m_bps": -10.0 if is_put else -25.0,
                "selected_option_ba_spread_pct": 1.0,
                "selected_option_volume": 1200 if is_put else 800,
                "selected_option_open_interest": 6000 if is_put else 4500,
                "selected_option_iv": 18.0,
                "option_chain_is_valid": True,
                "option_chain_is_stale": False,
                "option_chain_validation_status": "VALID",
                "market_data_provenance_status": "OK",
                "data_quality_score": 95,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def _candidate_bundle() -> dict:
    return {
        "artifact_type": "segmented_probability_calibration_candidate_bundle",
        "generated_at": "2026-05-10T04:00:00+00:00",
        "research_only": True,
        "approval_required_for_runtime_use": True,
        "candidate_count": 1,
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
                    "base_rate": 0.95,
                },
            }
        ],
    }


def test_ev_shadow_evaluation_scores_shadow_top_bucket_payoff():
    report = build_segmented_probability_ev_shadow_evaluation_report(
        _ev_shadow_frame(),
        candidate_bundle=_candidate_bundle(),
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        routing_policies=("candidate_priority",),
        top_fraction=0.25,
        min_ev_sample=20,
        min_top_sample=4,
        min_candidate_sample=5,
        min_regime_sample=3,
        min_risk_adjusted_improvement_bps=2.0,
    )

    assert report["report_type"] == "segmented_probability_ev_shadow_evaluation"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["ev_shadow_status"] == EV_SHADOW_PASS
    assert report["selection_summary"]["recommended_routing_policy"] == "candidate_priority"
    policy = report["policy_results"][0]
    assert policy["ev_shadow_status"] == EV_SHADOW_PASS
    assert policy["shadow_vs_raw_top_risk_adjusted_return_delta_bps"] > 0
    assert policy["shadow_vs_raw_top_hit_rate_delta"] > 0
    assert report["candidate_route_results"][0]["ev_route_status"] == "EV_ROUTE_HELPFUL"
    assert report["regime_payoff_results"]


def test_ev_shadow_evaluation_respects_sample_guardrails():
    report = build_segmented_probability_ev_shadow_evaluation_report(
        _ev_shadow_frame(20),
        candidate_bundle=_candidate_bundle(),
        validation_mode="holdout_replay",
        routing_policies=("candidate_priority",),
        min_ev_sample=100,
        min_top_sample=25,
    )

    assert report["ev_shadow_status"] == EV_SHADOW_NEEDS_MORE_DATA
    assert report["policy_results"][0]["status_reason"] == "sample_size_guardrail_failed"


def test_ev_shadow_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_ev_shadow_evaluation_report(
        _ev_shadow_frame(),
        candidate_bundle=_candidate_bundle(),
        candidate_bundle_path="unit_bundle.json",
        dataset_path="unit.csv",
        validation_mode="holdout_replay",
        routing_policies=("candidate_priority",),
        top_fraction=0.25,
        min_ev_sample=20,
        min_top_sample=4,
        min_candidate_sample=5,
        min_regime_sample=3,
        output_dir=tmp_path,
        report_name="unit_segmented_probability_ev_shadow",
    )

    for key in [
        "json_path",
        "markdown_path",
        "policies_csv_path",
        "candidates_csv_path",
        "regimes_csv_path",
        "routes_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_policies_csv_path",
        "latest_candidates_csv_path",
        "latest_regimes_csv_path",
        "latest_routes_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["selection_summary"]["recommended_policy_status"] == EV_SHADOW_PASS
