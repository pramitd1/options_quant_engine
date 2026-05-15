from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_ev_rejection_attribution import (
    EV_REJECTION_ATTRIBUTION_ACTIONABLE,
    EV_REJECTION_ATTRIBUTION_NOT_REJECTED,
    build_segmented_probability_ev_rejection_attribution_report,
    write_segmented_probability_ev_rejection_attribution_report,
)


def _route_rows(*, shadow_promotes_bad_puts: bool = True) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    for idx in range(12):
        raw_top = idx < 3
        shadow_top = 3 <= idx < 6 if shadow_promotes_bad_puts else idx < 3
        is_bad_put = 3 <= idx < 6
        return_bps = -30.0 if is_bad_put else (35.0 if raw_top else 5.0)
        mae_bps = -20.0 if is_bad_put else -8.0
        rows.append(
            {
                "route_policy": "regime_first",
                "row_index": idx,
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 5)).isoformat(),
                "direction": "PUT" if is_bad_put else "CALL",
                "raw_probability": 0.95 - (idx * 0.01) if raw_top else 0.50 - (idx * 0.01),
                "shadow_probability": 0.95 - (idx * 0.01) if shadow_top else 0.40 - (idx * 0.01),
                "label": 0.0 if is_bad_put else 1.0,
                "assigned_candidate_key": (
                    "regime_segment:direction=PUT"
                    if is_bad_put
                    else "recency_window:train_recency_window=last_25_pct_train"
                ),
                "assigned_candidate_type": "regime_segment" if is_bad_put else "recency_window",
                "assigned_segment_field": "direction" if is_bad_put else "train_recency_window",
                "assigned_segment_value": "PUT" if is_bad_put else "last_25_pct_train",
                "assigned_calibrator": "linear_shrink" if is_bad_put else "temperature_score",
                "matched_candidate_count": 1,
                "signed_return_60m_bps": return_bps,
                "mfe_60m_bps": 10.0 if is_bad_put else 45.0,
                "mae_60m_bps": mae_bps,
                "selected_option_ba_spread_pct": 0.25,
                "selected_option_volume": 1200,
                "selected_option_open_interest": 5000,
                "macro_regime": "RISK_OFF" if is_bad_put else "RISK_ON",
                "gamma_regime": "NEGATIVE_GAMMA" if is_bad_put else "POSITIVE_GAMMA",
                "volatility_regime": "ELEVATED_VOL" if is_bad_put else "NORMAL_VOL",
                "_return_bps": return_bps,
                "_risk_adjusted_return_bps": return_bps - (abs(mae_bps) * 0.25),
                "_liquidity_adjusted_return_bps": return_bps - (abs(mae_bps) * 0.25) - 0.5,
            }
        )
    return pd.DataFrame(rows)


def _ev_shadow_report(*, status: str = "EV_SHADOW_EVALUATION_REJECTED") -> dict:
    candidate_route_results = []
    if status == "EV_SHADOW_EVALUATION_REJECTED":
        candidate_route_results = [
            {
                "route_policy": "regime_first",
                "candidate_key": "regime_segment:direction=PUT",
                "ev_route_status": "EV_ROUTE_NEGATIVE",
                "sample_count": 3,
                "avg_risk_adjusted_return_bps": -35.0,
                "hit_rate": 0.0,
                "assigned_candidate_type": "regime_segment",
                "assigned_segment_field": "direction",
                "assigned_segment_value": "PUT",
                "assigned_calibrator": "linear_shrink",
            }
        ]
    return {
        "report_type": "segmented_probability_ev_shadow_evaluation",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "top_fraction": 0.25,
        "ev_shadow_status": status,
        "selection_summary": {
            "recommended_routing_policy": "regime_first",
            "evaluated_routing_policy_count": 1,
        },
        "policy_results": [
            {
                "route_policy": "regime_first",
                "ev_shadow_status": status,
                "status_reason": "shadow_top_bucket_worsened_risk_adjusted_return",
                "policy_score": -20.0,
                "shadow_vs_raw_top_risk_adjusted_return_delta_bps": -50.0,
                "shadow_vs_raw_top_hit_rate_delta": -1.0,
                "shadow_top_vs_bottom_risk_adjusted_return_spread_bps": -10.0,
                "liquidity_status": "OK",
            }
        ],
        "candidate_route_results": candidate_route_results,
    }


def test_ev_rejection_attribution_finds_damaging_shadow_routes():
    report = build_segmented_probability_ev_rejection_attribution_report(
        _ev_shadow_report(),
        _route_rows(),
        ev_shadow_report_path="ev_shadow.json",
        ev_shadow_routes_path="routes.csv",
        top_fraction=0.25,
        min_bucket_sample=3,
        min_candidate_sample=2,
        min_regime_sample=2,
    )

    assert report["report_type"] == "segmented_probability_ev_rejection_attribution"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["attribution_status"] == EV_REJECTION_ATTRIBUTION_ACTIONABLE
    assert "shadow_top_worse_than_raw_top" in report["attribution_reasons"]
    assert "shadow_only_rows_negative_ev" in report["attribution_reasons"]
    assert report["candidate_attribution"][0]["candidate_key"] == "regime_segment:direction=PUT"
    assert report["candidate_attribution"][0]["attribution_status"] == "PRUNE_CANDIDATE_ROUTE"
    assert report["negative_route_candidates"][0]["candidate_key"] == "regime_segment:direction=PUT"
    assert report["routing_diagnostics"]["likely_failure_mode"] == "DAMAGING_CANDIDATE_ROUTE_PROMOTED_WEAK_ROWS"


def test_ev_rejection_attribution_marks_non_rejected_shadow_as_clean():
    report = build_segmented_probability_ev_rejection_attribution_report(
        _ev_shadow_report(status="EV_SHADOW_EVALUATION_PASS"),
        _route_rows(shadow_promotes_bad_puts=False),
        top_fraction=0.25,
        min_bucket_sample=3,
        min_candidate_sample=2,
        min_regime_sample=2,
    )

    assert report["attribution_status"] == EV_REJECTION_ATTRIBUTION_NOT_REJECTED
    assert report["rejection_summary"]["top_bucket_risk_adjusted_return_delta_bps"] == 0.0
    assert report["routing_diagnostics"]["likely_failure_mode"] == "NO_REJECTION_ATTRIBUTED"


def test_ev_rejection_attribution_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_ev_rejection_attribution_report(
        _ev_shadow_report(),
        _route_rows(),
        ev_shadow_report_path="ev_shadow.json",
        ev_shadow_routes_path="routes.csv",
        top_fraction=0.25,
        min_bucket_sample=3,
        min_candidate_sample=2,
        min_regime_sample=2,
        output_dir=tmp_path,
        report_name="unit_segmented_probability_ev_rejection_attribution",
    )

    for key in [
        "json_path",
        "markdown_path",
        "candidates_csv_path",
        "shadow_only_candidates_csv_path",
        "regimes_csv_path",
        "policies_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_candidates_csv_path",
        "latest_shadow_only_candidates_csv_path",
        "latest_regimes_csv_path",
        "latest_policies_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["attribution_status"] == EV_REJECTION_ATTRIBUTION_ACTIONABLE
