from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.segmented_probability_guarded_ev_experiment import (
    GUARDED_EV_EXPERIMENT_PASS,
    VARIANT_QUARANTINE_PLUS_RANK_GUARD,
    build_segmented_probability_guarded_ev_experiment_report,
    write_segmented_probability_guarded_ev_experiment_report,
)


def _routes() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-04-01T09:20:00+05:30")
    for idx in range(12):
        raw_top = idx < 3
        shadow_bad_top = 3 <= idx < 6
        return_bps = -30.0 if shadow_bad_top else (35.0 if raw_top else 3.0)
        mae_bps = -20.0 if shadow_bad_top else -8.0
        rows.append(
            {
                "route_policy": "regime_first",
                "row_index": idx,
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx * 5)).isoformat(),
                "direction": "PUT" if shadow_bad_top else "CALL",
                "raw_probability": 0.95 - (idx * 0.01) if raw_top else 0.50 - (idx * 0.01),
                "shadow_probability": 0.95 - (idx * 0.01) if shadow_bad_top else 0.40 - (idx * 0.01),
                "label": 0.0 if shadow_bad_top else 1.0,
                "assigned_candidate_key": (
                    "regime_segment:direction=PUT"
                    if shadow_bad_top
                    else "recency_window:train_recency_window=last_25_pct_train"
                ),
                "assigned_candidate_type": "regime_segment" if shadow_bad_top else "recency_window",
                "assigned_segment_field": "direction" if shadow_bad_top else "train_recency_window",
                "assigned_segment_value": "PUT" if shadow_bad_top else "last_25_pct_train",
                "assigned_calibrator": "linear_shrink" if shadow_bad_top else "temperature_score",
                "matched_candidate_count": 1,
                "signed_return_60m_bps": return_bps,
                "mfe_60m_bps": 10.0 if shadow_bad_top else 45.0,
                "mae_60m_bps": mae_bps,
                "selected_option_ba_spread_pct": 0.25,
                "_return_bps": return_bps,
                "_risk_adjusted_return_bps": return_bps - (abs(mae_bps) * 0.25),
                "_liquidity_adjusted_return_bps": return_bps - (abs(mae_bps) * 0.25) - 0.5,
            }
        )
    return pd.DataFrame(rows)


def _ev_shadow_report() -> dict:
    return {
        "report_type": "segmented_probability_ev_shadow_evaluation",
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "unit.csv",
        "candidate_bundle_path": "bundle.json",
        "probability_field": "hybrid_move_probability",
        "label_field": "correct_60m",
        "return_field": "signed_return_60m_bps",
        "top_fraction": 0.25,
        "train_fraction": 0.70,
        "selection_summary": {
            "recommended_routing_policy": "regime_first",
        },
        "validation_window": {
            "validation_mode_used": "holdout_replay",
        },
    }


def _attribution_report() -> dict:
    return {
        "report_type": "segmented_probability_ev_rejection_attribution",
        "attribution_status": "EV_REJECTION_ATTRIBUTION_ACTIONABLE",
        "analysis_policy": "regime_first",
        "negative_route_candidates": [
            {
                "candidate_key": "regime_segment:direction=PUT",
                "sample_count": 3,
                "avg_risk_adjusted_return_bps": -35.0,
            }
        ],
    }


def test_guarded_ev_experiment_repairs_shadow_top_with_rank_guard_and_quarantine():
    report = build_segmented_probability_guarded_ev_experiment_report(
        _ev_shadow_report(),
        _attribution_report(),
        _routes(),
        top_fraction=0.25,
        min_top_sample=3,
        raw_rank_ceiling_multiplier=1.0,
    )

    assert report["report_type"] == "segmented_probability_guarded_ev_experiment"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["guarded_ev_status"] == GUARDED_EV_EXPERIMENT_PASS
    assert report["selection_summary"]["recommended_guarded_variant"] == VARIANT_QUARANTINE_PLUS_RANK_GUARD
    assert report["selection_summary"]["recommended_variant_risk_delta_improvement_vs_baseline_bps"] > 0
    variants = {row["variant_name"]: row for row in report["variant_results"]}
    assert variants["baseline_shadow"]["variant_status"] == "GUARDED_EV_EXPERIMENT_REJECTED"
    assert variants[VARIANT_QUARANTINE_PLUS_RANK_GUARD]["variant_status"] == GUARDED_EV_EXPERIMENT_PASS
    assert variants[VARIANT_QUARANTINE_PLUS_RANK_GUARD]["ev_negative_route_top_rate"] == 0.0


def test_guarded_ev_experiment_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_segmented_probability_guarded_ev_experiment_report(
        _ev_shadow_report(),
        _attribution_report(),
        _routes(),
        top_fraction=0.25,
        min_top_sample=3,
        output_dir=tmp_path,
        report_name="unit_segmented_probability_guarded_ev_experiment",
    )

    for key in [
        "json_path",
        "markdown_path",
        "variants_csv_path",
        "candidates_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_variants_csv_path",
        "latest_candidates_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["guarded_ev_status"] == GUARDED_EV_EXPERIMENT_PASS
