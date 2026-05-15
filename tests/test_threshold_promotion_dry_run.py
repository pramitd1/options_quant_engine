from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_promotion_dry_run import (
    PROMOTION_DRY_RUN_COMPLETE,
    PROMOTION_DRY_RUN_SKIPPED_PACKAGE_NOT_READY,
    run_threshold_promotion_dry_run,
)


def _promotion_package(value: float = 82.0) -> dict:
    return {
        "report_type": "threshold_promotion_review",
        "promotion_review_status": "PROMOTION_REVIEW_READY",
        "dataset_path": "unit.csv",
        "runtime_config_changed": False,
        "status_chain": {
            "governance_status": "PROMOTE_TO_REVIEW",
            "policy_experiment_status": "APPROVED_FOR_POLICY_EXPERIMENT",
            "shadow_simulation_status": "SHADOW_SIMULATION_READY",
            "shadow_review_status": "PROMOTION_READY",
        },
        "promotion_candidate": {
            "source_candidate_key": f"composite_signal_score>={value}",
            "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
            "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": value},
            "overrides": {"evaluation_thresholds.selection.composite_signal_score_floor": value},
            "runtime_config_changed": False,
        },
        "impact_summary": {
            "eligible_signal_count": 60,
            "retained_signal_count": 40,
            "suppressed_signal_count": 20,
            "false_positive_removed_count": 20,
            "true_positive_lost_count": 0,
            "false_positive_removal_rate": 1.0,
            "retention_ratio": 0.6667,
        },
        "retained_metrics": {
            "signal_count": 40,
            "label_count_60m": 40,
            "hit_rate_60m": 1.0,
            "avg_signed_return_60m_bps": 24.0,
        },
        "retained_vs_baseline_delta": {
            "hit_rate_delta": 0.3333,
            "avg_return_delta_bps": 10.7,
        },
    }


def _post_frame(days: int = 60) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-02-02 09:20:00+05:30")
    for idx in range(days):
        high_score = idx >= 20
        rows.append(
            {
                "signal_id": f"dry-run-{idx}",
                "signal_timestamp": (base + pd.Timedelta(days=idx)).isoformat(),
                "symbol": "NIFTY",
                "direction": "CALL" if idx % 2 == 0 else "PUT",
                "trade_status": "TRADE",
                "signal_regime": "EXPANSION_BIAS" if high_score else "CONFLICTED",
                "macro_regime": "RISK_ON" if idx % 3 else "RISK_OFF",
                "gamma_regime": "SHORT_GAMMA_ZONE" if idx % 2 else "LONG_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "global_risk_state": "CALM",
                "composite_signal_score": 82.0 if high_score else 52.0,
                "move_probability": 0.72 if high_score else 0.48,
                "ml_confidence_score": 0.74 if high_score else 0.42,
                "correct_60m": 1.0 if high_score else 0.0,
                "signed_return_60m_bps": 23.0 if high_score else -7.0,
                "calibration_label": 1.0 if high_score else 0.0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 23.0 if high_score else -7.0,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_promotion_dry_run_uses_sandbox_approval_without_touching_real_ledger(tmp_path: Path):
    real_ledger = tmp_path / "real_promotion_ledger.csv"
    artifact = run_threshold_promotion_dry_run(
        _post_frame(),
        promotion_package_report=_promotion_package(),
        promotion_package_report_path=tmp_path / "promotion.json",
        dataset_path="unit.csv",
        real_ledger_path=real_ledger,
        output_dir=tmp_path,
        report_name="unit_threshold_promotion_dry_run",
    )
    report = artifact["report"]

    assert report["dry_run_status"] == PROMOTION_DRY_RUN_COMPLETE
    assert report["runtime_config_changed"] is False
    assert report["real_promotion_ledger_changed"] is False
    assert not real_ledger.exists()
    assert Path(report["sandbox_ledger_path"]).exists()
    assert report["sandbox_approval_decision"]["review_action"] == "APPROVED"
    assert report["approval_timestamp_strategy"] == "before_first_signal_timestamp"
    assert report["post_promotion_monitor"]["monitor_status"] == "POST_PROMOTION_HEALTHY"
    assert report["adoption_reconciliation"]["adoption_status"] == "APPROVED_BUT_NOT_ADOPTED"

    for key in ["json_path", "markdown_path", "latest_json_path", "latest_markdown_path"]:
        assert Path(artifact[key]).exists()
    assert Path(report["post_promotion_monitor_artifact"]["latest_json_path"]).exists()
    assert Path(report["adoption_reconciliation_artifact"]["latest_json_path"]).exists()


def test_promotion_dry_run_skips_when_package_not_ready(tmp_path: Path):
    artifact = run_threshold_promotion_dry_run(
        _post_frame(),
        promotion_package_report={"promotion_review_status": "SKIPPED_SHADOW_REVIEW_NOT_READY"},
        dataset_path="unit.csv",
        output_dir=tmp_path,
        report_name="unit_threshold_promotion_dry_run",
    )
    report = artifact["report"]

    assert report["dry_run_status"] == PROMOTION_DRY_RUN_SKIPPED_PACKAGE_NOT_READY
    assert report["runtime_config_changed"] is False
    assert report["real_promotion_ledger_changed"] is False
    assert report["sandbox_approval_decision"] == {}
    assert Path(artifact["latest_json_path"]).exists()
