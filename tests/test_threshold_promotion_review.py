from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_promotion_review import (
    PROMOTION_REVIEW_READY,
    SKIPPED_SHADOW_REVIEW_NOT_READY,
    build_threshold_promotion_review_package,
    record_threshold_promotion_review_decision,
    write_threshold_promotion_review_package,
)


def _ready_shadow_review() -> dict:
    return {
        "report_type": "threshold_shadow_review",
        "review_status": "PROMOTION_READY",
        "shadow_status": "SHADOW_SIMULATION_READY",
        "policy_experiment_status": "APPROVED_FOR_POLICY_EXPERIMENT",
        "dataset_path": "unit.csv",
        "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
        "candidate_policy_pack": {
            "name": "candidate_threshold_composite_signal_score_75_0",
            "source_candidate_key": "composite_signal_score>=75.0",
            "source_governance_status": "PROMOTE_TO_REVIEW",
            "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
            "overrides": {"evaluation_thresholds.selection.composite_signal_score_floor": 75.0},
            "research_only": True,
            "runtime_config_changed": False,
        },
        "observation_summary": {"distinct_signal_dates": 60, "calendar_span_days": 60},
        "impact_summary": {
            "eligible_signal_count": 100,
            "retained_signal_count": 70,
            "suppressed_signal_count": 30,
            "true_positive_lost_count": 0,
        },
        "guardrail_summary": {
            "false_positive_removed_count": 25,
            "false_positive_removal_rate": 0.83,
            "true_positive_loss_rate": 0.0,
            "bad_regime_count": 0,
            "bad_bucket_count": 0,
        },
        "baseline_metrics": {"hit_rate_60m": 0.55, "avg_signed_return_60m_bps": 4.0},
        "retained_metrics": {"hit_rate_60m": 0.72, "avg_signed_return_60m_bps": 13.0},
        "suppressed_metrics": {"hit_rate_60m": 0.2, "avg_signed_return_60m_bps": -5.0},
        "retained_vs_baseline_delta": {"hit_rate_delta": 0.17, "avg_return_delta_bps": 9.0},
        "segment_failures": [],
    }


def test_promotion_review_package_is_ready_only_after_promotion_ready_shadow_review():
    report = build_threshold_promotion_review_package(_ready_shadow_review(), shadow_review_report_path="shadow_review.json")

    assert report["promotion_review_status"] == PROMOTION_REVIEW_READY
    assert report["manual_review_required"] is True
    assert report["runtime_config_changed"] is False
    assert report["status_chain"]["governance_status"] == "PROMOTE_TO_REVIEW"
    assert report["promotion_candidate"]["threshold_rule"]["field"] == "composite_signal_score"
    assert report["promotion_candidate"]["overrides"] == {
        "evaluation_thresholds.selection.composite_signal_score_floor": 75.0,
    }
    assert report["monitoring_plan"]
    assert report["rollback_notes"]


def test_promotion_review_package_skips_when_shadow_review_not_ready():
    report = build_threshold_promotion_review_package({"review_status": "NEEDS_MORE_SHADOW_DATA"})

    assert report["promotion_review_status"] == SKIPPED_SHADOW_REVIEW_NOT_READY
    assert report["manual_review_required"] is False
    assert report["runtime_config_changed"] is False


def test_promotion_review_writer_outputs_latest_package(tmp_path: Path):
    artifact = write_threshold_promotion_review_package(
        _ready_shadow_review(),
        shadow_review_report_path="shadow_review.json",
        output_dir=tmp_path,
        report_name="unit_threshold_promotion_review",
    )

    for key in ["json_path", "markdown_path", "latest_json_path", "latest_markdown_path", "review_ledger_path"]:
        if key == "review_ledger_path":
            assert Path(artifact[key]).parent.exists()
        else:
            assert Path(artifact[key]).exists()

    payload = json.loads(Path(artifact["latest_json_path"]).read_text(encoding="utf-8"))
    assert payload["promotion_review_status"] == PROMOTION_REVIEW_READY


def test_promotion_review_decision_appends_locked_ledger(tmp_path: Path):
    artifact = write_threshold_promotion_review_package(
        _ready_shadow_review(),
        output_dir=tmp_path,
        report_name="unit_threshold_promotion_review",
    )

    decision = record_threshold_promotion_review_decision(
        report_json_path=artifact["latest_json_path"],
        review_action="APPROVED",
        reviewer="unit-test",
        review_note="Looks ready for manual config review.",
    )

    ledger = pd.read_csv(decision["review_ledger_path"])
    assert len(ledger) == 1
    assert ledger.loc[0, "review_action"] == "APPROVED"
    assert ledger.loc[0, "reviewer"] == "unit-test"
    assert bool(ledger.loc[0, "runtime_config_changed"]) is False
