from __future__ import annotations

import json
from pathlib import Path

from research.signal_evaluation.threshold_shadow_review import (
    PROMOTION_READY,
    REJECTED_REGIME_DEGRADATION,
    REJECTED_TRUE_POSITIVE_LOSS,
    SKIPPED_SHADOW_NOT_READY,
    build_threshold_shadow_review_report,
    write_threshold_shadow_review_report,
    write_threshold_shadow_review_skip,
)
from research.signal_evaluation.threshold_shadow_simulation import SHADOW_SIMULATION_READY


def _signal_records(*, retained: bool, count: int, start_idx: int = 0) -> list[dict]:
    base = "2026-01-01T09:20:00+05:30"
    import pandas as pd

    start = pd.Timestamp(base)
    decision = "RETAINED" if retained else "SUPPRESSED"
    classification = "TRUE_POSITIVE_RETAINED" if retained else "FALSE_POSITIVE_REMOVED"
    return [
        {
            "signal_id": f"{decision.lower()}-{idx}",
            "signal_timestamp": (start + pd.Timedelta(days=start_idx + idx)).isoformat(),
            "shadow_decision": decision,
            "correct_60m": 1.0 if retained else 0.0,
            "signed_return_60m_bps": 24.0 if retained else -8.0,
            "shadow_outcome_classification": classification,
        }
        for idx in range(count)
    ]


def _ready_shadow_report() -> dict:
    return {
        "report_type": "threshold_shadow_simulation",
        "shadow_status": SHADOW_SIMULATION_READY,
        "runtime_config_changed": False,
        "dataset_path": "unit.csv",
        "policy_experiment_status": "APPROVED_FOR_POLICY_EXPERIMENT",
        "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
        "candidate_policy_pack": {"research_only": True, "runtime_config_changed": False},
        "impact_summary": {
            "eligible_signal_count": 60,
            "retained_signal_count": 40,
            "suppressed_signal_count": 20,
            "suppressed_label_count_60m": 20,
            "false_positive_removed_count": 20,
            "true_positive_lost_count": 0,
            "false_positive_removal_rate": 1.0,
            "avoided_suppressed_return_bps": 160.0,
        },
        "baseline_metrics": {
            "signal_count": 60,
            "label_count_60m": 60,
            "hit_rate_60m": 0.667,
            "avg_signed_return_60m_bps": 13.3,
        },
        "retained_metrics": {
            "signal_count": 40,
            "label_count_60m": 40,
            "hit_rate_60m": 1.0,
            "avg_signed_return_60m_bps": 24.0,
        },
        "suppressed_metrics": {
            "signal_count": 20,
            "label_count_60m": 20,
            "hit_rate_60m": 0.0,
            "avg_signed_return_60m_bps": -8.0,
        },
        "retained_vs_baseline_delta": {
            "signal_count_delta": -20,
            "label_count_delta": -20,
            "hit_rate_delta": 0.333,
            "avg_return_delta_bps": 10.7,
        },
        "regime_shadow": [],
        "bucket_shadow": [],
        "retained_signal_records": _signal_records(retained=True, count=40, start_idx=0),
        "suppressed_signal_records": _signal_records(retained=False, count=20, start_idx=40),
    }


def test_shadow_review_promotes_ready_shadow_evidence():
    report = build_threshold_shadow_review_report(_ready_shadow_report(), shadow_simulation_report_path="shadow.json")

    assert report["review_status"] == PROMOTION_READY
    assert report["runtime_config_changed"] is False
    assert report["requires_manual_promotion_review"] is True
    assert report["guardrail_summary"]["true_positive_loss_rate"] == 0.0
    assert report["observation_summary"]["distinct_signal_dates"] == 60


def test_shadow_review_skips_when_shadow_simulation_is_not_ready():
    report = build_threshold_shadow_review_report({"shadow_status": "SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED"})

    assert report["review_status"] == SKIPPED_SHADOW_NOT_READY
    assert report["requires_manual_promotion_review"] is False
    assert "SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED" in report["review_reasons"][0]


def test_shadow_review_rejects_true_positive_loss():
    shadow = _ready_shadow_report()
    shadow["impact_summary"]["true_positive_lost_count"] = 2
    shadow["impact_summary"]["avg_suppressed_return_60m_bps"] = 6.0
    shadow["impact_summary"]["avoided_suppressed_return_bps"] = -120.0

    report = build_threshold_shadow_review_report(shadow)

    assert report["review_status"] == REJECTED_TRUE_POSITIVE_LOSS
    assert "True positives lost" in report["review_reasons"][0]


def test_shadow_review_rejects_regime_degradation():
    shadow = _ready_shadow_report()
    shadow["regime_shadow"] = [
        {
            "segment_kind": "regime",
            "segment_field": "macro_regime",
            "segment_value": "RISK_OFF",
            "retained_signal_count": 15,
            "suppressed_signal_count": 15,
            "false_positive_removed_count": 8,
            "true_positive_lost_count": 1,
            "hit_rate_delta": -0.2,
            "avg_return_delta_bps": -8.0,
            "suppressed_avg_return_60m_bps": 4.0,
        }
    ]

    report = build_threshold_shadow_review_report(shadow)

    assert report["review_status"] == REJECTED_REGIME_DEGRADATION
    assert report["segment_failures"]


def test_shadow_review_writer_outputs_latest_artifacts(tmp_path: Path):
    artifact = write_threshold_shadow_review_report(
        _ready_shadow_report(),
        shadow_simulation_report_path="shadow.json",
        output_dir=tmp_path,
        report_name="unit_threshold_shadow_review",
    )

    for key in [
        "json_path",
        "markdown_path",
        "segments_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_segments_csv_path",
    ]:
        assert Path(artifact[key]).exists()

    payload = json.loads(Path(artifact["latest_json_path"]).read_text(encoding="utf-8"))
    assert payload["review_status"] == PROMOTION_READY


def test_shadow_review_skip_writer_prevents_stale_latest(tmp_path: Path):
    artifact = write_threshold_shadow_review_skip(
        reason="Shadow simulation unavailable.",
        shadow_report={"shadow_status": "ERROR"},
        output_dir=tmp_path,
        report_name="unit_threshold_shadow_review_skip",
    )

    payload = json.loads(Path(artifact["latest_json_path"]).read_text(encoding="utf-8"))
    assert payload["review_status"] == SKIPPED_SHADOW_NOT_READY
    assert payload["runtime_config_changed"] is False
    assert Path(artifact["latest_markdown_path"]).exists()
