from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_post_promotion_monitor import (
    POST_PROMOTION_DETERIORATING,
    POST_PROMOTION_HEALTHY,
    POST_PROMOTION_SKIPPED_NO_APPROVAL,
    build_threshold_post_promotion_monitor_report,
    write_threshold_post_promotion_monitor_report,
)


def _promotion_package() -> dict:
    return {
        "report_type": "threshold_promotion_review",
        "promotion_review_status": "PROMOTION_REVIEW_READY",
        "dataset_path": "unit.csv",
        "runtime_config_changed": False,
        "promotion_candidate": {
            "source_candidate_key": "composite_signal_score>=75.0",
            "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
            "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
            "research_only": True,
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


def _approval_decision() -> dict:
    return {
        "reviewed_at": "2026-02-01T00:00:00Z",
        "report_json": "promotion.json",
        "review_action": "APPROVED",
        "reviewer": "unit-test",
        "candidate_key": "composite_signal_score>=75.0",
        "threshold_field": "composite_signal_score",
        "threshold_value": 75.0,
        "runtime_config_changed": False,
    }


def _post_frame(days: int = 60, *, deteriorating: bool = False) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-02-02 09:20:00+05:30")
    for idx in range(days):
        high_score = idx >= 20
        if deteriorating:
            hit = 0.0 if high_score else 1.0
            ret = -10.0 if high_score else 8.0
        else:
            hit = 1.0 if high_score else 0.0
            ret = 23.0 if high_score else -7.0
        rows.append(
            {
                "signal_id": f"post-{idx}",
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
                "correct_60m": hit,
                "signed_return_60m_bps": ret,
                "calibration_label": hit,
                "calibration_label_available": True,
                "primary_outcome_return_bps": ret,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_post_promotion_monitor_skips_without_approved_ledger(tmp_path: Path):
    report = build_threshold_post_promotion_monitor_report(
        _post_frame(),
        promotion_package_report=_promotion_package(),
        ledger_path=tmp_path / "missing_ledger.csv",
    )

    assert report["monitor_status"] == POST_PROMOTION_SKIPPED_NO_APPROVAL
    assert report["runtime_config_changed"] is False


def test_post_promotion_monitor_marks_healthy_when_post_approval_matches_shadow():
    report = build_threshold_post_promotion_monitor_report(
        _post_frame(),
        promotion_package_report=_promotion_package(),
        approval_decision=_approval_decision(),
        dataset_path="unit.csv",
    )

    assert report["monitor_status"] == POST_PROMOTION_HEALTHY
    assert report["runtime_config_changed"] is False
    assert report["post_approval_impact"]["retained_signal_count"] == 40
    assert report["post_approval_impact"]["false_positive_removed_count"] == 20
    assert report["drift_from_shadow_expectation"]["retained_avg_return_delta_bps_vs_shadow"] == -1.0


def test_post_promotion_monitor_marks_deteriorating_when_retained_edge_collapses():
    report = build_threshold_post_promotion_monitor_report(
        _post_frame(deteriorating=True),
        promotion_package_report=_promotion_package(),
        approval_decision=_approval_decision(),
        dataset_path="unit.csv",
    )

    assert report["monitor_status"] == POST_PROMOTION_DETERIORATING
    assert "deterioration guardrails" in report["monitor_reasons"][0]


def test_post_promotion_monitor_writer_outputs_latest_artifacts(tmp_path: Path):
    artifact = write_threshold_post_promotion_monitor_report(
        _post_frame(),
        promotion_package_report=_promotion_package(),
        approval_decision=_approval_decision(),
        dataset_path="unit.csv",
        output_dir=tmp_path,
        report_name="unit_threshold_post_promotion_monitor",
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
    assert payload["monitor_status"] == POST_PROMOTION_HEALTHY
