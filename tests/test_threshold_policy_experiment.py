from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_governance import PROMOTE_TO_REVIEW, build_threshold_governance_report
from research.signal_evaluation.threshold_policy_experiment import (
    APPROVED_FOR_POLICY_EXPERIMENT,
    INSUFFICIENT_EVIDENCE,
    REJECTED_FOR_POLICY_EXPERIMENT,
    SKIPPED_NO_PROMOTED_CANDIDATE,
    build_candidate_policy_pack,
    build_threshold_policy_experiment_report,
    write_threshold_policy_experiment_report,
    write_threshold_policy_experiment_skip,
)


def _experiment_frame(days: int = 120, *, candidate_good: bool = True) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01 09:20:00+05:30")
    for idx in range(days):
        high_score = idx >= 30
        if candidate_good:
            hit = 1.0 if high_score else 0.0
            ret = 24.0 if high_score else -8.0
        else:
            hit = 0.0 if high_score else 1.0
            ret = -18.0 if high_score else 16.0
        rows.append(
            {
                "signal_id": f"sig-{idx}",
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
                "tradeability_score": 78.0 if high_score else 45.0,
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


def _low_retention_experiment_frame(days: int = 120, rows_per_day: int = 12) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01 09:20:00+05:30")
    for day_idx in range(days):
        for row_idx in range(rows_per_day):
            idx = day_idx * rows_per_day + row_idx
            high_score = row_idx == 0
            rows.append(
                {
                    "signal_id": f"low-retention-{idx}",
                    "signal_timestamp": (base + pd.Timedelta(days=day_idx, minutes=row_idx)).isoformat(),
                    "symbol": "NIFTY",
                    "direction": "CALL" if idx % 2 == 0 else "PUT",
                    "trade_status": "TRADE",
                    "signal_regime": "EXPANSION_BIAS" if high_score else "CONFLICTED",
                    "macro_regime": "RISK_ON",
                    "gamma_regime": "SHORT_GAMMA_ZONE",
                    "volatility_regime": "NORMAL",
                    "global_risk_state": "CALM",
                    "composite_signal_score": 85.0 if high_score else 52.0,
                    "tradeability_score": 80.0 if high_score else 45.0,
                    "move_probability": 0.72 if high_score else 0.48,
                    "ml_confidence_score": 0.74 if high_score else 0.42,
                    "correct_60m": 1.0 if high_score else 0.0,
                    "signed_return_60m_bps": 24.0 if high_score else -8.0,
                    "calibration_label": 1.0 if high_score else 0.0,
                    "calibration_label_available": True,
                    "primary_outcome_return_bps": 24.0 if high_score else -8.0,
                    "label_quality_status": "CLEAN",
                }
            )
    return pd.DataFrame(rows)


def test_policy_experiment_approves_governed_candidate_for_research_sandbox():
    frame = _experiment_frame()
    governance = build_threshold_governance_report(frame, dataset_path="unit.csv")

    report = build_threshold_policy_experiment_report(
        frame,
        governance_report=governance,
        dataset_path="unit.csv",
        governance_report_path="governance.json",
    )

    assert report["experiment_status"] == APPROVED_FOR_POLICY_EXPERIMENT
    assert report["runtime_config_changed"] is False
    assert report["candidate_policy_pack"]["research_only"] is True
    assert report["candidate_policy_pack"]["runtime_config_changed"] is False
    assert report["candidate_policy_pack"]["overrides"]
    assert report["full_sample_comparison"]["delta"]["avg_return_delta_bps"] > 0
    assert report["full_sample_comparison"]["delta"]["hit_rate_delta"] > 0
    assert report["walk_forward_comparison"]["summary"]["robustness_status"] == "ROBUST"
    assert report["quality_bucket_comparison"]


def test_policy_experiment_treats_low_but_useful_retention_as_shadow_review_issue():
    candidate_review = {
        "candidate_key": "composite_signal_score>=85.0",
        "governance_status": PROMOTE_TO_REVIEW,
        "threshold_field": "composite_signal_score",
        "threshold_value": 85.0,
        "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
    }

    report = build_threshold_policy_experiment_report(
        _low_retention_experiment_frame(),
        candidate_review=candidate_review,
    )

    assert report["experiment_status"] == APPROVED_FOR_POLICY_EXPERIMENT
    assert report["full_sample_comparison"]["candidate"]["retention_ratio"] < 0.10
    assert any("below target" in reason for reason in report["experiment_reasons"])


def test_candidate_policy_pack_is_advisory_and_maps_config_hint():
    candidate_review = {
        "candidate_key": "composite_signal_score>=75.0",
        "governance_status": PROMOTE_TO_REVIEW,
        "threshold_field": "composite_signal_score",
        "threshold_value": 75.0,
        "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
    }

    policy_pack = build_candidate_policy_pack(candidate_review)

    assert policy_pack["research_only"] is True
    assert policy_pack["runtime_config_changed"] is False
    assert policy_pack["overrides"] == {
        "evaluation_thresholds.selection.composite_signal_score_floor": 75.0,
    }


def test_policy_experiment_rejects_candidate_that_deteriorates_full_sample():
    frame = _experiment_frame(candidate_good=False)
    candidate_review = {
        "candidate_key": "composite_signal_score>=75.0",
        "governance_status": PROMOTE_TO_REVIEW,
        "threshold_field": "composite_signal_score",
        "threshold_value": 75.0,
        "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
    }

    report = build_threshold_policy_experiment_report(frame, candidate_review=candidate_review)

    assert report["experiment_status"] == REJECTED_FOR_POLICY_EXPERIMENT
    assert any("return delta" in reason or "hit-rate delta" in reason for reason in report["experiment_reasons"])
    assert report["runtime_config_changed"] is False


def test_policy_experiment_rejects_missing_concrete_candidate_as_insufficient():
    report = build_threshold_policy_experiment_report(_experiment_frame(), candidate_review={})

    assert report["experiment_status"] == INSUFFICIENT_EVIDENCE
    assert report["runtime_config_changed"] is False


def test_policy_experiment_writer_outputs_artifacts(tmp_path: Path):
    frame = _experiment_frame()
    governance = build_threshold_governance_report(frame, dataset_path="unit.csv")
    artifact = write_threshold_policy_experiment_report(
        frame,
        governance_report=governance,
        dataset_path="unit.csv",
        governance_report_path="governance.json",
        output_dir=tmp_path,
        report_name="unit_threshold_policy_experiment",
    )

    for key in [
        "json_path",
        "markdown_path",
        "candidate_policy_pack_path",
        "splits_csv_path",
        "regimes_csv_path",
        "quality_buckets_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_candidate_policy_pack_path",
        "latest_splits_csv_path",
        "latest_regimes_csv_path",
        "latest_quality_buckets_csv_path",
    ]:
        assert Path(artifact[key]).exists()

    payload = json.loads(Path(artifact["json_path"]).read_text(encoding="utf-8"))
    policy_pack = json.loads(Path(artifact["candidate_policy_pack_path"]).read_text(encoding="utf-8"))
    assert payload["experiment_status"] == APPROVED_FOR_POLICY_EXPERIMENT
    assert policy_pack["runtime_config_changed"] is False
    assert pd.read_csv(artifact["splits_csv_path"]).empty is False


def test_policy_experiment_skip_writer_prevents_stale_latest(tmp_path: Path):
    artifact = write_threshold_policy_experiment_skip(
        reason="No promoted candidate.",
        output_dir=tmp_path,
        report_name="unit_threshold_policy_experiment_skip",
    )

    payload = json.loads(Path(artifact["latest_json_path"]).read_text(encoding="utf-8"))
    assert payload["experiment_status"] == SKIPPED_NO_PROMOTED_CANDIDATE
    assert payload["runtime_config_changed"] is False
    assert Path(artifact["latest_markdown_path"]).exists()
    assert Path(artifact["latest_candidate_policy_pack_path"]).exists()
