from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_governance import (
    PROMOTE_TO_REVIEW,
    REJECT_INSUFFICIENT_EVIDENCE,
    build_threshold_governance_report,
    record_threshold_governance_review,
    write_threshold_governance_report,
)


def _governance_frame(days: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01 09:20:00+05:30")
    for idx in range(days):
        high_score = idx >= 30
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(days=idx)).isoformat(),
                "direction": "CALL" if idx % 2 == 0 else "PUT",
                "composite_signal_score": 82.0 if high_score else 52.0,
                "tradeability_score": 78.0 if high_score else 45.0,
                "move_probability": 0.72 if high_score else 0.48,
                "macro_regime": "RISK_ON" if idx % 3 else "RISK_OFF",
                "correct_60m": 1.0 if high_score else 0.0,
                "signed_return_60m_bps": 22.0 if high_score else -9.0,
                "calibration_label": 1.0 if high_score else 0.0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 22.0 if high_score else -9.0,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_threshold_governance_promotes_robust_candidate_to_review():
    report = build_threshold_governance_report(_governance_frame(), dataset_path="unit.csv")

    assert report["overall_status"] == PROMOTE_TO_REVIEW
    assert report["runtime_config_changed"] is False
    top = report["top_candidate_review"]
    assert top["governance_status"] == PROMOTE_TO_REVIEW
    assert top["requires_manual_review"] is True
    assert top["config_hint"]
    assert report["walk_forward_summary"]["robustness_status"] == "ROBUST"


def test_threshold_governance_rejects_insufficient_evidence():
    report = build_threshold_governance_report(_governance_frame(days=12), dataset_path="thin.csv")

    assert report["overall_status"] == REJECT_INSUFFICIENT_EVIDENCE
    assert report["top_candidate_review"]["governance_status"] == REJECT_INSUFFICIENT_EVIDENCE


def test_threshold_governance_writer_and_review_ledger(tmp_path: Path):
    artifact = write_threshold_governance_report(
        _governance_frame(),
        dataset_path="unit.csv",
        output_dir=tmp_path,
        report_name="unit_threshold_governance",
    )

    assert Path(artifact["json_path"]).exists()
    assert Path(artifact["markdown_path"]).exists()
    assert Path(artifact["candidates_csv_path"]).exists()
    assert Path(artifact["latest_json_path"]).exists()
    assert Path(artifact["latest_markdown_path"]).exists()
    assert Path(artifact["latest_candidates_csv_path"]).exists()

    review = record_threshold_governance_review(
        report_json_path=artifact["json_path"],
        review_action="ACKNOWLEDGED",
        reviewer="unit-test",
        review_note="Reviewed promotion candidate.",
        ledger_path=artifact["review_ledger_path"],
        next_review_at="2026-05-15",
    )

    ledger = pd.read_csv(review["review_ledger_path"])
    assert ledger["review_action"].iloc[-1] == "ACKNOWLEDGED"
    assert ledger["reviewer"].iloc[-1] == "unit-test"
    assert ledger["runtime_config_changed"].iloc[-1] == False
