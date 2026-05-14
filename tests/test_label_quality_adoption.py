from __future__ import annotations

import pandas as pd

from research.ml_evaluation.ml_calibration_report import build_calibration_report
from research.ml_evaluation.ml_comparison_report import build_comparison_report
from research.ml_evaluation.ml_filter_simulation import build_filter_simulation_report
from research.ml_evaluation.ml_ranking_report import build_ranking_report
from research.signal_evaluation.label_quality import (
    apply_quality_label_view,
    label_quality_summary,
    select_quality_labeled_rows,
)


def _quality_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_id": "good-win",
                "signal_timestamp": "2026-04-01T09:15:00+05:30",
                "trade_status": "TRADE",
                "correct_60m": 0,
                "signed_return_60m_bps": -99.0,
                "calibration_label": 1,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 25.0,
                "label_quality_status": "CLEAN",
                "ml_rank_score": 0.90,
                "ml_rank_bucket": "Q5_highest",
                "ml_confidence_score": 0.80,
                "ml_confidence_bucket": "Q5_highest",
                "ml_agreement_with_engine": "YES",
            },
            {
                "signal_id": "rejected",
                "signal_timestamp": "2026-04-01T09:20:00+05:30",
                "trade_status": "TRADE",
                "correct_60m": 1,
                "signed_return_60m_bps": 120.0,
                "calibration_label": None,
                "calibration_label_available": False,
                "primary_outcome_return_bps": None,
                "label_quality_status": "UNUSABLE",
                "label_quality_reasons": "direction_unresolved",
                "ml_rank_score": 0.50,
                "ml_rank_bucket": "Q3_mid",
                "ml_confidence_score": 0.60,
                "ml_confidence_bucket": "Q3_mid",
                "ml_agreement_with_engine": "YES",
            },
            {
                "signal_id": "good-loss",
                "signal_timestamp": "2026-04-01T09:25:00+05:30",
                "trade_status": "TRADE",
                "correct_60m": 1,
                "signed_return_60m_bps": 88.0,
                "calibration_label": 0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": -10.0,
                "label_quality_status": "CLEAN",
                "ml_rank_score": 0.10,
                "ml_rank_bucket": "Q1_lowest",
                "ml_confidence_score": 0.20,
                "ml_confidence_bucket": "Q1_lowest",
                "ml_agreement_with_engine": "NO",
            },
        ]
    )


def test_quality_label_view_replaces_raw_primary_label_values():
    frame = _quality_frame()

    view = apply_quality_label_view(frame)
    selected = select_quality_labeled_rows(frame)

    assert view["correct_60m"].tolist()[0] == 1
    assert pd.isna(view["correct_60m"].tolist()[1])
    assert view["correct_60m"].tolist()[2] == 0
    assert view["signed_return_60m_bps"].tolist()[0] == 25.0
    assert len(selected) == 2


def test_quality_label_view_falls_back_to_legacy_only_for_unannotated_rows():
    frame = pd.DataFrame(
        [
            {
                "signal_id": "legacy-good",
                "correct_60m": 1,
                "signed_return_60m_bps": 15.0,
            },
            {
                "signal_id": "quality-good",
                "correct_60m": 0,
                "signed_return_60m_bps": -25.0,
                "calibration_label": 1,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 30.0,
                "label_quality_status": "CLEAN",
            },
            {
                "signal_id": "quality-rejected",
                "correct_60m": 1,
                "signed_return_60m_bps": 45.0,
                "calibration_label": None,
                "calibration_label_available": False,
                "primary_outcome_return_bps": None,
                "label_quality_status": "UNUSABLE",
                "label_quality_reasons": "direction_unresolved",
            },
        ]
    )

    view = apply_quality_label_view(frame)
    selected = select_quality_labeled_rows(frame)
    quality_only = select_quality_labeled_rows(frame, fallback_to_legacy=False)
    summary = label_quality_summary(frame)

    assert view["correct_60m"].tolist()[0] == 1
    assert view["correct_60m"].tolist()[1] == 1
    assert pd.isna(view["correct_60m"].tolist()[2])
    assert view["signed_return_60m_bps"].tolist()[0] == 15.0
    assert view["signed_return_60m_bps"].tolist()[1] == 30.0
    assert len(selected) == 2
    assert quality_only["signal_id"].tolist() == ["quality-good"]
    assert summary["raw_labeled_rows"] == 3
    assert summary["quality_labeled_rows"] == 2
    assert summary["excluded_labeled_rows"] == 1
    assert summary["label_source"] == "mixed_calibration_label_and_legacy"


def test_ml_reports_use_quality_approved_labels():
    frame = _quality_frame()

    ranking = build_ranking_report(frame)
    calibration = build_calibration_report(frame)
    filtering = build_filter_simulation_report(frame)
    comparison = build_comparison_report(frame)

    assert ranking["label_quality_summary"]["quality_labeled_rows"] == 2
    assert ranking["label_quality_summary"]["excluded_labeled_rows"] == 1
    assert any(row["n_labeled_60m"] == 0 for row in ranking["quintile_analysis"])

    assert calibration["label_quality_summary"]["quality_labeled_rows"] == 2
    assert any(row["n_labeled_60m"] == 0 for row in calibration["bucket_analysis"])
    assert any(
        row["bucket"] == "Q3_mid" and row["avg_confidence_score"] is None
        for row in calibration["bucket_analysis"]
    )

    baseline = filtering["filter_results"][0]
    assert baseline["n_labeled_60m"] == 2
    assert filtering["label_quality_summary"]["excluded_labeled_rows"] == 1

    assert comparison["label_quality_summary"]["quality_labeled_rows"] == 2
    assert comparison["summary"]["engine_hit_rate_60m"] == 0.5
