from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.regime_cell_counterfactual import (
    build_regime_cell_counterfactual_report,
    write_regime_cell_counterfactual_report,
)


def _review() -> dict:
    return {
        "cells": [
            {
                "group_name": "gamma_regime+volatility_regime+direction+macro_risk_bucket",
                "cell": "gamma_regime=NEGATIVE_GAMMA<br>volatility_regime=VOL_EXPANSION<br>direction=CALL<br>macro_risk_bucket=RISK_OFF",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "direction": "CALL",
                "macro_risk_bucket": "RISK_OFF",
                "action_class": "HOLD_TIME_SPECIAL",
                "best_horizon": "120m",
                "score_adjustment_research": 2,
                "threshold_adjustment_research": -1,
            },
            {
                "group_name": "gamma_regime+volatility_regime+direction+macro_risk_bucket",
                "cell": "gamma_regime=NEGATIVE_GAMMA<br>volatility_regime=VOL_EXPANSION<br>direction=PUT<br>macro_risk_bucket=RISK_OFF",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "direction": "PUT",
                "macro_risk_bucket": "RISK_OFF",
                "action_class": "DOWNGRADE_OR_AVOID",
                "best_horizon": "5m",
                "score_adjustment_research": -4,
                "threshold_adjustment_research": 3,
            },
        ]
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_id": "c1",
                "signal_timestamp": "2026-05-18T09:30:00+05:30",
                "trade_status": "TRADE",
                "direction": "CALL",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "global_risk_state": "RISK_OFF",
                "trade_strength": 80,
                "signed_return_60m_bps": 10.0,
                "signed_return_120m_bps": 40.0,
                "correct_60m": 1,
                "correct_120m": 1,
            },
            {
                "signal_id": "c2",
                "signal_timestamp": "2026-05-18T09:31:00+05:30",
                "trade_status": "TRADE",
                "direction": "CALL",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "global_risk_state": "RISK_OFF",
                "trade_strength": 72,
                "signed_return_60m_bps": -5.0,
                "signed_return_120m_bps": 5.0,
                "correct_60m": 0,
                "correct_120m": 1,
            },
            {
                "signal_id": "c3",
                "signal_timestamp": "2026-05-18T09:32:00+05:30",
                "trade_status": "WATCHLIST",
                "direction": "CALL",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "global_risk_state": "RISK_OFF",
                "trade_strength": 60,
                "signed_return_60m_bps": 2.0,
                "signed_return_120m_bps": 20.0,
                "correct_60m": 1,
                "correct_120m": 1,
            },
            {
                "signal_id": "p1",
                "signal_timestamp": "2026-05-18T09:33:00+05:30",
                "trade_status": "TRADE",
                "direction": "PUT",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "global_risk_state": "RISK_OFF",
                "trade_strength": 88,
                "signed_return_60m_bps": -15.0,
                "correct_60m": 0,
            },
            {
                "signal_id": "p2",
                "signal_timestamp": "2026-05-18T09:34:00+05:30",
                "trade_status": "TRADE",
                "direction": "PUT",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "global_risk_state": "RISK_OFF",
                "trade_strength": 92,
                "signed_return_60m_bps": 5.0,
                "correct_60m": 1,
            },
        ]
    )


def test_regime_cell_counterfactual_suppresses_and_changes_hold_time():
    report, cells, details = build_regime_cell_counterfactual_report(
        _frame(),
        review=_review(),
        dataset_path="unit.csv",
        review_path="review.json",
        min_cell_labels=1,
    )

    assert report["report_type"] == "regime_cell_counterfactual"
    assert report["runtime_config_changed"] is False
    assert report["matched_signal_count"] == 5
    assert report["baseline_trade_count"] == 4
    assert report["counterfactual_selected_count"] == 2
    assert report["suppressed_existing_trade_count"] == 2
    assert report["promotion_candidate_count_sandbox"] == 1
    assert report["conservative_total_return_delta_bps"] == 50.0
    assert report["avoided_suppressed_return_60m_bps"] == 10.0
    assert report["hold_time_sum_delta_bps"] == 40.0
    assert set(cells["impact_status"]) == {"REPLAY_SUPPORTS_HOLD_CHANGE", "REPLAY_SUPPORTS_SUPPRESSION"}
    assert "PROMOTION_CANDIDATE_SANDBOX" in set(details["counterfactual_decision"])


def test_regime_cell_counterfactual_writer_outputs_artifacts(tmp_path: Path):
    dataset_path = tmp_path / "dataset.csv"
    review_path = tmp_path / "review.json"
    _frame().to_csv(dataset_path, index=False)
    review_path.write_text(json.dumps(_review()), encoding="utf-8")

    artifact = write_regime_cell_counterfactual_report(
        dataset_path=dataset_path,
        review_path=review_path,
        output_dir=tmp_path / "out",
        documentation_dir=tmp_path / "docs",
        min_cell_labels=1,
    )

    assert Path(artifact["json_path"]).exists()
    assert Path(artifact["markdown_path"]).exists()
    assert Path(artifact["cells_csv_path"]).exists()
    assert Path(artifact["details_csv_path"]).exists()
    assert Path(artifact["documentation_markdown_path"]).exists()
