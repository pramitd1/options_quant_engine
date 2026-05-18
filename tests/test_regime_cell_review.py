from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.regime_cell_review import (
    build_regime_cell_review_report,
    classify_regime_cell,
    write_regime_cell_review_report,
)


def test_classify_regime_cell_marks_long_horizon_favorable_as_hold_special():
    row = {
        "hit_rate": 0.79,
        "avg_signed_return_bps": 140.0,
        "hit_rate_delta_vs_all": 0.25,
        "avg_return_delta_vs_all_bps": 142.0,
        "label_count": 461,
        "best_horizon": "120m",
    }
    profile = {
        "early_best_avg_return_bps": 12.0,
        "late_best_avg_return_bps": 140.0,
        "best_late_minus_early_bps": 128.0,
    }

    proposal = classify_regime_cell(row, profile)

    assert proposal["action_class"] == "HOLD_TIME_SPECIAL"
    assert proposal["allow_trade_research"] is True
    assert proposal["threshold_adjustment_research"] < 1


def test_classify_regime_cell_marks_bad_short_cell_as_downgrade():
    row = {
        "hit_rate": 0.41,
        "avg_signed_return_bps": -6.0,
        "hit_rate_delta_vs_all": -0.06,
        "avg_return_delta_vs_all_bps": -4.8,
        "label_count": 852,
        "best_horizon": "5m",
    }

    proposal = classify_regime_cell(row, {})

    assert proposal["action_class"] == "DOWNGRADE_OR_AVOID"
    assert proposal["allow_trade_research"] is False
    assert proposal["threshold_adjustment_research"] > 0


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    best = pd.DataFrame(
        [
            {
                "group_name": "gamma_regime+volatility_regime+direction",
                "sample_quality": "RELIABLE",
                "signal_count": 500,
                "label_count": 480,
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "direction": "CALL",
                "best_horizon": "120m",
                "hit_rate": 0.80,
                "avg_signed_return_bps": 120.0,
                "hit_rate_delta_vs_all": 0.22,
                "avg_return_delta_vs_all_bps": 100.0,
            },
            {
                "group_name": "gamma_regime+volatility_regime+direction+macro_risk_bucket",
                "sample_quality": "RELIABLE",
                "signal_count": 300,
                "label_count": 300,
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "direction": "PUT",
                "macro_risk_bucket": "RISK_OFF",
                "best_horizon": "5m",
                "hit_rate": 0.39,
                "avg_signed_return_bps": -5.0,
                "hit_rate_delta_vs_all": -0.08,
                "avg_return_delta_vs_all_bps": -4.0,
            },
            {
                "group_name": "gamma_regime+volatility_regime+direction+macro_risk_bucket+pcr_bucket",
                "sample_quality": "RELIABLE",
                "signal_count": 100,
                "label_count": 100,
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "direction": "CALL",
                "macro_risk_bucket": "RISK_OFF",
                "pcr_bucket": "HIGH",
                "best_horizon": "120m",
                "hit_rate": 0.90,
                "avg_signed_return_bps": 150.0,
                "hit_rate_delta_vs_all": 0.30,
                "avg_return_delta_vs_all_bps": 130.0,
            },
        ]
    )
    horizon_rows = []
    for row in best.iloc[:2].to_dict("records"):
        for horizon, ret in [("5m", -1.0), ("15m", 4.0), ("30m", 10.0), ("60m", 40.0), ("120m", 120.0)]:
            item = dict(row)
            item["horizon"] = horizon
            item["avg_signed_return_bps"] = ret if row["direction"] == "CALL" else -5.0
            item["sample_quality"] = "RELIABLE"
            horizon_rows.append(item)
    return best, pd.DataFrame(horizon_rows)


def test_regime_cell_review_filters_to_reliable_three_and_four_factor_cells():
    best, by_horizon = _frames()

    report = build_regime_cell_review_report(best_horizon=best, by_horizon=by_horizon, source_dir="unit")

    assert report["report_type"] == "regime_cell_review"
    assert report["runtime_config_changed"] is False
    assert report["reviewed_cell_count"] == 2
    assert {row["group_name"] for row in report["cells"]} == {
        "gamma_regime+volatility_regime+direction",
        "gamma_regime+volatility_regime+direction+macro_risk_bucket",
    }
    assert report["action_counts"]["HOLD_TIME_SPECIAL"] == 1
    assert report["action_counts"]["DOWNGRADE_OR_AVOID"] == 1


def test_regime_cell_review_writer_outputs_artifacts(tmp_path: Path):
    best, by_horizon = _frames()
    best_path = tmp_path / "best.csv"
    by_path = tmp_path / "by.csv"
    best.to_csv(best_path, index=False)
    by_horizon.to_csv(by_path, index=False)

    artifact = write_regime_cell_review_report(
        best_horizon_csv=best_path,
        by_horizon_csv=by_path,
        output_dir=tmp_path / "out",
        documentation_dir=tmp_path / "docs",
    )

    assert Path(artifact["json_path"]).exists()
    assert Path(artifact["markdown_path"]).exists()
    assert Path(artifact["csv_path"]).exists()
    assert Path(artifact["documentation_markdown_path"]).exists()
