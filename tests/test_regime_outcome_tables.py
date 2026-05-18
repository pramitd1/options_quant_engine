from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.regime_outcome_tables import (
    build_regime_outcome_table_report,
    write_regime_outcome_table_report,
)


def _sample_frame() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-05-18T09:20:00+05:30")
    specs = [
        ("NEGATIVE_GAMMA", "VOL_EXPANSION", "PUT", "RISK_OFF", "high", 18, 1),
        ("NEGATIVE_GAMMA", "VOL_EXPANSION", "PUT", "RISK_OFF", "high", 22, 1),
        ("NEGATIVE_GAMMA", "VOL_EXPANSION", "PUT", "RISK_OFF", "high", 15, 1),
        ("NEGATIVE_GAMMA", "VOL_EXPANSION", "CALL", "RISK_OFF", "high", -12, 0),
        ("POSITIVE_GAMMA", "NORMAL_VOL", "CALL", "GLOBAL_NEUTRAL", "mid", 8, 1),
        ("POSITIVE_GAMMA", "NORMAL_VOL", "CALL", "GLOBAL_NEUTRAL", "mid", 9, 1),
        ("POSITIVE_GAMMA", "NORMAL_VOL", "PUT", "GLOBAL_NEUTRAL", "mid", -6, 0),
        ("NEUTRAL_GAMMA", "LOW_VOL", "CALL", "RISK_OFF", "low", -10, 0),
    ]
    for idx, (gamma, vol, direction, risk, pcr, ret60, hit60) in enumerate(specs):
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(minutes=idx)).isoformat(),
                "direction": direction,
                "gamma_regime": gamma,
                "volatility_regime": vol,
                "global_risk_state": risk,
                "macro_regime": risk,
                "statistical_context_bucket_state": f'{{"pcr_oi_bucket": "{pcr}"}}',
                "signed_return_5m_bps": ret60 / 3.0,
                "signed_return_15m_bps": ret60 / 2.0,
                "signed_return_30m_bps": ret60 * 0.8,
                "signed_return_60m_bps": ret60,
                "signed_return_120m_bps": ret60 * 1.2,
                "signed_return_session_close_bps": ret60 * 1.5,
                "correct_5m": hit60,
                "correct_15m": hit60,
                "correct_30m": hit60,
                "correct_60m": hit60,
                "correct_120m": hit60,
                "correct_session_close": hit60,
                "calibration_label": hit60,
                "calibration_label_available": True,
                "primary_outcome_return_bps": ret60,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_regime_outcome_tables_builds_sparse_aware_empirical_tables():
    report = build_regime_outcome_table_report(
        _sample_frame(),
        dataset_path="unit.csv",
        min_label_sample=2,
        strong_label_sample=4,
        top_n=5,
    )

    assert report["report_type"] == "regime_outcome_tables"
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["directional_row_count"] == 8
    assert report["by_horizon_row_count"] > 0
    assert report["best_horizon_row_count"] > 0
    assert report["top_favorable"]
    assert any(row.get("pcr_bucket") == "HIGH" for row in report["best_horizon"])
    assert any(
        row.get("gamma_regime") == "NEGATIVE_GAMMA"
        and row.get("volatility_regime") == "VOL_EXPANSION"
        and row.get("direction") == "PUT"
        for row in report["best_horizon"]
    )


def test_regime_outcome_tables_prefers_canonical_pcr_bucket():
    frame = _sample_frame()
    frame["pcr_bucket"] = "LOW_PCR"
    frame["statistical_context_bucket_state"] = '{"pcr_oi_bucket": "high"}'

    report = build_regime_outcome_table_report(
        frame,
        dataset_path="unit.csv",
        min_label_sample=2,
        strong_label_sample=4,
        top_n=5,
    )

    assert any(row.get("pcr_bucket") == "LOW" for row in report["best_horizon"])
    assert not any(row.get("pcr_bucket") == "HIGH" for row in report["best_horizon"])


def test_regime_outcome_tables_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_regime_outcome_table_report(
        _sample_frame(),
        dataset_path="unit.csv",
        output_dir=tmp_path,
        report_name="unit_regime_outcome_tables",
        min_label_sample=2,
        strong_label_sample=4,
    )

    for key in [
        "json_path",
        "markdown_path",
        "by_horizon_csv_path",
        "best_horizon_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_by_horizon_csv_path",
        "latest_best_horizon_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["best_horizon_row_count"] > 0


def test_regime_outcome_tables_handles_empty_frame():
    report = build_regime_outcome_table_report(pd.DataFrame(), min_label_sample=2)

    assert report["directional_row_count"] == 0
    assert report["by_horizon_row_count"] == 0
    assert report["best_horizon_row_count"] == 0
    assert report["top_favorable"] == []
