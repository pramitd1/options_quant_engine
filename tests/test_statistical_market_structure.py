from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.signal_evaluation.statistical_market_structure import (
    _volatility_series_for_display,
    build_distribution_summary,
    build_statistical_market_structure_report,
    build_target_correlation_table,
    write_statistical_market_structure_artifacts,
)


def _sample_panel(rows: int = 620) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-02", periods=rows, freq="B")
    market_shock = rng.normal(0, 1, rows)
    vol_state = np.abs(rng.normal(0.9, 0.25, rows))
    ret = 12 * market_shock + rng.normal(0, 18, rows)
    range_bps = 80 + 25 * vol_state + np.abs(rng.normal(0, 35, rows))
    close = 18000 * np.cumprod(1 + ret / 10000.0)
    pcr = 0.95 + 0.15 * np.tanh(-market_shock) + rng.normal(0, 0.04, rows)
    india_vix = 13 + 3.2 * vol_state + rng.normal(0, 0.6, rows)
    fwd_ret = np.roll(ret, -1)
    fwd_ret[-1] = np.nan
    fwd_range = np.roll(range_bps, -1)
    fwd_range[-1] = np.nan
    panel = pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.001, rows)),
            "high": close * (1 + range_bps / 20000.0),
            "low": close * (1 - range_bps / 20000.0),
            "close": close,
            "ret_1d_bps": ret,
            "ret_5d_bps": pd.Series(ret).rolling(5).sum().to_numpy(),
            "ret_20d_bps": pd.Series(ret).rolling(20).sum().to_numpy(),
            "gap_bps": rng.normal(0, 25, rows),
            "intraday_bps": ret + rng.normal(0, 12, rows),
            "range_bps": range_bps,
            "realized_vol_5d": pd.Series(ret).rolling(5).std().fillna(20).to_numpy() / 100.0,
            "realized_vol_20d": pd.Series(ret).rolling(20).std().fillna(20).to_numpy() / 100.0,
            "realized_vol_60d": pd.Series(ret).rolling(60).std().fillna(20).to_numpy() / 100.0,
            "oil_change_24h": rng.normal(0, 1, rows),
            "gold_change_24h": rng.normal(0, 0.6, rows),
            "copper_change_24h": rng.normal(0, 1.2, rows),
            "vix_change_24h": rng.normal(0, 2, rows),
            "india_vix_change_24h": rng.normal(0, 1.8, rows),
            "india_vix_level": india_vix,
            "sp500_change_24h": 0.25 * ret + rng.normal(0, 20, rows),
            "nasdaq_change_24h": 0.3 * ret + rng.normal(0, 25, rows),
            "us10y_change_bp": rng.normal(0, 5, rows),
            "usdinr_change_24h": rng.normal(0, 0.2, rows),
            "nifty50_realized_vol_5d": pd.Series(ret).rolling(5).std().fillna(20).to_numpy() / 100.0,
            "nifty50_realized_vol_30d": pd.Series(ret).rolling(30).std().fillna(20).to_numpy() / 100.0,
            "banknifty_realized_vol_5d": pd.Series(ret).rolling(5).std().fillna(25).to_numpy() / 90.0,
            "banknifty_realized_vol_30d": pd.Series(ret).rolling(30).std().fillna(25).to_numpy() / 90.0,
            "front_dte": rng.integers(0, 8, rows),
            "pcr_oi": pcr,
            "pcr_volume": pcr + rng.normal(0, 0.08, rows),
            "pcr_chg_oi": pcr + rng.normal(0, 0.2, rows),
            "near_atm_pcr_oi": pcr + rng.normal(0, 0.05, rows),
            "near_atm_pcr_volume": pcr + rng.normal(0, 0.08, rows),
            "oi_top5_concentration": rng.uniform(0.15, 0.45, rows),
            "atm_straddle_pct": 0.6 + 0.04 * india_vix + rng.normal(0, 0.08, rows),
            "max_pain_dist_pct": rng.normal(0, 0.7, rows),
            "max_pain_abs_dist_pct": np.abs(rng.normal(0.4, 0.3, rows)),
            "call_wall_dist_pct": rng.normal(0.7, 0.4, rows),
            "put_wall_dist_pct": rng.normal(-0.7, 0.4, rows),
            "wall_width_pct": rng.normal(1.5, 0.5, rows),
            "fwd_ret_1d_bps": fwd_ret,
            "fwd_ret_3d_bps": pd.Series(ret).shift(-3).to_numpy(),
            "fwd_ret_5d_bps": pd.Series(ret).shift(-5).to_numpy(),
            "fwd_abs_ret_1d_bps": np.abs(fwd_ret),
            "fwd_abs_ret_3d_bps": np.abs(pd.Series(ret).shift(-3).to_numpy()),
            "fwd_abs_ret_5d_bps": np.abs(pd.Series(ret).shift(-5).to_numpy()),
            "next_day_range_bps": fwd_range,
            "weekday": dates.day_name(),
            "month": dates.month_name(),
            "expiry_bucket": np.where(np.arange(rows) % 5 == 0, "0-1d", "4-7d"),
            "india_vix_bucket": pd.qcut(india_vix, 5, labels=["low", "q2", "q3", "q4", "high"]).astype(str),
            "pcr_oi_bucket": pd.qcut(pcr, 5, labels=["low", "q2", "q3", "q4", "high"]).astype(str),
            "trend_20d_bucket": np.where(pd.Series(ret).rolling(20).sum().fillna(0) > 0, "strong", "weak"),
            "is_expiry_weekday_thursday": dates.weekday == 3,
            "near_call_wall": rng.random(rows) > 0.7,
            "near_put_wall": rng.random(rows) > 0.7,
            "macro_major_event": rng.integers(0, 2, rows),
        }
    )
    return panel


def _sample_signal_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-05-18 09:20", periods=rows, freq="min", tz="Asia/Kolkata")
    strength = rng.normal(65, 8, rows)
    returns = (strength - 65) * 0.8 + rng.normal(0, 9, rows)
    return pd.DataFrame(
        {
            "signal_timestamp": ts,
            "direction": np.where(np.arange(rows) % 2 == 0, "CALL", "PUT"),
            "trade_strength": strength,
            "move_probability": rng.uniform(0.25, 0.65, rows),
            "signal_confidence_score": rng.uniform(30, 75, rows),
            "signed_return_15m_bps": returns * 0.4,
            "signed_return_30m_bps": returns * 0.7,
            "signed_return_60m_bps": returns,
            "correct_60m": returns > 0,
        }
    )


def test_distribution_and_target_correlation_tables_have_signal():
    panel = _sample_panel()

    distribution = build_distribution_summary(panel)
    correlations = build_target_correlation_table(panel)

    assert {"ret_1d_bps", "range_bps", "pcr_oi"}.issubset(set(distribution["feature"]))
    assert not correlations.empty
    assert {"feature", "target", "spearman"}.issubset(correlations.columns)


def test_volatility_series_for_display_converts_decimal_realized_vol_to_percent():
    decimal_values, decimal_unit = _volatility_series_for_display(pd.Series([0.12, 0.18, 0.25]))
    percent_values, percent_unit = _volatility_series_for_display(pd.Series([12.0, 18.0, 25.0]))

    assert decimal_unit == "%"
    assert decimal_values.round(2).tolist() == [12.0, 18.0, 25.0]
    assert percent_unit == "index"
    assert percent_values.tolist() == [12.0, 18.0, 25.0]


def test_statistical_market_structure_report_is_research_only():
    report, tables = build_statistical_market_structure_report(
        _sample_panel(),
        signal_frame=_sample_signal_frame(),
        run_id="unit",
    )

    assert report["report_type"] == "statistical_market_structure_study"
    assert report["runtime_config_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["research_only"] is True
    assert report["coverage"]["panel_rows"] == 620
    assert report["findings"]
    assert not tables["distribution_summary"].empty
    assert not tables["spearman_correlation_matrix"].empty
    assert not tables["macro_target_correlations"].empty
    assert not tables["macro_shock_distributions"].empty
    assert not tables["macro_interaction_distributions"].empty
    assert not tables["macro_spearman_correlation_matrix"].empty
    assert report["macro_summary"]["pca_summary"]["status"] == "OK"
    assert report["pca_summary"]["status"] == "OK"


def test_statistical_market_structure_writer_outputs_pdf_and_tables(tmp_path: Path):
    panel_path = tmp_path / "panel.csv"
    signal_path = tmp_path / "signals.csv"
    _sample_panel().to_csv(panel_path, index=False)
    _sample_signal_frame().to_csv(signal_path, index=False)

    artifact = write_statistical_market_structure_artifacts(
        panel_path=panel_path,
        signal_dataset_path=signal_path,
        output_dir=tmp_path / "out",
        report_name="unit_market_structure",
    )

    for key in [
        "json_path",
        "markdown_path",
        "pdf_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_pdf_path",
    ]:
        assert Path(artifact[key]).exists()
    assert Path(artifact["pdf_path"]).stat().st_size > 1000
    assert artifact["report"]["coverage"]["signal_rows"] == 80
