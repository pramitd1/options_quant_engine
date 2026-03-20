"""Tests for the daily signal research report generator."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research.signal_evaluation.daily_research_report import (
    generate_daily_report,
    _directional_rows,
    _section_executive_summary,
    _section_signal_generation,
    _section_horizon_performance,
    _section_alpha_decay,
    _section_score_calibration,
    _section_dataset_summary,
    _section_information_coefficient,
    _section_edge_distribution,
    _section_exit_horizon,
    _section_feature_variance,
    _filter_day,
)


def _sample_dataset() -> pd.DataFrame:
    """Build a minimal synthetic signals dataset for testing."""
    rows = []
    base_ts = pd.Timestamp("2026-03-16 10:00:00+05:30")
    for i in range(10):
        ts = base_ts + pd.Timedelta(minutes=i * 5)
        row = {
            "signal_id": f"sig_{i}",
            "signal_timestamp": ts.isoformat(),
            "symbol": "NIFTY",
            "direction": "CALL" if i < 7 else None,
            "trade_status": "TRADE" if i < 7 else "NO_SIGNAL",
            "macro_regime": "RISK_OFF",
            "gamma_regime": "POSITIVE_GAMMA" if i < 5 else "NEGATIVE_GAMMA",
            "volatility_regime": "VOL_EXPANSION",
            "global_risk_state": "RISK_OFF",
            "composite_signal_score": 80 + i if i < 3 else 40 + i,
            "spot_at_signal": 23200 + i * 10,
            "day_high": 23300,
            "day_low": 23100,
            "global_risk_score": 30.0,
            "volatility_shock_score": 0.0,
            "dealer_hedging_pressure_score": 20.0,
            "gamma_vol_acceleration_score": 10.0,
            "move_probability": 0.60,
            "target_reachability_score": 85.0,
            "signed_return_5m_bps": 5.0 - i,
            "signed_return_15m_bps": 8.0 - i,
            "signed_return_30m_bps": 6.0 - i * 0.5,
            "signed_return_60m_bps": 10.0 - i,
            "signed_return_120m_bps": 12.0 - i,
            "signed_return_session_close_bps": -5.0 + i,
            "correct_5m": 1.0 if i < 6 else 0.0,
            "correct_15m": 1.0 if i < 5 else 0.0,
            "correct_30m": 1.0 if i < 4 else 0.0,
            "correct_60m": 1.0 if i < 6 else 0.0,
            "correct_120m": 1.0 if i < 5 else 0.0,
            "correct_session_close": 1.0 if i > 5 else 0.0,
            "realized_return_5m": 0.0005 - i * 0.0001,
            "realized_return_15m": 0.0008 - i * 0.0001,
            "realized_return_30m": 0.0006 - i * 0.00005,
            "realized_return_60m": 0.001 - i * 0.0001,
            "mfe_60m_bps": 15.0 + i,
            "mae_60m_bps": -10.0 - i,
            "mfe_120m_bps": 20.0 + i,
            "mae_120m_bps": -12.0 - i,
            "lookback_avg_range_pct": 1.38,
            "expected_move_pct": 1.70,
            "tradeability_score": 60.0,
            "option_efficiency_score": 75.0,
            "data_quality_score": 92.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TestDirectionalRows:
    def test_includes_directional_non_trade_rows(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        # Ensure directional rows are not filtered by trade_status.
        df.loc[0, "trade_status"] = "WATCHLIST"
        result = _directional_rows(df)
        assert len(result) == 7
        assert "WATCHLIST" in set(result["trade_status"].astype(str))

    def test_empty_when_no_direction(self):
        df = _sample_dataset()
        df["direction"] = None
        result = _directional_rows(df)
        assert len(result) == 0


class TestFilterDay:
    def test_filters_correct_date(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        result = _filter_day(df, date(2026, 3, 16))
        assert len(result) == 10

    def test_returns_empty_for_wrong_date(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        result = _filter_day(df, date(2026, 3, 17))
        assert len(result) == 0


class TestSectionBuilders:
    def test_executive_summary_contains_key_fields(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_executive_summary(df, df, date(2026, 3, 16))
        text = "\n".join(lines)
        assert "Executive Summary" in text
        assert "10 signal snapshots" in text

    def test_horizon_performance_has_all_horizons(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        dir_df = _directional_rows(df)
        lines = _section_horizon_performance(df)
        text = "\n".join(lines)
        assert "5m" in text
        assert "60m" in text
        assert "session_close" in text

    def test_alpha_decay_has_ascii_chart(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_alpha_decay(df)
        text = "\n".join(lines)
        assert "Decay Curve (ASCII)" in text
        assert "bps" in text

    def test_score_calibration_has_buckets(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_score_calibration(df)
        text = "\n".join(lines)
        assert "Score Bucket" in text

    def test_dataset_summary_has_metrics(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_dataset_summary(df, df)
        text = "\n".join(lines)
        assert "Signal Dataset Summary" in text
        assert "Full Dataset" in text
        assert "10" in text  # total signal count

    def test_information_coefficient_has_ic_table(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_information_coefficient(df, df)
        text = "\n".join(lines)
        assert "Pearson IC" in text
        assert "Rank IC" in text

    def test_edge_distribution_has_stats(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_edge_distribution(df)
        text = "\n".join(lines)
        assert "Signal Edge Distribution" in text
        assert "Mean" in text
        assert "Median" in text

    def test_exit_horizon_has_recommendation(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_exit_horizon(df)
        text = "\n".join(lines)
        assert "Exit Horizon Diagnostic" in text
        assert "Peak alpha" in text

    def test_feature_variance_has_flags(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_feature_variance(df)
        text = "\n".join(lines)
        assert "Feature Variance Check" in text
        assert "Std Dev" in text

    def test_signal_generation_summary_count_parity(self):
        df = pd.DataFrame(
            {
                "direction": ["CALL", "PUT", "CALL", None, None, "PUT"],
                "trade_status": ["TRADE", "WATCHLIST", "NO_SIGNAL", "NO_SIGNAL", "NO_TRADE", "WATCHLIST"],
            }
        )

        lines = _section_signal_generation(df)
        text = "\n".join(lines)

        assert "| Total signal snapshots | 6 |" in text
        assert "| Directional signals | 4 |" in text
        assert "| Neutral / no-direction | 2 |" in text
        assert "| Qualified trade signals | 1 |" in text
        assert "| Watchlist / no-signal | 5 |" in text


class TestGenerateReport:
    def test_generate_report_creates_file(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
        )
        assert report_path.exists()
        assert report_path.name == "signal_research_report_20260316.md"

    def test_report_contains_all_sections(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
        )
        content = report_path.read_text()
        required_sections = [
            "Executive Summary",
            "Signal Dataset Summary",
            "Macroeconomic Environment",
            "Market Structure Context",
            "Signal Generation Summary",
            "Horizon Performance",
            "Signal Alpha Decay Curve",
            "Decay Curve by Score Bucket",
            "Decay Curve by Regime",
            "Directional Accuracy",
            "Magnitude Adequacy",
            "Tradeability Analysis",
            "Score Calibration",
            "Probability Calibration",
            "Reversal Diagnostics",
            "Regime Performance",
            "Information Coefficient",
            "Signal Edge Distribution",
            "Signal Stability Metrics",
            "Exit Horizon Diagnostic",
            "Regime Coverage Tracker",
            "Feature Variance Check",
            "Model Diagnostics",
            "Key Insights",
            "Research Actions",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_report_has_header_metadata(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
        )
        content = report_path.read_text()
        assert "Pramit Dutta" in content
        assert "Quant Engines" in content
        assert "Signal Terminology" in content

    def test_report_auto_detects_latest_date(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            dataset_path=csv_path,
            output_dir=output_dir,
        )
        assert "20260316" in report_path.name

    def test_cumulative_report_creates_file(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
            mode="cumulative",
        )
        assert report_path.exists()
        assert report_path.name == "signal_research_report_cumulative.md"

    def test_cumulative_report_contains_all_sections(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
            mode="cumulative",
        )
        content = report_path.read_text()
        assert "Cumulative Signal Research Report" in content
        assert "Executive Summary" in content
        assert "Research Actions" in content

    def test_cumulative_report_analyses_full_dataset(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
            mode="cumulative",
        )
        content = report_path.read_text()
        # Footer should show all 10 signals (full dataset) not just day
        assert "Signals analysed: 10" in content
        assert "Mode: cumulative" in content

    def test_both_modes_produce_separate_files(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        daily_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
            mode="daily",
        )
        cumulative_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
            mode="cumulative",
        )
        assert daily_path != cumulative_path
        assert daily_path.exists()
        assert cumulative_path.exists()
        assert "daily" in daily_path.read_text().lower().split("mode:")[1][:20]
        assert "cumulative" in cumulative_path.read_text().lower().split("mode:")[1][:20]

    def test_invalid_mode_raises(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="mode must be"):
            generate_daily_report(
                report_date=date(2026, 3, 16),
                dataset_path=csv_path,
                output_dir=tmp_path,
                mode="invalid",
            )
