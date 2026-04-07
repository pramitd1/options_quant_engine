"""Tests for the daily signal research report generator."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from research.signal_evaluation.daily_research_report import (
    generate_daily_report,
    _append_summary,
    _interpretation_conflicts_with_kpis,
    _directional_rows,
    _summarize_alpha_decay,
    _summarize_research_actions,
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


class TestSummaryHelpers:
    def test_alpha_decay_summary_reports_no_positive_alpha_when_all_horizons_negative(self):
        df = pd.DataFrame(
            {
                "direction": ["CALL", "PUT", "CALL"],
                "signed_return_5m_bps": [-6.0, -7.0, -5.0],
                "signed_return_15m_bps": [-8.0, -9.0, -7.0],
                "signed_return_30m_bps": [-10.0, -11.0, -9.0],
                "signed_return_60m_bps": [-12.0, -13.0, -11.0],
                "signed_return_120m_bps": [-15.0, -16.0, -14.0],
                "signed_return_session_close_bps": [-20.0, -19.0, -18.0],
            }
        )

        summary = _summarize_alpha_decay(df)

        assert "No positive alpha horizon identified" in summary
        assert "remained positive" not in summary

    def test_research_actions_flags_large_probability_gap_and_negative_close_edge(self):
        df = pd.DataFrame(
            {
                "direction": ["CALL"] * 30,
                "macro_regime": ["RISK_OFF"] * 30,
                "move_probability": [0.70] * 30,
                "correct_60m": [1.0] * 27 + [0.0] * 3,
                "signed_return_60m_bps": [-5.0] * 30,
                "signed_return_session_close_bps": [-18.0] * 30,
            }
        )

        summary = _summarize_research_actions(df)

        assert "Recalibrate probability model" in summary
        assert "Session-close edge remains negative" in summary


class TestInterpretationConflictChecker:
    def test_horizon_performance_flags_short_lived_claim_when_best_horizon_is_long(self):
        payload = {
            "tables": [
                {
                    "table": {
                        "rows": [
                            {"Horizon": "5m", "Hit Rate": "52.0%", "Avg Signed Return (bps)": "4.0"},
                            {"Horizon": "60m", "Hit Rate": "68.0%", "Avg Signed Return (bps)": "22.0"},
                            {"Horizon": "120m", "Hit Rate": "74.0%", "Avg Signed Return (bps)": "35.0"},
                            {"Horizon": "session_close", "Hit Rate": "70.0%", "Avg Signed Return (bps)": "19.0"},
                        ]
                    }
                }
            ]
        }

        conflict = _interpretation_conflicts_with_kpis(
            "Horizon Performance",
            "The edge looks short-lived alpha that fades quickly after the first few minutes.",
            payload,
            "Best accuracy at 120m (74.0%), weakest at 5m (52.0%). Session close averaged 19.0 bps.",
        )

        assert conflict is True

    def test_horizon_performance_flags_positive_close_claim_when_session_close_is_negative(self):
        payload = {
            "tables": [
                {
                    "table": {
                        "rows": [
                            {"Horizon": "5m", "Hit Rate": "64.0%", "Avg Signed Return (bps)": "8.0"},
                            {"Horizon": "60m", "Hit Rate": "61.0%", "Avg Signed Return (bps)": "6.0"},
                            {"Horizon": "session_close", "Hit Rate": "42.0%", "Avg Signed Return (bps)": "-11.0"},
                        ]
                    }
                }
            ]
        }

        conflict = _interpretation_conflicts_with_kpis(
            "Horizon Performance",
            "The signal held its gains into the close and remained positive by close.",
            payload,
            "Best accuracy at 5m (64.0%), weakest at session_close (42.0%). Session close averaged -11.0 bps.",
        )

        assert conflict is True

    def test_regime_coverage_flags_reliable_coverage_claim_when_some_buckets_are_sparse(self):
        payload = {
            "tables": [
                {
                    "table": {
                        "rows": [
                            {"Macro Regime": "RISK_OFF", "N": "18", "% of Total": "62.1%"},
                            {"Macro Regime": "RISK_ON", "N": "7", "% of Total": "24.1%"},
                            {"Macro Regime": "RISK_NEUTRAL", "N": "4", "% of Total": "13.8%"},
                        ]
                    }
                }
            ]
        }

        conflict = _interpretation_conflicts_with_kpis(
            "Regime Coverage Tracker",
            "The sample has broad regime coverage and statistically reliable regime estimates.",
            payload,
            "Some regime states remain sparse and require more data before conditional metrics are reliable.",
        )

        assert conflict is True

    def test_regime_coverage_allows_balanced_coverage_commentary_when_counts_are_sufficient(self):
        payload = {
            "tables": [
                {
                    "table": {
                        "rows": [
                            {"Macro Regime": "RISK_OFF", "N": "14", "% of Total": "33.3%"},
                            {"Macro Regime": "RISK_ON", "N": "15", "% of Total": "35.7%"},
                            {"Macro Regime": "RISK_NEUTRAL", "N": "13", "% of Total": "31.0%"},
                        ]
                    }
                }
            ]
        }

        conflict = _interpretation_conflicts_with_kpis(
            "Regime Coverage Tracker",
            "The sample now has broad regime coverage with enough observations for more reliable conditional reads.",
            payload,
            "Coverage is balanced enough for conditional analysis.",
        )

        assert conflict is False

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


class TestAppendSummaryNarrativeSuppression:
    """Integration tests: _append_summary with a mocked AI narrative.

    Verifies that conflicting interpretations are suppressed in the rendered
    markdown output and that non-conflicting interpretations are preserved.
    """

    _MOCK_TARGET = "research.signal_evaluation.daily_research_report._ai_narrative"

    # ---------- helpers ---------------------------------------------------

    @staticmethod
    def _section_lines_overconfident_calibration() -> list[str]:
        """Minimal section lines for a Probability Calibration section where
        the model is overconfident (predicted > realized)."""
        return [
            "## 13. Probability Calibration",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            "| Avg Predicted Probability | 75.0% |",
            "| Avg Realized Hit Rate | 52.0% |",
            "",
        ]

    @staticmethod
    def _fallback_overconfident() -> str:
        return (
            "Predicted 75.0% vs realized 52.0% — significantly overconfident (gap: 0.230). "
            "The model systematically overestimates the probability of favorable outcomes."
        )

    @staticmethod
    def _section_lines_underconfident_calibration() -> list[str]:
        return [
            "## 13. Probability Calibration",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            "| Avg Predicted Probability | 50.0% |",
            "| Avg Realized Hit Rate | 74.0% |",
            "",
        ]

    @staticmethod
    def _fallback_underconfident() -> str:
        return (
            "Predicted 50.0% vs realized 74.0% — significantly underconfident (gap: 0.240). "
            "The model systematically underestimates its predictive power."
        )

    # ---------- tests -------------------------------------------------------

    def test_conflicting_ai_interpretation_is_absent_from_rendered_markdown(self):
        """Overconfident fallback summary + AI claiming 'underconfident' must be suppressed."""
        conflicting_ai = (
            "The model appears underconfident — predicted probabilities fall well below "
            "realized outcomes, suggesting alpha is being left on the table."
        )
        lines: list[str] = []
        section_lines = self._section_lines_overconfident_calibration()
        with patch(self._MOCK_TARGET, return_value=conflicting_ai):
            _append_summary(
                lines,
                "Probability Calibration",
                section_lines,
                self._fallback_overconfident(),
                narrative=True,
            )
        rendered = "\n".join(lines)
        assert "**Interpretation:**" not in rendered, (
            "Conflicting AI interpretation should have been suppressed"
        )
        assert "**Summary:**" in rendered, "Authoritative fallback summary must always be present"

    def test_aligned_ai_interpretation_is_present_in_rendered_markdown(self):
        """Aligned AI commentary must be preserved in the markdown output."""
        aligned_ai = (
            "The model is significantly overconfident — it overestimates move probability by "
            "roughly 23 percentage points. Recalibrate with isotonic regression or Platt scaling "
            "to restore reliable probability estimates for sizing."
        )
        lines: list[str] = []
        section_lines = self._section_lines_overconfident_calibration()
        with patch(self._MOCK_TARGET, return_value=aligned_ai):
            _append_summary(
                lines,
                "Probability Calibration",
                section_lines,
                self._fallback_overconfident(),
                narrative=True,
            )
        rendered = "\n".join(lines)
        assert "**Interpretation:**" in rendered, (
            "Aligned AI interpretation should be present in the rendered markdown"
        )
        assert aligned_ai in rendered

    def test_conflicting_alpha_decay_interpretation_is_suppressed(self):
        """AI claiming positive alpha when fallback says 'No positive alpha' must be suppressed."""
        conflicting_ai = (
            "Alpha remained positive by close, suggesting durable informational edge. "
            "Longer holds viable beyond the 5m peak horizon to capture more alpha."
        )
        section_lines = [
            "## Exit Horizon Diagnostic",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            "| Peak alpha horizon | 5m |",
            "| Peak alpha (bps) | -6.70 |",
            "| Session close alpha (bps) | -70.60 |",
            "",
        ]
        fallback = (
            "No positive alpha horizon identified. Best horizon was 5m (-6.7 bps) and "
            "session close was -70.6 bps. Realized edge remained negative."
        )
        lines: list[str] = []
        with patch(self._MOCK_TARGET, return_value=conflicting_ai):
            _append_summary(
                lines,
                "Exit Horizon Diagnostic",
                section_lines,
                fallback,
                narrative=True,
            )
        rendered = "\n".join(lines)
        assert "**Interpretation:**" not in rendered
        assert "**Summary:**" in rendered

    def test_no_ai_call_when_narrative_is_false(self):
        """When narrative=False the AI must never be called and only Summary appears."""
        lines: list[str] = []
        with patch(self._MOCK_TARGET) as mock_ai:
            _append_summary(
                lines,
                "Probability Calibration",
                self._section_lines_overconfident_calibration(),
                self._fallback_overconfident(),
                narrative=False,
            )
            mock_ai.assert_not_called()
        rendered = "\n".join(lines)
        assert "**Summary:**" in rendered
        assert "**Interpretation:**" not in rendered

    def test_empty_ai_response_does_not_add_interpretation_block(self):
        """An empty string from the AI must not produce an empty Interpretation block."""
        lines: list[str] = []
        with patch(self._MOCK_TARGET, return_value=""):
            _append_summary(
                lines,
                "Probability Calibration",
                self._section_lines_overconfident_calibration(),
                self._fallback_overconfident(),
                narrative=True,
            )
        rendered = "\n".join(lines)
        assert "**Interpretation:**" not in rendered
        assert "**Summary:**" in rendered
