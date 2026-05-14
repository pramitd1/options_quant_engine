"""Tests for the daily signal research report generator."""

from __future__ import annotations

import json
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
    _section_premium_analytics,
    _section_score_calibration,
    _section_threshold_replay,
    _section_threshold_governance,
    _section_threshold_policy_experiment,
    _section_threshold_shadow_simulation,
    _section_threshold_shadow_review,
    _section_threshold_promotion_review,
    _section_threshold_post_promotion_monitor,
    _section_threshold_adoption_reconciliation,
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
            "option_entry_premium": 110.0 + i,
            "option_target_premium": 143.0 + i,
            "option_stop_loss_premium": 93.5 + i,
            "option_premium_pct_of_spot": 0.47 + i * 0.001,
            "premium_efficiency_score": 72.0,
            "option_efficiency_score": 75.0,
            "data_quality_score": 92.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_experiment_dataset(days: int = 120) -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2026-01-01 09:20:00+05:30")
    for i in range(days):
        ts = base_ts + pd.Timedelta(days=i)
        high_score = i >= 30
        rows.append(
            {
                "signal_id": f"policy_sig_{i}",
                "signal_timestamp": ts.isoformat(),
                "symbol": "NIFTY",
                "direction": "CALL" if i % 2 == 0 else "PUT",
                "trade_status": "TRADE",
                "signal_regime": "EXPANSION_BIAS" if high_score else "CONFLICTED",
                "macro_regime": "RISK_ON" if i % 3 else "RISK_OFF",
                "gamma_regime": "SHORT_GAMMA_ZONE" if i % 2 else "LONG_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "global_risk_state": "CALM",
                "composite_signal_score": 82.0 if high_score else 52.0,
                "tradeability_score": 78.0 if high_score else 45.0,
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
        assert "Hit Rate 95% CI" in text
        assert "INSUFFICIENT_EVIDENCE" in text

    def test_threshold_replay_section_has_candidate_table(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_threshold_replay(df)
        text = "\n".join(lines)
        assert "Threshold Replay Diagnostics" in text
        assert "Overall Candidate Thresholds" in text
        assert "Regime-Conditioned Candidates" in text
        assert "Walk-Forward Threshold Validation" in text

    def test_threshold_governance_section_shows_review_status(self):
        artifact = {
            "json_path": "threshold_governance.json",
            "markdown_path": "threshold_governance.md",
            "review_ledger_path": "threshold_governance_review_ledger.csv",
            "report": {
                "overall_status": "WATCHLIST",
                "walk_forward_summary": {
                    "robustness_status": "MIXED",
                    "evaluated_split_count": 2,
                    "split_count": 3,
                    "positive_holdout_rate": 0.5,
                    "avg_holdout_return_60m_bps": 3.2,
                },
                "top_candidate_review": {
                    "candidate_key": "composite_signal_score>=75.0",
                    "governance_status": "WATCHLIST",
                    "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
                    "recommended_next_action": "Keep collecting labels.",
                    "reasons": ["More walk-forward evidence required."],
                },
                "candidate_reviews": [],
            },
        }
        text = "\n".join(_section_threshold_governance(artifact))

        assert "Threshold Governance" in text
        assert "WATCHLIST" in text
        assert "composite_signal_score>=75.0" in text

    def test_threshold_policy_experiment_section_shows_status(self):
        artifact = {
            "json_path": "threshold_policy_experiment.json",
            "markdown_path": "threshold_policy_experiment.md",
            "candidate_policy_pack_path": "candidate_pack.json",
            "report": {
                "experiment_status": "APPROVED_FOR_POLICY_EXPERIMENT",
                "runtime_config_changed": False,
                "experiment_reasons": ["Candidate passed sandbox guardrails."],
                "candidate_policy_pack": {
                    "source_candidate_key": "composite_signal_score>=75.0",
                    "source_governance_status": "PROMOTE_TO_REVIEW",
                    "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
                    "research_only": True,
                },
                "full_sample_comparison": {
                    "baseline": {"signal_count": 100, "label_count_60m": 100, "hit_rate_60m": 0.55, "avg_signed_return_60m_bps": 4.0},
                    "candidate": {"signal_count": 45, "label_count_60m": 45, "hit_rate_60m": 0.7, "avg_signed_return_60m_bps": 12.0},
                    "delta": {"signal_count_delta": -55, "label_count_delta": -55, "hit_rate_delta": 0.15, "avg_return_delta_bps": 8.0},
                },
                "walk_forward_comparison": {
                    "summary": {
                        "robustness_status": "ROBUST",
                        "evaluated_split_count": 3,
                        "split_count": 3,
                        "positive_delta_rate": 1.0,
                        "avg_holdout_return_delta_bps": 6.0,
                    },
                },
                "regime_comparison": [],
            },
        }
        text = "\n".join(_section_threshold_policy_experiment(artifact))

        assert "Threshold Policy Experiment" in text
        assert "APPROVED_FOR_POLICY_EXPERIMENT" in text
        assert "composite_signal_score>=75.0" in text

    def test_threshold_shadow_simulation_section_shows_signal_stream_impact(self):
        artifact = {
            "json_path": "threshold_shadow_simulation.json",
            "markdown_path": "threshold_shadow_simulation.md",
            "retained_signals_csv_path": "retained.csv",
            "suppressed_signals_csv_path": "suppressed.csv",
            "report": {
                "shadow_status": "SHADOW_SIMULATION_READY",
                "runtime_config_changed": False,
                "shadow_reasons": ["Approved threshold replayed as shadow simulation."],
                "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
                "impact_summary": {
                    "eligible_signal_count": 100,
                    "retained_signal_count": 60,
                    "suppressed_signal_count": 40,
                    "retention_ratio": 0.6,
                    "false_positive_removed_count": 25,
                    "true_positive_lost_count": 5,
                    "avoided_suppressed_return_bps": 120.0,
                },
                "baseline_metrics": {"signal_count": 100, "label_count_60m": 100, "hit_rate_60m": 0.55, "avg_signed_return_60m_bps": 4.0},
                "retained_metrics": {"signal_count": 60, "label_count_60m": 60, "hit_rate_60m": 0.7, "avg_signed_return_60m_bps": 12.0},
                "suppressed_metrics": {"signal_count": 40, "label_count_60m": 40, "hit_rate_60m": 0.35, "avg_signed_return_60m_bps": -3.0},
                "retained_vs_baseline_delta": {"signal_count_delta": -40, "label_count_delta": -40, "hit_rate_delta": 0.15, "avg_return_delta_bps": 8.0},
                "regime_shadow": [],
            },
        }
        text = "\n".join(_section_threshold_shadow_simulation(artifact))

        assert "Threshold Shadow Simulation" in text
        assert "SHADOW_SIMULATION_READY" in text
        assert "False positives removed" in text

    def test_threshold_shadow_review_section_shows_promotion_readiness(self):
        artifact = {
            "json_path": "threshold_shadow_review.json",
            "markdown_path": "threshold_shadow_review.md",
            "segments_csv_path": "segments.csv",
            "report": {
                "review_status": "PROMOTION_READY",
                "runtime_config_changed": False,
                "requires_manual_promotion_review": True,
                "recommended_next_action": "Open a human promotion review.",
                "review_reasons": ["Shadow evidence meets guardrails."],
                "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
                "observation_summary": {"distinct_signal_dates": 60},
                "impact_summary": {
                    "eligible_signal_count": 100,
                    "retained_signal_count": 70,
                    "suppressed_signal_count": 30,
                    "true_positive_lost_count": 0,
                },
                "guardrail_summary": {
                    "false_positive_removed_count": 25,
                    "false_positive_removal_rate": 0.83,
                    "true_positive_loss_rate": 0.0,
                    "bad_regime_count": 0,
                    "bad_bucket_count": 0,
                },
                "retained_vs_baseline_delta": {"hit_rate_delta": 0.14, "avg_return_delta_bps": 8.0},
                "segment_failures": [],
            },
        }
        text = "\n".join(_section_threshold_shadow_review(artifact))

        assert "Threshold Shadow Review" in text
        assert "PROMOTION_READY" in text
        assert "Manual promotion review required" in text

    def test_threshold_promotion_review_section_shows_manual_package(self):
        artifact = {
            "json_path": "threshold_promotion_review.json",
            "markdown_path": "threshold_promotion_review.md",
            "review_ledger_path": "threshold_promotion_review_ledger.csv",
            "report": {
                "promotion_review_status": "PROMOTION_REVIEW_READY",
                "runtime_config_changed": False,
                "manual_review_required": True,
                "recommended_next_action": "Open the promotion review ledger.",
                "promotion_review_reasons": ["Shadow review is PROMOTION_READY."],
                "status_chain": {
                    "governance_status": "PROMOTE_TO_REVIEW",
                    "policy_experiment_status": "APPROVED_FOR_POLICY_EXPERIMENT",
                    "shadow_simulation_status": "SHADOW_SIMULATION_READY",
                    "shadow_review_status": "PROMOTION_READY",
                },
                "promotion_candidate": {
                    "source_candidate_key": "composite_signal_score>=75.0",
                    "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
                    "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
                },
                "impact_summary": {
                    "eligible_signal_count": 100,
                    "retained_signal_count": 70,
                    "suppressed_signal_count": 30,
                },
                "guardrail_summary": {"false_positive_removal_rate": 0.83, "true_positive_loss_rate": 0.0},
                "risk_flags": {
                    "false_positive_removed_count": 25,
                    "true_positive_lost_count": 0,
                    "segment_failure_count": 0,
                },
                "retained_vs_baseline_delta": {"hit_rate_delta": 0.14, "avg_return_delta_bps": 8.0},
                "monitoring_plan": ["Track retained/suppressed counts."],
                "rollback_notes": ["No rollback required from package generation."],
            },
        }
        text = "\n".join(_section_threshold_promotion_review(artifact))

        assert "Threshold Promotion Review Package" in text
        assert "PROMOTION_REVIEW_READY" in text
        assert "composite_signal_score>=75.0" in text

    def test_threshold_post_promotion_monitor_section_shows_health_status(self):
        artifact = {
            "json_path": "threshold_post_promotion_monitor.json",
            "markdown_path": "threshold_post_promotion_monitor.md",
            "segments_csv_path": "threshold_post_promotion_monitor_segments.csv",
            "report": {
                "monitor_status": "POST_PROMOTION_HEALTHY",
                "runtime_config_changed": False,
                "recommended_next_action": "Reaffirm the approved candidate.",
                "monitor_reasons": ["Post-approval evidence remains consistent."],
                "threshold_rule": {"field": "composite_signal_score", "operator": ">=", "value": 75.0},
                "post_approval_window": {
                    "approval_timestamp": "2026-02-01T00:00:00Z",
                    "post_approval_signal_count": 60,
                    "post_approval_signal_dates": 60,
                },
                "post_approval_impact": {
                    "eligible_signal_count": 60,
                    "retained_signal_count": 40,
                    "suppressed_signal_count": 20,
                    "false_positive_removed_count": 20,
                    "true_positive_lost_count": 0,
                },
                "post_approval_retained_metrics": {
                    "label_count_60m": 40,
                    "hit_rate_60m": 1.0,
                    "avg_signed_return_60m_bps": 23.0,
                },
                "drift_from_shadow_expectation": {
                    "retained_hit_rate_delta_vs_shadow": 0.0,
                    "retained_avg_return_delta_bps_vs_shadow": -1.0,
                    "true_positive_lost_delta_vs_shadow": 0,
                },
                "segment_summary": {
                    "deteriorating_segment_count": 0,
                    "watch_segment_count": 0,
                },
                "segment_monitoring": [],
            },
        }
        text = "\n".join(_section_threshold_post_promotion_monitor(artifact))

        assert "Threshold Post-Promotion Monitor" in text
        assert "POST_PROMOTION_HEALTHY" in text
        assert "Avg-return drift" in text

    def test_threshold_adoption_reconciliation_section_shows_adoption_status(self):
        artifact = {
            "json_path": "threshold_adoption_reconciliation.json",
            "markdown_path": "threshold_adoption_reconciliation.md",
            "comparison_csv_path": "threshold_adoption_reconciliation_comparison.csv",
            "report": {
                "adoption_status": "ADOPTED_MANUALLY",
                "runtime_config_changed": False,
                "recommended_next_action": "Continue post-promotion monitoring.",
                "adoption_reasons": ["Active runtime value matches the approved threshold candidate."],
                "comparison": {
                    "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
                    "candidate_value": 82.0,
                    "observed_runtime_value": 82.0,
                    "observed_runtime_source": "runtime_config",
                    "default_runtime_value": 75.0,
                    "matches_candidate": True,
                    "matches_default": False,
                },
                "runtime_state": {
                    "active_parameter_pack": {"name": "baseline_v1", "layers": ["baseline_v1"], "override_keys": []},
                },
                "post_promotion_monitor_summary": {"monitor_status": "POST_PROMOTION_HEALTHY"},
            },
        }
        text = "\n".join(_section_threshold_adoption_reconciliation(artifact))

        assert "Threshold Adoption Reconciliation" in text
        assert "ADOPTED_MANUALLY" in text
        assert "Active runtime value" in text

    def test_alpha_decay_has_ascii_chart(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_alpha_decay(df)
        text = "\n".join(lines)
        assert "Decay Curve (ASCII)" in text
        assert "bps" in text

    def test_premium_analytics_has_premium_tables(self):
        df = _sample_dataset()
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"])
        lines = _section_premium_analytics(df)
        text = "\n".join(lines)
        assert "Premium Analytics" in text
        assert "Avg Entry Premium" in text
        assert "Premium Band" in text

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

    def test_generate_report_writes_reproducibility_manifest(self, tmp_path):
        df = _sample_dataset()
        df.loc[1, "signal_timestamp"] = "2026-03-16 10:05:00+05:30"
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"

        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
            run_evaluation=False,
            narrative=False,
        )

        manifest_path = report_path.with_suffix(".manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["report_kind"] == "daily_signal_research_report"
        assert manifest["report_date"] == "2026-03-16"
        assert manifest["mode"] == "daily"
        assert manifest["dataset"]["sha256"]
        assert manifest["report"]["sha256"]
        assert manifest["frame"]["row_count"] == 10
        assert manifest["timestamp_parse"]["parsed_count"] == 10
        assert manifest["timestamp_parse"]["failed_count"] == 0

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
            "Signal Drift Monitor",
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
            "Premium Analytics",
            "Score Calibration",
            "Threshold Replay Diagnostics",
            "Threshold Governance",
            "Threshold Policy Experiment",
            "Threshold Shadow Simulation",
            "Threshold Shadow Review",
            "Threshold Promotion Review Package",
            "Threshold Post-Promotion Monitor",
            "Threshold Adoption Reconciliation",
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

    def test_report_writes_latest_signal_drift_artifact(self, tmp_path):
        df = _sample_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 3, 16),
            dataset_path=csv_path,
            output_dir=output_dir,
        )

        latest_json = output_dir / "drift_monitoring" / "latest_signal_drift.json"
        latest_md = output_dir / "drift_monitoring" / "latest_signal_drift.md"
        trend_history = output_dir / "drift_monitoring" / "signal_drift_trend_history.csv"
        trend_json = output_dir / "drift_monitoring" / "latest_signal_drift_trend.json"
        trend_md = output_dir / "drift_monitoring" / "latest_signal_drift_trend.md"
        threshold_json = output_dir / "threshold_governance" / "latest_threshold_governance.json"
        threshold_md = output_dir / "threshold_governance" / "latest_threshold_governance.md"
        threshold_candidates = output_dir / "threshold_governance" / "latest_threshold_governance_candidates.csv"
        policy_experiment_json = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment.json"
        policy_experiment_md = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment.md"
        policy_experiment_pack = output_dir / "threshold_policy_experiments" / "latest_candidate_threshold_policy_pack.json"
        shadow_json = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation.json"
        shadow_md = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation.md"
        shadow_retained = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_retained_signals.csv"
        shadow_suppressed = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_suppressed_signals.csv"
        shadow_review_json = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review.json"
        shadow_review_md = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review.md"
        shadow_review_segments = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review_segments.csv"
        promotion_review_json = output_dir / "threshold_promotion_review" / "latest_threshold_promotion_review.json"
        promotion_review_md = output_dir / "threshold_promotion_review" / "latest_threshold_promotion_review.md"
        promotion_review_ledger = output_dir / "threshold_promotion_review" / "threshold_promotion_review_ledger.csv"
        post_promotion_json = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor.json"
        post_promotion_md = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor.md"
        post_promotion_segments = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor_segments.csv"
        assert report_path.exists()
        assert latest_json.exists()
        assert latest_md.exists()
        assert trend_history.exists()
        assert trend_json.exists()
        assert trend_md.exists()
        assert threshold_json.exists()
        assert threshold_md.exists()
        assert threshold_candidates.exists()
        assert policy_experiment_json.exists()
        assert policy_experiment_md.exists()
        assert policy_experiment_pack.exists()
        assert shadow_json.exists()
        assert shadow_md.exists()
        assert shadow_retained.exists()
        assert shadow_suppressed.exists()
        assert shadow_review_json.exists()
        assert shadow_review_md.exists()
        assert shadow_review_segments.exists()
        assert promotion_review_json.exists()
        assert promotion_review_md.exists()
        assert promotion_review_ledger.parent.exists()
        assert post_promotion_json.exists()
        assert post_promotion_md.exists()
        assert post_promotion_segments.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Signal Drift Monitor" in content
        assert "Trend dashboard" in content
        assert "Threshold Governance" in content
        assert "Threshold Policy Experiment" in content
        assert "Threshold Shadow Simulation" in content
        assert "Threshold Shadow Review" in content
        assert "Threshold Promotion Review Package" in content
        assert "Threshold Post-Promotion Monitor" in content
        policy_payload = json.loads(policy_experiment_json.read_text(encoding="utf-8"))
        assert policy_payload["experiment_status"] == "SKIPPED_NO_PROMOTED_CANDIDATE"
        shadow_payload = json.loads(shadow_json.read_text(encoding="utf-8"))
        assert shadow_payload["shadow_status"] == "SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED"
        shadow_review_payload = json.loads(shadow_review_json.read_text(encoding="utf-8"))
        assert shadow_review_payload["review_status"] == "SKIPPED_SHADOW_NOT_READY"
        promotion_review_payload = json.loads(promotion_review_json.read_text(encoding="utf-8"))
        assert promotion_review_payload["promotion_review_status"] == "SKIPPED_SHADOW_REVIEW_NOT_READY"
        post_promotion_payload = json.loads(post_promotion_json.read_text(encoding="utf-8"))
        assert post_promotion_payload["monitor_status"] == "POST_PROMOTION_SKIPPED_NO_APPROVAL"

    def test_report_runs_threshold_policy_experiment_for_promoted_candidate(self, tmp_path):
        df = _policy_experiment_dataset()
        csv_path = tmp_path / "signals_dataset.csv"
        df.to_csv(csv_path, index=False)
        output_dir = tmp_path / "reports"
        report_path = generate_daily_report(
            report_date=date(2026, 4, 30),
            dataset_path=csv_path,
            output_dir=output_dir,
            run_evaluation=False,
        )

        policy_experiment_json = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment.json"
        policy_experiment_md = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment.md"
        policy_experiment_pack = output_dir / "threshold_policy_experiments" / "latest_candidate_threshold_policy_pack.json"
        policy_experiment_splits = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment_splits.csv"
        policy_experiment_regimes = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment_regimes.csv"
        policy_experiment_quality = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment_quality_buckets.csv"
        shadow_json = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation.json"
        shadow_md = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation.md"
        shadow_retained = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_retained_signals.csv"
        shadow_suppressed = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_suppressed_signals.csv"
        shadow_regimes = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_regimes.csv"
        shadow_buckets = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_buckets.csv"
        shadow_review_json = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review.json"
        shadow_review_md = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review.md"
        shadow_review_segments = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review_segments.csv"
        promotion_review_json = output_dir / "threshold_promotion_review" / "latest_threshold_promotion_review.json"
        promotion_review_md = output_dir / "threshold_promotion_review" / "latest_threshold_promotion_review.md"
        promotion_review_ledger = output_dir / "threshold_promotion_review" / "threshold_promotion_review_ledger.csv"
        post_promotion_json = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor.json"
        post_promotion_md = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor.md"
        post_promotion_segments = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor_segments.csv"

        assert report_path.exists()
        assert policy_experiment_json.exists()
        assert policy_experiment_md.exists()
        assert policy_experiment_pack.exists()
        assert policy_experiment_splits.exists()
        assert policy_experiment_regimes.exists()
        assert policy_experiment_quality.exists()
        assert shadow_json.exists()
        assert shadow_md.exists()
        assert shadow_retained.exists()
        assert shadow_suppressed.exists()
        assert shadow_regimes.exists()
        assert shadow_buckets.exists()
        assert shadow_review_json.exists()
        assert shadow_review_md.exists()
        assert shadow_review_segments.exists()
        assert promotion_review_json.exists()
        assert promotion_review_md.exists()
        assert promotion_review_ledger.parent.exists()
        assert post_promotion_json.exists()
        assert post_promotion_md.exists()
        assert post_promotion_segments.exists()
        policy_payload = json.loads(policy_experiment_json.read_text(encoding="utf-8"))
        assert policy_payload["experiment_status"] == "APPROVED_FOR_POLICY_EXPERIMENT"
        assert policy_payload["runtime_config_changed"] is False
        assert policy_payload["candidate_policy_pack"]["research_only"] is True
        assert policy_payload["full_sample_comparison"]["delta"]["avg_return_delta_bps"] > 0
        shadow_payload = json.loads(shadow_json.read_text(encoding="utf-8"))
        assert shadow_payload["shadow_status"] == "SHADOW_SIMULATION_READY"
        assert shadow_payload["impact_summary"]["false_positive_removed_count"] == 30
        assert shadow_payload["impact_summary"]["true_positive_lost_count"] == 0
        shadow_review_payload = json.loads(shadow_review_json.read_text(encoding="utf-8"))
        assert shadow_review_payload["review_status"] == "PROMOTION_READY"
        assert shadow_review_payload["requires_manual_promotion_review"] is True
        promotion_review_payload = json.loads(promotion_review_json.read_text(encoding="utf-8"))
        assert promotion_review_payload["promotion_review_status"] == "PROMOTION_REVIEW_READY"
        assert promotion_review_payload["manual_review_required"] is True
        assert promotion_review_payload["runtime_config_changed"] is False
        post_promotion_payload = json.loads(post_promotion_json.read_text(encoding="utf-8"))
        assert post_promotion_payload["monitor_status"] == "POST_PROMOTION_SKIPPED_NO_APPROVAL"

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
