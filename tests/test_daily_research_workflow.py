from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPT = ROOT / "scripts" / "ops" / "run_daily_research_workflow.py"


def _workflow_dataset() -> pd.DataFrame:
    rows = []
    for idx, day in enumerate(pd.date_range("2026-04-01", periods=4, freq="D")):
        rows.append(
            {
                "signal_id": f"base-{idx}",
                "signal_timestamp": f"{day.date()}T09:20:00+05:30",
                "symbol": "NIFTY",
                "source": "unit",
                "mode": "TEST",
                "direction": "CALL",
                "trade_status": "TRADE",
                "signal_regime": "EXPANSION_BIAS",
                "macro_regime": "RISK_ON",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "global_risk_state": "CALM",
                "correct_60m": 1,
                "signed_return_60m_bps": 20.0,
                "calibration_label": 1,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 20.0,
                "label_quality_status": "CLEAN",
                "hybrid_move_probability": 0.72,
                "move_probability": 0.72,
                "ml_confidence_score": 0.70,
                "ml_rank_score": 0.80,
                "trade_strength": 80,
                "composite_signal_score": 78,
                "tradeability_score": 76,
                "entry_price": 110.0,
            }
        )
    for idx in range(2):
        rows.append(
            {
                "signal_id": f"recent-{idx}",
                "signal_timestamp": "2026-04-05T09:20:00+05:30",
                "symbol": "NIFTY",
                "source": "unit",
                "mode": "TEST",
                "direction": "PUT",
                "trade_status": "TRADE",
                "signal_regime": "CONFLICTED",
                "macro_regime": "RISK_OFF",
                "gamma_regime": "LONG_GAMMA_ZONE",
                "volatility_regime": "ELEVATED",
                "global_risk_state": "CAUTION",
                "correct_60m": 0,
                "signed_return_60m_bps": -15.0,
                "calibration_label": 0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": -15.0,
                "label_quality_status": "CLEAN",
                "hybrid_move_probability": 0.48,
                "move_probability": 0.48,
                "ml_confidence_score": 0.45,
                "ml_rank_score": 0.35,
                "trade_strength": 42,
                "composite_signal_score": 40,
                "tradeability_score": 44,
                "entry_price": 95.0,
            }
        )
    return pd.DataFrame(rows)


def test_daily_research_workflow_writes_checkpoint_history_and_drift(tmp_path: Path):
    dataset_path = tmp_path / "signals_dataset.csv"
    output_dir = tmp_path / "daily_reports"
    history_dir = tmp_path / "daily_ops_runs"
    _workflow_dataset().to_csv(dataset_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(WORKFLOW_SCRIPT),
            "--date",
            "2026-04-05",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--history-dir",
            str(history_dir),
            "--skip-evaluation",
            "--include-cumulative",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["status"] == "SUCCESS"

    daily_report = Path(payload["daily_report_path"])
    cumulative_report = Path(payload["cumulative_report_path"])
    drift_latest = output_dir / "drift_monitoring" / "latest_signal_drift.json"
    drift_trend_history = output_dir / "drift_monitoring" / "signal_drift_trend_history.csv"
    drift_trend_dashboard = output_dir / "drift_monitoring" / "latest_signal_drift_trend.json"
    drift_alert = output_dir / "drift_monitoring" / "latest_signal_drift_alert.json"
    threshold_governance_json = output_dir / "threshold_governance" / "latest_threshold_governance.json"
    threshold_governance_markdown = output_dir / "threshold_governance" / "latest_threshold_governance.md"
    threshold_governance_candidates = output_dir / "threshold_governance" / "latest_threshold_governance_candidates.csv"
    threshold_policy_experiment_json = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment.json"
    threshold_policy_experiment_markdown = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment.md"
    threshold_policy_experiment_pack = output_dir / "threshold_policy_experiments" / "latest_candidate_threshold_policy_pack.json"
    threshold_policy_experiment_splits = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment_splits.csv"
    threshold_policy_experiment_regimes = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment_regimes.csv"
    threshold_policy_experiment_quality = output_dir / "threshold_policy_experiments" / "latest_threshold_policy_experiment_quality_buckets.csv"
    threshold_shadow_json = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation.json"
    threshold_shadow_markdown = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation.md"
    threshold_shadow_retained = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_retained_signals.csv"
    threshold_shadow_suppressed = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_suppressed_signals.csv"
    threshold_shadow_regimes = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_regimes.csv"
    threshold_shadow_buckets = output_dir / "threshold_shadow_simulation" / "latest_threshold_shadow_simulation_buckets.csv"
    threshold_shadow_review_json = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review.json"
    threshold_shadow_review_markdown = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review.md"
    threshold_shadow_review_segments = output_dir / "threshold_shadow_review" / "latest_threshold_shadow_review_segments.csv"
    threshold_promotion_review_json = output_dir / "threshold_promotion_review" / "latest_threshold_promotion_review.json"
    threshold_promotion_review_markdown = output_dir / "threshold_promotion_review" / "latest_threshold_promotion_review.md"
    threshold_promotion_review_ledger = output_dir / "threshold_promotion_review" / "threshold_promotion_review_ledger.csv"
    threshold_post_promotion_json = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor.json"
    threshold_post_promotion_markdown = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor.md"
    threshold_post_promotion_segments = output_dir / "threshold_post_promotion_monitoring" / "latest_threshold_post_promotion_monitor_segments.csv"
    threshold_adoption_reconciliation_json = output_dir / "threshold_adoption_reconciliation" / "latest_threshold_adoption_reconciliation.json"
    threshold_adoption_reconciliation_markdown = output_dir / "threshold_adoption_reconciliation" / "latest_threshold_adoption_reconciliation.md"
    threshold_adoption_reconciliation_comparison = output_dir / "threshold_adoption_reconciliation" / "latest_threshold_adoption_reconciliation_comparison.csv"
    history_csv = history_dir / "run_history.csv"
    checkpoint_json = Path(payload["checkpoint_json"])

    assert daily_report.exists()
    assert cumulative_report.exists()
    assert drift_latest.exists()
    assert drift_trend_history.exists()
    assert drift_trend_dashboard.exists()
    assert drift_alert.exists()
    assert threshold_governance_json.exists()
    assert threshold_governance_markdown.exists()
    assert threshold_governance_candidates.exists()
    assert threshold_policy_experiment_json.exists()
    assert threshold_policy_experiment_markdown.exists()
    assert threshold_policy_experiment_pack.exists()
    assert threshold_policy_experiment_splits.exists()
    assert threshold_policy_experiment_regimes.exists()
    assert threshold_policy_experiment_quality.exists()
    assert threshold_shadow_json.exists()
    assert threshold_shadow_markdown.exists()
    assert threshold_shadow_retained.exists()
    assert threshold_shadow_suppressed.exists()
    assert threshold_shadow_regimes.exists()
    assert threshold_shadow_buckets.exists()
    assert threshold_shadow_review_json.exists()
    assert threshold_shadow_review_markdown.exists()
    assert threshold_shadow_review_segments.exists()
    assert threshold_promotion_review_json.exists()
    assert threshold_promotion_review_markdown.exists()
    assert threshold_promotion_review_ledger.parent.exists()
    assert threshold_post_promotion_json.exists()
    assert threshold_post_promotion_markdown.exists()
    assert threshold_post_promotion_segments.exists()
    assert threshold_adoption_reconciliation_json.exists()
    assert threshold_adoption_reconciliation_markdown.exists()
    assert threshold_adoption_reconciliation_comparison.exists()
    assert history_csv.exists()
    assert checkpoint_json.exists()
    assert payload["drift_latest_json"] == str(drift_latest)
    assert payload["drift_trend_history_csv"] == str(drift_trend_history)
    assert payload["drift_trend_dashboard_json"] == str(drift_trend_dashboard)
    assert payload["drift_alert_json"] == str(drift_alert)
    assert payload["drift_alert_status"] in {"STABLE", "WATCH", "DETERIORATING", "NO_HISTORY"}
    assert payload["threshold_governance_json"] == str(threshold_governance_json)
    assert payload["threshold_governance_markdown"] == str(threshold_governance_markdown)
    assert payload["threshold_governance_candidates_csv"] == str(threshold_governance_candidates)
    assert payload["threshold_governance_status"] in {
        "PROMOTE_TO_REVIEW",
        "WATCHLIST",
        "REJECT_INSUFFICIENT_EVIDENCE",
        "REJECT_UNSTABLE",
    }
    assert payload["threshold_policy_experiment_json"] == str(threshold_policy_experiment_json)
    assert payload["threshold_policy_experiment_markdown"] == str(threshold_policy_experiment_markdown)
    assert payload["threshold_policy_experiment_policy_pack_json"] == str(threshold_policy_experiment_pack)
    assert payload["threshold_policy_experiment_splits_csv"] == str(threshold_policy_experiment_splits)
    assert payload["threshold_policy_experiment_regimes_csv"] == str(threshold_policy_experiment_regimes)
    assert payload["threshold_policy_experiment_quality_buckets_csv"] == str(threshold_policy_experiment_quality)
    assert payload["threshold_policy_experiment_status"] in {
        "APPROVED_FOR_POLICY_EXPERIMENT",
        "REVIEW_REQUIRED",
        "REJECTED_FOR_POLICY_EXPERIMENT",
        "INSUFFICIENT_EVIDENCE",
        "SKIPPED_NO_PROMOTED_CANDIDATE",
    }
    assert payload["threshold_shadow_simulation_json"] == str(threshold_shadow_json)
    assert payload["threshold_shadow_simulation_markdown"] == str(threshold_shadow_markdown)
    assert payload["threshold_shadow_simulation_retained_signals_csv"] == str(threshold_shadow_retained)
    assert payload["threshold_shadow_simulation_suppressed_signals_csv"] == str(threshold_shadow_suppressed)
    assert payload["threshold_shadow_simulation_regimes_csv"] == str(threshold_shadow_regimes)
    assert payload["threshold_shadow_simulation_buckets_csv"] == str(threshold_shadow_buckets)
    assert payload["threshold_shadow_simulation_status"] in {
        "SHADOW_SIMULATION_READY",
        "SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED",
        "INSUFFICIENT_SHADOW_EVIDENCE",
    }
    assert payload["threshold_shadow_review_json"] == str(threshold_shadow_review_json)
    assert payload["threshold_shadow_review_markdown"] == str(threshold_shadow_review_markdown)
    assert payload["threshold_shadow_review_segments_csv"] == str(threshold_shadow_review_segments)
    assert payload["threshold_shadow_review_status"] in {
        "PROMOTION_READY",
        "NEEDS_MORE_SHADOW_DATA",
        "REJECTED_SHADOW_REGRESSION",
        "REJECTED_TRUE_POSITIVE_LOSS",
        "REJECTED_REGIME_DEGRADATION",
        "SKIPPED_SHADOW_NOT_READY",
    }
    assert payload["threshold_promotion_review_json"] == str(threshold_promotion_review_json)
    assert payload["threshold_promotion_review_markdown"] == str(threshold_promotion_review_markdown)
    assert payload["threshold_promotion_review_ledger_csv"] == str(threshold_promotion_review_ledger)
    assert payload["threshold_promotion_review_status"] in {
        "PROMOTION_REVIEW_READY",
        "SKIPPED_SHADOW_REVIEW_NOT_READY",
    }
    assert payload["threshold_post_promotion_monitor_json"] == str(threshold_post_promotion_json)
    assert payload["threshold_post_promotion_monitor_markdown"] == str(threshold_post_promotion_markdown)
    assert payload["threshold_post_promotion_monitor_segments_csv"] == str(threshold_post_promotion_segments)
    assert payload["threshold_post_promotion_monitor_status"] in {
        "POST_PROMOTION_HEALTHY",
        "POST_PROMOTION_WATCH",
        "POST_PROMOTION_DETERIORATING",
        "POST_PROMOTION_INSUFFICIENT_DATA",
        "POST_PROMOTION_SKIPPED_NO_APPROVAL",
    }
    assert payload["threshold_adoption_reconciliation_json"] == str(threshold_adoption_reconciliation_json)
    assert payload["threshold_adoption_reconciliation_markdown"] == str(threshold_adoption_reconciliation_markdown)
    assert payload["threshold_adoption_reconciliation_comparison_csv"] == str(threshold_adoption_reconciliation_comparison)
    assert payload["threshold_adoption_reconciliation_status"] in {
        "APPROVED_BUT_NOT_ADOPTED",
        "ADOPTED_MANUALLY",
        "ADOPTION_MISMATCH",
        "ROLLED_BACK_MANUALLY",
        "UNKNOWN_ADOPTION_STATE",
    }

    checkpoint = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    assert checkpoint["run_id"] == payload["run_id"]
    assert checkpoint["history_csv"] == str(history_csv)

    with history_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "SUCCESS"
    assert rows[0]["checkpoint_json"] == str(checkpoint_json)
    assert rows[0]["drift_alert_json"] == str(drift_alert)
    assert rows[0]["threshold_governance_status"] == payload["threshold_governance_status"]
    assert rows[0]["threshold_policy_experiment_status"] == payload["threshold_policy_experiment_status"]
    assert rows[0]["threshold_shadow_simulation_status"] == payload["threshold_shadow_simulation_status"]
    assert rows[0]["threshold_shadow_review_status"] == payload["threshold_shadow_review_status"]
    assert rows[0]["threshold_promotion_review_status"] == payload["threshold_promotion_review_status"]
    assert rows[0]["threshold_post_promotion_monitor_status"] == payload["threshold_post_promotion_monitor_status"]
    assert rows[0]["threshold_adoption_reconciliation_status"] == payload["threshold_adoption_reconciliation_status"]
