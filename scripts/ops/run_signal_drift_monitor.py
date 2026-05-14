#!/usr/bin/env python3
"""Run the quality-aware signal drift monitor against a signal dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, SIGNAL_DATASET_PATH, load_signals_dataset
from research.signal_evaluation.drift_monitor import write_signal_drift_report
from config.signal_drift_policy import get_signal_drift_monitor_policy


def _default_dataset_path() -> Path:
    return CUMULATIVE_DATASET_PATH if CUMULATIVE_DATASET_PATH.exists() else SIGNAL_DATASET_PATH


def _parse_args() -> argparse.Namespace:
    policy = get_signal_drift_monitor_policy()
    parser = argparse.ArgumentParser(description="Build a quality-aware signal drift monitoring report.")
    parser.add_argument("--dataset", type=Path, default=_default_dataset_path())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--recent-days", type=int, default=None, help=f"Recent window in trading days (default: {policy['recent_days']}).")
    parser.add_argument("--baseline-days", type=int, default=None, help=f"Baseline window in trading days; 0 means all prior days (default: {policy['baseline_days']}).")
    parser.add_argument("--min-recent-labeled", type=int, default=None, help=f"Recent quality-label sample floor (default: {policy['min_recent_labeled']}).")
    parser.add_argument("--min-baseline-labeled", type=int, default=None, help=f"Baseline quality-label sample floor (default: {policy['min_baseline_labeled']}).")
    parser.add_argument("--hit-rate-drop-warn", type=float, default=None, help=f"Warn when recent hit rate drops by this amount (default: {policy['hit_rate_drop_warn']}).")
    parser.add_argument("--return-drop-bps-warn", type=float, default=None, help=f"Warn when recent avg return drops by this many bps (default: {policy['return_drop_bps_warn']}).")
    parser.add_argument("--calibration-gap-delta-warn", type=float, default=None, help=f"Warn when absolute calibration gap widens by this amount (default: {policy['calibration_gap_delta_warn']}).")
    parser.add_argument("--label-coverage-drop-warn", type=float, default=None, help=f"Warn when quality-label coverage drops by this amount (default: {policy['label_coverage_drop_warn']}).")
    parser.add_argument("--retention-delta-warn", type=float, default=None, help=f"Warn when policy retention changes by this amount (default: {policy['retention_delta_warn']}).")
    parser.add_argument("--trend-lookback-runs", type=int, default=20, help="Number of recent drift runs to summarize in the trend dashboard.")
    parser.add_argument("--no-trend-history", action="store_true", help="Do not append trend history or refresh the trend dashboard.")
    parser.add_argument("--no-policy-application", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    frame = load_signals_dataset(args.dataset)
    artifact = write_signal_drift_report(
        frame,
        dataset_path=str(args.dataset),
        output_dir=args.output_dir,
        report_name=args.report_name,
        recent_days=args.recent_days,
        baseline_days=args.baseline_days,
        min_recent_labeled=args.min_recent_labeled,
        min_baseline_labeled=args.min_baseline_labeled,
        hit_rate_drop_warn=args.hit_rate_drop_warn,
        return_drop_bps_warn=args.return_drop_bps_warn,
        calibration_gap_delta_warn=args.calibration_gap_delta_warn,
        label_coverage_drop_warn=args.label_coverage_drop_warn,
        retention_delta_warn=args.retention_delta_warn,
        apply_missing_policies=False if args.no_policy_application else None,
        write_trend_history=not args.no_trend_history,
        trend_lookback_runs=args.trend_lookback_runs,
    )
    report = artifact["report"]
    dashboard = artifact.get("trend_dashboard") or {}
    print(
        json.dumps(
            {
                "monitor_status": report.get("monitor_status"),
                "json_path": artifact.get("json_path"),
                "markdown_path": artifact.get("markdown_path"),
                "trend_history_path": artifact.get("trend_history_path"),
                "trend_dashboard_json_path": artifact.get("trend_dashboard_json_path"),
                "trend_dashboard_markdown_path": artifact.get("trend_dashboard_markdown_path"),
                "trend_assessment": dashboard.get("trend_assessment"),
                "warning_count": len(report.get("warnings", [])),
                "warnings": report.get("warnings", [])[:10],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
