#!/usr/bin/env python3
"""Append threshold adoption status to history and refresh the dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_adoption_history import (  # noqa: E402
    DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH,
    DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH,
    DEFAULT_ROLLOUT_MONITOR_REPORT_PATH,
    DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR,
    LIFECYCLE_MISMATCHED,
    LIFECYCLE_ROLLED_BACK,
    write_threshold_adoption_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append the latest threshold adoption/rollout state to an ops history. "
            "This command reads report artifacts only; it does not run the engine or change runtime config."
        )
    )
    parser.add_argument("--rollout-monitor", type=Path, default=DEFAULT_ROLLOUT_MONITOR_REPORT_PATH)
    parser.add_argument("--adoption-reconciliation", type=Path, default=DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH)
    parser.add_argument("--post-promotion-monitor", type=Path, default=DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR)
    parser.add_argument("--history-filename", default="threshold_adoption_history.csv")
    parser.add_argument("--lookback-runs", type=int, default=20)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with status 2 when the latest lifecycle status is MISMATCHED or ROLLED_BACK.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_threshold_adoption_history(
        rollout_report_path=args.rollout_monitor,
        adoption_reconciliation_report_path=args.adoption_reconciliation,
        post_promotion_monitor_report_path=args.post_promotion_monitor,
        output_dir=args.output_dir,
        history_filename=args.history_filename,
        lookback_runs=args.lookback_runs,
    )
    row = artifact.get("history_row", {}) or {}
    dashboard = artifact.get("history_dashboard", {}) or {}
    payload = {
        "history_path": artifact.get("history_path"),
        "history_dashboard_json_path": artifact.get("history_dashboard_json_path"),
        "history_dashboard_markdown_path": artifact.get("history_dashboard_markdown_path"),
        "trend_assessment": dashboard.get("trend_assessment"),
        "adoption_lifecycle_status": row.get("adoption_lifecycle_status"),
        "runtime_signal_status": row.get("runtime_signal_status"),
        "adoption_reconciliation_status": row.get("adoption_reconciliation_status"),
        "rollout_status": row.get("rollout_status"),
        "post_adoption_signal_count": row.get("post_adoption_signal_count"),
        "candidate_pack_signal_count": row.get("candidate_pack_signal_count"),
        "non_candidate_pack_signal_count": row.get("non_candidate_pack_signal_count"),
        "operator_message": dashboard.get("operator_message"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.fail_on_mismatch and row.get("adoption_lifecycle_status") in {LIFECYCLE_MISMATCHED, LIFECYCLE_ROLLED_BACK}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
