#!/usr/bin/env python3
"""Run drift alerting from the latest signal drift trend dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.drift_alerts import (  # noqa: E402
    REVIEW_ACTIONS,
    default_trend_dashboard_path,
    run_signal_drift_alert_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build signal drift alert artifacts from the trend dashboard.")
    parser.add_argument("--trend-dashboard", type=Path, default=default_trend_dashboard_path())
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for latest alert JSON/Markdown.")
    parser.add_argument("--review-ledger", type=Path, default=None, help="CSV ledger for human review actions.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when ops status is DETERIORATING.")
    parser.add_argument("--review-action", choices=sorted(REVIEW_ACTIONS), default=None)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--review-note", default=None)
    parser.add_argument("--next-review-at", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = run_signal_drift_alert_workflow(
        trend_dashboard_path=args.trend_dashboard,
        output_dir=args.output_dir,
        review_ledger_path=args.review_ledger,
        review_action=args.review_action,
        reviewer=args.reviewer,
        review_note=args.review_note,
        next_review_at=args.next_review_at,
    )
    summary = artifact["alert_summary"]
    payload = {
        "ops_status": summary.get("ops_status"),
        "trend_assessment": summary.get("trend_assessment"),
        "strict_failure": summary.get("strict_failure"),
        "alert_json_path": artifact.get("alert_json_path"),
        "alert_markdown_path": artifact.get("alert_markdown_path"),
        "review_ledger_path": artifact.get("review_ledger_path"),
        "review_action": (artifact.get("review_artifact") or {}).get("review_row", {}).get("review_action"),
        "alert_reasons": summary.get("alert_reasons", []),
        "warning_digest": summary.get("warning_digest", [])[:10],
    }
    print(json.dumps(payload, indent=2, default=str))
    if args.strict and summary.get("ops_status") == "DETERIORATING":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
