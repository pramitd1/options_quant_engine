#!/usr/bin/env python3
"""
Generate a structured current signal evaluation report for human review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(Path(__file__))

from research.signal_evaluation import (
    SIGNAL_DATASET_PATH,
    load_signals_dataset,
    write_signal_evaluation_report,
)
from tuning.promotion import get_active_live_pack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a structured signal evaluation report for the canonical signal dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(SIGNAL_DATASET_PATH),
        help="Path to the canonical signals dataset CSV.",
    )
    parser.add_argument(
        "--production-pack-name",
        default=None,
        help="Optional production pack name. Defaults to the live pack from promotion state.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for report artifacts.",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional stable report name.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maximum rows to keep in top-level grouped summaries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = load_signals_dataset(args.dataset_path)
    production_pack_name = args.production_pack_name or get_active_live_pack()
    report = write_signal_evaluation_report(
        frame,
        production_pack_name=production_pack_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir or (PROJECT_ROOT / "research" / "signal_evaluation" / "reports"),
        report_name=args.report_name,
        top_n=args.top_n,
    )

    summary = report["summary"]
    period = summary["evaluation_period"]
    frequency = summary["signal_frequency"]

    print(f"dataset_path: {args.dataset_path}")
    print(f"production_pack_name: {production_pack_name}")
    print(f"total_signal_count: {summary['total_signal_count']}")
    print(f"evaluation_period: {period['start']} -> {period['end']}")
    print(f"trading_days: {period['trading_days']}")
    print(f"average_signals_per_day: {frequency['average_signals_per_day']}")
    print(f"markdown_report: {report['markdown_path']}")
    print(f"json_report: {report['json_path']}")
    print("top_symbols:")
    for row in summary.get("signals_by_symbol", [])[:5]:
        print(
            f"  - {row.get('symbol')}: count={row.get('signal_count')}, "
            f"hit_rate_60m={row.get('hit_rate_60m')}, "
            f"avg_composite={row.get('avg_composite_signal_score')}"
        )


if __name__ == "__main__":
    main()
