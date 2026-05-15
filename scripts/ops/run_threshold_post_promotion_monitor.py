#!/usr/bin/env python3
"""Run post-promotion monitoring for manually approved threshold candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.daily_research_report import DEFAULT_CUMULATIVE_DATASET_PATH, DEFAULT_DATASET_PATH  # noqa: E402
from research.signal_evaluation.threshold_post_promotion_monitor import (  # noqa: E402
    DEFAULT_PROMOTION_REVIEW_LEDGER_PATH,
    DEFAULT_PROMOTION_REVIEW_REPORT_PATH,
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    write_threshold_post_promotion_monitor_report,
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-promotion monitoring for an approved threshold candidate.")
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--promotion-package", type=Path, default=DEFAULT_PROMOTION_REVIEW_REPORT_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_PROMOTION_REVIEW_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--candidate-key", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    promotion_package = (
        json.loads(args.promotion_package.read_text(encoding="utf-8"))
        if args.promotion_package.exists()
        else None
    )
    artifact = write_threshold_post_promotion_monitor_report(
        frame,
        promotion_package_report=promotion_package,
        promotion_package_report_path=args.promotion_package,
        ledger_path=args.ledger,
        dataset_path=str(dataset_path),
        candidate_key=args.candidate_key,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    print(json.dumps({key: value for key, value in artifact.items() if key != "report"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
