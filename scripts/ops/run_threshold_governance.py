#!/usr/bin/env python3
"""Run advisory threshold governance artifacts from a signal dataset."""

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
from research.signal_evaluation.threshold_governance import (  # noqa: E402
    DEFAULT_THRESHOLD_GOVERNANCE_DIR,
    record_threshold_governance_review,
    write_threshold_governance_report,
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run advisory threshold governance.")
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_GOVERNANCE_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--review-action", default=None, help="Optional human review action to append.")
    parser.add_argument("--reviewer", default=None, help="Reviewer name for --review-action.")
    parser.add_argument("--review-note", default="")
    parser.add_argument("--next-review-at", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    artifact = write_threshold_governance_report(
        frame,
        dataset_path=str(dataset_path),
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    if args.review_action:
        if not args.reviewer:
            raise ValueError("--reviewer is required when --review-action is provided")
        artifact["review_artifact"] = record_threshold_governance_review(
            report_json_path=artifact["json_path"],
            review_action=args.review_action,
            reviewer=args.reviewer,
            review_note=args.review_note,
            ledger_path=artifact["review_ledger_path"],
            next_review_at=args.next_review_at,
        )
    print(json.dumps({key: value for key, value in artifact.items() if key != "report"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
