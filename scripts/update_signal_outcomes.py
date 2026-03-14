#!/usr/bin/env python3
"""
Refresh pending signal outcomes in the canonical signal evaluation dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation import SIGNAL_DATASET_PATH, update_signal_dataset_outcomes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refresh pending realized outcomes in the canonical signal evaluation dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(SIGNAL_DATASET_PATH),
        help="Path to the canonical signals dataset CSV.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional timestamp cutoff for outcome enrichment, e.g. 2026-03-14T15:25:00+05:30",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = update_signal_dataset_outcomes(
        dataset_path=args.dataset_path,
        as_of=args.as_of,
    )

    pending = int((frame.get("outcome_status") == "PENDING").sum()) if not frame.empty else 0
    partial = int((frame.get("outcome_status") == "PARTIAL").sum()) if not frame.empty else 0
    complete = int((frame.get("outcome_status") == "COMPLETE").sum()) if not frame.empty else 0

    print(f"dataset_path: {args.dataset_path}")
    print(f"rows: {len(frame)}")
    print(f"pending: {pending}")
    print(f"partial: {partial}")
    print(f"complete: {complete}")


if __name__ == "__main__":
    main()
