#!/usr/bin/env python3
"""
Print grouped research reporting tables for the canonical signal evaluation dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation import SIGNAL_DATASET_PATH, build_research_report, load_signals_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build grouped research reports for the signal evaluation dataset.")
    parser.add_argument(
        "--dataset-path",
        default=str(SIGNAL_DATASET_PATH),
        help="Path to the canonical signals dataset CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = load_signals_dataset(args.dataset_path)
    report = build_research_report(frame)

    print(f"dataset_path: {args.dataset_path}")
    print(f"rows: {len(frame)}")

    for section_name, section_df in report.items():
        print(f"\n=== {section_name} ===")
        if section_df.empty:
            print("(empty)")
            continue
        print(section_df.to_string(index=False))


if __name__ == "__main__":
    main()
