#!/usr/bin/env python3
"""Backfill selected-option premium paths from saved option-chain snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation import CUMULATIVE_DATASET_PATH
from research.signal_evaluation.option_premium_path import (
    DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    OPTION_PREMIUM_MAX_LAG_SECONDS,
    OPTION_PREMIUM_PRICE_BASIS,
    backfill_option_premium_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill selected-contract option premium paths for P&L research."
    )
    parser.add_argument("--dataset-path", default=str(CUMULATIVE_DATASET_PATH))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-lag-seconds", type=int, default=OPTION_PREMIUM_MAX_LAG_SECONDS)
    parser.add_argument(
        "--price-basis",
        choices=["LTP", "MID", "BID", "ASK"],
        default=OPTION_PREMIUM_PRICE_BASIS,
        help="Premium mark to use from future option-chain snapshots.",
    )
    parser.add_argument("--write", action="store_true", help="Persist the backfilled dataset. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = backfill_option_premium_dataset(
        dataset_path=args.dataset_path,
        snapshot_dir=args.snapshot_dir,
        max_lag_seconds=args.max_lag_seconds,
        price_basis=args.price_basis,
        as_of=args.as_of,
        dry_run=not args.write,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
