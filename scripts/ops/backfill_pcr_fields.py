#!/usr/bin/env python3
"""Backfill canonical PCR fields in signal-evaluation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH  # noqa: E402
from research.signal_evaluation.pcr_backfill import (  # noqa: E402
    DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    backfill_pcr_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CUMULATIVE_DATASET_PATH)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR)
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=None,
        help="Also use the nearest past snapshot within this age. Omit for saved-path-only backfill.",
    )
    parser.add_argument("--write", action="store_true", help="Write the updated dataset. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = backfill_pcr_dataset(
        dataset_path=args.dataset,
        snapshot_dir=args.snapshot_dir,
        max_age_seconds=args.max_age_seconds,
        dry_run=not args.write,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
