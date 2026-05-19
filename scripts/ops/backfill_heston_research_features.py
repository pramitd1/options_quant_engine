#!/usr/bin/env python3
"""Backfill research-only Heston diagnostics from saved option-chain snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    HESTON_CALIBRATION_ERROR_REJECT,
    HESTON_CALIBRATION_MAX_ROWS,
    HESTON_CALIBRATION_MIN_ROWS,
    HESTON_CALIBRATION_TIMEOUT_SECONDS,
)
from research.signal_evaluation import CUMULATIVE_DATASET_PATH  # noqa: E402
from research.signal_evaluation.heston_backfill import backfill_heston_research_dataset  # noqa: E402
from research.signal_evaluation.pcr_backfill import DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill research-only Heston diagnostics from stored option-chain "
            "snapshots. The live Black-Scholes Greek engine and trade decisions "
            "are not changed."
        )
    )
    parser.add_argument("--dataset-path", default=str(CUMULATIVE_DATASET_PATH))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR))
    parser.add_argument("--min-rows", type=int, default=HESTON_CALIBRATION_MIN_ROWS)
    parser.add_argument("--max-rows", type=int, default=HESTON_CALIBRATION_MAX_ROWS)
    parser.add_argument("--reject-error", type=float, default=HESTON_CALIBRATION_ERROR_REJECT)
    parser.add_argument("--timeout-seconds", type=float, default=HESTON_CALIBRATION_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to update in this run.")
    parser.add_argument("--force", action="store_true", help="Recompute rows that already have Heston diagnostics.")
    parser.add_argument("--write", action="store_true", help="Persist the backfilled dataset. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = backfill_heston_research_dataset(
        dataset_path=args.dataset_path,
        snapshot_dir=args.snapshot_dir,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        reject_error=args.reject_error,
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        limit=args.limit,
        dry_run=not args.write,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
