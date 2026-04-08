#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(Path(__file__))

from research.signal_evaluation import (
    CUMULATIVE_DATASET_PATH,
    load_cumulative_dataset,
    resolve_research_as_of,
    sync_live_to_cumulative,
    update_signal_dataset_outcomes,
    write_signals_dataset,
)
from research.signal_evaluation.legacy_backfill import backfill_signal_contract_fields
from tuning.walk_forward import build_walk_forward_splits_with_fallback


def _readiness_summary(
    frame: pd.DataFrame,
    *,
    minimum_trading_days: int,
    minimum_completed_signals: int,
    minimum_splits: int,
) -> dict[str, object]:
    timestamps = pd.to_datetime(frame.get("signal_timestamp", pd.Series(dtype=object)), errors="coerce")
    trading_days = int(timestamps.dropna().dt.normalize().nunique()) if not timestamps.empty else 0
    outcome_status = frame.get("outcome_status", pd.Series(dtype=object)).fillna("")
    completed_signals = int(outcome_status.isin(["PARTIAL", "COMPLETE"]).sum())

    splits, split_config = build_walk_forward_splits_with_fallback(
        frame,
        split_type="anchored",
        train_window_days=365,
        validation_window_days=90,
        step_size_days=90,
        minimum_train_rows=250,
        minimum_validation_rows=80,
        timestamp_col="signal_timestamp",
    )

    split_count = int(len(splits))
    return {
        "trading_days": trading_days,
        "completed_signals": completed_signals,
        "split_count": split_count,
        "minimum_trading_days": int(minimum_trading_days),
        "minimum_completed_signals": int(minimum_completed_signals),
        "minimum_splits": int(minimum_splits),
        "trading_days_gap": max(int(minimum_trading_days) - trading_days, 0),
        "completed_signals_gap": max(int(minimum_completed_signals) - completed_signals, 0),
        "split_count_gap": max(int(minimum_splits) - split_count, 0),
        "split_fallback_used": bool(split_config.get("used_fallback", False)),
        "split_config_reason": split_config.get("reason"),
        "effective_train_window_days": int(split_config.get("effective_train_window_days", 365)),
        "effective_validation_window_days": int(split_config.get("effective_validation_window_days", 90)),
        "effective_step_size_days": int(split_config.get("effective_step_size_days", 90)),
        "effective_minimum_train_rows": int(split_config.get("effective_minimum_train_rows", 250)),
        "effective_minimum_validation_rows": int(split_config.get("effective_minimum_validation_rows", 80)),
        "ready": (
            trading_days >= int(minimum_trading_days)
            and completed_signals >= int(minimum_completed_signals)
            and split_count >= int(minimum_splits)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync, mature, and backfill cumulative signal dataset; then report calibration-floor readiness."
    )
    parser.add_argument("--dataset-path", default=str(CUMULATIVE_DATASET_PATH))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--skip-outcome-update", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--minimum-trading-days", type=int, default=20)
    parser.add_argument("--minimum-completed-signals", type=int, default=500)
    parser.add_argument("--minimum-splits", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    resolved_as_of = resolve_research_as_of(args.as_of)

    synced_rows = int(sync_live_to_cumulative())
    frame = load_cumulative_dataset(dataset_path)

    if not args.skip_outcome_update:
        frame = update_signal_dataset_outcomes(dataset_path=dataset_path, as_of=resolved_as_of)

    backfill_stats = {
        "rows_seen": 0,
        "rows_with_chain_path": 0,
        "rows_backfilled": 0,
        "rows_skipped_missing_snapshot": 0,
        "rows_skipped_no_contract": 0,
    }
    if not args.skip_backfill:
        updated, backfill_stats = backfill_signal_contract_fields(frame, project_root=PROJECT_ROOT)
        if not updated.equals(frame):
            write_signals_dataset(updated, dataset_path)
            frame = updated

    readiness = _readiness_summary(
        frame,
        minimum_trading_days=args.minimum_trading_days,
        minimum_completed_signals=args.minimum_completed_signals,
        minimum_splits=args.minimum_splits,
    )

    print(f"dataset_path: {dataset_path}")
    print(f"as_of: {resolved_as_of.isoformat()}")
    print(f"synced_live_rows_into_cumulative: {synced_rows}")
    print(f"rows: {len(frame)}")
    for key, value in backfill_stats.items():
        print(f"backfill_{key}: {value}")
    for key, value in readiness.items():
        print(f"readiness_{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
