#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(Path(__file__))

from research.signal_evaluation import CUMULATIVE_DATASET_PATH, load_signals_dataset, write_signals_dataset
from research.signal_evaluation.legacy_backfill import apply_repair_proposals_to_dataset, backfill_signal_contract_fields


def _parse_signal_ids(args: argparse.Namespace) -> set[str]:
    signal_ids: set[str] = set()
    if args.signal_ids:
        signal_ids.update({token.strip() for token in str(args.signal_ids).split(",") if token.strip()})

    if args.signal_ids_file:
        path = Path(args.signal_ids_file)
        if not path.exists():
            raise FileNotFoundError(f"signal id file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token:
                signal_ids.add(token)

    return signal_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approve selected repair-review rows and apply them directly to a dataset without rerunning proposal generation."
        )
    )
    parser.add_argument("--review-queue-csv", required=True, help="Path to repair_review_queue.csv generated earlier.")
    parser.add_argument("--dataset-path", default=str(CUMULATIVE_DATASET_PATH), help="Dataset CSV path to patch.")
    parser.add_argument(
        "--signal-ids",
        default=None,
        help="Comma-separated signal_id values to approve from the review queue.",
    )
    parser.add_argument(
        "--signal-ids-file",
        default=None,
        help="Optional newline-delimited file with signal_id values to approve.",
    )
    parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Allow LOW-confidence proposals in addition to MEDIUM when approving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and emit artifacts without overwriting the source dataset.",
    )
    parser.add_argument(
        "--promote-approved",
        action="store_true",
        help="Overwrite source dataset with approved repairs and backfilled output (writes backup first).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for approval artifacts (default: sibling folder near review queue).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_queue_path = Path(args.review_queue_csv)
    if not review_queue_path.exists():
        raise FileNotFoundError(f"review queue csv not found: {review_queue_path}")

    selected_ids = _parse_signal_ids(args)
    if not selected_ids:
        raise ValueError("No signal IDs supplied. Use --signal-ids and/or --signal-ids-file.")

    dataset_path = Path(args.dataset_path)
    source_frame = load_signals_dataset(dataset_path)

    review_queue = pd.read_csv(review_queue_path)
    if review_queue.empty:
        raise ValueError("Review queue is empty; no rows available to approve.")

    review_queue["signal_id"] = review_queue.get("signal_id", pd.Series(index=review_queue.index, dtype=object)).astype(str)
    filtered = review_queue.loc[review_queue["signal_id"].isin(selected_ids)].copy()

    if not args.allow_low_confidence:
        filtered = filtered.loc[
            filtered.get("proposal_confidence", pd.Series(index=filtered.index, dtype=object)).astype(str).str.upper() == "MEDIUM"
        ].copy()

    if filtered.empty:
        raise ValueError("No matching review-queue rows found for the selected IDs and confidence policy.")

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = review_queue_path.parent / f"review_queue_approval_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    repaired_frame, repair_summary = apply_repair_proposals_to_dataset(
        source_frame,
        filtered,
        min_confidence="LOW",
    )
    repaired_backfilled, repaired_backfill_stats = backfill_signal_contract_fields(
        repaired_frame,
        project_root=PROJECT_ROOT,
    )

    repaired_dataset_path = output_dir / f"{dataset_path.stem}_approved_repairs{dataset_path.suffix}"
    write_signals_dataset(repaired_backfilled, repaired_dataset_path)

    approved_csv_path = output_dir / "approved_review_repairs.csv"
    approved_json_path = output_dir / "approved_review_repairs.json"
    filtered.to_csv(approved_csv_path, index=False)
    filtered.to_json(approved_json_path, orient="records", indent=2)

    remaining_queue = review_queue.loc[~review_queue["signal_id"].isin(set(filtered["signal_id"].astype(str)))].copy()
    remaining_csv_path = output_dir / "remaining_review_queue.csv"
    remaining_json_path = output_dir / "remaining_review_queue.json"
    remaining_queue.to_csv(remaining_csv_path, index=False)
    remaining_queue.to_json(remaining_json_path, orient="records", indent=2)

    promoted = False
    backup_path = None
    if args.promote_approved and not args.dry_run:
        backup_path = output_dir / f"{dataset_path.stem}_pre_review_promotion_backup{dataset_path.suffix}"
        write_signals_dataset(source_frame, backup_path)
        write_signals_dataset(repaired_backfilled, dataset_path)
        promoted = True

    summary = {
        "review_queue_csv": str(review_queue_path),
        "dataset_path": str(dataset_path),
        "selected_signal_ids": sorted(selected_ids),
        "allow_low_confidence": bool(args.allow_low_confidence),
        "approved_rows": int(len(filtered)),
        "remaining_review_queue_rows": int(len(remaining_queue)),
        "repair_summary": repair_summary,
        "repaired_backfill_stats": repaired_backfill_stats,
        "approved_repairs_csv": str(approved_csv_path),
        "approved_repairs_json": str(approved_json_path),
        "remaining_review_queue_csv": str(remaining_csv_path),
        "remaining_review_queue_json": str(remaining_json_path),
        "repaired_dataset_path": str(repaired_dataset_path),
        "dry_run": bool(args.dry_run),
        "promoted_to_source_dataset": promoted,
        "source_backup_path": str(backup_path) if backup_path is not None else None,
    }

    summary_path = output_dir / "review_queue_approval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"review_queue_csv: {review_queue_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"approved_rows: {len(filtered)}")
    print(f"remaining_review_queue_rows: {len(remaining_queue)}")
    print(f"repaired_dataset_path: {repaired_dataset_path}")
    print(f"approved_repairs_csv: {approved_csv_path}")
    print(f"remaining_review_queue_csv: {remaining_csv_path}")
    print(f"summary_path: {summary_path}")
    if promoted:
        print(f"promoted_source_dataset_path: {dataset_path}")
        print(f"source_backup_path: {backup_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
