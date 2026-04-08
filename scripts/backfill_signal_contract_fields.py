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

from research.signal_evaluation import CUMULATIVE_DATASET_PATH, SIGNAL_DATASET_PATH, load_signals_dataset, write_signals_dataset
from research.signal_evaluation.legacy_backfill import (
    apply_repair_proposals_to_dataset,
    audit_unresolved_signal_contract_matches,
    backfill_signal_contract_fields,
    partition_repair_proposals,
    propose_repairs_for_unresolved_signal_contract_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill legacy signal rows with selected option diagnostics from saved chain snapshots."
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Dataset CSV path to backfill (default: cumulative signal dataset).",
    )
    parser.add_argument(
        "--dataset",
        choices=["cumulative", "live"],
        default="cumulative",
        help="Named dataset to use when --dataset-path is not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute backfill stats without writing updates to disk.",
    )
    parser.add_argument(
        "--emit-audit",
        action="store_true",
        help="Write unresolved contract match audit artifacts for rows that still cannot be matched.",
    )
    parser.add_argument(
        "--audit-top-n",
        type=int,
        default=5,
        help="Number of nearest candidate contracts to include per unresolved row in the audit artifact.",
    )
    parser.add_argument(
        "--emit-repair-proposals",
        action="store_true",
        help="Write heuristic repair proposals for unresolved rows alongside the audit artifact.",
    )
    parser.add_argument(
        "--apply-repair-proposals",
        action="store_true",
        help="Apply accepted repair proposals into a copied dataset and rerun backfill on that copy.",
    )
    parser.add_argument(
        "--min-proposal-confidence",
        choices=["MEDIUM", "HIGH"],
        default="MEDIUM",
        help="Minimum repair-proposal confidence required when --apply-repair-proposals is used.",
    )
    parser.add_argument(
        "--high-confidence-only",
        action="store_true",
        help="Shortcut for applying only HIGH-confidence proposals and routing the rest to a review queue.",
    )
    parser.add_argument(
        "--promote-repaired-dataset",
        action="store_true",
        help="After repaired-copy backfill succeeds, overwrite the source dataset with the repaired result and write a backup copy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.high_confidence_only:
        args.min_proposal_confidence = "HIGH"
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = CUMULATIVE_DATASET_PATH if args.dataset == "cumulative" else SIGNAL_DATASET_PATH

    frame = load_signals_dataset(dataset_path)
    updated, stats = backfill_signal_contract_fields(frame, project_root=PROJECT_ROOT)

    apply_repairs = bool(args.apply_repair_proposals)

    if not args.dry_run and not apply_repairs and not updated.equals(frame):
        write_signals_dataset(updated, dataset_path)

    print(f"dataset_path: {dataset_path}")
    print(f"dry_run: {args.dry_run}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    rows_changed = int(len(updated.compare(frame).index.unique())) if not frame.empty else 0
    print(f"rows_changed: {rows_changed}")

    if args.emit_audit or args.emit_repair_proposals or apply_repairs:
        audit_frame = audit_unresolved_signal_contract_matches(
            updated,
            project_root=PROJECT_ROOT,
            top_n=args.audit_top_n,
        )
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        audit_dir = PROJECT_ROOT / "research" / "signal_evaluation" / "backfill_audit" / f"contract_match_audit_{run_id}"
        audit_dir.mkdir(parents=True, exist_ok=True)
        csv_path = audit_dir / "unresolved_contract_match_audit.csv"
        json_path = audit_dir / "unresolved_contract_match_audit.json"
        audit_frame.to_csv(csv_path, index=False)
        audit_frame.to_json(json_path, orient="records", indent=2)
        print(f"audit_rows: {len(audit_frame)}")
        print(f"audit_csv_path: {csv_path}")
        print(f"audit_json_path: {json_path}")

        proposals = None
        if args.emit_repair_proposals or apply_repairs:
            proposals = propose_repairs_for_unresolved_signal_contract_matches(
                audit_frame,
                project_root=PROJECT_ROOT,
            )
            proposal_csv_path = audit_dir / "proposed_contract_repairs.csv"
            proposal_json_path = audit_dir / "proposed_contract_repairs.json"
            proposals.to_csv(proposal_csv_path, index=False)
            proposals.to_json(proposal_json_path, orient="records", indent=2)
            print(f"repair_proposal_rows: {len(proposals)}")
            print(f"repair_proposal_csv_path: {proposal_csv_path}")
            print(f"repair_proposal_json_path: {proposal_json_path}")

        if apply_repairs:
            repaired_frame, repair_summary = apply_repair_proposals_to_dataset(
                updated,
                proposals if proposals is not None else audit_frame.iloc[0:0],
                min_confidence=args.min_proposal_confidence,
            )
            repaired_dataset_path = audit_dir / f"{dataset_path.stem}_repaired{dataset_path.suffix}"
            repaired_backfilled, repaired_backfill_stats = backfill_signal_contract_fields(
                repaired_frame,
                project_root=PROJECT_ROOT,
            )
            write_signals_dataset(repaired_backfilled, repaired_dataset_path)

            applied_proposals = pd.DataFrame()
            review_queue = pd.DataFrame()
            if proposals is not None and not proposals.empty:
                applied_proposals, review_queue = partition_repair_proposals(
                    proposals,
                    min_confidence=args.min_proposal_confidence,
                )
                applied_csv_path = audit_dir / "applied_contract_repairs.csv"
                applied_json_path = audit_dir / "applied_contract_repairs.json"
                applied_proposals.to_csv(applied_csv_path, index=False)
                applied_proposals.to_json(applied_json_path, orient="records", indent=2)
                print(f"applied_repair_rows: {len(applied_proposals)}")
                print(f"applied_repair_csv_path: {applied_csv_path}")
                print(f"applied_repair_json_path: {applied_json_path}")

                review_csv_path = audit_dir / "repair_review_queue.csv"
                review_json_path = audit_dir / "repair_review_queue.json"
                review_queue.to_csv(review_csv_path, index=False)
                review_queue.to_json(review_json_path, orient="records", indent=2)
                print(f"review_queue_rows: {len(review_queue)}")
                print(f"review_queue_csv_path: {review_csv_path}")
                print(f"review_queue_json_path: {review_json_path}")

            remaining_audit = audit_unresolved_signal_contract_matches(
                repaired_backfilled,
                project_root=PROJECT_ROOT,
                top_n=args.audit_top_n,
            )
            remaining_csv_path = audit_dir / "remaining_unresolved_contract_match_audit.csv"
            remaining_json_path = audit_dir / "remaining_unresolved_contract_match_audit.json"
            remaining_audit.to_csv(remaining_csv_path, index=False)
            remaining_audit.to_json(remaining_json_path, orient="records", indent=2)

            summary = {
                "source_dataset_path": str(dataset_path),
                "repaired_dataset_path": str(repaired_dataset_path),
                "min_proposal_confidence": args.min_proposal_confidence,
                "high_confidence_only": bool(args.high_confidence_only),
                "repair_summary": repair_summary,
                "repaired_backfill_stats": repaired_backfill_stats,
                "review_queue_rows": int(len(review_queue)),
                "remaining_unresolved_rows": int(len(remaining_audit)),
            }

            promoted_backup_path = None
            if args.promote_repaired_dataset:
                backup_path = audit_dir / f"{dataset_path.stem}_pre_promotion_backup{dataset_path.suffix}"
                write_signals_dataset(frame, backup_path)
                write_signals_dataset(repaired_backfilled, dataset_path)
                promoted_backup_path = str(backup_path)
                summary["promoted_to_source_dataset"] = True
                summary["source_backup_path"] = promoted_backup_path
                print(f"promoted_source_dataset_path: {dataset_path}")
                print(f"source_backup_path: {backup_path}")
            else:
                summary["promoted_to_source_dataset"] = False

            summary_path = audit_dir / "repair_application_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"repaired_dataset_path: {repaired_dataset_path}")
            print(f"repair_application_summary_path: {summary_path}")
            print(f"remaining_audit_csv_path: {remaining_csv_path}")
            print(f"remaining_audit_json_path: {remaining_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
