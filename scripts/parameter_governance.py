#!/usr/bin/env python3
"""
Governed parameter research workflow entrypoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(Path(__file__))

from research.signal_evaluation import SIGNAL_DATASET_PATH
from tuning.governance import (
    evaluate_current_production_signal_quality,
    get_candidate_review_context,
    run_controlled_tuning_workflow,
)
from tuning.promotion import (
    get_active_live_pack,
    promote_candidate,
    record_manual_approval,
)


def _add_common_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-path",
        default=str(SIGNAL_DATASET_PATH),
        help="Path to the canonical signal evaluation dataset CSV.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled research-to-production workflow for parameter tuning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate-current",
        help="Generate a structured current signal evaluation report for the active production pack.",
    )
    _add_common_dataset_argument(evaluate_parser)
    evaluate_parser.add_argument("--production-pack-name", default=None)
    evaluate_parser.add_argument("--report-name", default=None)
    evaluate_parser.add_argument("--output-dir", default=None)
    evaluate_parser.add_argument("--top-n", type=int, default=10)

    tune_parser = subparsers.add_parser(
        "tune",
        help="Run governed tuning, create a separate candidate pack, and write a candidate-vs-production report.",
    )
    _add_common_dataset_argument(tune_parser)
    tune_parser.add_argument("--production-pack-name", default=None)
    tune_parser.add_argument("--candidate-pack-name", default=None)
    tune_parser.add_argument("--group", action="append", dest="groups", default=None)
    tune_parser.add_argument("--seed", type=int, default=19)
    tune_parser.add_argument("--created-by", default="research")
    tune_parser.add_argument("--notes", default=None)
    tune_parser.add_argument("--allow-live-unsafe", action="store_true")
    tune_parser.add_argument("--train-window-days", type=int, default=180)
    tune_parser.add_argument("--validation-window-days", type=int, default=60)
    tune_parser.add_argument("--step-size-days", type=int, default=30)
    tune_parser.add_argument("--minimum-train-rows", type=int, default=50)
    tune_parser.add_argument("--minimum-validation-rows", type=int, default=20)

    approve_parser = subparsers.add_parser(
        "approve-candidate",
        help="Record explicit human approval or rejection for the current candidate pack.",
    )
    approve_parser.add_argument("--pack-name", default=None)
    approve_parser.add_argument("--reviewer", required=True)
    approve_parser.add_argument("--notes", default=None)
    approve_parser.add_argument("--reject", action="store_true")

    promote_parser = subparsers.add_parser(
        "promote-candidate",
        help="Promote the approved candidate pack to live. This never runs during tuning.",
    )
    promote_parser.add_argument("--pack-name", default=None)
    promote_parser.add_argument("--approved-by", required=True)
    promote_parser.add_argument("--reason", default="candidate_promoted_to_live")
    promote_parser.add_argument("--source-experiment-id", default=None)
    promote_parser.add_argument("--source-validation-experiment-id", default=None)
    promote_parser.add_argument(
        "--expected-improvement-json",
        default=None,
        help="Optional JSON string with expected improvement summary to include in the audit trail.",
    )

    review_parser = subparsers.add_parser(
        "review-context",
        help="Print the current production pack, candidate pack, and recorded approval state.",
    )
    _add_common_dataset_argument(review_parser)
    return parser


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate-current":
        report = evaluate_current_production_signal_quality(
            dataset_path=args.dataset_path,
            production_pack_name=args.production_pack_name or get_active_live_pack(),
            output_dir=args.output_dir,
            report_name=args.report_name,
            top_n=args.top_n,
        )
        payload = {
            "production_pack_name": report["summary"]["production_pack_name"],
            "total_signal_count": report["summary"]["total_signal_count"],
            "evaluation_period": report["summary"]["evaluation_period"],
            "markdown_report": report["markdown_path"],
            "json_report": report["json_path"],
        }
        _print_json(payload)
        return

    if args.command == "tune":
        workflow = run_controlled_tuning_workflow(
            dataset_path=args.dataset_path,
            production_pack_name=args.production_pack_name,
            candidate_pack_name=args.candidate_pack_name,
            groups=args.groups,
            allow_live_unsafe=args.allow_live_unsafe,
            walk_forward_config={
                "split_type": "rolling",
                "train_window_days": args.train_window_days,
                "validation_window_days": args.validation_window_days,
                "step_size_days": args.step_size_days,
                "minimum_train_rows": args.minimum_train_rows,
                "minimum_validation_rows": args.minimum_validation_rows,
            },
            seed=args.seed,
            created_by=args.created_by,
            notes=args.notes,
        )
        payload = {
            "production_pack_name": workflow["production_pack_name"],
            "candidate_pack_name": workflow["candidate_pack_name"],
            "candidate_pack_path": workflow["candidate_pack_path"],
            "signal_evaluation_report": {
                "markdown_path": workflow["signal_evaluation_report"]["markdown_path"],
                "json_path": workflow["signal_evaluation_report"]["json_path"],
            },
            "candidate_report_paths": workflow["candidate_report_paths"],
            "expected_improvement_summary": workflow["candidate_report"]["expected_improvement_summary"],
            "live_pack_unchanged": True,
        }
        _print_json(payload)
        return

    if args.command == "approve-candidate":
        context = get_candidate_review_context()
        pack_name = args.pack_name or context.get("candidate_pack_name")
        if not pack_name:
            raise ValueError("No candidate pack is currently assigned")
        payload = record_manual_approval(
            pack_name=pack_name,
            approved=not args.reject,
            reviewer=args.reviewer,
            notes=args.notes,
        )
        _print_json(
            {
                "candidate_pack_name": pack_name,
                "approved": not args.reject,
                "manual_approval": payload.get("manual_approvals", {}).get(pack_name),
            }
        )
        return

    if args.command == "promote-candidate":
        context = get_candidate_review_context()
        pack_name = args.pack_name or context.get("candidate_pack_name")
        if not pack_name:
            raise ValueError("No candidate pack is currently assigned")
        expected_improvement = json.loads(args.expected_improvement_json) if args.expected_improvement_json else None
        promote_candidate(
            pack_name,
            approved_by=args.approved_by,
            reason=args.reason,
            source_experiment_id=args.source_experiment_id,
            source_validation_experiment_id=args.source_validation_experiment_id,
            expected_improvement_summary=expected_improvement,
        )
        _print_json(
            {
                "promoted_pack_name": pack_name,
                "approved_by": args.approved_by,
                "reason": args.reason,
                "live_pack_updated": True,
            }
        )
        return

    if args.command == "review-context":
        _print_json(get_candidate_review_context())
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
