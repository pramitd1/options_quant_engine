#!/usr/bin/env python3
"""Run research-only forward-shadow validation for segmented calibration candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_forward_shadow import (  # noqa: E402
    DEFAULT_ROUTING_POLICIES,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR,
    write_segmented_probability_forward_shadow_report_from_path,
)
from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    DEFAULT_PROBABILITY_FIELD,
    default_signal_quality_dataset_path,
)


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-shadow validate segmented probability calibration candidates. "
            "This command writes advisory artifacts only and does not change runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
        help="Segmented probability calibration candidate bundle JSON.",
    )
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default="correct_60m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument(
        "--validation-mode",
        choices=["auto", "after_candidate_generated", "holdout_replay"],
        default="auto",
        help="Use true rows after candidate generation, holdout replay, or auto fallback.",
    )
    parser.add_argument("--routing-policies", default=",".join(DEFAULT_ROUTING_POLICIES))
    parser.add_argument("--min-shadow-sample", type=int, default=100)
    parser.add_argument("--min-candidate-sample", type=int, default=50)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece-regression", type=float, default=0.01)
    parser.add_argument("--max-candidate-brier-regression", type=float, default=0.002)
    parser.add_argument("--max-candidate-ece-regression", type=float, default=0.02)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_segmented_probability_forward_shadow_report_from_path(
        dataset_path=dataset_path,
        candidate_bundle_path=args.candidate_bundle,
        probability_field=args.probability_field,
        label_field=args.label_field,
        train_fraction=args.train_fraction,
        validation_mode=args.validation_mode,
        routing_policies=_csv_tuple(args.routing_policies),
        min_shadow_sample=args.min_shadow_sample,
        min_candidate_sample=args.min_candidate_sample,
        min_brier_improvement=args.min_brier_improvement,
        max_ece_regression=args.max_ece_regression,
        max_candidate_brier_regression=args.max_candidate_brier_regression,
        max_candidate_ece_regression=args.max_candidate_ece_regression,
        n_bins=args.n_bins,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    selection = report.get("selection_summary", {}) or {}
    window = report.get("validation_window", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["row_count"] = report.get("row_count")
    payload["quality_labeled_row_count"] = report.get("quality_labeled_row_count")
    payload["candidate_count"] = report.get("candidate_count")
    payload["validation_mode_used"] = window.get("validation_mode_used")
    payload["strict_forward_row_count"] = window.get("strict_forward_row_count")
    payload["holdout_replay_row_count"] = window.get("holdout_replay_row_count")
    payload["shadow_validation_status"] = report.get("shadow_validation_status")
    payload["recommended_routing_policy"] = selection.get("recommended_routing_policy")
    payload["recommended_policy_status"] = selection.get("recommended_policy_status")
    payload["recommended_policy_brier_improvement"] = selection.get("recommended_policy_brier_improvement")
    payload["recommended_policy_ece_change"] = selection.get("recommended_policy_ece_change")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
