#!/usr/bin/env python3
"""Run and record segmented probability forward-shadow accumulation."""

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
)
from research.signal_evaluation.segmented_probability_forward_shadow_accumulator import (  # noqa: E402
    ACCUMULATION_TRUE_FORWARD_REJECTED,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME,
    write_segmented_probability_forward_shadow_accumulation,
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
            "Run segmented probability forward-shadow validation in auto mode, append true-forward "
            "accumulation history, and refresh the latest dashboard. This command is research-only "
            "and does not change runtime config, parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--candidate-bundle", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH)
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default="correct_60m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--routing-policies", default=",".join(DEFAULT_ROUTING_POLICIES))
    parser.add_argument("--min-shadow-sample", type=int, default=100)
    parser.add_argument("--min-candidate-sample", type=int, default=50)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece-regression", type=float, default=0.01)
    parser.add_argument("--max-candidate-brier-regression", type=float, default=0.002)
    parser.add_argument("--max-candidate-ece-regression", type=float, default=0.02)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--shadow-output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR)
    parser.add_argument("--history-filename", default=SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME)
    parser.add_argument("--lookback-runs", type=int, default=20)
    parser.add_argument(
        "--fail-on-true-forward-rejection",
        action="store_true",
        help="Exit with status 2 when the latest true-forward accumulation status is rejected.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_segmented_probability_forward_shadow_accumulation(
        dataset_path=dataset_path,
        candidate_bundle_path=args.candidate_bundle,
        shadow_output_dir=args.shadow_output_dir,
        output_dir=args.output_dir,
        history_filename=args.history_filename,
        lookback_runs=args.lookback_runs,
        min_shadow_sample=args.min_shadow_sample,
        probability_field=args.probability_field,
        label_field=args.label_field,
        train_fraction=args.train_fraction,
        routing_policies=_csv_tuple(args.routing_policies),
        min_candidate_sample=args.min_candidate_sample,
        min_brier_improvement=args.min_brier_improvement,
        max_ece_regression=args.max_ece_regression,
        max_candidate_brier_regression=args.max_candidate_brier_regression,
        max_candidate_ece_regression=args.max_candidate_ece_regression,
        n_bins=args.n_bins,
    )
    row = artifact.get("history_row", {}) or {}
    dashboard = artifact.get("accumulation_dashboard", {}) or {}
    payload = {
        "history_path": artifact.get("history_path"),
        "accumulation_dashboard_json_path": artifact.get("accumulation_dashboard_json_path"),
        "accumulation_dashboard_markdown_path": artifact.get("accumulation_dashboard_markdown_path"),
        "forward_shadow_json_path": artifact.get("forward_shadow_artifact", {}).get("latest_json_path"),
        "trend_assessment": dashboard.get("trend_assessment"),
        "accumulation_status": row.get("accumulation_status"),
        "shadow_validation_status": row.get("shadow_validation_status"),
        "validation_mode_used": row.get("validation_mode_used"),
        "strict_forward_row_count": row.get("strict_forward_row_count"),
        "min_shadow_sample": row.get("min_shadow_sample"),
        "forward_sample_gap": row.get("forward_sample_gap"),
        "recommended_routing_policy": row.get("recommended_routing_policy"),
        "recommended_policy_brier_improvement": row.get("recommended_policy_brier_improvement"),
        "recommended_policy_ece_change": row.get("recommended_policy_ece_change"),
        "operator_message": dashboard.get("operator_message"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.fail_on_true_forward_rejection and row.get("accumulation_status") == ACCUMULATION_TRUE_FORWARD_REJECTED:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
