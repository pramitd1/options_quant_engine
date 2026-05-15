#!/usr/bin/env python3
"""Run the research-only probability calibration experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.probability_calibration_experiment import (  # noqa: E402
    DEFAULT_METHODS,
    DEFAULT_PROBABILITY_CALIBRATION_EXPERIMENT_DIR,
    DEFAULT_PROBABILITY_FIELD,
    write_probability_calibration_experiment_report_from_path,
)
from research.signal_evaluation.signal_quality_model_audit import default_signal_quality_dataset_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and compare research-only probability calibration mappings on a chronological "
            "holdout split. This command writes artifacts only and does not change runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default="correct_60m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--min-train-sample", type=int, default=100)
    parser.add_argument("--min-holdout-sample", type=int, default=50)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece-regression", type=float, default=0.01)
    parser.add_argument("--max-candidate-ece", type=float, default=0.12)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROBABILITY_CALIBRATION_EXPERIMENT_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_probability_calibration_experiment_report_from_path(
        dataset_path=dataset_path,
        probability_field=args.probability_field,
        label_field=args.label_field,
        train_fraction=args.train_fraction,
        methods=tuple(args.methods),
        min_train_sample=args.min_train_sample,
        min_holdout_sample=args.min_holdout_sample,
        min_brier_improvement=args.min_brier_improvement,
        max_ece_regression=args.max_ece_regression,
        max_candidate_ece=args.max_candidate_ece,
        n_bins=args.n_bins,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    selection = report.get("selection", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["row_count"] = report.get("row_count")
    payload["quality_labeled_row_count"] = report.get("quality_labeled_row_count")
    payload["train_count"] = report.get("train_count")
    payload["holdout_count"] = report.get("holdout_count")
    payload["calibration_status"] = report.get("calibration_status")
    payload["selected_calibrator"] = report.get("selected_calibrator")
    payload["holdout_brier_improvement"] = selection.get("holdout_brier_improvement")
    payload["holdout_ece_change"] = selection.get("holdout_ece_change")
    payload["candidate_ready_for_review"] = selection.get("candidate_ready_for_review")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
