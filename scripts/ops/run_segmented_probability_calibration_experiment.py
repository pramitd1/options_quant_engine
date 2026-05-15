#!/usr/bin/env python3
"""Run the research-only segmented probability calibration experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.probability_calibration_experiment import DEFAULT_METHODS  # noqa: E402
from research.signal_evaluation.segmented_probability_calibration_experiment import (  # noqa: E402
    DEFAULT_PROBABILITY_FIELD,
    DEFAULT_RECENCY_WINDOWS,
    DEFAULT_SEGMENT_FIELDS,
    DEFAULT_SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_DIR,
    write_segmented_probability_calibration_experiment_report_from_path,
)
from research.signal_evaluation.signal_quality_model_audit import default_signal_quality_dataset_path  # noqa: E402


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _float_csv_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(value or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search for research-only regime and recency probability calibration candidates. "
            "This command writes advisory artifacts only and does not change runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default="correct_60m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--segment-fields", default=",".join(DEFAULT_SEGMENT_FIELDS))
    parser.add_argument(
        "--recency-windows",
        default=",".join(str(value) for value in DEFAULT_RECENCY_WINDOWS),
        help="Comma-separated fractions of the train slice to test, e.g. 0.25,0.5,1.0.",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--min-train-sample", type=int, default=100)
    parser.add_argument("--min-holdout-sample", type=int, default=50)
    parser.add_argument("--min-segment-train-sample", type=int, default=120)
    parser.add_argument("--min-segment-holdout-sample", type=int, default=60)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece-regression", type=float, default=0.01)
    parser.add_argument("--max-candidate-ece", type=float, default=0.12)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-segments-per-field", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_segmented_probability_calibration_experiment_report_from_path(
        dataset_path=dataset_path,
        probability_field=args.probability_field,
        label_field=args.label_field,
        train_fraction=args.train_fraction,
        segment_fields=_csv_tuple(args.segment_fields),
        recency_windows=_float_csv_tuple(args.recency_windows),
        methods=tuple(args.methods),
        min_train_sample=args.min_train_sample,
        min_holdout_sample=args.min_holdout_sample,
        min_segment_train_sample=args.min_segment_train_sample,
        min_segment_holdout_sample=args.min_segment_holdout_sample,
        min_brier_improvement=args.min_brier_improvement,
        max_ece_regression=args.max_ece_regression,
        max_candidate_ece=args.max_candidate_ece,
        n_bins=args.n_bins,
        max_segments_per_field=args.max_segments_per_field,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    summary = report.get("selection_summary", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["row_count"] = report.get("row_count")
    payload["quality_labeled_row_count"] = report.get("quality_labeled_row_count")
    payload["train_count"] = report.get("train_count")
    payload["holdout_count"] = report.get("holdout_count")
    payload["calibration_status"] = report.get("calibration_status")
    payload["review_ready_candidate_count"] = summary.get("review_ready_candidate_count")
    payload["evaluated_regime_segment_count"] = summary.get("evaluated_regime_segment_count")
    payload["evaluated_recency_window_count"] = summary.get("evaluated_recency_window_count")
    payload["best_ready_candidate"] = summary.get("best_ready_candidate")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
