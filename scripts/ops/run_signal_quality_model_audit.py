#!/usr/bin/env python3
"""Run the signal-quality calibration and EV/risk ranking audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    DEFAULT_PROBABILITY_FIELD,
    DEFAULT_SIGNAL_QUALITY_MODEL_AUDIT_DIR,
    default_signal_quality_dataset_path,
    write_signal_quality_model_audit_report_from_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a research-only signal-quality model audit covering probability calibration, "
            "regime bias, feature stability, and EV/risk ranking. This command does not run the "
            "engine or change runtime configuration."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--min-label-sample", type=int, default=30)
    parser.add_argument("--strong-label-sample", type=int, default=100)
    parser.add_argument("--min-regime-sample", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SIGNAL_QUALITY_MODEL_AUDIT_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_signal_quality_model_audit_report_from_path(
        dataset_path=dataset_path,
        probability_field=args.probability_field,
        min_label_sample=args.min_label_sample,
        strong_label_sample=args.strong_label_sample,
        min_regime_sample=args.min_regime_sample,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    calibration = report.get("calibration_summary", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["row_count"] = report.get("row_count")
    payload["quality_labeled_row_count"] = report.get("quality_labeled_row_count")
    payload["calibration_status"] = calibration.get("calibration_status")
    payload["brier_score"] = calibration.get("brier_score")
    payload["expected_calibration_error"] = calibration.get("expected_calibration_error")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
