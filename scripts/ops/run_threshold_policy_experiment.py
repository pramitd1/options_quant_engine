#!/usr/bin/env python3
"""Run a research-only policy experiment for a governed threshold candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.daily_research_report import DEFAULT_CUMULATIVE_DATASET_PATH, DEFAULT_DATASET_PATH  # noqa: E402
from research.signal_evaluation.threshold_policy_experiment import (  # noqa: E402
    DEFAULT_GOVERNANCE_REPORT_PATH,
    DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR,
    write_threshold_policy_experiment_report,
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run threshold candidate policy experiment sandbox.")
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--governance-report", type=Path, default=DEFAULT_GOVERNANCE_REPORT_PATH)
    parser.add_argument("--candidate-key", default=None, help="Optional candidate key from governance candidate_reviews.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    governance_report = json.loads(args.governance_report.read_text(encoding="utf-8"))
    artifact = write_threshold_policy_experiment_report(
        frame,
        governance_report=governance_report,
        candidate_key=args.candidate_key,
        dataset_path=str(dataset_path),
        governance_report_path=str(args.governance_report),
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    print(json.dumps({key: value for key, value in artifact.items() if key != "report"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
