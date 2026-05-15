#!/usr/bin/env python3
"""Run research-only shadow simulation for an approved threshold policy experiment."""

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
from research.signal_evaluation.threshold_policy_experiment import APPROVED_FOR_POLICY_EXPERIMENT  # noqa: E402
from research.signal_evaluation.threshold_shadow_simulation import (  # noqa: E402
    DEFAULT_POLICY_EXPERIMENT_REPORT_PATH,
    DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR,
    write_threshold_shadow_simulation_report,
    write_threshold_shadow_simulation_skip,
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run threshold shadow simulation.")
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--policy-experiment", type=Path, default=DEFAULT_POLICY_EXPERIMENT_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    policy_experiment = json.loads(args.policy_experiment.read_text(encoding="utf-8"))
    if policy_experiment.get("experiment_status") != APPROVED_FOR_POLICY_EXPERIMENT:
        artifact = write_threshold_shadow_simulation_skip(
            reason=(
                "Policy experiment is not approved for shadow simulation; "
                f"status is {policy_experiment.get('experiment_status') or 'UNKNOWN'}."
            ),
            policy_experiment_report=policy_experiment,
            dataset_path=str(dataset_path),
            policy_experiment_report_path=str(args.policy_experiment),
            output_dir=args.output_dir,
            report_name=args.report_name,
            write_latest=True,
        )
    else:
        artifact = write_threshold_shadow_simulation_report(
            frame,
            policy_experiment_report=policy_experiment,
            dataset_path=str(dataset_path),
            policy_experiment_report_path=str(args.policy_experiment),
            output_dir=args.output_dir,
            report_name=args.report_name,
            write_latest=True,
        )
    print(json.dumps({key: value for key, value in artifact.items() if key != "report"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
