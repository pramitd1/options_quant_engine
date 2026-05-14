#!/usr/bin/env python3
"""Run the automated research-only threshold shadow-mode workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_shadow_mode import run_threshold_shadow_mode  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run threshold governance, policy experiment, shadow simulation, shadow review, "
            "promotion packaging, post-promotion monitoring, and adoption reconciliation "
            "in one research-only command."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional root directory for all shadow-mode artifacts.",
    )
    parser.add_argument("--report-name", default=None, help="Optional stable run id/report stem.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_threshold_shadow_mode(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
