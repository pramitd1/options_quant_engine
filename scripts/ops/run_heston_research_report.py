#!/usr/bin/env python3
"""Build Heston research diagnostics from the signal-evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.heston_research_report import (  # noqa: E402
    DEFAULT_HESTON_REPORT_DIR,
    default_heston_dataset_path,
    write_heston_research_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build research-only Heston calibration diagnostics. This command "
            "does not change runtime config, parameter packs, or trade decisions."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HESTON_REPORT_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--min-sample", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_heston_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False) if dataset_path.exists() else pd.DataFrame()
    artifact = write_heston_research_report(
        frame,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        report_name=args.report_name,
        min_sample=args.min_sample,
        write_latest=True,
    )
    report = artifact["report"]
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload.update(
        {
            "report_type": report.get("report_type"),
            "row_count": report.get("row_count"),
            "heston_row_count": report.get("heston_row_count"),
            "heston_calibrated_row_count": report.get("heston_calibrated_row_count"),
            "surface_quality_counts": report.get("surface_quality_counts"),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
