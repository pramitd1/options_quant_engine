#!/usr/bin/env python3
"""Build empirical regime outcome tables from the signal-evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.regime_outcome_tables import (  # noqa: E402
    DEFAULT_REGIME_OUTCOME_TABLE_DIR,
    default_regime_outcome_dataset_path,
    write_regime_outcome_table_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build research-only empirical outcome tables by regime combinations. "
            "The command writes CSV/JSON/Markdown artifacts and never changes runtime behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGIME_OUTCOME_TABLE_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--min-label-sample", type=int, default=30)
    parser.add_argument("--strong-label-sample", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_regime_outcome_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    artifact = write_regime_outcome_table_report(
        frame,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        report_name=args.report_name,
        min_label_sample=args.min_label_sample,
        strong_label_sample=args.strong_label_sample,
        top_n=args.top_n,
        write_latest=True,
    )
    report = artifact["report"]
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload.update(
        {
            "report_type": report.get("report_type"),
            "directional_row_count": report.get("directional_row_count"),
            "by_horizon_row_count": report.get("by_horizon_row_count"),
            "best_horizon_row_count": report.get("best_horizon_row_count"),
            "reliable_or_low_confidence_best_row_count": report.get(
                "reliable_or_low_confidence_best_row_count"
            ),
            "sparse_best_row_count": report.get("sparse_best_row_count"),
            "top_favorable": report.get("top_favorable", [])[:5],
            "top_unfavorable": report.get("top_unfavorable", [])[:5],
            "runtime_config_changed": report.get("runtime_config_changed"),
            "parameter_pack_file_changed": report.get("parameter_pack_file_changed"),
            "execution_behavior_changed": report.get("execution_behavior_changed"),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
