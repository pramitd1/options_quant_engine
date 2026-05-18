#!/usr/bin/env python3
"""Run the regime-cell counterfactual replay comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.daily_research_report import DEFAULT_CUMULATIVE_DATASET_PATH  # noqa: E402
from research.signal_evaluation.regime_cell_counterfactual import (  # noqa: E402
    DEFAULT_BASE_TRADE_STRENGTH_THRESHOLD,
    DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR,
    DEFAULT_MIN_CELL_LABELS,
    DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR,
    write_regime_cell_counterfactual_report,
)
from research.signal_evaluation.regime_cell_review import (  # noqa: E402
    DEFAULT_REGIME_CELL_REVIEW_DIR,
    LATEST_REVIEW_JSON_FILENAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CUMULATIVE_DATASET_PATH)
    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REGIME_CELL_REVIEW_DIR / LATEST_REVIEW_JSON_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR)
    parser.add_argument("--documentation-dir", type=Path, default=DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR)
    parser.add_argument("--report-name", default="regime_cell_counterfactual")
    parser.add_argument("--base-trade-strength-threshold", type=float, default=DEFAULT_BASE_TRADE_STRENGTH_THRESHOLD)
    parser.add_argument("--min-cell-labels", type=int, default=DEFAULT_MIN_CELL_LABELS)
    parser.add_argument("--no-doc-copy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = write_regime_cell_counterfactual_report(
        dataset_path=args.dataset,
        review_path=args.review,
        output_dir=args.output_dir,
        documentation_dir=None if args.no_doc_copy else args.documentation_dir,
        report_name=args.report_name,
        base_trade_strength_threshold=args.base_trade_strength_threshold,
        min_cell_labels=args.min_cell_labels,
    )
    report = artifact["report"]
    print(
        json.dumps(
            {
                "report_type": report.get("report_type"),
                "assessment_status": report.get("assessment_status"),
                "matched_signal_count": report.get("matched_signal_count"),
                "baseline_trade_count": report.get("baseline_trade_count"),
                "counterfactual_selected_count": report.get("counterfactual_selected_count"),
                "suppressed_existing_trade_count": report.get("suppressed_existing_trade_count"),
                "promotion_candidate_count_sandbox": report.get("promotion_candidate_count_sandbox"),
                "conservative_total_return_delta_bps": report.get("conservative_total_return_delta_bps"),
                "impact_status_counts": report.get("impact_status_counts"),
                "json_path": artifact.get("json_path"),
                "markdown_path": artifact.get("markdown_path"),
                "cells_csv_path": artifact.get("cells_csv_path"),
                "details_csv_path": artifact.get("details_csv_path"),
                "latest_json_path": artifact.get("latest_json_path"),
                "latest_markdown_path": artifact.get("latest_markdown_path"),
                "latest_cells_csv_path": artifact.get("latest_cells_csv_path"),
                "latest_details_csv_path": artifact.get("latest_details_csv_path"),
                "documentation_markdown_path": artifact.get("documentation_markdown_path"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
