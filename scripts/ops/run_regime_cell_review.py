#!/usr/bin/env python3
"""Generate the reliable 3-factor/4-factor regime-cell review report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.regime_cell_review import (  # noqa: E402
    DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR,
    DEFAULT_REGIME_CELL_REVIEW_DIR,
    DEFAULT_REGIME_OUTCOME_TABLE_DIR,
    REGIME_OUTCOME_BEST_HORIZON_CSV_FILENAME,
    REGIME_OUTCOME_BY_HORIZON_CSV_FILENAME,
    write_regime_cell_review_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-horizon-csv",
        type=Path,
        default=DEFAULT_REGIME_OUTCOME_TABLE_DIR / REGIME_OUTCOME_BEST_HORIZON_CSV_FILENAME,
    )
    parser.add_argument(
        "--by-horizon-csv",
        type=Path,
        default=DEFAULT_REGIME_OUTCOME_TABLE_DIR / REGIME_OUTCOME_BY_HORIZON_CSV_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGIME_CELL_REVIEW_DIR)
    parser.add_argument("--documentation-dir", type=Path, default=DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR)
    parser.add_argument("--report-name", default="regime_cell_review")
    parser.add_argument("--no-doc-copy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = write_regime_cell_review_report(
        best_horizon_csv=args.best_horizon_csv,
        by_horizon_csv=args.by_horizon_csv,
        output_dir=args.output_dir,
        documentation_dir=None if args.no_doc_copy else args.documentation_dir,
        report_name=args.report_name,
    )
    report = artifact["report"]
    print(
        json.dumps(
            {
                "report_type": report.get("report_type"),
                "proposal_status": report.get("proposal_status"),
                "reviewed_cell_count": report.get("reviewed_cell_count"),
                "action_counts": report.get("action_counts"),
                "json_path": artifact.get("json_path"),
                "markdown_path": artifact.get("markdown_path"),
                "csv_path": artifact.get("csv_path"),
                "latest_json_path": artifact.get("latest_json_path"),
                "latest_markdown_path": artifact.get("latest_markdown_path"),
                "latest_csv_path": artifact.get("latest_csv_path"),
                "documentation_markdown_path": artifact.get("documentation_markdown_path"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
