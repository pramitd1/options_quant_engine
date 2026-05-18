#!/usr/bin/env python3
"""Build the research-only statistical market structure PDF pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.statistical_market_structure import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    write_statistical_market_structure_artifacts,
)
from research.signal_evaluation.statistical_market_context_artifact import (  # noqa: E402
    write_statistical_market_context_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Study historical NIFTY spot, option structure, macro/global data, "
            "and live signal outcomes. Produces CSV tables, JSON/Markdown, and a PDF."
        )
    )
    parser.add_argument("--panel", type=Path, default=None, help="Historical daily feature panel CSV.")
    parser.add_argument("--signal-dataset", type=Path, default=None, help="Optional signal evaluation dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_statistical_market_structure_artifacts(
        panel_path=args.panel,
        signal_dataset_path=args.signal_dataset,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    context_artifact = write_statistical_market_context_artifact(report_path=Path(artifact["json_path"]))
    report = artifact["report"]
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["statistical_context_artifact_path"] = context_artifact["artifact_path"]
    payload.update(
        {
            "report_type": report.get("report_type"),
            "run_id": report.get("run_id"),
            "panel_rows": (report.get("coverage") or {}).get("panel_rows"),
            "option_feature_rows": (report.get("coverage") or {}).get("option_feature_rows"),
            "macro_global_rows": (report.get("coverage") or {}).get("macro_global_rows"),
            "signal_rows": (report.get("coverage") or {}).get("signal_rows"),
            "findings": report.get("findings", []),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
