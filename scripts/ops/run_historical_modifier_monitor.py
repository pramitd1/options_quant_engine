#!/usr/bin/env python3
"""Run the historical modifier helped/hurt monitor from signal-evaluation rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.historical_modifier_monitor import (  # noqa: E402
    DEFAULT_HISTORICAL_MODIFIER_MONITOR_DIR,
    default_historical_modifier_dataset_path,
    write_historical_modifier_monitor_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a research-only monitor showing whether historical modifiers "
            "aligned with realized 15m/30m/60m outcomes."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HISTORICAL_MODIFIER_MONITOR_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--min-label-sample", type=int, default=30)
    parser.add_argument("--recent-row-limit", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_historical_modifier_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    artifact = write_historical_modifier_monitor_report(
        frame,
        dataset_path=dataset_path,
        min_label_sample=args.min_label_sample,
        recent_row_limit=args.recent_row_limit,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact["report"]
    summary_60m = (report.get("horizon_summary") or {}).get("60m", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload.update(
        {
            "monitor_status": report.get("monitor_status"),
            "assessment_basis": report.get("assessment_basis"),
            "row_count": report.get("row_count"),
            "modifier_row_count": report.get("modifier_row_count"),
            "score_adjustment_nonzero_count": report.get("score_adjustment_nonzero_count"),
            "probability_adjustment_nonzero_count": report.get("probability_adjustment_nonzero_count"),
            "direction_override_count": report.get("direction_override_count"),
            "interaction_nonzero_count": report.get("interaction_nonzero_count"),
            "label_count_60m": summary_60m.get("label_count"),
            "helped_count_60m": summary_60m.get("helped_count"),
            "hurt_count_60m": summary_60m.get("hurt_count"),
            "help_rate_60m": summary_60m.get("help_rate"),
            "avg_signed_return_60m_bps": summary_60m.get("avg_signed_return_bps"),
            "recommended_next_actions": report.get("recommended_next_actions", []),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
