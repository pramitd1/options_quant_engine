#!/usr/bin/env python3
"""Run the read-only pre-adoption threshold replay gate."""

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
from research.signal_evaluation.threshold_adoption_replay_gate import (  # noqa: E402
    ADOPTION_REPLAY_READY,
    DEFAULT_ADOPTION_PLAN_REPORT_PATH,
    DEFAULT_THRESHOLD_ADOPTION_REPLAY_GATE_DIR,
    write_threshold_adoption_replay_gate_report,
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an approved threshold adoption plan in a temporary runtime "
            "context before any manual parameter-pack adoption."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--adoption-plan", type=Path, default=DEFAULT_ADOPTION_PLAN_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_ADOPTION_REPLAY_GATE_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit with status 2 unless replay status is ADOPTION_REPLAY_READY.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    adoption_plan = _load_json(args.adoption_plan)
    artifact = write_threshold_adoption_replay_gate_report(
        frame,
        adoption_plan_report=adoption_plan,
        adoption_plan_report_path=args.adoption_plan,
        dataset_path=str(dataset_path),
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact["report"]
    comparison = report.get("replay_comparison", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["replay_status"] = report.get("replay_status")
    payload["runtime_config_changed"] = report.get("runtime_config_changed")
    payload["parameter_pack_file_changed"] = report.get("parameter_pack_file_changed")
    payload["baseline_signal_count"] = comparison.get("baseline_signal_count")
    payload["candidate_signal_count"] = comparison.get("candidate_signal_count")
    payload["signal_count_delta"] = comparison.get("signal_count_delta")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.require_ready and report.get("replay_status") != ADOPTION_REPLAY_READY:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
