#!/usr/bin/env python3
"""Run a sandbox threshold promotion dry-run on real artifacts."""

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
from research.signal_evaluation.threshold_promotion_dry_run import (  # noqa: E402
    DEFAULT_PROMOTION_REVIEW_LEDGER_PATH,
    DEFAULT_PROMOTION_REVIEW_REPORT_PATH,
    DEFAULT_THRESHOLD_PROMOTION_DRY_RUN_DIR,
    run_threshold_promotion_dry_run,
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dry-run APPROVED threshold promotion using a sandbox ledger, "
            "real promotion package, and real signal dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--promotion-package", type=Path, default=DEFAULT_PROMOTION_REVIEW_REPORT_PATH)
    parser.add_argument("--real-ledger", type=Path, default=DEFAULT_PROMOTION_REVIEW_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_PROMOTION_DRY_RUN_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--approval-timestamp",
        default=None,
        help="Optional simulated approval timestamp. Defaults to just before the first dataset signal.",
    )
    parser.add_argument("--reviewer", default="promotion-dry-run")
    parser.add_argument("--review-note", default="Dry-run APPROVED decision; sandbox ledger only.")
    parser.add_argument("--candidate-key", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    promotion_package = _load_json(args.promotion_package)
    artifact = run_threshold_promotion_dry_run(
        frame,
        promotion_package_report=promotion_package,
        promotion_package_report_path=args.promotion_package,
        dataset_path=str(dataset_path),
        real_ledger_path=args.real_ledger,
        output_dir=args.output_dir,
        report_name=args.report_name,
        approval_timestamp=args.approval_timestamp,
        reviewer=args.reviewer,
        review_note=args.review_note,
        candidate_key=args.candidate_key,
        write_latest=True,
    )
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["dry_run_status"] = artifact["report"].get("dry_run_status")
    payload["post_promotion_monitor_status"] = (
        (artifact["report"].get("post_promotion_monitor", {}) or {}).get("monitor_status")
    )
    payload["adoption_reconciliation_status"] = (
        (artifact["report"].get("adoption_reconciliation", {}) or {}).get("adoption_status")
    )
    payload["runtime_config_changed"] = artifact["report"].get("runtime_config_changed")
    payload["real_promotion_ledger_changed"] = artifact["report"].get("real_promotion_ledger_changed")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
