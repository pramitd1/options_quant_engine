#!/usr/bin/env python3
"""Run read-only reconciliation for approved threshold adoption state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_adoption_reconciliation import (  # noqa: E402
    ADOPTED_MANUALLY,
    DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH,
    DEFAULT_PROMOTION_REVIEW_LEDGER_PATH,
    DEFAULT_PROMOTION_REVIEW_REPORT_PATH,
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    write_threshold_adoption_reconciliation_report,
)


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile APPROVED threshold promotion intent with the active runtime threshold policy."
    )
    parser.add_argument("--promotion-package", type=Path, default=DEFAULT_PROMOTION_REVIEW_REPORT_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_PROMOTION_REVIEW_LEDGER_PATH)
    parser.add_argument("--post-promotion-monitor", type=Path, default=DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH)
    parser.add_argument("--runtime-config", type=Path, default=None, help="Optional JSON snapshot of runtime config.")
    parser.add_argument("--parameter-pack", type=Path, default=None, help="Optional parameter-pack JSON to compare.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--candidate-key", default=None)
    parser.add_argument(
        "--require-adopted",
        action="store_true",
        help="Exit with status 2 unless the approved threshold resolves as ADOPTED_MANUALLY.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    promotion_package = _load_json(args.promotion_package)
    post_promotion_monitor = _load_json(args.post_promotion_monitor)
    runtime_config = _load_json(args.runtime_config)
    parameter_pack = _load_json(args.parameter_pack)
    artifact = write_threshold_adoption_reconciliation_report(
        promotion_package_report=promotion_package,
        promotion_package_report_path=args.promotion_package,
        ledger_path=args.ledger,
        post_promotion_monitor_report=post_promotion_monitor,
        post_promotion_monitor_report_path=args.post_promotion_monitor,
        runtime_config=runtime_config,
        parameter_pack=parameter_pack,
        parameter_pack_path=args.parameter_pack,
        candidate_key=args.candidate_key,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["adoption_status"] = artifact["report"].get("adoption_status")
    payload["runtime_config_changed"] = artifact["report"].get("runtime_config_changed")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_adopted and artifact["report"].get("adoption_status") != ADOPTED_MANUALLY:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
