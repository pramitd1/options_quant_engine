#!/usr/bin/env python3
"""Build or record a manual threshold promotion review package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_promotion_review import (  # noqa: E402
    DEFAULT_SHADOW_REVIEW_REPORT_PATH,
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    record_threshold_promotion_review_decision,
    write_threshold_promotion_review_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a manual threshold promotion review package.")
    parser.add_argument("--shadow-review", type=Path, default=DEFAULT_SHADOW_REVIEW_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--review-action", choices=["APPROVED", "REJECTED", "DEFERRED"], default=None)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--review-note", default="")
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--next-review-at", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    shadow_review = json.loads(args.shadow_review.read_text(encoding="utf-8"))
    artifact = write_threshold_promotion_review_package(
        shadow_review,
        shadow_review_report_path=args.shadow_review,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["promotion_review_status"] = artifact["report"].get("promotion_review_status")
    payload["runtime_config_changed"] = artifact["report"].get("runtime_config_changed")
    if args.review_action:
        if not args.reviewer:
            raise SystemExit("--reviewer is required when --review-action is supplied")
        payload["review_artifact"] = record_threshold_promotion_review_decision(
            report_json_path=artifact["latest_json_path"],
            review_action=args.review_action,
            reviewer=args.reviewer,
            review_note=args.review_note,
            ledger_path=args.ledger_path,
            next_review_at=args.next_review_at,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
