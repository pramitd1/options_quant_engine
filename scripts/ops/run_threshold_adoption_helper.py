#!/usr/bin/env python3
"""Build or apply an advisory parameter-pack patch for threshold adoption."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_adoption_helper import (  # noqa: E402
    DEFAULT_TARGET_PARAMETER_PACK_PATH,
    DEFAULT_THRESHOLD_ADOPTION_PLAN_DIR,
    write_threshold_adoption_plan,
)
from research.signal_evaluation.threshold_adoption_reconciliation import (  # noqa: E402
    DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH,
    DEFAULT_PROMOTION_REVIEW_LEDGER_PATH,
    DEFAULT_PROMOTION_REVIEW_REPORT_PATH,
)


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an advisory parameter-pack patch from an APPROVED threshold "
            "promotion package. Dry-run is the default; --apply is required to write."
        )
    )
    parser.add_argument("--promotion-package", type=Path, default=DEFAULT_PROMOTION_REVIEW_REPORT_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_PROMOTION_REVIEW_LEDGER_PATH)
    parser.add_argument("--post-promotion-monitor", type=Path, default=DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH)
    parser.add_argument("--runtime-config", type=Path, default=None, help="Optional JSON snapshot of runtime config.")
    parser.add_argument("--parameter-pack", type=Path, default=None, help="Optional parameter-pack JSON to compare.")
    parser.add_argument(
        "--target-parameter-pack",
        type=Path,
        default=DEFAULT_TARGET_PARAMETER_PACK_PATH,
        help="Parameter-pack JSON to patch when --apply is supplied.",
    )
    parser.add_argument("--candidate-key", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_ADOPTION_PLAN_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--apply", action="store_true", help="Write the proposed override to --target-parameter-pack.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    promotion_package = _load_json(args.promotion_package)
    post_promotion_monitor = _load_json(args.post_promotion_monitor)
    runtime_config = _load_json(args.runtime_config)
    parameter_pack = _load_json(args.parameter_pack)
    target_parameter_pack = _load_json(args.target_parameter_pack)

    artifact = write_threshold_adoption_plan(
        promotion_package_report=promotion_package,
        promotion_package_report_path=args.promotion_package,
        ledger_path=args.ledger,
        post_promotion_monitor_report=post_promotion_monitor,
        post_promotion_monitor_report_path=args.post_promotion_monitor,
        runtime_config=runtime_config,
        parameter_pack=parameter_pack,
        parameter_pack_path=args.parameter_pack,
        target_parameter_pack=target_parameter_pack,
        target_parameter_pack_path=args.target_parameter_pack,
        candidate_key=args.candidate_key,
        apply_changes=args.apply,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["plan_status"] = artifact["report"].get("plan_status")
    payload["runtime_config_changed"] = artifact["report"].get("runtime_config_changed")
    payload["parameter_pack_file_changed"] = artifact["report"].get("parameter_pack_file_changed")
    payload["adoption_status"] = (
        (artifact["report"].get("adoption_reconciliation", {}) or {}).get("adoption_status")
    )
    payload["target_parameter_pack_path"] = artifact["report"].get("target_parameter_pack_path")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
