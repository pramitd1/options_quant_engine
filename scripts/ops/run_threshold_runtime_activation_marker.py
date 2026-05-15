#!/usr/bin/env python3
"""Record deliberate runtime activation of an approved threshold candidate pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_runtime_activation import (  # noqa: E402
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR,
    write_threshold_runtime_activation_marker,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record the timestamp from which candidate-pack signal rows should be treated "
            "as the active rollout window. This writes an ops marker only."
        )
    )
    parser.add_argument("--candidate-pack", default="candidate_v1")
    parser.add_argument("--activated-at", default=None, help="Activation timestamp. Defaults to now in UTC.")
    parser.add_argument("--activated-by", default="operator")
    parser.add_argument("--activation-note", default=None)
    parser.add_argument(
        "--config-hint",
        default="evaluation_thresholds.selection.composite_signal_score_floor",
    )
    parser.add_argument("--threshold-value", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_threshold_runtime_activation_marker(
        candidate_pack_name=args.candidate_pack,
        activated_at=args.activated_at,
        activated_by=args.activated_by,
        activation_note=args.activation_note,
        config_hint=args.config_hint,
        threshold_value=args.threshold_value,
        output_dir=args.output_dir,
    )
    marker = artifact["activation_marker"]
    print(
        json.dumps(
            {
                "activation_marker_json_path": artifact["activation_marker_json_path"],
                "activation_marker_markdown_path": artifact["activation_marker_markdown_path"],
                "candidate_pack_name": marker.get("candidate_pack_name"),
                "activated_at": marker.get("activated_at"),
                "runtime_config_changed": marker.get("runtime_config_changed"),
                "execution_behavior_changed": marker.get("execution_behavior_changed"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
