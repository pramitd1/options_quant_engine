#!/usr/bin/env python3
"""Generate a guarded segmented-probability candidate bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_forward_shadow import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
)
from research.signal_evaluation.segmented_probability_guarded_candidate_bundle import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR,
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_PATH,
    GUARDED_CANDIDATE_BUNDLE_READY,
    write_segmented_probability_guarded_candidate_bundle_report_from_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a research-only guarded segmented-probability candidate bundle from the latest guarded EV "
            "experiment. The command writes advisory artifacts only and does not change runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument(
        "--source-candidate-bundle",
        type=Path,
        default=None,
        help=f"Source candidate bundle JSON. Defaults to {DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH}.",
    )
    parser.add_argument(
        "--guarded-ev-experiment",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_PATH,
        help="Guarded EV experiment JSON artifact.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--allow-watch",
        action="store_true",
        help="Write a WATCH bundle when non-blocking warnings exist. Blocking failures still remain advisory only.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit 2 unless the generated guarded candidate bundle is READY.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_segmented_probability_guarded_candidate_bundle_report_from_paths(
        source_candidate_bundle_path=args.source_candidate_bundle,
        guarded_ev_experiment_path=args.guarded_ev_experiment,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
        allow_watch=args.allow_watch,
    )
    report = artifact.get("report", {}) or {}
    payload = {key: value for key, value in artifact.items() if key not in {"report", "candidate_bundle"}}
    payload["guarded_candidate_bundle_status"] = report.get("guarded_candidate_bundle_status")
    payload["guarded_candidate_bundle_reasons"] = report.get("guarded_candidate_bundle_reasons")
    payload["source_candidate_count"] = report.get("source_candidate_count")
    payload["kept_candidate_count"] = report.get("kept_candidate_count")
    payload["quarantined_candidate_count"] = report.get("quarantined_candidate_count")
    payload["quarantined_candidate_keys"] = report.get("quarantined_candidate_keys")
    payload["required_next_validations"] = report.get("required_next_validations")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.require_ready and report.get("guarded_candidate_bundle_status") != GUARDED_CANDIDATE_BUNDLE_READY:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
