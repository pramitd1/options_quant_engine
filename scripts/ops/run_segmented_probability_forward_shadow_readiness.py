#!/usr/bin/env python3
"""Run the segmented probability forward-shadow readiness gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_forward_shadow_readiness import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_DIR,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_REPORT_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_PATH,
    FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW,
    write_segmented_probability_forward_shadow_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read latest segmented probability forward-shadow artifacts and emit a hard "
            "manual-review readiness gate. This command is research-only and does not change "
            "runtime config, parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument(
        "--accumulation-dashboard",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_PATH,
    )
    parser.add_argument(
        "--forward-shadow-report",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_REPORT_PATH,
    )
    parser.add_argument(
        "--candidate-staleness-report",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_PATH,
    )
    parser.add_argument(
        "--ev-shadow-report",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_PATH,
    )
    parser.add_argument(
        "--guarded-shadow-report",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_PATH,
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_DIR)
    parser.add_argument("--min-forward-sample", type=int, default=100)
    parser.add_argument("--min-policy-stability-runs", type=int, default=1)
    parser.add_argument("--policy-lookback-runs", type=int, default=5)
    parser.add_argument("--max-candidate-age-days", type=float, default=14.0)
    parser.add_argument(
        "--allow-holdout-replay-guarded-validation",
        action="store_true",
        help="Allow a guarded holdout-replay PASS to satisfy validation/sample gates for research review only.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit with status 2 unless the gate emits FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard_path=args.accumulation_dashboard,
        forward_shadow_report_path=args.forward_shadow_report,
        candidate_staleness_path=args.candidate_staleness_report,
        ev_shadow_path=args.ev_shadow_report,
        guarded_shadow_path=args.guarded_shadow_report,
        history_path=args.history,
        output_dir=args.output_dir,
        min_forward_sample=args.min_forward_sample,
        min_policy_stability_runs=args.min_policy_stability_runs,
        policy_lookback_runs=args.policy_lookback_runs,
        max_candidate_age_days=args.max_candidate_age_days,
        allow_holdout_replay_guarded_validation=args.allow_holdout_replay_guarded_validation,
    )
    report = artifact.get("readiness_report", {}) or {}
    checks = report.get("checked_conditions", {}) or {}
    payload = {
        "readiness_json_path": artifact.get("readiness_json_path"),
        "readiness_markdown_path": artifact.get("readiness_markdown_path"),
        "readiness_status": report.get("readiness_status"),
        "readiness_reasons": report.get("readiness_reasons"),
        "validation_mode_used": checks.get("validation_mode_used"),
        "strict_forward_row_count": checks.get("strict_forward_row_count"),
        "min_forward_sample": checks.get("min_forward_sample"),
        "shadow_validation_status": checks.get("shadow_validation_status"),
        "accumulation_status": checks.get("accumulation_status"),
        "recommended_routing_policy": checks.get("recommended_routing_policy"),
        "routing_policy_stable": checks.get("routing_policy_stable"),
        "candidate_route_regression_count": checks.get("candidate_route_regression_count"),
        "candidate_age_days": checks.get("candidate_age_days"),
        "candidate_staleness_status": checks.get("candidate_staleness_status"),
        "candidate_staleness_active": checks.get("candidate_staleness_status_active_review"),
        "candidate_staleness_schema_passed": checks.get("candidate_staleness_schema_passed"),
        "candidate_bundle_not_superseded": checks.get("candidate_bundle_not_superseded"),
        "candidate_forward_label_population_stable": checks.get(
            "candidate_forward_label_population_stable"
        ),
        "ev_shadow_status": checks.get("ev_shadow_status"),
        "ev_shadow_schema_passed": checks.get("ev_shadow_schema_passed"),
        "ev_shadow_top_bucket_risk_adjusted_return_delta_bps": checks.get(
            "ev_shadow_top_bucket_risk_adjusted_return_delta_bps"
        ),
        "ev_shadow_top_bucket_hit_rate_delta": checks.get("ev_shadow_top_bucket_hit_rate_delta"),
        "ev_shadow_liquidity_status": checks.get("ev_shadow_liquidity_status"),
        "ev_shadow_key_candidate_negative_route_count": checks.get(
            "ev_shadow_key_candidate_negative_route_count"
        ),
        "guarded_shadow_status": checks.get("guarded_shadow_status"),
        "guarded_shadow_schema_passed": checks.get("guarded_shadow_schema_passed"),
        "guarded_shadow_validation_mode_used": checks.get("guarded_shadow_validation_mode_used"),
        "guarded_shadow_top_bucket_risk_adjusted_return_delta_bps": checks.get(
            "guarded_shadow_top_bucket_risk_adjusted_return_delta_bps"
        ),
        "guarded_shadow_top_bucket_hit_rate_delta": checks.get("guarded_shadow_top_bucket_hit_rate_delta"),
        "guarded_shadow_quarantined_route_top_count": checks.get(
            "guarded_shadow_quarantined_route_top_count"
        ),
        "allow_holdout_replay_guarded_validation": checks.get(
            "allow_holdout_replay_guarded_validation"
        ),
        "recommended_next_actions": report.get("recommended_next_actions"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.fail_on_blocked and report.get("readiness_status") != FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
