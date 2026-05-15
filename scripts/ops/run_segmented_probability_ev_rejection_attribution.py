#!/usr/bin/env python3
"""Run attribution for rejected segmented-probability EV shadow candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_ev_rejection_attribution import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_DIR,
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH,
    EV_REJECTION_ATTRIBUTION_ACTIONABLE,
    write_segmented_probability_ev_rejection_attribution_report_from_paths,
)
from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    DEFAULT_REGIME_FIELDS,
    DEFAULT_RETURN_FIELD,
)


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute why a segmented-probability EV shadow candidate was rejected. "
            "This command writes advisory artifacts only and does not change runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument(
        "--ev-shadow-report",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH,
        help="EV shadow evaluation JSON artifact.",
    )
    parser.add_argument(
        "--ev-shadow-routes",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH,
        help="EV shadow route decisions CSV artifact.",
    )
    parser.add_argument(
        "--route-policy",
        default=None,
        help="Routing policy to attribute. Defaults to the EV shadow recommended policy.",
    )
    parser.add_argument("--return-field", default=DEFAULT_RETURN_FIELD)
    parser.add_argument("--regime-fields", default=",".join(DEFAULT_REGIME_FIELDS))
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--min-bucket-sample", type=int, default=25)
    parser.add_argument("--min-candidate-sample", type=int, default=30)
    parser.add_argument("--min-regime-sample", type=int, default=10)
    parser.add_argument("--max-spread-pct", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--fail-on-actionable",
        action="store_true",
        help="Exit 2 when attribution status is actionable. Useful for CI/review gates.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_segmented_probability_ev_rejection_attribution_report_from_paths(
        ev_shadow_report_path=args.ev_shadow_report,
        ev_shadow_routes_path=args.ev_shadow_routes,
        route_policy=args.route_policy,
        return_field=args.return_field,
        regime_fields=_csv_tuple(args.regime_fields),
        top_fraction=args.top_fraction,
        min_bucket_sample=args.min_bucket_sample,
        min_candidate_sample=args.min_candidate_sample,
        min_regime_sample=args.min_regime_sample,
        max_spread_pct=args.max_spread_pct,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    summary = report.get("rejection_summary", {}) or {}
    diagnostics = report.get("routing_diagnostics", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["ev_shadow_status"] = report.get("ev_shadow_status")
    payload["attribution_status"] = report.get("attribution_status")
    payload["attribution_reasons"] = report.get("attribution_reasons")
    payload["analysis_policy"] = report.get("analysis_policy")
    payload["likely_failure_mode"] = diagnostics.get("likely_failure_mode")
    payload["top_bucket_risk_adjusted_return_delta_bps"] = summary.get(
        "top_bucket_risk_adjusted_return_delta_bps"
    )
    payload["top_bucket_hit_rate_delta"] = summary.get("top_bucket_hit_rate_delta")
    payload["shadow_top_raw_top_overlap_rate"] = summary.get("shadow_top_raw_top_overlap_rate")
    payload["selected_policy_negative_candidate_route_count"] = diagnostics.get(
        "selected_policy_negative_candidate_route_count"
    )
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.fail_on_actionable and report.get("attribution_status") == EV_REJECTION_ATTRIBUTION_ACTIONABLE:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
