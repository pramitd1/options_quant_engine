#!/usr/bin/env python3
"""Run segmented probability candidate staleness governance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_candidate_staleness import (  # noqa: E402
    ACTIVE_REVIEW,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH,
    write_segmented_probability_candidate_staleness_report,
)
from research.signal_evaluation.segmented_probability_forward_shadow import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
)
from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    DEFAULT_LABEL_FIELD,
    DEFAULT_PROBABILITY_FIELD,
    default_signal_quality_dataset_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the latest segmented probability candidate bundle and emit a research-only "
            "staleness governance report. This command does not change runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=default_signal_quality_dataset_path())
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    )
    parser.add_argument("--candidate-bundle-search-dir", type=Path, default=None)
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR,
    )
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD)
    parser.add_argument("--stale-after-days", type=float, default=7.0)
    parser.add_argument("--expire-after-days", type=float, default=14.0)
    parser.add_argument("--max-new-rows-before-stale", type=int, default=500)
    parser.add_argument("--max-new-labeled-rows-before-stale", type=int, default=100)
    parser.add_argument("--min-shift-sample", type=int, default=100)
    parser.add_argument("--material-hit-rate-delta", type=float, default=0.10)
    parser.add_argument("--material-probability-delta", type=float, default=0.08)
    parser.add_argument("--material-distribution-psi", type=float, default=0.20)
    parser.add_argument("--policy-lookback-runs", type=int, default=10)
    parser.add_argument(
        "--fail-on-not-active",
        action="store_true",
        help="Exit with status 2 unless the report emits ACTIVE_REVIEW.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_segmented_probability_candidate_staleness_report(
        dataset_path=args.dataset,
        candidate_bundle_path=args.candidate_bundle,
        candidate_bundle_search_dir=args.candidate_bundle_search_dir,
        history_path=args.history,
        output_dir=args.output_dir,
        probability_field=args.probability_field,
        label_field=args.label_field,
        stale_after_days=args.stale_after_days,
        expire_after_days=args.expire_after_days,
        max_new_rows_before_stale=args.max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=args.max_new_labeled_rows_before_stale,
        min_shift_sample=args.min_shift_sample,
        material_hit_rate_delta=args.material_hit_rate_delta,
        material_probability_delta=args.material_probability_delta,
        material_distribution_psi=args.material_distribution_psi,
        policy_lookback_runs=args.policy_lookback_runs,
    )
    report = artifact.get("staleness_report", {}) or {}
    candidate = report.get("candidate_summary", {}) or {}
    currency = report.get("dataset_currency", {}) or {}
    shift = report.get("forward_label_population_shift", {}) or {}
    routing = report.get("routing_policy_stability", {}) or {}
    payload = {
        "staleness_json_path": artifact.get("staleness_json_path"),
        "staleness_markdown_path": artifact.get("staleness_markdown_path"),
        "staleness_status": report.get("staleness_status"),
        "staleness_reasons": report.get("staleness_reasons"),
        "candidate_generated_at": candidate.get("candidate_generated_at"),
        "candidate_age_days": candidate.get("candidate_age_days"),
        "candidate_count": candidate.get("candidate_count"),
        "dataset_currency_status": currency.get("dataset_currency_status"),
        "rows_after_candidate_generated": currency.get("rows_after_candidate_generated"),
        "quality_labeled_rows_after_candidate_generated": currency.get(
            "quality_labeled_rows_after_candidate_generated"
        ),
        "forward_label_shift_status": shift.get("shift_status"),
        "post_candidate_label_count": shift.get("post_candidate_label_count"),
        "routing_policy_status": routing.get("policy_stability_status"),
        "latest_recommended_routing_policy": routing.get("latest_recommended_routing_policy"),
        "recommended_next_actions": report.get("recommended_next_actions"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.fail_on_not_active and report.get("staleness_status") != ACTIVE_REVIEW:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
