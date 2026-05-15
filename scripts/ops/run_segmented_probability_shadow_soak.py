#!/usr/bin/env python3
"""Run the segmented probability shadow-soak workflow in one command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_forward_shadow import (  # noqa: E402
    DEFAULT_ROUTING_POLICIES,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
)
from research.signal_evaluation.segmented_probability_guarded_shadow_validation import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH,
)
from research.signal_evaluation.segmented_probability_shadow_soak import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_SHADOW_SOAK_DIR,
    OUTCOME_REFRESH_DEFAULT_PROVIDER,
    OUTCOME_REFRESH_LOCAL_SPOT_HISTORY,
    OUTCOME_REFRESH_SKIP,
    SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED,
    SOAK_CANDIDATE_STALENESS_BLOCKED,
    SOAK_GUARDED_VALIDATION_REJECTED,
    SOAK_READY_FOR_MANUAL_REVIEW,
    write_segmented_probability_shadow_soak_report,
)
from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    DEFAULT_LABEL_FIELD,
    DEFAULT_PROBABILITY_FIELD,
    DEFAULT_REGIME_FIELDS,
    DEFAULT_RETURN_FIELD,
    default_signal_quality_dataset_path,
)


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full research-only segmented-probability shadow soak: forward-shadow "
            "accumulation, candidate staleness, EV shadow context, guard-aware validation, "
            "readiness gating, and a compact soak-status report. This command does not "
            "change runtime config, parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
        help="Original segmented probability candidate bundle JSON.",
    )
    parser.add_argument(
        "--guarded-candidate-bundle",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH,
        help="Guarded segmented probability candidate bundle JSON.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_SHADOW_SOAK_DIR)
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD)
    parser.add_argument("--return-field", default=DEFAULT_RETURN_FIELD)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--routing-policies", default=",".join(DEFAULT_ROUTING_POLICIES))
    parser.add_argument("--regime-fields", default=",".join(DEFAULT_REGIME_FIELDS))
    parser.add_argument("--min-shadow-sample", type=int, default=100)
    parser.add_argument("--min-forward-sample", type=int, default=None)
    parser.add_argument("--min-candidate-sample", type=int, default=50)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece-regression", type=float, default=0.01)
    parser.add_argument("--max-candidate-brier-regression", type=float, default=0.002)
    parser.add_argument("--max-candidate-ece-regression", type=float, default=0.02)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--raw-rank-ceiling-multiplier", type=float, default=None)
    parser.add_argument("--min-ev-sample", type=int, default=100)
    parser.add_argument("--min-top-sample", type=int, default=25)
    parser.add_argument("--min-risk-adjusted-improvement-bps", type=float, default=0.0)
    parser.add_argument("--max-hit-rate-regression", type=float, default=0.02)
    parser.add_argument("--max-spread-pct", type=float, default=5.0)
    parser.add_argument("--max-liquidity-watch-rate", type=float, default=0.35)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--lookback-runs", type=int, default=20)
    parser.add_argument("--stale-after-days", type=float, default=7.0)
    parser.add_argument("--expire-after-days", type=float, default=14.0)
    parser.add_argument("--max-new-rows-before-stale", type=int, default=500)
    parser.add_argument("--max-new-labeled-rows-before-stale", type=int, default=100)
    parser.add_argument("--min-shift-sample", type=int, default=100)
    parser.add_argument("--material-hit-rate-delta", type=float, default=0.10)
    parser.add_argument("--material-probability-delta", type=float, default=0.08)
    parser.add_argument("--material-distribution-psi", type=float, default=0.20)
    parser.add_argument("--policy-lookback-runs", type=int, default=5)
    parser.add_argument("--min-policy-stability-runs", type=int, default=1)
    parser.add_argument("--max-candidate-age-days", type=float, default=14.0)
    parser.add_argument(
        "--outcome-refresh-source",
        choices=[
            OUTCOME_REFRESH_LOCAL_SPOT_HISTORY,
            OUTCOME_REFRESH_DEFAULT_PROVIDER,
            OUTCOME_REFRESH_SKIP,
        ],
        default=OUTCOME_REFRESH_LOCAL_SPOT_HISTORY,
        help=(
            "Refresh pending outcomes before the soak. The default uses local spot history only, "
            "avoiding network/provider dependence."
        ),
    )
    parser.add_argument(
        "--allow-holdout-replay-guarded-validation",
        action="store_true",
        help="Allow guarded holdout replay to satisfy readiness for research review only.",
    )
    parser.add_argument(
        "--skip-legacy-ev-shadow",
        action="store_true",
        help="Skip refreshing the legacy EV shadow artifact; guarded validation remains the primary gate.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit 2 unless the soak reaches SOAK_READY_FOR_MANUAL_REVIEW.",
    )
    parser.add_argument(
        "--fail-on-guarded-rejection",
        action="store_true",
        help="Exit 2 when guard-aware validation rejects the candidate.",
    )
    parser.add_argument(
        "--fail-on-staleness",
        action="store_true",
        help="Exit 2 when candidate staleness blocks the soak.",
    )
    parser.add_argument(
        "--fail-on-no-new-forward-rows",
        action="store_true",
        help="Exit 2 when the latest soak adds no true-forward rows and still needs more sample.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_segmented_probability_shadow_soak_report(
        dataset_path=dataset_path,
        candidate_bundle_path=args.candidate_bundle,
        guarded_candidate_bundle_path=args.guarded_candidate_bundle,
        output_dir=args.output_dir,
        outcome_refresh_source=args.outcome_refresh_source,
        refresh_legacy_ev_shadow=not args.skip_legacy_ev_shadow,
        probability_field=args.probability_field,
        label_field=args.label_field,
        return_field=args.return_field,
        train_fraction=args.train_fraction,
        routing_policies=_csv_tuple(args.routing_policies),
        regime_fields=_csv_tuple(args.regime_fields),
        min_shadow_sample=args.min_shadow_sample,
        min_forward_sample=args.min_forward_sample,
        min_candidate_sample=args.min_candidate_sample,
        min_brier_improvement=args.min_brier_improvement,
        max_ece_regression=args.max_ece_regression,
        max_candidate_brier_regression=args.max_candidate_brier_regression,
        max_candidate_ece_regression=args.max_candidate_ece_regression,
        top_fraction=args.top_fraction,
        raw_rank_ceiling_multiplier=args.raw_rank_ceiling_multiplier,
        min_ev_sample=args.min_ev_sample,
        min_top_sample=args.min_top_sample,
        min_risk_adjusted_improvement_bps=args.min_risk_adjusted_improvement_bps,
        max_hit_rate_regression=args.max_hit_rate_regression,
        max_spread_pct=args.max_spread_pct,
        max_liquidity_watch_rate=args.max_liquidity_watch_rate,
        n_bins=args.n_bins,
        lookback_runs=args.lookback_runs,
        stale_after_days=args.stale_after_days,
        expire_after_days=args.expire_after_days,
        max_new_rows_before_stale=args.max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=args.max_new_labeled_rows_before_stale,
        min_shift_sample=args.min_shift_sample,
        material_hit_rate_delta=args.material_hit_rate_delta,
        material_probability_delta=args.material_probability_delta,
        material_distribution_psi=args.material_distribution_psi,
        policy_lookback_runs=args.policy_lookback_runs,
        min_policy_stability_runs=args.min_policy_stability_runs,
        max_candidate_age_days=args.max_candidate_age_days,
        allow_holdout_replay_guarded_validation=args.allow_holdout_replay_guarded_validation,
    )
    report = artifact.get("shadow_soak_report", {}) or {}
    progress = report.get("forward_sample_progress", {}) or {}
    outcome = report.get("outcome_refresh_summary", {}) or {}
    guarded = report.get("guarded_validation_summary", {}) or {}
    readiness = report.get("readiness_summary", {}) or {}
    staleness = report.get("candidate_staleness_summary", {}) or {}
    guarded_staleness = report.get("guarded_candidate_staleness_summary", {}) or {}
    payload = {
        "shadow_soak_json_path": artifact.get("shadow_soak_json_path"),
        "shadow_soak_markdown_path": artifact.get("shadow_soak_markdown_path"),
        "shadow_soak_history_path": artifact.get("shadow_soak_history_path"),
        "soak_status": report.get("soak_status"),
        "soak_reasons": report.get("soak_reasons"),
        "validation_mode_used": progress.get("validation_mode_used"),
        "strict_forward_row_count": progress.get("strict_forward_row_count"),
        "min_forward_sample": progress.get("min_forward_sample"),
        "forward_sample_gap": progress.get("forward_sample_gap"),
        "new_true_forward_rows_since_previous_soak": progress.get(
            "new_true_forward_rows_since_previous_soak"
        ),
        "outcome_refresh_source": outcome.get("outcome_refresh_source"),
        "outcome_refresh_attempted": outcome.get("outcome_refresh_attempted"),
        "outcome_refresh_error": outcome.get("outcome_refresh_error"),
        "new_quality_labeled_rows_after_candidate": outcome.get(
            "new_quality_labeled_rows_after_candidate"
        ),
        "post_candidate_quality_labeled_rows": (
            outcome.get("post_refresh", {}) or {}
        ).get("quality_labeled_rows_after_candidate"),
        "post_guarded_quality_labeled_rows": (
            outcome.get("guarded_post_refresh", {}) or {}
        ).get("quality_labeled_rows_after_candidate"),
        "new_quality_labeled_rows_after_guarded_candidate": outcome.get(
            "new_quality_labeled_rows_after_guarded_candidate"
        ),
        "accumulation_status": progress.get("accumulation_status"),
        "guarded_shadow_status": guarded.get("guarded_shadow_status"),
        "guarded_validation_mode_used": guarded.get("validation_mode_used"),
        "guarded_strict_forward_row_count": guarded.get("strict_forward_row_count"),
        "guarded_forward_sample_gap": guarded.get("forward_sample_gap"),
        "guarded_risk_delta_bps": guarded.get("top_bucket_risk_delta_bps"),
        "guarded_hit_rate_delta": guarded.get("top_bucket_hit_rate_delta"),
        "guarded_quarantined_route_top_count": guarded.get("quarantined_route_top_count"),
        "readiness_status": readiness.get("readiness_status"),
        "staleness_status": staleness.get("staleness_status"),
        "guarded_staleness_status": guarded_staleness.get("guarded_staleness_status"),
        "guarded_staleness_routing_policy_status": guarded_staleness.get(
            "guarded_routing_policy_status"
        ),
        "recommended_next_actions": report.get("recommended_next_actions"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    if args.fail_on_no_new_forward_rows and "no_new_true_forward_rows_since_previous_soak" in (
        report.get("soak_reasons", []) or []
    ):
        return 2
    if args.fail_on_guarded_rejection and report.get("soak_status") == SOAK_GUARDED_VALIDATION_REJECTED:
        return 2
    if args.fail_on_staleness and report.get("soak_status") in {
        SOAK_CANDIDATE_STALENESS_BLOCKED,
        SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED,
    }:
        return 2
    if args.fail_on_blocked and report.get("soak_status") != SOAK_READY_FOR_MANUAL_REVIEW:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
