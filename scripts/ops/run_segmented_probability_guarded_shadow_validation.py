#!/usr/bin/env python3
"""Run guard-aware shadow validation for segmented-probability candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_forward_shadow import DEFAULT_ROUTING_POLICIES  # noqa: E402
from research.signal_evaluation.segmented_probability_guarded_shadow_validation import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR,
    GUARDED_SHADOW_VALIDATION_PASS,
    write_segmented_probability_guarded_shadow_validation_report_from_path,
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
            "Run guard-aware forward and EV shadow validation for a guarded segmented-probability bundle. "
            "The command writes advisory artifacts only and does not change runtime config, parameter packs, "
            "data sources, or execution behavior."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH,
        help="Guarded candidate bundle JSON.",
    )
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD)
    parser.add_argument("--return-field", default=DEFAULT_RETURN_FIELD)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument(
        "--validation-mode",
        choices=["auto", "after_candidate_generated", "holdout_replay"],
        default="auto",
    )
    parser.add_argument("--routing-policies", default=",".join(DEFAULT_ROUTING_POLICIES))
    parser.add_argument("--regime-fields", default=",".join(DEFAULT_REGIME_FIELDS))
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--raw-rank-ceiling-multiplier", type=float, default=None)
    parser.add_argument("--min-shadow-sample", type=int, default=100)
    parser.add_argument("--min-ev-sample", type=int, default=100)
    parser.add_argument("--min-top-sample", type=int, default=25)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece-regression", type=float, default=0.01)
    parser.add_argument("--min-risk-adjusted-improvement-bps", type=float, default=0.0)
    parser.add_argument("--max-hit-rate-regression", type=float, default=0.02)
    parser.add_argument("--downside-penalty-weight", type=float, default=0.25)
    parser.add_argument("--spread-penalty-per-pct", type=float, default=2.0)
    parser.add_argument("--max-spread-pct", type=float, default=5.0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 unless guard-aware shadow validation passes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_quality_dataset_path()
    artifact = write_segmented_probability_guarded_shadow_validation_report_from_path(
        dataset_path=dataset_path,
        candidate_bundle_path=args.candidate_bundle,
        probability_field=args.probability_field,
        label_field=args.label_field,
        return_field=args.return_field,
        train_fraction=args.train_fraction,
        validation_mode=args.validation_mode,
        routing_policies=_csv_tuple(args.routing_policies),
        regime_fields=_csv_tuple(args.regime_fields),
        top_fraction=args.top_fraction,
        raw_rank_ceiling_multiplier=args.raw_rank_ceiling_multiplier,
        min_shadow_sample=args.min_shadow_sample,
        min_ev_sample=args.min_ev_sample,
        min_top_sample=args.min_top_sample,
        min_brier_improvement=args.min_brier_improvement,
        max_ece_regression=args.max_ece_regression,
        min_risk_adjusted_improvement_bps=args.min_risk_adjusted_improvement_bps,
        max_hit_rate_regression=args.max_hit_rate_regression,
        downside_penalty_weight=args.downside_penalty_weight,
        spread_penalty_per_pct=args.spread_penalty_per_pct,
        max_spread_pct=args.max_spread_pct,
        n_bins=args.n_bins,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    selection = report.get("selection_summary", {}) or {}
    window = report.get("validation_window", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["guarded_shadow_status"] = report.get("guarded_shadow_status")
    payload["candidate_count"] = report.get("candidate_count")
    payload["validation_mode_used"] = window.get("validation_mode_used")
    payload["strict_forward_row_count"] = window.get("strict_forward_row_count")
    payload["holdout_replay_row_count"] = window.get("holdout_replay_row_count")
    payload["recommended_routing_policy"] = selection.get("recommended_routing_policy")
    payload["recommended_policy_status"] = selection.get("recommended_policy_status")
    payload["recommended_policy_score"] = selection.get("recommended_policy_score")
    payload["recommended_policy_risk_delta_vs_raw_bps"] = selection.get(
        "recommended_policy_risk_delta_vs_raw_bps"
    )
    payload["recommended_policy_hit_delta_vs_raw"] = selection.get("recommended_policy_hit_delta_vs_raw")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.require_pass and report.get("guarded_shadow_status") != GUARDED_SHADOW_VALIDATION_PASS:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
