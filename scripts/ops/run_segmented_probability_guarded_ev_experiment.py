#!/usr/bin/env python3
"""Run guarded EV experiment for rejected segmented-probability candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.segmented_probability_guarded_ev_experiment import (  # noqa: E402
    DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_DIR,
    GUARDED_EV_EXPERIMENT_PASS,
    write_segmented_probability_guarded_ev_experiment_report_from_paths,
)
from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    DEFAULT_LABEL_FIELD,
    DEFAULT_PROBABILITY_FIELD,
    DEFAULT_REGIME_FIELDS,
    DEFAULT_RETURN_FIELD,
)


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a research-only guarded EV experiment for a rejected segmented-probability candidate. "
            "The command evaluates route quarantine and raw-rank preservation without changing runtime config, "
            "parameter packs, data sources, or execution behavior."
        )
    )
    parser.add_argument("--ev-shadow-report", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH)
    parser.add_argument(
        "--ev-rejection-attribution",
        type=Path,
        default=DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_PATH,
    )
    parser.add_argument("--ev-shadow-routes", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--candidate-bundle", type=Path, default=None)
    parser.add_argument("--route-policy", default=None)
    parser.add_argument("--probability-field", default=DEFAULT_PROBABILITY_FIELD)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD)
    parser.add_argument("--return-field", default=DEFAULT_RETURN_FIELD)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--validation-mode", choices=["auto", "after_candidate_generated", "holdout_replay"], default=None)
    parser.add_argument("--regime-fields", default=",".join(DEFAULT_REGIME_FIELDS))
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--min-shadow-sample", type=int, default=100)
    parser.add_argument("--min-top-sample", type=int, default=25)
    parser.add_argument("--min-risk-adjusted-improvement-bps", type=float, default=0.0)
    parser.add_argument("--max-hit-rate-regression", type=float, default=0.02)
    parser.add_argument("--downside-penalty-weight", type=float, default=0.25)
    parser.add_argument("--spread-penalty-per-pct", type=float, default=2.0)
    parser.add_argument("--max-spread-pct", type=float, default=5.0)
    parser.add_argument("--raw-rank-ceiling-multiplier", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 unless the guarded experiment clears its pass guardrail.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = write_segmented_probability_guarded_ev_experiment_report_from_paths(
        ev_shadow_report_path=args.ev_shadow_report,
        ev_rejection_attribution_path=args.ev_rejection_attribution,
        ev_shadow_routes_path=args.ev_shadow_routes,
        dataset_path=args.dataset,
        candidate_bundle_path=args.candidate_bundle,
        route_policy=args.route_policy,
        probability_field=args.probability_field,
        label_field=args.label_field,
        return_field=args.return_field,
        train_fraction=args.train_fraction,
        validation_mode=args.validation_mode,
        regime_fields=_csv_tuple(args.regime_fields),
        top_fraction=args.top_fraction,
        min_shadow_sample=args.min_shadow_sample,
        min_top_sample=args.min_top_sample,
        min_risk_adjusted_improvement_bps=args.min_risk_adjusted_improvement_bps,
        max_hit_rate_regression=args.max_hit_rate_regression,
        downside_penalty_weight=args.downside_penalty_weight,
        spread_penalty_per_pct=args.spread_penalty_per_pct,
        max_spread_pct=args.max_spread_pct,
        raw_rank_ceiling_multiplier=args.raw_rank_ceiling_multiplier,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact.get("report", {}) or {}
    selection = report.get("selection_summary", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["guarded_ev_status"] = report.get("guarded_ev_status")
    payload["analysis_policy"] = report.get("analysis_policy")
    payload["simulation_mode"] = report.get("simulation_mode")
    payload["recommended_guarded_variant"] = selection.get("recommended_guarded_variant")
    payload["recommended_guarded_variant_status"] = selection.get("recommended_guarded_variant_status")
    payload["recommended_variant_risk_delta_vs_raw_bps"] = selection.get(
        "recommended_variant_risk_delta_vs_raw_bps"
    )
    payload["recommended_variant_hit_delta_vs_raw"] = selection.get("recommended_variant_hit_delta_vs_raw")
    payload["recommended_variant_risk_delta_improvement_vs_baseline_bps"] = selection.get(
        "recommended_variant_risk_delta_improvement_vs_baseline_bps"
    )
    payload["quarantined_candidate_keys"] = report.get("quarantined_candidate_keys")
    payload["recommended_next_actions"] = report.get("recommended_next_actions")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.require_pass and report.get("guarded_ev_status") != GUARDED_EV_EXPERIMENT_PASS:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
