#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.analytics_feature_policy import (
    get_dealer_flow_policy_config,
    get_iv_hv_spread_policy_config,
)
from research.signal_evaluation.dataset import (
    CUMULATIVE_DATASET_PATH,
    load_signals_dataset,
    sync_live_to_cumulative,
)
from tuning.walk_forward import (
    apply_walk_forward_split,
    build_walk_forward_splits_with_fallback,
)


OUTPUT_DIR = PROJECT_ROOT / "research" / "parameter_tuning"


@dataclass(frozen=True)
class IvHvCandidate:
    rich_threshold_relative: float
    cheap_threshold_relative: float


@dataclass(frozen=True)
class DealerCandidate:
    gamma_weight: float
    charm_weight: float


@dataclass(frozen=True)
class RecommendationDecision:
    status: str
    reasons: tuple[str, ...]
    mean_objective_lift: float | None
    lower_confidence_lift: float | None
    mean_hit_rate_lift: float | None
    split_count: int
    recommended_overrides: dict[str, float]


def _safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _compute_y_up_1d(frame: pd.DataFrame) -> pd.Series:
    spot = _safe_series(frame, "spot_at_signal", default=np.nan)
    spot_1d = _safe_series(frame, "spot_1d", default=np.nan)
    valid = spot.notna() & spot_1d.notna() & (spot > 0)
    y = pd.Series(np.nan, index=frame.index, dtype=float)
    y.loc[valid] = (spot_1d.loc[valid] > spot.loc[valid]).astype(float)
    return y


def _iv_hv_relative(frame: pd.DataFrame) -> pd.Series:
    iv = _safe_series(frame, "atm_iv_scaled", default=np.nan)
    hv_pct = _safe_series(frame, "hist_vol_20d", default=np.nan)
    hv = hv_pct / 100.0
    valid = iv.notna() & hv.notna() & (hv > 0)
    rel = pd.Series(np.nan, index=frame.index, dtype=float)
    rel.loc[valid] = (iv.loc[valid] - hv.loc[valid]) / hv.loc[valid].clip(lower=1e-4)
    return rel


def _zscore(train: pd.Series, values: pd.Series) -> pd.Series:
    mu = float(train.mean()) if len(train) else 0.0
    sigma = float(train.std()) if len(train) else 0.0
    sigma = sigma if sigma > 1e-9 else 1.0
    return (values - mu) / sigma


def _eval_iv_hv_candidate(train: pd.DataFrame, val: pd.DataFrame, cand: IvHvCandidate) -> dict:
    y_val = _compute_y_up_1d(val)
    rel_val = _iv_hv_relative(val)

    valid = y_val.notna() & rel_val.notna()
    if int(valid.sum()) == 0:
        return {
            "n_valid": 0,
            "n_signaled": 0,
            "coverage": 0.0,
            "hit_rate": 0.5,
            "edge": 0.0,
            "objective": 0.0,
        }

    pred = pd.Series(np.nan, index=val.index, dtype=float)
    pred.loc[rel_val > cand.rich_threshold_relative] = 0.0
    pred.loc[rel_val < cand.cheap_threshold_relative] = 1.0

    signaled = valid & pred.notna()
    n_valid = int(valid.sum())
    n_signaled = int(signaled.sum())
    coverage = float(n_signaled / max(n_valid, 1))

    if n_signaled == 0:
        hit_rate = 0.5
    else:
        hit_rate = float((pred.loc[signaled] == y_val.loc[signaled]).mean())

    edge = hit_rate - 0.5
    objective = edge * np.sqrt(max(coverage, 0.0))
    return {
        "n_valid": n_valid,
        "n_signaled": n_signaled,
        "coverage": round(coverage, 6),
        "hit_rate": round(hit_rate, 6),
        "edge": round(edge, 6),
        "objective": round(float(objective), 6),
    }


def _dealer_signal_components(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    flow = _safe_series(frame, "dealer_hedging_flow", default=np.nan)
    if flow.isna().all():
        flow = _safe_series(frame, "hedging_flow_ratio", default=0.0)

    gamma_component = _safe_series(frame, "market_gamma_exposure", default=np.nan)
    if gamma_component.isna().all():
        upside = _safe_series(frame, "upside_hedging_pressure", default=np.nan)
        downside = _safe_series(frame, "downside_hedging_pressure", default=np.nan)
        gamma_component = (upside - downside).fillna(_safe_series(frame, "dealer_hedging_pressure_score", default=0.0) / 100.0)

    charm_component = _safe_series(frame, "market_charm_exposure", default=np.nan)
    if charm_component.isna().all():
        charm_component = _safe_series(frame, "gamma_vol_acceleration_score", default=0.0) / 100.0

    return flow.fillna(0.0), gamma_component.fillna(0.0), charm_component.fillna(0.0)


def _summary_row_for_params(summary: pd.DataFrame, param_cols: list[str], params: dict[str, float]) -> dict:
    if summary.empty:
        return {}
    mask = pd.Series(True, index=summary.index)
    for col in param_cols:
        mask &= summary[col] == params[col]
    if not bool(mask.any()):
        return {}
    return summary.loc[mask].iloc[0].to_dict()


def _split_rows_for_params(split_df: pd.DataFrame, param_cols: list[str], params: dict[str, float]) -> pd.DataFrame:
    if split_df.empty:
        return split_df.copy()
    mask = pd.Series(True, index=split_df.index)
    for col in param_cols:
        mask &= split_df[col] == params[col]
    return split_df.loc[mask].copy().reset_index(drop=True)


def _paired_objective_lift(best_rows: pd.DataFrame, baseline_rows: pd.DataFrame) -> tuple[float | None, float | None, float | None, int]:
    if best_rows.empty or baseline_rows.empty:
        return None, None, None, 0
    merged = best_rows[["split_id", "objective", "hit_rate"]].merge(
        baseline_rows[["split_id", "objective", "hit_rate"]],
        on="split_id",
        how="inner",
        suffixes=("_best", "_baseline"),
    )
    if merged.empty:
        return None, None, None, 0
    objective_delta = merged["objective_best"] - merged["objective_baseline"]
    hit_rate_delta = merged["hit_rate_best"] - merged["hit_rate_baseline"]
    mean_delta = float(objective_delta.mean())
    mean_hit_rate_delta = float(hit_rate_delta.mean())
    if len(merged) < 2:
        return mean_delta, None, mean_hit_rate_delta, int(len(merged))
    stderr = float(objective_delta.std(ddof=1) / np.sqrt(len(merged)))
    lower_ci = mean_delta - (1.96 * stderr)
    return mean_delta, lower_ci, mean_hit_rate_delta, int(len(merged))


def _govern_recommendation(
    *,
    best_params: dict[str, float],
    baseline_params: dict[str, float],
    param_cols: list[str],
    split_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    minimum_trading_days: int,
    observed_trading_days: int,
    minimum_completed_signals: int,
    observed_completed_signals: int,
    minimum_splits: int,
    minimum_objective_lift: float,
    override_prefix: str,
) -> RecommendationDecision:
    reasons: list[str] = []
    recommended_overrides: dict[str, float] = {}
    best_summary = _summary_row_for_params(summary_df, param_cols, best_params)
    best_rows = _split_rows_for_params(split_df, param_cols, best_params)
    baseline_rows = _split_rows_for_params(split_df, param_cols, baseline_params)
    mean_lift, lower_ci, mean_hit_rate_lift, split_count = _paired_objective_lift(best_rows, baseline_rows)

    if observed_trading_days < minimum_trading_days:
        reasons.append("insufficient_trading_days")
    if observed_completed_signals < minimum_completed_signals:
        reasons.append("insufficient_completed_signals")
    if split_count < minimum_splits:
        reasons.append("insufficient_walk_forward_splits")
    if best_params == baseline_params:
        reasons.append("best_candidate_matches_current_policy")
    if float(best_summary.get("avg_coverage", 0.0) or 0.0) <= 0.0:
        reasons.append("zero_signal_coverage")
    if mean_lift is None or mean_lift <= minimum_objective_lift:
        reasons.append("objective_lift_below_floor")
    if lower_ci is None or lower_ci <= 0.0:
        reasons.append("objective_lift_confidence_not_positive")

    if not reasons:
        for col in param_cols:
            if best_params[col] != baseline_params[col]:
                recommended_overrides[f"{override_prefix}.{col}"] = float(best_params[col])

    return RecommendationDecision(
        status="RECOMMEND" if not reasons else "BLOCKED",
        reasons=tuple(reasons),
        mean_objective_lift=round(float(mean_lift), 6) if mean_lift is not None else None,
        lower_confidence_lift=round(float(lower_ci), 6) if lower_ci is not None else None,
        mean_hit_rate_lift=round(float(mean_hit_rate_lift), 6) if mean_hit_rate_lift is not None else None,
        split_count=int(split_count),
        recommended_overrides=recommended_overrides,
    )


def _eval_dealer_candidate(train: pd.DataFrame, val: pd.DataFrame, cand: DealerCandidate) -> dict:
    y_val = _compute_y_up_1d(val)
    y_train = _compute_y_up_1d(train)

    flow_tr, gamma_tr, charm_tr = _dealer_signal_components(train)
    flow_va, gamma_va, charm_va = _dealer_signal_components(val)

    flow_z = _zscore(flow_tr[y_train.notna()], flow_va)
    gamma_z = _zscore(gamma_tr[y_train.notna()], gamma_va)
    charm_z = _zscore(charm_tr[y_train.notna()], charm_va)

    score = flow_z + cand.gamma_weight * gamma_z + cand.charm_weight * charm_z
    valid = y_val.notna() & score.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "n_valid": 0,
            "coverage": 0.0,
            "hit_rate": 0.5,
            "edge": 0.0,
            "objective": 0.0,
        }

    pred_up = (score.loc[valid] > 0).astype(float)
    hit_rate = float((pred_up == y_val.loc[valid]).mean())
    edge = hit_rate - 0.5
    return {
        "n_valid": n_valid,
        "coverage": 1.0,
        "hit_rate": round(hit_rate, 6),
        "edge": round(edge, 6),
        "objective": round(edge, 6),
    }


def _summarize_candidate(df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(param_cols, dropna=False)
        .agg(
            splits=("split_id", "count"),
            avg_objective=("objective", "mean"),
            avg_hit_rate=("hit_rate", "mean"),
            avg_coverage=("coverage", "mean"),
            min_objective=("objective", "min"),
            max_objective=("objective", "max"),
        )
        .reset_index()
        .sort_values(["avg_objective", "avg_hit_rate"], ascending=False)
        .reset_index(drop=True)
    )
    return agg


def _build_splits_with_fallback(
    frame: pd.DataFrame,
    *,
    split_type: str,
    train_window_days: int,
    validation_window_days: int,
    step_size_days: int,
    minimum_train_rows: int,
    minimum_validation_rows: int,
) -> tuple[list, dict]:
    return build_walk_forward_splits_with_fallback(
        frame,
        split_type=split_type,
        train_window_days=train_window_days,
        validation_window_days=validation_window_days,
        step_size_days=step_size_days,
        minimum_train_rows=minimum_train_rows,
        minimum_validation_rows=minimum_validation_rows,
        timestamp_col="signal_timestamp",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward calibration for IV-HV and dealer-flow coefficients")
    parser.add_argument("--dataset", type=Path, default=CUMULATIVE_DATASET_PATH, help="Signal evaluation dataset CSV")
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--validation-window-days", type=int, default=90)
    parser.add_argument("--step-size-days", type=int, default=90)
    parser.add_argument("--minimum-train-rows", type=int, default=250)
    parser.add_argument("--minimum-validation-rows", type=int, default=80)
    parser.add_argument("--minimum-trading-days", type=int, default=20)
    parser.add_argument("--minimum-completed-signals", type=int, default=500)
    parser.add_argument("--minimum-splits", type=int, default=4)
    parser.add_argument("--minimum-objective-lift", type=float, default=0.005)
    args = parser.parse_args()

    synced_rows = 0
    if args.dataset == CUMULATIVE_DATASET_PATH:
        synced_rows = int(sync_live_to_cumulative())

    frame = load_signals_dataset(args.dataset)
    if frame.empty:
        raise RuntimeError("Dataset is empty; cannot calibrate")

    signal_timestamps = pd.to_datetime(frame.get("signal_timestamp", pd.Series(dtype=object)), errors="coerce")
    observed_trading_days = int(signal_timestamps.dropna().dt.normalize().nunique()) if not signal_timestamps.empty else 0
    outcome_status = frame.get("outcome_status", pd.Series(dtype=object)).fillna("")
    observed_completed_signals = int(outcome_status.isin(["PARTIAL", "COMPLETE"]).sum())
    current_iv_policy = get_iv_hv_spread_policy_config()
    current_dealer_policy = get_dealer_flow_policy_config()

    splits, split_config = _build_splits_with_fallback(
        frame,
        split_type="anchored",
        train_window_days=args.train_window_days,
        validation_window_days=args.validation_window_days,
        step_size_days=args.step_size_days,
        minimum_train_rows=args.minimum_train_rows,
        minimum_validation_rows=args.minimum_validation_rows,
    )
    if not splits:
        raise RuntimeError("No walk-forward splits generated; adjust window sizes")

    iv_hv_candidates = [
        IvHvCandidate(rich_threshold_relative=rt, cheap_threshold_relative=ct)
        for rt in (0.10, 0.12, 0.15, 0.18, 0.20, 0.25)
        for ct in (-0.10, -0.12, -0.15, -0.18, -0.20, -0.25)
        if ct < 0 and rt > 0
    ]
    current_iv_candidate = IvHvCandidate(
        rich_threshold_relative=float(current_iv_policy.rich_threshold_relative),
        cheap_threshold_relative=float(current_iv_policy.cheap_threshold_relative),
    )
    if current_iv_candidate not in iv_hv_candidates:
        iv_hv_candidates.append(current_iv_candidate)
    dealer_candidates = [
        DealerCandidate(gamma_weight=gw, charm_weight=cw)
        for gw in (0.10, 0.25, 0.50, 0.75, 1.00, 1.25)
        for cw in (0.00, 0.10, 0.25, 0.50, 0.75)
    ]
    current_dealer_candidate = DealerCandidate(
        gamma_weight=float(current_dealer_policy.gamma_weight),
        charm_weight=float(current_dealer_policy.charm_weight),
    )
    if current_dealer_candidate not in dealer_candidates:
        dealer_candidates.append(current_dealer_candidate)

    iv_rows: list[dict] = []
    dealer_rows: list[dict] = []

    for split in splits:
        split_frames = apply_walk_forward_split(frame, split)
        train = split_frames.train
        val = split_frames.validation

        for cand in iv_hv_candidates:
            metrics = _eval_iv_hv_candidate(train, val, cand)
            iv_rows.append(
                {
                    "split_id": split.split_id,
                    "rich_threshold_relative": cand.rich_threshold_relative,
                    "cheap_threshold_relative": cand.cheap_threshold_relative,
                    **metrics,
                }
            )

        for cand in dealer_candidates:
            metrics = _eval_dealer_candidate(train, val, cand)
            dealer_rows.append(
                {
                    "split_id": split.split_id,
                    "gamma_weight": cand.gamma_weight,
                    "charm_weight": cand.charm_weight,
                    **metrics,
                }
            )

    iv_df = pd.DataFrame(iv_rows)
    dealer_df = pd.DataFrame(dealer_rows)

    iv_summary = _summarize_candidate(
        iv_df,
        ["rich_threshold_relative", "cheap_threshold_relative"],
    )
    dealer_summary = _summarize_candidate(
        dealer_df,
        ["gamma_weight", "charm_weight"],
    )

    best_iv = iv_summary.iloc[0].to_dict() if not iv_summary.empty else {}
    best_dealer = dealer_summary.iloc[0].to_dict() if not dealer_summary.empty else {}
    baseline_iv = {
        "rich_threshold_relative": float(current_iv_candidate.rich_threshold_relative),
        "cheap_threshold_relative": float(current_iv_candidate.cheap_threshold_relative),
    }
    baseline_dealer = {
        "gamma_weight": float(current_dealer_candidate.gamma_weight),
        "charm_weight": float(current_dealer_candidate.charm_weight),
    }
    best_iv_params = {
        "rich_threshold_relative": float(best_iv.get("rich_threshold_relative", baseline_iv["rich_threshold_relative"])),
        "cheap_threshold_relative": float(best_iv.get("cheap_threshold_relative", baseline_iv["cheap_threshold_relative"])),
    }
    best_dealer_params = {
        "gamma_weight": float(best_dealer.get("gamma_weight", baseline_dealer["gamma_weight"])),
        "charm_weight": float(best_dealer.get("charm_weight", baseline_dealer["charm_weight"])),
    }
    iv_governance = _govern_recommendation(
        best_params=best_iv_params,
        baseline_params=baseline_iv,
        param_cols=["rich_threshold_relative", "cheap_threshold_relative"],
        split_df=iv_df,
        summary_df=iv_summary,
        minimum_trading_days=args.minimum_trading_days,
        observed_trading_days=observed_trading_days,
        minimum_completed_signals=args.minimum_completed_signals,
        observed_completed_signals=observed_completed_signals,
        minimum_splits=args.minimum_splits,
        minimum_objective_lift=args.minimum_objective_lift,
        override_prefix="analytics.iv_hv_spread",
    )
    dealer_governance = _govern_recommendation(
        best_params=best_dealer_params,
        baseline_params=baseline_dealer,
        param_cols=["gamma_weight", "charm_weight"],
        split_df=dealer_df,
        summary_df=dealer_summary,
        minimum_trading_days=args.minimum_trading_days,
        observed_trading_days=observed_trading_days,
        minimum_completed_signals=args.minimum_completed_signals,
        observed_completed_signals=observed_completed_signals,
        minimum_splits=args.minimum_splits,
        minimum_objective_lift=args.minimum_objective_lift,
        override_prefix="analytics.dealer_flow",
    )

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"ivhv_dealer_walkforward_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    iv_df.to_csv(run_dir / "iv_hv_candidate_split_metrics.csv", index=False)
    iv_summary.to_csv(run_dir / "iv_hv_candidate_summary.csv", index=False)
    dealer_df.to_csv(run_dir / "dealer_candidate_split_metrics.csv", index=False)
    dealer_summary.to_csv(run_dir / "dealer_candidate_summary.csv", index=False)

    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": str(args.dataset),
        "rows": int(len(frame)),
        "synced_live_rows_into_cumulative": synced_rows,
        "observed_trading_days": observed_trading_days,
        "observed_completed_signals": observed_completed_signals,
        "splits": [split.to_dict() for split in splits],
        "split_configuration": split_config,
        "search_space": {
            "iv_hv": {
                "rich_threshold_relative": sorted({c.rich_threshold_relative for c in iv_hv_candidates}),
                "cheap_threshold_relative": sorted({c.cheap_threshold_relative for c in iv_hv_candidates}),
            },
            "dealer": {
                "gamma_weight": sorted({c.gamma_weight for c in dealer_candidates}),
                "charm_weight": sorted({c.charm_weight for c in dealer_candidates}),
            },
        },
        "best": {
            "iv_hv": best_iv,
            "dealer": best_dealer,
        },
        "baseline": {
            "iv_hv": baseline_iv,
            "dealer": baseline_dealer,
        },
        "governance": {
            "iv_hv": {
                "status": iv_governance.status,
                "reasons": list(iv_governance.reasons),
                "mean_objective_lift": iv_governance.mean_objective_lift,
                "lower_confidence_lift": iv_governance.lower_confidence_lift,
                "mean_hit_rate_lift": iv_governance.mean_hit_rate_lift,
                "split_count": iv_governance.split_count,
                "recommended_overrides": iv_governance.recommended_overrides,
            },
            "dealer": {
                "status": dealer_governance.status,
                "reasons": list(dealer_governance.reasons),
                "mean_objective_lift": dealer_governance.mean_objective_lift,
                "lower_confidence_lift": dealer_governance.lower_confidence_lift,
                "mean_hit_rate_lift": dealer_governance.mean_hit_rate_lift,
                "split_count": dealer_governance.split_count,
                "recommended_overrides": dealer_governance.recommended_overrides,
            },
        },
        "assumptions": [
            "IV-HV relative uses atm_iv_scaled vs hist_vol_20d/100 proxy from signal artifacts.",
            "Dealer calibration uses actual chain-level gamma/charm exposures when present in the signal dataset and falls back to legacy proxy features for older rows.",
        ],
    }

    with (run_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    md = []
    md.append(f"# IV-HV + Dealer Coefficient Walk-Forward Calibration ({run_id})")
    md.append("")
    md.append(f"- Dataset: {args.dataset}")
    md.append(f"- Rows: {len(frame)}")
    md.append(f"- Synced live rows into cumulative archive: {synced_rows}")
    md.append(f"- Trading days observed: {observed_trading_days}")
    md.append(f"- Completed or partial outcome rows: {observed_completed_signals}")
    md.append(f"- Splits: {len(splits)}")
    md.append(f"- Train window days: {args.train_window_days}")
    md.append(f"- Validation window days: {args.validation_window_days}")
    md.append(f"- Step size days: {args.step_size_days}")
    md.append(f"- Minimum train rows: {args.minimum_train_rows}")
    md.append(f"- Minimum validation rows: {args.minimum_validation_rows}")
    if split_config.get("used_fallback"):
        md.append("- Fallback split configuration: enabled")
        md.append(
            f"- Effective split config: train={split_config.get('effective_train_window_days')}d, "
            f"validation={split_config.get('effective_validation_window_days')}d, "
            f"step={split_config.get('effective_step_size_days')}d, "
            f"min_train={split_config.get('effective_minimum_train_rows')}, "
            f"min_validation={split_config.get('effective_minimum_validation_rows')}"
        )
    else:
        md.append("- Fallback split configuration: disabled")
    md.append("")
    if best_iv:
        md.append("## Recommended IV-HV Relative Thresholds")
        md.append("")
        md.append(f"- rich_threshold_relative: {best_iv['rich_threshold_relative']}")
        md.append(f"- cheap_threshold_relative: {best_iv['cheap_threshold_relative']}")
        md.append(f"- avg_objective: {round(float(best_iv['avg_objective']), 6)}")
        md.append(f"- avg_hit_rate: {round(float(best_iv['avg_hit_rate']), 6)}")
        md.append(f"- avg_coverage: {round(float(best_iv['avg_coverage']), 6)}")
        md.append(f"- governance_status: {iv_governance.status}")
        if iv_governance.reasons:
            md.append(f"- governance_reasons: {', '.join(iv_governance.reasons)}")
        md.append("")
    if best_dealer:
        md.append("## Recommended Dealer-Flow Weights")
        md.append("")
        md.append(f"- gamma_weight: {best_dealer['gamma_weight']}")
        md.append(f"- charm_weight: {best_dealer['charm_weight']}")
        md.append(f"- avg_objective: {round(float(best_dealer['avg_objective']), 6)}")
        md.append(f"- avg_hit_rate: {round(float(best_dealer['avg_hit_rate']), 6)}")
        md.append(f"- governance_status: {dealer_governance.status}")
        if dealer_governance.reasons:
            md.append(f"- governance_reasons: {', '.join(dealer_governance.reasons)}")
        md.append("")

    md.append("## Notes")
    md.append("")
    md.append("- IV-HV tuning is directly calibrated on walk-forward artifacts.")
    md.append("- Dealer-flow tuning now uses actual stored chain-level gamma/charm exposures when available and legacy proxies only for older rows.")

    (run_dir / "calibration_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Calibration artifacts written to: {run_dir}")
    if iv_governance.status == "RECOMMEND":
        print(
            "Best IV-HV thresholds:",
            best_iv["rich_threshold_relative"],
            best_iv["cheap_threshold_relative"],
        )
    else:
        print("IV-HV overrides blocked by governance:", ", ".join(iv_governance.reasons))
    if dealer_governance.status == "RECOMMEND":
        print(
            "Best dealer weights:",
            best_dealer["gamma_weight"],
            best_dealer["charm_weight"],
        )
    else:
        print("Dealer overrides blocked by governance:", ", ".join(dealer_governance.reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
