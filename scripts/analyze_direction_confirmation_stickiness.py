from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.policy_resolver import temporary_parameter_pack
from config.probability_feature_policy import get_probability_feature_policy_config
from config.symbol_microstructure import get_microstructure_config
from strategy.confirmation_filters import compute_confirmation_filters
from utils.numerics import clip, safe_float

DATASET_PATH = ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"
OUTPUT_DIR = ROOT / "research" / "reviews" / "direction_confirmation_stickiness_2026-03-23"
SWEEP_PENALTIES = [round(value * 0.5, 1) for value in range(0, 13)]

# Two-parameter decay sweep grid.
# We pair the hardest-hitting base_penalty candidates with a range of
# decay_steps/decay_factor combinations to find the setting where the
# post-reversal decay window is sufficient to demote confirmation status.
DECAY_BASE_PENALTIES = [3.0, 4.0, 6.0]
DECAY_STEPS_GRID = [0, 1, 2, 3, 4, 5]
DECAY_FACTOR_GRID = [0.3, 0.5, 0.7, 1.0]

# Veto sweep: test different grace period lengths (0-6 steps).
# This is a 1D sweep: we're looking for the minimum veto_steps that breaks stickiness.
VETO_STEPS_GRID = [0, 1, 2, 3, 4, 5, 6]


def _compute_intraday_range_pct(
    *,
    symbol,
    spot_price,
    day_high,
    day_low,
    day_open,
    prev_close,
    lookback_avg_range_pct,
):
    cfg = get_probability_feature_policy_config()
    micro_cfg = get_microstructure_config(symbol)
    spot = safe_float(spot_price, None)
    if spot in (None, 0):
        return None

    high = safe_float(day_high, None)
    low = safe_float(day_low, None)
    open_px = safe_float(day_open, None)
    prev_close_px = safe_float(prev_close, None)
    avg_range = safe_float(lookback_avg_range_pct, None)

    realized_range_pct = None
    if high is not None and low is not None and high >= low:
        realized_range_pct = ((high - low) / spot) * 100.0
    else:
        anchor_moves = []
        if open_px not in (None, 0):
            anchor_moves.append(abs(spot - open_px) / spot * 100.0)
        if prev_close_px not in (None, 0):
            anchor_moves.append(abs(spot - prev_close_px) / spot * 100.0)
        if anchor_moves:
            realized_range_pct = max(anchor_moves) * cfg.intraday_range_anchor_multiplier

    if realized_range_pct is None:
        return None

    baseline_floor = safe_float(
        micro_cfg.get("range_baseline_floor_pct"),
        cfg.intraday_range_baseline_floor_pct,
    )
    baseline = avg_range if avg_range not in (None, 0) else baseline_floor
    baseline = max(baseline, baseline_floor)
    normalized = realized_range_pct / max(baseline, cfg.intraday_range_denominator_floor_pct)
    return round(clip(normalized, 0.0, cfg.intraday_range_clip_cap), 4)


def _recomputed_confirmation_frame(
    frame: pd.DataFrame,
    *,
    direction_change_penalty: float,
    direction_change_decay_steps: int = 0,
    direction_change_decay_factor: float = 0.5,
    reversal_veto_steps: int = 0,
) -> pd.DataFrame:
    rows = []
    previous_direction = None
    reversal_age = None  # steps since last direction change; None = no prior flip

    with temporary_parameter_pack(
        overrides={
            "confirmation_filter.core.direction_change_penalty": direction_change_penalty,
            "confirmation_filter.core.direction_change_decay_steps": direction_change_decay_steps,
            "confirmation_filter.core.direction_change_decay_factor": direction_change_decay_factor,
            "confirmation_filter.core.reversal_veto_steps": reversal_veto_steps,
        },
    ):
        for row in frame.itertuples(index=False):
            # Compute reversal_age before re-evaluating confirmation so that
            # step-0 and step-k penalties fire at the right moment.
            current_dir = str(row.direction or "").strip().upper()
            current_dir = current_dir if current_dir in {"CALL", "PUT"} else None

            if previous_direction is not None and current_dir is not None:
                if current_dir != previous_direction:
                    reversal_age = 0  # this IS the flip snapshot
                else:
                    # same direction: advance the age counter if we are
                    # still inside the decay window
                    reversal_age = (reversal_age + 1) if reversal_age is not None else None
            else:
                reversal_age = None

            intraday_range_pct = _compute_intraday_range_pct(
                symbol=row.symbol,
                spot_price=row.spot_at_signal,
                day_high=row.day_high,
                day_low=row.day_low,
                day_open=row.day_open,
                prev_close=row.prev_close,
                lookback_avg_range_pct=row.lookback_avg_range_pct,
            )
            confirmation = compute_confirmation_filters(
                direction=row.direction,
                previous_direction=previous_direction,
                reversal_age=reversal_age,
                spot=row.spot_at_signal,
                symbol=row.symbol,
                day_open=row.day_open,
                prev_close=row.prev_close,
                intraday_range_pct=intraday_range_pct,
                final_flow_signal=row.final_flow_signal,
                hedging_bias=row.dealer_hedging_bias,
                gamma_event=None,
                hybrid_move_probability=row.hybrid_move_probability,
                spot_vs_flip=row.spot_vs_flip,
                gamma_regime=row.gamma_regime,
            )
            rows.append(
                {
                    "signal_timestamp": row.signal_timestamp,
                    "created_at": row.created_at,
                    "direction": row.direction,
                    "previous_direction": previous_direction,
                    "reversal_age": reversal_age,
                    "dataset_confirmation_status": row.confirmation_status,
                    "recomputed_confirmation_status": confirmation["status"],
                    "recomputed_confirmation_score": confirmation["score_adjustment"],
                    "direction_change_penalty": direction_change_penalty,
                    "direction_change_decay_steps": direction_change_decay_steps,
                    "direction_change_decay_factor": direction_change_decay_factor,
                    "breakdown_direction_change_penalty": confirmation["breakdown"].get("direction_change_penalty", 0.0),
                    "breakdown_decay_penalty": confirmation["breakdown"].get("direction_change_decay_penalty", 0.0),
                    "trade_status": row.trade_status,
                    "direction_source": row.direction_source,
                    "spot_at_signal": row.spot_at_signal,
                }
            )
            previous_direction = current_dir

    return pd.DataFrame(rows)


def _sweep_direction_change_penalty(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_rows = []
    best_frames = []

    for penalty in SWEEP_PENALTIES:
        recomputed = _recomputed_confirmation_frame(frame, direction_change_penalty=penalty)
        metrics = _frame_metrics(
            recomputed.rename(columns={"recomputed_confirmation_status": "confirmation_status"})
        )
        dataset_match_rate = round(
            float((recomputed["dataset_confirmation_status"] == recomputed["recomputed_confirmation_status"]).mean()),
            4,
        )
        strong_share = (
            recomputed["recomputed_confirmation_status"].isin(["STRONG_CONFIRMATION", "CONFIRMED"]).mean()
            if not recomputed.empty
            else 0.0
        )
        reversal_rows = recomputed[
            recomputed["previous_direction"].isin(["CALL", "PUT"])
            & recomputed["direction"].isin(["CALL", "PUT"])
            & recomputed["direction"].ne(recomputed["previous_direction"])
        ].copy()
        strong_reversal_share = (
            reversal_rows["recomputed_confirmation_status"].isin(["STRONG_CONFIRMATION", "CONFIRMED"]).mean()
            if not reversal_rows.empty
            else 0.0
        )
        sweep_rows.append(
            {
                "direction_change_penalty": penalty,
                "rows": int(len(recomputed)),
                "dataset_match_rate": dataset_match_rate,
                "self_transition_rate": metrics["self_transition_rate"],
                "strong_or_confirmed_share": round(float(strong_share), 4),
                "directional_flip_count": metrics["directional_flip_count"],
                "strong_or_confirmed_persist_on_direction_flip_ratio": metrics[
                    "strong_or_confirmed_persist_on_direction_flip_ratio"
                ],
                "strong_or_confirmed_on_reversal_row_share": round(float(strong_reversal_share), 4),
            }
        )
        if penalty in {0.0, 2.0, 3.0, 4.0, 6.0}:
            best_frames.append(recomputed)

    return pd.DataFrame(sweep_rows), pd.concat(best_frames, ignore_index=True) if best_frames else pd.DataFrame()


def _sweep_decay_model(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2-D sweep: base_penalty × decay_steps × decay_factor.

    For each combination, recomputes confirmation scores for every row in the
    collapsed stream (including reversal_age tracking), then summarises:
    - reversal_strong_share: share of step-0 reversal rows still STRONG/CONFIRMED
    - post_reversal_strong_share: share of step-1..N post-reversal rows still STRONG/CONFIRMED
    - self_transition_rate: overall confirmation stickiness after applying decay
    """
    sweep_rows = []
    sample_frames: list[pd.DataFrame] = []

    for base_penalty in DECAY_BASE_PENALTIES:
        for decay_steps in DECAY_STEPS_GRID:
            for decay_factor in DECAY_FACTOR_GRID:
                recomputed = _recomputed_confirmation_frame(
                    frame,
                    direction_change_penalty=base_penalty,
                    direction_change_decay_steps=decay_steps,
                    direction_change_decay_factor=decay_factor,
                )
                metrics = _frame_metrics(
                    recomputed.rename(columns={"recomputed_confirmation_status": "confirmation_status"})
                )

                # Step-0 reversal rows
                rev_mask = (
                    recomputed["previous_direction"].isin(["CALL", "PUT"])
                    & recomputed["direction"].isin(["CALL", "PUT"])
                    & recomputed["direction"].ne(recomputed["previous_direction"])
                )
                rev_rows = recomputed.loc[rev_mask]
                step0_strong_share = (
                    rev_rows["recomputed_confirmation_status"].isin(["STRONG_CONFIRMATION", "CONFIRMED"]).mean()
                    if not rev_rows.empty else float("nan")
                )

                # Post-reversal decay rows (reversal_age in 1..decay_steps)
                if decay_steps > 0:
                    post_mask = (
                        recomputed["reversal_age"].notna()
                        & recomputed["reversal_age"].between(1, decay_steps)
                    )
                    post_rows = recomputed.loc[post_mask]
                    post_strong_share = (
                        post_rows["recomputed_confirmation_status"].isin(["STRONG_CONFIRMATION", "CONFIRMED"]).mean()
                        if not post_rows.empty else float("nan")
                    )
                    post_count = int(len(post_rows))
                else:
                    post_strong_share = float("nan")
                    post_count = 0

                sweep_rows.append(
                    {
                        "direction_change_penalty": base_penalty,
                        "direction_change_decay_steps": decay_steps,
                        "direction_change_decay_factor": decay_factor,
                        "self_transition_rate": metrics["self_transition_rate"],
                        "directional_flip_count": metrics["directional_flip_count"],
                        "step0_reversal_count": int(rev_mask.sum()),
                        "step0_strong_or_confirmed_share": round(float(step0_strong_share), 4) if step0_strong_share == step0_strong_share else None,
                        "post_reversal_row_count": post_count,
                        "post_reversal_strong_or_confirmed_share": round(float(post_strong_share), 4) if post_strong_share == post_strong_share else None,
                        "flip_persist_ratio": metrics["strong_or_confirmed_persist_on_direction_flip_ratio"],
                    }
                )
                # Keep sample frame for the most informative combos
                if base_penalty == 6.0 and decay_steps in {3, 5} and decay_factor in {0.5, 0.7}:
                    sample_frames.append(recomputed)

    return (
        pd.DataFrame(sweep_rows),
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(),
    )


def _sweep_reversal_veto(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-D veto sweep: test reversal_veto_steps from 0 to 6.

    For each veto_steps value, measure:
    - Fraction of reversals (reversal_age=0) forced to MIXED
    - Fraction of post-reversals (reversal_age > 0) forced to MIXED
    - Overall stickiness metrics (self_transition_rate, flip_persist_ratio)
    """
    sweep_rows = []
    sample_frames: list[pd.DataFrame] = []

    for veto_steps in VETO_STEPS_GRID:
        recomputed = _recomputed_confirmation_frame(
            frame,
            direction_change_penalty=0.0,  # No penalty, only veto
            direction_change_decay_steps=0,
            direction_change_decay_factor=0.5,
            reversal_veto_steps=veto_steps,
        )
        metrics = _frame_metrics(
            recomputed.rename(columns={"recomputed_confirmation_status": "confirmation_status"})
        )

        # Reversal snapshot rows (reversal_age=0): count those forced to MIXED
        rev_mask = (
            recomputed["reversal_age"].eq(0)
            & recomputed["previous_direction"].isin(["CALL", "PUT"])
            & recomputed["direction"].isin(["CALL", "PUT"])
            & recomputed["direction"].ne(recomputed["previous_direction"])
        )
        rev_rows = recomputed.loc[rev_mask]
        reversal_mixed_count = (
            (rev_rows["recomputed_confirmation_status"] == "MIXED").sum()
            if not rev_rows.empty else 0
        )
        reversal_mixed_share = (
            int(reversal_mixed_count) / int(len(rev_rows))
            if not rev_rows.empty else float("nan")
        )

        # Post-reversal rows (reversal_age > 0): count those forced to MIXED
        if veto_steps > 0:
            post_mask = (
                recomputed["reversal_age"].notna()
                & (recomputed["reversal_age"] > 0)
                & (recomputed["reversal_age"] < veto_steps)
            )
            post_rows = recomputed.loc[post_mask]
            post_mixed_count = (
                (post_rows["recomputed_confirmation_status"] == "MIXED").sum()
                if not post_rows.empty else 0
            )
            post_mixed_share = (
                int(post_mixed_count) / int(len(post_rows))
                if not post_rows.empty else float("nan")
            )
            post_count = int(len(post_rows))
        else:
            post_mixed_share = float("nan")
            post_count = 0

        sweep_rows.append(
            {
                "reversal_veto_steps": veto_steps,
                "self_transition_rate": metrics["self_transition_rate"],
                "directional_flip_count": metrics["directional_flip_count"],
                "reversal_snapshot_count": int(rev_mask.sum()),
                "reversal_mixed_by_veto_count": int(reversal_mixed_count),
                "reversal_mixed_by_veto_share": round(float(reversal_mixed_share), 4) if reversal_mixed_share == reversal_mixed_share else None,
                "post_reversal_snapshot_count": post_count,
                "post_reversal_mixed_by_veto_share": round(float(post_mixed_share), 4) if post_mixed_share == post_mixed_share else None,
                "flip_persist_ratio": metrics["strong_or_confirmed_persist_on_direction_flip_ratio"],
            }
        )
        if veto_steps in {0, 2, 4, 6}:
            sample_frames.append(recomputed)

    return (
        pd.DataFrame(sweep_rows),
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(),
    )


def _frame_metrics(frame: pd.DataFrame) -> dict:
    status = frame["confirmation_status"].fillna("NA")
    prev_status = status.shift(1)
    direction = frame["direction"].fillna("NONE")
    prev_direction = direction.shift(1)

    run_id = (status != status.shift()).cumsum()
    runs = frame.assign(run_id=run_id).groupby("run_id").agg(
        status=("confirmation_status", "first"),
        run_len=("confirmation_status", "size"),
        start=("signal_timestamp", "min"),
        end=("signal_timestamp", "max"),
    )
    runs["duration_min"] = (runs["end"] - runs["start"]).dt.total_seconds().div(60.0).fillna(0.0)

    directional_flip = (
        direction.isin(["CALL", "PUT"])
        & prev_direction.isin(["CALL", "PUT"])
        & direction.ne(prev_direction)
    )
    strong_conf = status.isin(["STRONG_CONFIRMATION", "CONFIRMED"])
    prev_strong_conf = prev_status.isin(["STRONG_CONFIRMATION", "CONFIRMED"])

    transition_matrix = pd.crosstab(prev_status.fillna("START"), status, normalize="index").round(4)

    return {
        "rows": int(len(frame)),
        "status_counts": {str(key): int(value) for key, value in status.value_counts().to_dict().items()},
        "self_transition_rate": round(float((status.eq(prev_status))[prev_status.notna()].mean()), 4),
        "run_length_mean_by_status": {
            str(key): round(float(value), 2)
            for key, value in runs.groupby("status")["run_len"].mean().to_dict().items()
        },
        "run_length_median_by_status": {
            str(key): round(float(value), 2)
            for key, value in runs.groupby("status")["run_len"].median().to_dict().items()
        },
        "run_length_p95_by_status": {
            str(key): round(float(value), 2)
            for key, value in runs.groupby("status")["run_len"].quantile(0.95).to_dict().items()
        },
        "run_duration_mean_min_by_status": {
            str(key): round(float(value), 2)
            for key, value in runs.groupby("status")["duration_min"].mean().to_dict().items()
        },
        "run_duration_p95_min_by_status": {
            str(key): round(float(value), 2)
            for key, value in runs.groupby("status")["duration_min"].quantile(0.95).to_dict().items()
        },
        "directional_flip_count": int(directional_flip.sum()),
        "same_confirmation_status_on_direction_flip_count": int((directional_flip & status.eq(prev_status)).sum()),
        "strong_or_confirmed_persist_on_direction_flip_count": int(
            (directional_flip & strong_conf & prev_strong_conf).sum()
        ),
        "strong_or_confirmed_persist_on_direction_flip_ratio": round(
            float((directional_flip & strong_conf & prev_strong_conf).sum() / max(int(directional_flip.sum()), 1)),
            4,
        ),
        "strong_or_confirmed_while_direction_missing_count": int(
            ((frame["direction"].isna() | frame["direction"].eq("")) & strong_conf).sum()
        ),
        "transition_matrix": transition_matrix.to_dict(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    df = df[(df["mode"] == "LIVE") & (df["source"] == "ICICI") & (df["symbol"] == "NIFTY")].copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce")
    df["spot_at_signal"] = pd.to_numeric(df["spot_at_signal"], errors="coerce")
    df = df.sort_values(["created_at", "signal_timestamp"]).reset_index(drop=True)

    collapsed = (
        df.sort_values(["signal_timestamp", "created_at"])
        .groupby("signal_timestamp", as_index=False)
        .tail(1)
        .sort_values("signal_timestamp")
        .reset_index(drop=True)
    )

    by_ts = df.groupby("signal_timestamp")
    duplicate_groups = by_ts.size()
    duplicate_groups = duplicate_groups[duplicate_groups > 1]
    intra_timestamp = []
    for ts, group in by_ts:
        if len(group) <= 1:
            continue
        intra_timestamp.append(
            {
                "signal_timestamp": ts.isoformat(),
                "rows": int(len(group)),
                "unique_confirmation_statuses": int(group["confirmation_status"].nunique(dropna=True)),
                "unique_directions": int(group["direction"].nunique(dropna=True)),
                "same_confirmation_status": bool(group["confirmation_status"].nunique(dropna=False) == 1),
                "same_direction": bool(group["direction"].nunique(dropna=False) == 1),
            }
        )
    intra_ts_df = pd.DataFrame(intra_timestamp)

    cf = collapsed.copy()
    status = cf["confirmation_status"].fillna("NA")
    prev_status = status.shift(1)
    direction = cf["direction"].fillna("NONE")
    prev_direction = direction.shift(1)
    flip_mask = direction.isin(["CALL", "PUT"]) & prev_direction.isin(["CALL", "PUT"]) & direction.ne(prev_direction)
    strong_persist_mask = flip_mask & status.isin(["STRONG_CONFIRMATION", "CONFIRMED"]) & prev_status.isin(
        ["STRONG_CONFIRMATION", "CONFIRMED"]
    )
    flip_cases = cf.loc[
        strong_persist_mask,
        ["signal_timestamp", "spot_at_signal", "trade_status", "direction", "confirmation_status", "direction_source"],
    ].copy()
    flip_cases.insert(3, "prev_direction", prev_direction[strong_persist_mask].values)
    flip_cases.insert(5, "prev_confirmation_status", prev_status[strong_persist_mask].values)

    status_collapsed = cf["confirmation_status"].fillna("NA")
    run_id = (status_collapsed != status_collapsed.shift()).cumsum()
    long_runs = (
        cf.assign(run_id=run_id)
        .groupby("run_id")
        .agg(
            confirmation_status=("confirmation_status", "first"),
            rows=("confirmation_status", "size"),
            start=("signal_timestamp", "min"),
            end=("signal_timestamp", "max"),
            first_direction=("direction", "first"),
            last_direction=("direction", "last"),
            trade_status_mode=(
                "trade_status",
                lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0],
            ),
        )
    )
    long_runs["duration_min"] = (long_runs["end"] - long_runs["start"]).dt.total_seconds().div(60.0).fillna(0.0)
    long_runs = long_runs.sort_values(["rows", "duration_min"], ascending=False).head(20).reset_index(drop=True)

    summary = {
        "analysis_date": "2026-03-23",
        "dataset_path": str(DATASET_PATH),
        "scope": {
            "symbol": "NIFTY",
            "mode": "LIVE",
            "source": "ICICI",
            "time_window": {
                "start": df["signal_timestamp"].min().isoformat() if not df.empty else None,
                "end": df["signal_timestamp"].max().isoformat() if not df.empty else None,
            },
        },
        "raw": _frame_metrics(df),
        "collapsed": _frame_metrics(collapsed),
        "duplicate_timestamp_analysis": {
            "duplicate_timestamp_groups": int(len(duplicate_groups)),
            "share_with_same_confirmation_status": round(float(intra_ts_df["same_confirmation_status"].mean()), 4)
            if not intra_ts_df.empty
            else None,
            "share_with_same_direction": round(float(intra_ts_df["same_direction"].mean()), 4)
            if not intra_ts_df.empty
            else None,
            "share_with_multiple_confirmation_statuses": round(float((~intra_ts_df["same_confirmation_status"]).mean()), 4)
            if not intra_ts_df.empty
            else None,
            "share_with_multiple_directions": round(float((~intra_ts_df["same_direction"]).mean()), 4)
            if not intra_ts_df.empty
            else None,
        },
        "assessment": {},
    }

    sweep_summary, sweep_samples = _sweep_direction_change_penalty(collapsed)

    # Two-parameter decay sweep
    print("Running 2-parameter decay sweep (this may take a moment)...")
    decay_sweep_summary, decay_sweep_samples = _sweep_decay_model(collapsed)

    # 1-parameter veto sweep
    print("Running reversal veto sweep...")
    veto_sweep_summary, veto_sweep_samples = _sweep_reversal_veto(collapsed)

    collapsed_self = summary["collapsed"]["self_transition_rate"]
    collapsed_flip_persist = summary["collapsed"]["strong_or_confirmed_persist_on_direction_flip_ratio"]
    no_dir_share = summary["collapsed"]["status_counts"].get("NO_DIRECTION", 0) / max(summary["collapsed"]["rows"], 1)
    strong_share = summary["collapsed"]["status_counts"].get("STRONG_CONFIRMATION", 0) / max(
        summary["collapsed"]["rows"], 1
    )

    if collapsed_self >= 0.78 or collapsed_flip_persist >= 0.4:
        verdict = "high_stickiness"
    elif collapsed_self >= 0.62 or collapsed_flip_persist >= 0.2:
        verdict = "moderate_stickiness"
    else:
        verdict = "low_stickiness"

    summary["assessment"] = {
        "verdict": verdict,
        "collapsed_self_transition_rate": collapsed_self,
        "collapsed_strong_share": round(strong_share, 4),
        "collapsed_no_direction_share": round(no_dir_share, 4),
        "collapsed_strong_persist_on_direction_flip_ratio": collapsed_flip_persist,
        "headline": (
            "Direction confirmation is structurally sticky: once the engine reaches STRONG_CONFIRMATION it tends to persist, and a non-trivial share of CALL/PUT reversals still retain confirmed/strong confirmation on the next observation."
            if verdict != "low_stickiness"
            else "Direction confirmation does not appear excessively sticky on the captured live dataset."
        ),
    }

    recommended_candidates = sweep_summary[
        (sweep_summary["dataset_match_rate"] >= 0.8)
        & (sweep_summary["strong_or_confirmed_persist_on_direction_flip_ratio"] <= 0.35)
        & (sweep_summary["strong_or_confirmed_share"] >= max(round(strong_share - 0.15, 4), 0.0))
    ].copy()
    recommended_penalty = (
        float(recommended_candidates.iloc[0]["direction_change_penalty"])
        if not recommended_candidates.empty
        else None
    )
    summary["direction_change_penalty_sweep"] = {
        "candidate_penalties": SWEEP_PENALTIES,
        "recommended_penalty": recommended_penalty,
        "recommended_range": [
            float(recommended_candidates["direction_change_penalty"].min()),
            float(recommended_candidates["direction_change_penalty"].max()),
        ]
        if not recommended_candidates.empty
        else None,
        "baseline_recomputed_match_rate": float(
            sweep_summary.loc[sweep_summary["direction_change_penalty"].eq(0.0), "dataset_match_rate"].iloc[0]
        ),
    }

    json_path = OUTPUT_DIR / "direction_confirmation_stickiness_summary.json"
    transition_csv = OUTPUT_DIR / "direction_confirmation_transition_matrix_collapsed.csv"
    flip_csv = OUTPUT_DIR / "direction_confirmation_flip_cases_collapsed.csv"
    long_runs_csv = OUTPUT_DIR / "direction_confirmation_long_runs_collapsed.csv"
    duplicate_csv = OUTPUT_DIR / "direction_confirmation_duplicate_timestamp_stability.csv"
    sweep_csv = OUTPUT_DIR / "direction_change_penalty_sweep.csv"
    sweep_samples_csv = OUTPUT_DIR / "direction_change_penalty_sweep_samples.csv"
    decay_sweep_csv = OUTPUT_DIR / "direction_change_decay_sweep.csv"
    decay_sweep_samples_csv = OUTPUT_DIR / "direction_change_decay_sweep_samples.csv"
    veto_sweep_csv = OUTPUT_DIR / "reversal_veto_sweep.csv"
    veto_sweep_samples_csv = OUTPUT_DIR / "reversal_veto_sweep_samples.csv"
    memo_path = OUTPUT_DIR / "direction_confirmation_stickiness_memo.md"

    json_path.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(summary["collapsed"]["transition_matrix"]).sort_index().to_csv(transition_csv)
    flip_cases.to_csv(flip_csv, index=False)
    long_runs.to_csv(long_runs_csv, index=False)
    intra_ts_df.to_csv(duplicate_csv, index=False)
    sweep_summary.to_csv(sweep_csv, index=False)
    sweep_samples.to_csv(sweep_samples_csv, index=False)
    decay_sweep_summary.to_csv(decay_sweep_csv, index=False)
    decay_sweep_samples.to_csv(decay_sweep_samples_csv, index=False)
    veto_sweep_summary.to_csv(veto_sweep_csv, index=False)
    veto_sweep_samples.to_csv(veto_sweep_samples_csv, index=False)

    contradiction_count = summary['collapsed']['strong_or_confirmed_while_direction_missing_count']

    memo = f"""# Direction Confirmation Stickiness Assessment

## Scope
- Dataset: `{DATASET_PATH}`
- Filter: `symbol=NIFTY`, `mode=LIVE`, `source=ICICI`
- Window: `{summary['scope']['time_window']['start']}` to `{summary['scope']['time_window']['end']}`
- Raw rows: `{summary['raw']['rows']}`
- Timestamp-collapsed rows: `{summary['collapsed']['rows']}`

## Verdict
- Verdict: `{summary['assessment']['verdict']}`
- Headline: {summary['assessment']['headline']}

## Key Metrics (timestamp-collapsed)
- Self-transition rate: `{summary['collapsed']['self_transition_rate']}`
- STRONG_CONFIRMATION share: `{summary['assessment']['collapsed_strong_share']}`
- NO_DIRECTION share: `{summary['assessment']['collapsed_no_direction_share']}`
- Directional CALL/PUT flips: `{summary['collapsed']['directional_flip_count']}`
- Strong/confirmed persistence across CALL/PUT flips: `{summary['collapsed']['strong_or_confirmed_persist_on_direction_flip_count']}` (`{summary['collapsed']['strong_or_confirmed_persist_on_direction_flip_ratio']}` of flips)
- Strong/confirmed while direction missing: `{summary['collapsed']['strong_or_confirmed_while_direction_missing_count']}`

## Interpretation
- The confirmation component is dominated by two terminal-like states: `STRONG_CONFIRMATION` and `NO_DIRECTION`.
- The raw append stream is stickier than the timestamp-collapsed stream, so duplicate captures amplify persistence but do not create it.
- On the collapsed stream, a meaningful fraction of directional reversals still keep confirmation in `CONFIRMED` or `STRONG_CONFIRMATION`, which is the clearest empirical sign of stickiness.
- Rare contradiction cases do exist: `{contradiction_count}` collapsed row(s) show strong/confirmed status with unresolved direction.
- Duration-based persistence can be inflated by overnight or weekend gaps, so row-count persistence and reversal behavior are more reliable than mean duration alone.

## Direction-Change Penalty Sweep
- Baseline recomputed-vs-dataset match rate: `{summary['direction_change_penalty_sweep']['baseline_recomputed_match_rate']}`
- Recommended penalty: `{summary['direction_change_penalty_sweep']['recommended_penalty']}`
- Recommended range: `{summary['direction_change_penalty_sweep']['recommended_range']}`

### Sweep Heuristic
- Keep recomputed behavior close to the captured engine stream (`dataset_match_rate >= 0.80`).
- Push strong/confirmed persistence across direction flips down to `<= 0.35`.
- Avoid collapsing all directional conviction by keeping strong/confirmed share within 15 percentage points of the baseline recomputed share.

## Two-Parameter Decay Model Sweep
- See `{decay_sweep_csv}` for full grid results (base_penalty × decay_steps × decay_factor).
- Sample rows for most informative combos: `{decay_sweep_samples_csv}`
- Interpretation: `post_reversal_strong_or_confirmed_share` measures how many of the N post-reversal snapshots still carry STRONG/CONFIRMED status after applying the decaying penalty.  Lower is better for breaking stickiness.

## Reversal Veto Sweep
- See `{veto_sweep_csv}` for veto_steps sweep results (0-6 steps).
- Sample rows: `{veto_sweep_samples_csv}`
- Interpretation: The veto forces newly-reversed directions to MIXED status for N snapshots (reversal_age ∈ [0, veto_steps)). Measure the fraction of reversal snapshots and post-reversal snapshots actually demoted to MIXED to find the threshold that breaks stickiness.

## Supporting Artifacts
- Transition matrix: `{transition_csv}`
- Direction-flip persistence cases: `{flip_csv}`
- Longest status runs: `{long_runs_csv}`
- Duplicate timestamp stability: `{duplicate_csv}`
- Penalty sweep table: `{sweep_csv}`
- Penalty sweep samples: `{sweep_samples_csv}`
- Decay model sweep table: `{decay_sweep_csv}`
- Decay model sweep samples: `{decay_sweep_samples_csv}`
- Reversal veto sweep table: `{veto_sweep_csv}`
- Reversal veto sweep samples: `{veto_sweep_samples_csv}`
- Structured summary: `{json_path}`
"""
    memo_path.write_text(memo)

    print(
        json.dumps(
            {
                "summary_json": str(json_path),
                "memo_md": str(memo_path),
                "transition_csv": str(transition_csv),
                "flip_csv": str(flip_csv),
                "long_runs_csv": str(long_runs_csv),
                "duplicate_timestamp_csv": str(duplicate_csv),
                "sweep_csv": str(sweep_csv),
                "sweep_samples_csv": str(sweep_samples_csv),
                "decay_sweep_csv": str(decay_sweep_csv),
                "decay_sweep_samples_csv": str(decay_sweep_samples_csv),
                "veto_sweep_csv": str(veto_sweep_csv),
                "veto_sweep_samples_csv": str(veto_sweep_samples_csv),
                "verdict": summary["assessment"]["verdict"],
                "collapsed_self_transition_rate": summary["collapsed"]["self_transition_rate"],
                "collapsed_flip_persist_ratio": summary["collapsed"]["strong_or_confirmed_persist_on_direction_flip_ratio"],
                "recommended_penalty": summary["direction_change_penalty_sweep"]["recommended_penalty"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()