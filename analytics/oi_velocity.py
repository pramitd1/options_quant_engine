"""
Module: oi_velocity.py

Purpose:
    Compute open-interest change velocity and acceleration for near-ATM strikes.

Role in the System:
    Part of the analytics layer.  Provides a rate-of-change view of OI accumulation
    that complements the instantaneous CHG_IN_OI column supplied by the option chain.
    Rapid OI build-up (velocity) or accelerating build (positive acceleration) signals
    fresh directional positioning entering the market rather than stale rolls.

Key Outputs:
    - oi_velocity: dict of { strike: velocity (contracts/snapshot) }
    - oi_acceleration: dict of { strike: acceleration (velocity change/snapshot) }
    - dominant_side: "CALL" | "PUT" | "NEUTRAL" — which side is accumulating fastest
    - velocity_score: float in [-1, +1]; positive = call-side accumulation dominant

Downstream Usage:
    Consumed by market-state assembly and signal engine as a supplementary feature.
    Can also be used in research to replay OI build-up dynamics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def compute_oi_velocity(
    snapshots: List[pd.DataFrame],
    spot: Optional[float] = None,
    strike_window_steps: int = 4,
) -> dict:
    """
    Compute OI change velocity from a rolling list of option-chain snapshots.

    Each snapshot must contain columns: STRIKE_PR (or strikePrice), OPTION_TYP,
    OPEN_INT (or openInterest).

    Parameters
    ----------
    snapshots:
        Ordered list of DataFrames from oldest to newest.  Intent: the last entry
        is the current snapshot.  At minimum 2 snapshots are needed.
    spot:
        Current underlying spot price used to restrict analysis to near-ATM strikes.
    strike_window_steps:
        Number of strike intervals to keep on each side of spot.

    Returns
    -------
    dict with keys:
        oi_velocity          – {strike_str: velocity_float}  (contracts gained per snapshot)
        oi_acceleration      – {strike_str: accel_float}     (change in velocity per snapshot)
        dominant_side        – "CALL" | "PUT" | "NEUTRAL"
        velocity_score       – float in [-1, +1]
        call_velocity_total  – total OI/snapshot on call side
        put_velocity_total   – total OI/snapshot on put side
        snapshot_count       – int
    """
    result: dict = {
        "oi_velocity": {},
        "oi_acceleration": {},
        "dominant_side": "NEUTRAL",
        "velocity_score": 0.0,
        "call_velocity_total": 0.0,
        "put_velocity_total": 0.0,
        "snapshot_count": len(snapshots),
    }

    if len(snapshots) < 2:
        return result

    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "STRIKE_PR" not in out.columns and "strikePrice" in out.columns:
            out["STRIKE_PR"] = out["strikePrice"]
        if "OPEN_INT" not in out.columns and "openInterest" in out.columns:
            out["OPEN_INT"] = out["openInterest"]
        out["STRIKE_PR"] = pd.to_numeric(out.get("STRIKE_PR"), errors="coerce")
        out["OPEN_INT"] = pd.to_numeric(out.get("OPEN_INT"), errors="coerce").fillna(0.0)
        out["OPTION_TYP"] = out.get("OPTION_TYP", pd.Series("", index=out.index))
        return out.dropna(subset=["STRIKE_PR"])

    frames = [_normalise(s) for s in snapshots]

    # Infer strike step and ATM window
    all_strikes = sorted(
        set(float(v) for f in frames for v in f["STRIKE_PR"].dropna().tolist())
    )
    strike_step: float = 50.0  # default for Nifty
    if len(all_strikes) >= 2:
        diffs = [all_strikes[i] - all_strikes[i - 1] for i in range(1, len(all_strikes))]
        pos_diffs = [d for d in diffs if d > 0]
        if pos_diffs:
            strike_step = min(pos_diffs)

    if spot is not None and strike_step > 0:
        half_window = strike_window_steps * strike_step
        lo = float(spot) - half_window
        hi = float(spot) + half_window
        frames = [f[f["STRIKE_PR"].between(lo, hi)] for f in frames]

    # Build per-(strike, side) OI timeseries
    oi_series: Dict[tuple, list] = {}
    for frame in frames:
        for _, row in frame.iterrows():
            key = (float(row["STRIKE_PR"]), str(row["OPTION_TYP"]).strip().upper())
            oi_series.setdefault(key, []).append(float(row["OPEN_INT"]))

    # Velocity = mean OI change per snapshot interval
    velocities: Dict[tuple, float] = {}
    for key, oi_list in oi_series.items():
        if len(oi_list) < 2:
            continue
        changes = [oi_list[i] - oi_list[i - 1] for i in range(1, len(oi_list))]
        velocities[key] = sum(changes) / len(changes)

    # Acceleration = change in velocity (last interval vs previous mean)
    accelerations: Dict[tuple, float] = {}
    for key, oi_list in oi_series.items():
        if len(oi_list) < 3:
            continue
        changes = [oi_list[i] - oi_list[i - 1] for i in range(1, len(oi_list))]
        if len(changes) >= 2:
            recent = changes[-1]
            prev_mean = sum(changes[:-1]) / len(changes[:-1])
            accelerations[key] = recent - prev_mean

    call_total = sum(v for (_, side), v in velocities.items() if side == "CE")
    put_total = sum(v for (_, side), v in velocities.items() if side == "PE")

    # Velocity score in [-1, +1]: positive = calls building faster
    total_abs = abs(call_total) + abs(put_total)
    velocity_score = 0.0
    if total_abs > 0:
        velocity_score = float((call_total - put_total) / total_abs)

    dominant_side = "NEUTRAL"
    if velocity_score > 0.15:
        dominant_side = "CALL"
    elif velocity_score < -0.15:
        dominant_side = "PUT"

    result["oi_velocity"] = {
        f"{k[0]}_{k[1]}": round(v, 1) for k, v in velocities.items()
    }
    result["oi_acceleration"] = {
        f"{k[0]}_{k[1]}": round(v, 1) for k, v in accelerations.items()
    }
    result["dominant_side"] = dominant_side
    result["velocity_score"] = round(velocity_score, 4)
    result["call_velocity_total"] = round(call_total, 1)
    result["put_velocity_total"] = round(put_total, 1)

    return result


def oi_velocity_regime(velocity_score: float) -> str:
    """Map OI velocity score to a human-readable regime label."""
    if velocity_score > 0.40:
        return "STRONG_CALL_BUILDUP"
    if velocity_score > 0.15:
        return "CALL_BUILDUP"
    if velocity_score < -0.40:
        return "STRONG_PUT_BUILDUP"
    if velocity_score < -0.15:
        return "PUT_BUILDUP"
    return "BALANCED"
