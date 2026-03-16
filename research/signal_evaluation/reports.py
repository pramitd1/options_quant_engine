"""
Module: reports.py

Purpose:
    Implement reports utilities for signal evaluation, reporting, or research diagnostics.

Role in the System:
    Part of the research layer that records signal-evaluation datasets and diagnostic reports.

Key Outputs:
    Signal-evaluation datasets, reports, and comparison artifacts.

Downstream Usage:
    Consumed by tuning, governance reviews, and post-trade analysis.
"""

from __future__ import annotations

import pandas as pd


RETURN_HORIZON_FIELDS = [
    "realized_return_5m",
    "realized_return_15m",
    "realized_return_30m",
    "realized_return_60m",
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Purpose:
        Safely normalize numeric while preserving fallback behavior.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        series (pd.Series): Input associated with series.
    
    Returns:
        pd.Series: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return pd.to_numeric(series, errors="coerce")


def _signal_hit_flag(frame: pd.DataFrame) -> pd.Series:
    """
    Purpose:
        Process signal hit flag for downstream use.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.Series: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    spot_at_signal = _safe_numeric(df.get("spot_at_signal", pd.Series(index=df.index)))
    spot_next_close = _safe_numeric(df.get("spot_next_close", pd.Series(index=df.index)))
    spot_close_same_day = _safe_numeric(df.get("spot_close_same_day", pd.Series(index=df.index)))
    spot_60m = _safe_numeric(df.get("spot_60m", pd.Series(index=df.index)))
    direction = df.get("direction", pd.Series(index=df.index)).astype(str).str.upper().str.strip()

    reference_spot = spot_next_close.copy()
    reference_spot = reference_spot.fillna(spot_close_same_day)
    reference_spot = reference_spot.fillna(spot_60m)

    signal_move = reference_spot - spot_at_signal
    hit_flag = pd.Series(pd.NA, index=df.index, dtype="object")
    hit_flag = hit_flag.mask(direction.eq("CALL") & signal_move.gt(0), 1)
    hit_flag = hit_flag.mask(direction.eq("CALL") & signal_move.le(0), 0)
    hit_flag = hit_flag.mask(direction.eq("PUT") & signal_move.lt(0), 1)
    hit_flag = hit_flag.mask(direction.eq("PUT") & signal_move.ge(0), 0)
    return pd.to_numeric(hit_flag, errors="coerce")


def _group_hit_rate(frame: pd.DataFrame, group_field: str) -> pd.DataFrame:
    """
    Purpose:
        Group records by hit rate.
    
    Context:
        Internal helper within the research layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        group_field (str): Input associated with group field.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    if group_field not in df.columns:
        return pd.DataFrame(columns=[group_field, "signal_count", "hit_rate"])

    df["hit_flag"] = _signal_hit_flag(df)
    grouped = (
        df.dropna(subset=[group_field])
        .groupby(group_field, dropna=False)
        .agg(
            signal_count=("signal_id", "count"),
            hit_rate=("hit_flag", "mean"),
        )
        .reset_index()
    )
    grouped["hit_rate"] = grouped["hit_rate"].round(4)
    return grouped.sort_values("signal_count", ascending=False).reset_index(drop=True)


def hit_rate_by_trade_strength(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Process hit rate by trade strength for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    group_field = "signal_calibration_bucket" if "signal_calibration_bucket" in frame.columns else "trade_strength"
    return _group_hit_rate(frame, group_field)


def hit_rate_by_macro_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Process hit rate by macro regime for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return _group_hit_rate(frame, "macro_regime")


def average_score_by_signal_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Compute the average value for score by signal quality.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    required = [
        "signal_quality",
        "direction_score",
        "magnitude_score",
        "timing_score",
        "tradeability_score",
        "composite_signal_score",
    ]
    if any(column not in df.columns for column in required):
        return pd.DataFrame(columns=["signal_quality", "signal_count"])

    for column in required[1:]:
        df[column] = _safe_numeric(df[column])

    grouped = (
        df.dropna(subset=["signal_quality"])
        .groupby("signal_quality", dropna=False)
        .agg(
            signal_count=("signal_id", "count"),
            avg_direction_score=("direction_score", "mean"),
            avg_magnitude_score=("magnitude_score", "mean"),
            avg_timing_score=("timing_score", "mean"),
            avg_tradeability_score=("tradeability_score", "mean"),
            avg_composite_signal_score=("composite_signal_score", "mean"),
        )
        .reset_index()
    )
    numeric_cols = [column for column in grouped.columns if column.startswith("avg_")]
    grouped[numeric_cols] = grouped[numeric_cols].round(2)
    return grouped.sort_values("signal_count", ascending=False).reset_index(drop=True)


def average_realized_return_by_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Compute the average value for realized return by horizon.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    available_fields = [field for field in RETURN_HORIZON_FIELDS if field in df.columns]
    if not available_fields:
        return pd.DataFrame(columns=["horizon", "avg_realized_return"])

    rows = []
    for field in available_fields:
        series = _safe_numeric(df[field])
        rows.append(
            {
                "horizon": field.replace("realized_return_", ""),
                "avg_realized_return": round(series.mean(), 6),
                "sample_count": int(series.notna().sum()),
            }
        )

    return pd.DataFrame(rows)


def signal_count_by_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Process signal count by regime for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    rows = []
    for regime_field in ["signal_regime", "macro_regime", "gamma_regime", "regime_fingerprint_id"]:
        if regime_field not in df.columns:
            continue
        grouped = (
            df.dropna(subset=[regime_field])
            .groupby(regime_field, dropna=False)
            .agg(signal_count=("signal_id", "count"))
            .reset_index()
        )
        grouped["regime_field"] = regime_field
        grouped = grouped.rename(columns={regime_field: "regime_value"})
        rows.append(grouped[["regime_field", "regime_value", "signal_count"]])

    if not rows:
        return pd.DataFrame(columns=["regime_field", "regime_value", "signal_count"])
    return pd.concat(rows, ignore_index=True).sort_values(["regime_field", "signal_count"], ascending=[True, False]).reset_index(drop=True)


def regime_fingerprint_performance(frame: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Purpose:
        Process regime fingerprint performance for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        top_n (int): Input associated with top n.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    required = ["regime_fingerprint_id", "regime_fingerprint", "composite_signal_score"]
    if any(column not in df.columns for column in required):
        return pd.DataFrame(columns=["regime_fingerprint_id", "regime_fingerprint", "signal_count", "avg_composite_signal_score", "hit_rate"])

    df["hit_flag"] = _signal_hit_flag(df)
    df["composite_signal_score"] = _safe_numeric(df["composite_signal_score"])
    grouped = (
        df.dropna(subset=["regime_fingerprint_id"])
        .groupby(["regime_fingerprint_id", "regime_fingerprint"], dropna=False)
        .agg(
            signal_count=("signal_id", "count"),
            avg_composite_signal_score=("composite_signal_score", "mean"),
            hit_rate=("hit_flag", "mean"),
        )
        .reset_index()
    )
    grouped["avg_composite_signal_score"] = grouped["avg_composite_signal_score"].round(2)
    grouped["hit_rate"] = grouped["hit_rate"].round(4)
    return grouped.sort_values(["avg_composite_signal_score", "signal_count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def move_probability_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Process move probability calibration for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    df = frame.copy()
    if "move_probability" not in df.columns:
        return pd.DataFrame(columns=["probability_calibration_bucket", "signal_count", "avg_move_probability", "actual_hit_rate"])

    df["move_probability"] = _safe_numeric(df["move_probability"])
    df["hit_flag"] = _signal_hit_flag(df)
    group_field = "probability_calibration_bucket" if "probability_calibration_bucket" in df.columns else "move_probability"

    grouped = (
        df.dropna(subset=[group_field])
        .groupby(group_field, dropna=False)
        .agg(
            signal_count=("signal_id", "count"),
            avg_move_probability=("move_probability", "mean"),
            actual_hit_rate=("hit_flag", "mean"),
            avg_composite_signal_score=("composite_signal_score", "mean"),
        )
        .reset_index()
    )
    grouped["avg_move_probability"] = grouped["avg_move_probability"].round(4)
    grouped["actual_hit_rate"] = grouped["actual_hit_rate"].round(4)
    grouped["avg_composite_signal_score"] = grouped["avg_composite_signal_score"].round(2)
    return grouped.sort_values("signal_count", ascending=False).reset_index(drop=True)


def build_research_report(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Purpose:
        Build the research report used by downstream components.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
    
    Returns:
        dict[str, pd.DataFrame]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return {
        "hit_rate_by_trade_strength": hit_rate_by_trade_strength(frame),
        "hit_rate_by_macro_regime": hit_rate_by_macro_regime(frame),
        "average_score_by_signal_quality": average_score_by_signal_quality(frame),
        "average_realized_return_by_horizon": average_realized_return_by_horizon(frame),
        "signal_count_by_regime": signal_count_by_regime(frame),
        "move_probability_calibration": move_probability_calibration(frame),
        "regime_fingerprint_performance": regime_fingerprint_performance(frame),
    }
