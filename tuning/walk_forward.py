"""
Module: walk_forward.py

Purpose:
    Implement walk forward utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tuning.models import WalkForwardSplit


DEFAULT_WALK_FORWARD_CONFIG = {
    "split_type": "anchored",
    "train_window_days": 60,
    "validation_window_days": 20,
    "step_size_days": 20,
    "minimum_train_rows": 20,
    "minimum_validation_rows": 10,
}


@dataclass(frozen=True)
class SplitFrames:
    """
    Purpose:
        Represent SplitFrames within the repository.
    
    Context:
        Used within the `walk forward` module. The class standardizes records that move through search, validation, shadow mode, and promotion.
    
    Attributes:
        train (pd.DataFrame): DataFrame containing train.
        validation (pd.DataFrame): DataFrame containing validation.
    
    Notes:
        The record is immutable so tuning artifacts can be compared, persisted, and promoted without accidental mutation.
    """
    train: pd.DataFrame
    validation: pd.DataFrame


def _prepare_frame(frame: pd.DataFrame, timestamp_col: str = "signal_timestamp") -> pd.DataFrame:
    """
    Purpose:
        Process prepare frame for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        timestamp_col (str): Input associated with timestamp col.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[timestamp_col])

    ordered = frame.copy()
    ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], errors="coerce")
    ordered = ordered.dropna(subset=[timestamp_col])
    return ordered.sort_values(timestamp_col, kind="stable").reset_index(drop=True)


def build_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    split_type: str = "anchored",
    train_window_days: int = 60,
    validation_window_days: int = 20,
    step_size_days: int | None = None,
    minimum_train_rows: int = 20,
    minimum_validation_rows: int = 10,
    timestamp_col: str = "signal_timestamp",
) -> list[WalkForwardSplit]:
    """
    Purpose:
        Build the walk forward splits used by downstream components.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        split_type (str): Input associated with split type.
        train_window_days (int): Input associated with train window days.
        validation_window_days (int): Input associated with validation window days.
        step_size_days (int | None): Input associated with step size days.
        minimum_train_rows (int): Input associated with minimum train rows.
        minimum_validation_rows (int): Input associated with minimum validation rows.
        timestamp_col (str): Input associated with timestamp col.
    
    Returns:
        list[WalkForwardSplit]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    ordered = _prepare_frame(frame, timestamp_col=timestamp_col)
    if ordered.empty:
        return []

    split_type = str(split_type or "anchored").lower().strip()
    if split_type not in {"anchored", "rolling"}:
        raise ValueError("split_type must be 'anchored' or 'rolling'")

    train_window = pd.Timedelta(days=max(int(train_window_days), 1))
    validation_window = pd.Timedelta(days=max(int(validation_window_days), 1))
    step_size = pd.Timedelta(days=max(int(step_size_days or validation_window_days), 1))

    min_ts = ordered[timestamp_col].min()
    max_ts = ordered[timestamp_col].max()
    cursor = min_ts
    split_index = 0
    splits: list[WalkForwardSplit] = []

    while True:
        train_start = min_ts if split_type == "anchored" else cursor
        train_end = (cursor + train_window) if split_type == "anchored" else (train_start + train_window)
        validation_start = train_end
        validation_end = validation_start + validation_window

        if validation_start > max_ts:
            break

        train_mask = (ordered[timestamp_col] >= train_start) & (ordered[timestamp_col] < train_end)
        validation_mask = (ordered[timestamp_col] >= validation_start) & (ordered[timestamp_col] < validation_end)
        train_frame = ordered.loc[train_mask]
        validation_frame = ordered.loc[validation_mask]

        if len(train_frame) >= minimum_train_rows and len(validation_frame) >= minimum_validation_rows:
            splits.append(
                WalkForwardSplit(
                    split_id=f"{split_type}_{split_index:03d}",
                    split_type=split_type,
                    train_start=train_frame[timestamp_col].min().isoformat() if not train_frame.empty else None,
                    train_end=train_frame[timestamp_col].max().isoformat() if not train_frame.empty else None,
                    validation_start=validation_frame[timestamp_col].min().isoformat() if not validation_frame.empty else None,
                    validation_end=validation_frame[timestamp_col].max().isoformat() if not validation_frame.empty else None,
                    train_count=int(len(train_frame)),
                    validation_count=int(len(validation_frame)),
                )
            )
            split_index += 1

        cursor = cursor + step_size
        if cursor > max_ts:
            break

    return splits


def build_walk_forward_splits_with_fallback(
    frame: pd.DataFrame,
    *,
    split_type: str,
    train_window_days: int,
    validation_window_days: int,
    step_size_days: int,
    minimum_train_rows: int,
    minimum_validation_rows: int,
    timestamp_col: str = "signal_timestamp",
) -> tuple[list[WalkForwardSplit], dict[str, Any]]:
    """
    Build walk-forward splits and progressively relax configuration when no split is found.
    """
    candidate_configs: list[dict[str, Any]] = [
        {
            "used_fallback": False,
            "effective_train_window_days": int(train_window_days),
            "effective_validation_window_days": int(validation_window_days),
            "effective_step_size_days": int(step_size_days),
            "effective_minimum_train_rows": int(minimum_train_rows),
            "effective_minimum_validation_rows": int(minimum_validation_rows),
        }
    ]

    signal_ts = frame[timestamp_col] if timestamp_col in frame.columns else pd.Series(dtype=object)
    ordered_ts = pd.to_datetime(signal_ts, errors="coerce").dropna()
    if ordered_ts.empty:
        return [], {
            "used_fallback": True,
            "reason": "no_valid_signal_timestamp",
        }

    span_days = max((ordered_ts.max() - ordered_ts.min()).total_seconds() / 86400.0, 1.0)
    fallback_train_window_days = max(1, int(round(min(span_days * 0.5, 30.0))))
    fallback_validation_window_days = max(1, int(round(min(span_days * 0.2, 10.0))))
    fallback_step_size_days = max(1, min(fallback_validation_window_days, int(step_size_days)))

    candidate_configs.extend(
        [
            {
                "used_fallback": True,
                "effective_train_window_days": int(fallback_train_window_days),
                "effective_validation_window_days": int(fallback_validation_window_days),
                "effective_step_size_days": int(fallback_step_size_days),
                "effective_minimum_train_rows": max(20, min(int(minimum_train_rows), 25)),
                "effective_minimum_validation_rows": max(10, min(int(minimum_validation_rows), 10)),
            },
            {
                "used_fallback": True,
                "effective_train_window_days": max(1, min(int(fallback_train_window_days), 2)),
                "effective_validation_window_days": 1,
                "effective_step_size_days": 1,
                "effective_minimum_train_rows": 20,
                "effective_minimum_validation_rows": 10,
            },
            {
                "used_fallback": True,
                "effective_train_window_days": 1,
                "effective_validation_window_days": 1,
                "effective_step_size_days": 1,
                "effective_minimum_train_rows": 10,
                "effective_minimum_validation_rows": 5,
            },
        ]
    )

    for config in candidate_configs:
        splits = build_walk_forward_splits(
            frame,
            split_type=split_type,
            train_window_days=config["effective_train_window_days"],
            validation_window_days=config["effective_validation_window_days"],
            step_size_days=config["effective_step_size_days"],
            minimum_train_rows=config["effective_minimum_train_rows"],
            minimum_validation_rows=config["effective_minimum_validation_rows"],
            timestamp_col=timestamp_col,
        )
        if splits:
            return splits, config

    return [], {
        "used_fallback": True,
        "reason": "no_valid_split_configuration_found",
    }


def apply_walk_forward_split(
    frame: pd.DataFrame,
    split: WalkForwardSplit | dict[str, Any],
    *,
    timestamp_col: str = "signal_timestamp",
) -> SplitFrames:
    """
    Purpose:
        Process apply walk forward split for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        frame (pd.DataFrame): Input associated with frame.
        split (WalkForwardSplit | dict[str, Any]): Input associated with split.
        timestamp_col (str): Input associated with timestamp col.
    
    Returns:
        SplitFrames: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    ordered = _prepare_frame(frame, timestamp_col=timestamp_col)
    split_payload = split.to_dict() if hasattr(split, "to_dict") else dict(split or {})
    if ordered.empty:
        return SplitFrames(train=ordered.copy(), validation=ordered.copy())

    train_start = pd.to_datetime(split_payload.get("train_start"), errors="coerce")
    train_end = pd.to_datetime(split_payload.get("train_end"), errors="coerce")
    validation_start = pd.to_datetime(split_payload.get("validation_start"), errors="coerce")
    validation_end = pd.to_datetime(split_payload.get("validation_end"), errors="coerce")

    train_mask = pd.Series(True, index=ordered.index)
    validation_mask = pd.Series(True, index=ordered.index)

    if pd.notna(train_start):
        train_mask &= ordered[timestamp_col] >= train_start
    if pd.notna(train_end):
        train_mask &= ordered[timestamp_col] <= train_end
    if pd.notna(validation_start):
        validation_mask &= ordered[timestamp_col] >= validation_start
    if pd.notna(validation_end):
        validation_mask &= ordered[timestamp_col] <= validation_end

    return SplitFrames(
        train=ordered.loc[train_mask].copy().reset_index(drop=True),
        validation=ordered.loc[validation_mask].copy().reset_index(drop=True),
    )
