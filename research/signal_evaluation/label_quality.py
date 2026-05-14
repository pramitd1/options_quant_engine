"""
Shared helpers for quality-aware realized-label selection.

These utilities keep research reports, calibration checks, and policy
evaluations aligned on which 60-minute labels are safe to trust.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


PRIMARY_HIT_COLUMN = "correct_60m"
PRIMARY_RETURN_COLUMN = "signed_return_60m_bps"
QUALITY_HIT_COLUMN = "calibration_label"
QUALITY_RETURN_COLUMN = "primary_outcome_return_bps"
QUALITY_AVAILABLE_COLUMN = "calibration_label_available"
QUALITY_STATUS_COLUMN = "label_quality_status"
QUALITY_REASONS_COLUMN = "label_quality_reasons"

QUALITY_LABEL_APPROVED_COLUMN = "_quality_label_approved"
QUALITY_LABEL_SOURCE_COLUMN = "_quality_label_source"
QUALITY_RETURN_SOURCE_COLUMN = "_quality_return_source"

QUALITY_LABEL_SOURCE = "calibration_label"
LEGACY_LABEL_SOURCE = "correct_60m_legacy"
MIXED_LABEL_SOURCE = "mixed_calibration_label_and_legacy"

_MISSING_TEXT_TOKENS = {"", "NA", "N/A", "NAN", "NONE", "NULL", "<NA>"}


def _empty_series(frame: pd.DataFrame, dtype: str = "float64") -> pd.Series:
    return pd.Series(index=frame.index, dtype=dtype)


def _nonempty_value_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    text = values.astype("string").str.strip().str.upper()
    return (values.notna() & ~text.isin(_MISSING_TEXT_TOKENS)).astype(bool)


def _truthy_flag(value: Any) -> bool:
    try:
        if value is None or pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        token = value.strip().upper()
        if token in {"TRUE", "T", "YES", "Y", "1", "1.0"}:
            return True
        if token in {"FALSE", "F", "NO", "N", "0", "0.0", ""}:
            return False
    try:
        return bool(int(float(value)))
    except Exception:
        return bool(value)


def quality_annotation_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows that carry explicit label-quality metadata."""
    if frame is None:
        return pd.Series(dtype=bool)
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)

    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in (
        QUALITY_AVAILABLE_COLUMN,
        QUALITY_HIT_COLUMN,
        QUALITY_RETURN_COLUMN,
        QUALITY_STATUS_COLUMN,
        QUALITY_REASONS_COLUMN,
    ):
        mask = mask | _nonempty_value_mask(frame, column)
    return mask.astype(bool)


def has_quality_label_annotations(frame: pd.DataFrame) -> bool:
    """Return True when any row carries explicit label-quality annotations."""
    if frame is None or frame.empty:
        return False
    return bool(quality_annotation_mask(frame).any())


def quality_label_mask(frame: pd.DataFrame, *, fallback_to_legacy: bool = True) -> pd.Series:
    """Return rows with a usable primary calibration label."""
    if frame is None:
        return pd.Series(dtype=bool)
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)

    raw_labels = pd.to_numeric(frame.get(PRIMARY_HIT_COLUMN, _empty_series(frame)), errors="coerce")

    if has_quality_label_annotations(frame):
        annotated = quality_annotation_mask(frame)
        available = (
            frame[QUALITY_AVAILABLE_COLUMN].map(_truthy_flag)
            if QUALITY_AVAILABLE_COLUMN in frame.columns
            else pd.Series(False, index=frame.index, dtype=bool)
        )
        labels = pd.to_numeric(frame.get(QUALITY_HIT_COLUMN, _empty_series(frame)), errors="coerce")
        approved_quality = annotated & available.astype(bool) & labels.notna()
        if fallback_to_legacy:
            approved_legacy = (~annotated) & raw_labels.notna()
            return (approved_quality | approved_legacy).astype(bool)
        return approved_quality.astype(bool)

    if not fallback_to_legacy:
        return pd.Series(False, index=frame.index, dtype=bool)

    return raw_labels.notna().astype(bool)


def apply_quality_label_view(
    frame: pd.DataFrame,
    *,
    hit_column: str = PRIMARY_HIT_COLUMN,
    return_column: str = PRIMARY_RETURN_COLUMN,
    fallback_to_legacy: bool = True,
    drop_unapproved: bool = False,
) -> pd.DataFrame:
    """
    Return a copy whose primary hit/return columns reflect quality-approved labels.

    Old datasets without label-quality columns keep legacy behavior. Newer
    datasets use ``calibration_label`` and blank out rejected labels so means,
    ECE, and hit-rate calculations do not count partial or ambiguous outcomes.
    """
    if frame is None:
        return pd.DataFrame()

    working = frame.copy()
    if working.empty:
        working[QUALITY_LABEL_APPROVED_COLUMN] = pd.Series(dtype=bool)
        working[QUALITY_LABEL_SOURCE_COLUMN] = LEGACY_LABEL_SOURCE
        working[QUALITY_RETURN_SOURCE_COLUMN] = PRIMARY_RETURN_COLUMN
        return working

    active_quality = has_quality_label_annotations(working)
    approved = quality_label_mask(working, fallback_to_legacy=fallback_to_legacy)

    if active_quality:
        annotated = quality_annotation_mask(working)
        labels = pd.to_numeric(working.get(QUALITY_HIT_COLUMN, _empty_series(working)), errors="coerce")
        primary_returns = pd.to_numeric(working.get(QUALITY_RETURN_COLUMN, _empty_series(working)), errors="coerce")
        legacy_returns = pd.to_numeric(working.get(PRIMARY_RETURN_COLUMN, _empty_series(working)), errors="coerce")
        legacy_labels = pd.to_numeric(working.get(PRIMARY_HIT_COLUMN, _empty_series(working)), errors="coerce")
        quality_returns = primary_returns.combine_first(legacy_returns)

        if fallback_to_legacy:
            labels = labels.where(annotated, legacy_labels)
            returns = quality_returns.where(annotated, legacy_returns)
            legacy_used = bool((~annotated & legacy_labels.notna() & approved).any())
            quality_used = bool((annotated & approved).any())
            label_source = MIXED_LABEL_SOURCE if legacy_used and quality_used else (
                LEGACY_LABEL_SOURCE if legacy_used else QUALITY_LABEL_SOURCE
            )
            return_source = MIXED_LABEL_SOURCE if legacy_used and quality_used else (
                PRIMARY_RETURN_COLUMN if legacy_used else QUALITY_RETURN_COLUMN
            )
            source_series = pd.Series(QUALITY_LABEL_SOURCE, index=working.index, dtype="object").where(
                annotated,
                LEGACY_LABEL_SOURCE,
            )
            return_source_series = pd.Series(QUALITY_RETURN_COLUMN, index=working.index, dtype="object").where(
                annotated,
                PRIMARY_RETURN_COLUMN,
            )
        else:
            returns = quality_returns
            label_source = QUALITY_LABEL_SOURCE
            return_source = QUALITY_RETURN_COLUMN
            source_series = pd.Series(QUALITY_LABEL_SOURCE, index=working.index, dtype="object")
            return_source_series = pd.Series(QUALITY_RETURN_COLUMN, index=working.index, dtype="object")
    else:
        labels = pd.to_numeric(working.get(PRIMARY_HIT_COLUMN, _empty_series(working)), errors="coerce")
        returns = pd.to_numeric(working.get(PRIMARY_RETURN_COLUMN, _empty_series(working)), errors="coerce")
        label_source = LEGACY_LABEL_SOURCE
        return_source = PRIMARY_RETURN_COLUMN
        source_series = pd.Series(LEGACY_LABEL_SOURCE, index=working.index, dtype="object")
        return_source_series = pd.Series(PRIMARY_RETURN_COLUMN, index=working.index, dtype="object")

    working[hit_column] = labels.where(approved)
    working[return_column] = returns.where(approved)
    working[QUALITY_LABEL_APPROVED_COLUMN] = approved
    working[QUALITY_LABEL_SOURCE_COLUMN] = source_series
    working[QUALITY_RETURN_SOURCE_COLUMN] = return_source_series
    working.attrs["quality_label_source"] = label_source
    working.attrs["quality_return_source"] = return_source

    if drop_unapproved:
        working = working.loc[approved].copy()
    return working


def select_quality_labeled_rows(frame: pd.DataFrame, *, fallback_to_legacy: bool = True) -> pd.DataFrame:
    """Return only rows with a usable primary calibration label."""
    if frame is None:
        return pd.DataFrame()
    if has_quality_label_annotations(frame) or fallback_to_legacy:
        return apply_quality_label_view(
            frame,
            fallback_to_legacy=fallback_to_legacy,
            drop_unapproved=True,
        )
    return frame.iloc[0:0].copy()


def label_quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize raw label coverage versus quality-approved label coverage."""
    if frame is None:
        frame = pd.DataFrame()

    raw_labels = pd.to_numeric(frame.get(PRIMARY_HIT_COLUMN, _empty_series(frame)), errors="coerce")
    quality_active = has_quality_label_annotations(frame)
    approved = quality_label_mask(frame, fallback_to_legacy=True)
    if quality_active:
        annotated = quality_annotation_mask(frame)
        legacy_used = bool((~annotated & raw_labels.notna() & approved).any())
        quality_used = bool((annotated & approved).any())
        source = MIXED_LABEL_SOURCE if legacy_used and quality_used else (
            LEGACY_LABEL_SOURCE if legacy_used else QUALITY_LABEL_SOURCE
        )
    else:
        source = LEGACY_LABEL_SOURCE

    status_counts: dict[str, int] = {}
    if QUALITY_STATUS_COLUMN in frame.columns:
        counts = frame[QUALITY_STATUS_COLUMN].fillna("UNKNOWN").astype(str).str.upper().value_counts()
        status_counts = {str(key): int(value) for key, value in counts.items()}

    reason_counter: Counter[str] = Counter()
    if QUALITY_REASONS_COLUMN in frame.columns:
        for value in frame[QUALITY_REASONS_COLUMN].dropna().astype(str):
            for reason in value.split("|"):
                reason = reason.strip()
                if reason:
                    reason_counter[reason] += 1

    total = int(len(frame))
    raw_count = int(raw_labels.notna().sum())
    quality_count = int(approved.sum())

    return {
        "label_source": source,
        "total_rows": total,
        "raw_labeled_rows": raw_count,
        "quality_labeled_rows": quality_count,
        "excluded_labeled_rows": max(raw_count - quality_count, 0) if quality_active else 0,
        "raw_label_coverage_ratio": round(raw_count / max(total, 1), 4),
        "quality_label_coverage_ratio": round(quality_count / max(total, 1), 4),
        "quality_annotations_active": bool(quality_active),
        "label_quality_status_counts": status_counts,
        "label_quality_reason_counts": dict(reason_counter.most_common(20)),
    }
