"""Confidence intervals and sample-size guardrails for signal evaluation."""

from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd


DEFAULT_MIN_RELIABLE_SAMPLE = 30
DEFAULT_STRONG_SAMPLE = 100
_Z_95 = 1.959963984540054


def _round_or_none(value: Any, digits: int) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def sample_quality(
    sample_count: int,
    *,
    min_sample: int = DEFAULT_MIN_RELIABLE_SAMPLE,
    strong_sample: int = DEFAULT_STRONG_SAMPLE,
) -> str:
    """Classify whether a metric has enough observations to trust."""
    count = int(sample_count or 0)
    min_sample = max(int(min_sample), 1)
    strong_sample = max(int(strong_sample), min_sample + 1)
    if count <= 0:
        return "NO_EVIDENCE"
    if count < min_sample:
        return "INSUFFICIENT_EVIDENCE"
    if count < strong_sample:
        return "LOW_CONFIDENCE"
    return "RELIABLE"


def sample_guardrail(
    sample_count: int,
    *,
    min_sample: int = DEFAULT_MIN_RELIABLE_SAMPLE,
    strong_sample: int = DEFAULT_STRONG_SAMPLE,
) -> dict[str, Any]:
    """Return JSON-friendly sample quality metadata."""
    return {
        "sample_quality": sample_quality(sample_count, min_sample=min_sample, strong_sample=strong_sample),
        "min_reliable_sample": int(max(min_sample, 1)),
        "strong_sample": int(max(strong_sample, max(min_sample, 1) + 1)),
    }


def wilson_interval(
    successes: int | float,
    total: int,
    *,
    z: float = _Z_95,
    digits: int = 4,
) -> tuple[float | None, float | None]:
    """Return a Wilson score interval for a binomial hit rate."""
    n = int(total or 0)
    if n <= 0:
        return None, None
    p_hat = max(min(float(successes) / n, 1.0), 0.0)
    denominator = 1.0 + (z * z / n)
    centre = p_hat + (z * z / (2.0 * n))
    margin = z * sqrt((p_hat * (1.0 - p_hat) / n) + (z * z / (4.0 * n * n)))
    low = (centre - margin) / denominator
    high = (centre + margin) / denominator
    return _round_or_none(max(low, 0.0), digits), _round_or_none(min(high, 1.0), digits)


def mean_confidence_interval(
    values: pd.Series | list[Any],
    *,
    z: float = _Z_95,
    digits: int = 4,
) -> tuple[float | None, float | None]:
    """Return a normal-approximation interval for a sample mean."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = int(series.count())
    if n < 2:
        return None, None
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    margin = z * std / sqrt(n)
    return _round_or_none(mean - margin, digits), _round_or_none(mean + margin, digits)


def confidence_intervals_overlap(
    left_low: Any,
    left_high: Any,
    right_low: Any,
    right_high: Any,
) -> bool | None:
    """Return whether two intervals overlap, or None when either is unavailable."""
    values = [left_low, left_high, right_low, right_high]
    if any(value is None or pd.isna(value) for value in values):
        return None
    return not (float(left_high) < float(right_low) or float(right_high) < float(left_low))


def outcome_confidence_fields(
    hit_labels: pd.Series | list[Any] | None,
    signed_returns_bps: pd.Series | list[Any] | None,
    *,
    sample_count: int | None = None,
    min_sample: int = DEFAULT_MIN_RELIABLE_SAMPLE,
    strong_sample: int = DEFAULT_STRONG_SAMPLE,
) -> dict[str, Any]:
    """Build confidence metadata for hit-rate and signed-return metrics."""
    hit_series = (
        pd.to_numeric(pd.Series(hit_labels), errors="coerce").dropna()
        if hit_labels is not None
        else pd.Series(dtype=float)
    )
    return_series = (
        pd.to_numeric(pd.Series(signed_returns_bps), errors="coerce").dropna()
        if signed_returns_bps is not None
        else pd.Series(dtype=float)
    )
    hit_n = int(hit_series.count())
    return_n = int(return_series.count())
    evidence_n = int(sample_count) if sample_count is not None else max(hit_n, return_n)
    successes = float(hit_series.clip(lower=0.0, upper=1.0).sum()) if hit_n else 0.0
    hit_low, hit_high = wilson_interval(successes, hit_n)
    ret_low, ret_high = mean_confidence_interval(return_series)
    fields = {
        "hit_rate_label_count": hit_n,
        "return_label_count": return_n,
        "hit_rate_ci_low": hit_low,
        "hit_rate_ci_high": hit_high,
        "return_ci_low_bps": ret_low,
        "return_ci_high_bps": ret_high,
    }
    fields.update(sample_guardrail(evidence_n, min_sample=min_sample, strong_sample=strong_sample))
    return fields
