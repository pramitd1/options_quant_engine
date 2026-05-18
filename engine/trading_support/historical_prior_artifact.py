"""
Module: historical_prior_artifact.py

Purpose:
    Load the versioned historical-prior artifact consumed by the live engine.

Role in the System:
    Keeps historical research evidence outside the hot scoring code while
    providing a deterministic fallback when the generated artifact is absent.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_PRIOR_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "research"
    / "ml_research"
    / "historical_priors"
    / "latest_historical_prior_artifact.json"
)


DEFAULT_HISTORICAL_PRIOR_ARTIFACT = {
    "artifact_version": "historical_prior_artifact_v1",
    "source": "code_fallback",
    "bucket_thresholds": {
        "pcr_oi": [
            {"label": "low", "upper": 0.8191},
            {"label": "q2", "upper": 1.0267},
            {"label": "q3", "upper": 1.2370},
            {"label": "q4", "upper": 1.2880},
            {"label": "high", "upper": None},
        ],
        "india_vix": [
            {"label": "low", "upper": 13.61},
            {"label": "q2", "upper": 15.84},
            {"label": "q3", "upper": 18.69},
            {"label": "q4", "upper": 24.06},
            {"label": "high", "upper": None},
        ],
    },
    "interactions": {
        "expiry_x_pcr": {
            "target": "fwd_ret_1d_bps",
            "key_fields": ["expiry_bucket", "pcr_oi_bucket"],
            "rows": {
                "2-3d|high": {"n": 122, "mean_bps": 17.84, "hit_up": 0.6475, "abs_mean_bps": 54.79},
                "0-1d|high": {"n": 165, "mean_bps": 12.61, "hit_up": 0.6242, "abs_mean_bps": 63.43},
                "4-7d|low": {"n": 152, "mean_bps": -21.30, "hit_up": 0.5197, "abs_mean_bps": 93.74},
                "0-1d|low": {"n": 285, "mean_bps": -3.17, "hit_up": 0.4632, "abs_mean_bps": 87.25},
            },
        },
        "india_vix_x_trend": {
            "target": "fwd_ret_1d_bps",
            "key_fields": ["india_vix_bucket", "trend_20d_bucket"],
            "rows": {
                "q4|selloff": {"n": 123, "mean_bps": 21.60, "hit_up": 0.5447, "abs_mean_bps": 103.19},
                "q3|selloff": {"n": 43, "mean_bps": 21.04, "hit_up": 0.6977, "abs_mean_bps": 92.47},
                "high|weak": {"n": 132, "mean_bps": 16.07, "hit_up": 0.5227, "abs_mean_bps": 146.55},
                "q3|weak": {"n": 196, "mean_bps": -12.70, "hit_up": 0.4490, "abs_mean_bps": 83.91},
            },
        },
        "weekday_x_vix": {
            "target": "next_day_range_bps",
            "key_fields": ["weekday", "india_vix_bucket"],
            "rows": {
                "Friday|high": {"n": 173, "mean_bps": 281.41, "hit_up": 1.0, "abs_mean_bps": 281.41},
                "Monday|low": {"n": 160, "mean_bps": 74.03, "hit_up": 1.0, "abs_mean_bps": 74.03},
                "Tuesday|low": {"n": 170, "mean_bps": 73.04, "hit_up": 1.0, "abs_mean_bps": 73.04},
            },
        },
    },
}


def _coerce_artifact(payload):
    if not isinstance(payload, dict):
        return DEFAULT_HISTORICAL_PRIOR_ARTIFACT
    if not isinstance(payload.get("interactions"), dict):
        return DEFAULT_HISTORICAL_PRIOR_ARTIFACT
    return payload


@lru_cache(maxsize=4)
def load_historical_prior_artifact(path: str | Path | None = None) -> dict:
    """Load the generated historical-prior artifact, falling back safely."""

    artifact_path = Path(path) if path is not None else DEFAULT_HISTORICAL_PRIOR_ARTIFACT_PATH
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_HISTORICAL_PRIOR_ARTIFACT
    return _coerce_artifact(payload)


def clear_historical_prior_artifact_cache() -> None:
    """Clear the artifact cache after regenerating priors in the same process."""

    load_historical_prior_artifact.cache_clear()
