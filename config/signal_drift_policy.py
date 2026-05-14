"""
Signal drift monitor policy configuration.

These defaults govern research/ops warnings only.  They do not gate live
signals, change data sources, or alter execution behavior.
"""

from __future__ import annotations

from typing import Any


SIGNAL_DRIFT_MONITOR_POLICY: dict[str, Any] = {
    "recent_days": 20,
    "baseline_days": 120,
    "min_recent_labeled": 25,
    "min_baseline_labeled": 75,
    "top_n": 20,
    "apply_missing_policies": True,
    "hit_rate_drop_warn": 0.08,
    "return_drop_bps_warn": 15.0,
    "calibration_gap_delta_warn": 0.08,
    "label_coverage_drop_warn": 0.15,
    "retention_delta_warn": 0.20,
    "dimensions": [
        "source",
        "mode",
        "gamma_regime",
        "volatility_regime",
        "macro_regime",
        "global_risk_state",
    ],
    "probability_fields": [
        "hybrid_move_probability",
        "ml_confidence_score",
    ],
    "score_fields": [
        "ml_rank_score",
        "ml_confidence_score",
        "hybrid_move_probability",
        "trade_strength",
        "composite_signal_score",
        "tradeability_score",
    ],
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def get_signal_drift_monitor_policy(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved drift monitor policy bundle."""
    from config.policy_resolver import resolve_mapping

    resolved = resolve_mapping("signal_drift.monitor", SIGNAL_DRIFT_MONITOR_POLICY)
    if overrides:
        resolved.update({key: value for key, value in overrides.items() if value is not None})

    normalized = dict(resolved)
    for key in ("recent_days", "min_recent_labeled", "min_baseline_labeled", "top_n"):
        normalized[key] = int(normalized[key])

    baseline_days = normalized.get("baseline_days")
    normalized["baseline_days"] = None if baseline_days is None else int(baseline_days)
    normalized["apply_missing_policies"] = _as_bool(normalized.get("apply_missing_policies", True))

    for key in (
        "hit_rate_drop_warn",
        "return_drop_bps_warn",
        "calibration_gap_delta_warn",
        "label_coverage_drop_warn",
        "retention_delta_warn",
    ):
        normalized[key] = float(normalized[key])

    for key in ("dimensions", "probability_fields", "score_fields"):
        normalized[key] = _as_list(normalized.get(key))

    return normalized
