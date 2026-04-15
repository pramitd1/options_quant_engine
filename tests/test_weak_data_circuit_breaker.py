from __future__ import annotations

from config.signal_policy import get_trade_runtime_thresholds
from engine.signal_engine import (
    _apply_bearish_bias_threshold_adjustments,
    _evaluate_weak_data_circuit_breaker,
)


def _runtime_thresholds(**overrides):
    base = {
        "enable_weak_data_circuit_breaker": 1,
        "weak_data_circuit_breaker_data_quality_statuses": ["WEAK", "CAUTION"],
        "weak_data_circuit_breaker_provider_statuses": ["WEAK", "CAUTION"],
        "weak_data_circuit_breaker_require_strong_confirmation": 1,
        "weak_data_circuit_breaker_min_trade_strength": 74,
        "weak_data_circuit_breaker_min_runtime_composite_score": 70,
        "weak_data_circuit_breaker_max_proxy_ratio": 0.35,
        "weak_data_circuit_breaker_min_trigger_count": 2,
        "enable_bearish_bias_guard": 1,
        "bearish_bias_guard_composite_add": 3,
        "bearish_bias_guard_strength_add": 2,
        "bearish_bias_guard_size_cap": 0.70,
    }
    base.update(overrides)
    return base


def test_bearish_bias_guard_applies_in_put_toxic_vol_expansion_context():
    strength, composite, details = _apply_bearish_bias_threshold_adjustments(
        runtime_thresholds=_runtime_thresholds(),
        direction="PUT",
        gamma_regime="NEGATIVE_GAMMA",
        vol_regime="VOL_EXPANSION",
        base_min_trade_strength=62,
        base_min_composite_score=58,
    )

    assert details["applied"] is True
    assert strength == 64
    assert composite == 61
    assert details["size_cap"] == 0.7


def test_bearish_bias_guard_not_applied_outside_context():
    strength, composite, details = _apply_bearish_bias_threshold_adjustments(
        runtime_thresholds=_runtime_thresholds(),
        direction="CALL",
        gamma_regime="NEGATIVE_GAMMA",
        vol_regime="VOL_EXPANSION",
        base_min_trade_strength=62,
        base_min_composite_score=58,
    )

    assert details["applied"] is False
    assert strength == 62
    assert composite == 58


def test_weak_data_circuit_breaker_triggers_on_fragile_cluster():
    triggered, details = _evaluate_weak_data_circuit_breaker(
        runtime_thresholds=_runtime_thresholds(),
        data_quality_status="WEAK",
        provider_health_summary="CAUTION",
        confirmation_status="MIXED",
        adjusted_trade_strength=66,
        runtime_composite_score=64,
        ranked_strikes=[{"iv_is_proxy": True, "delta_is_proxy": False} for _ in range(4)],
        direction="PUT",
        gamma_regime="NEGATIVE_GAMMA",
        vol_regime="VOL_EXPANSION",
    )

    assert triggered is True
    assert details["triggered"] is True
    assert details["trigger_count"] >= 2
    assert "put_toxic_regime_context" in details["trigger_reasons"]


def test_weak_data_circuit_breaker_skips_when_data_quality_is_strong():
    triggered, details = _evaluate_weak_data_circuit_breaker(
        runtime_thresholds=_runtime_thresholds(),
        data_quality_status="STRONG",
        provider_health_summary="GOOD",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=90,
        runtime_composite_score=85,
        ranked_strikes=[{"iv_is_proxy": False, "delta_is_proxy": False} for _ in range(4)],
        direction="CALL",
        gamma_regime="POSITIVE_GAMMA",
        vol_regime="NORMAL_VOL",
    )

    assert triggered is False
    assert details["reason"] == "data_quality_not_in_breaker_scope"


def test_runtime_defaults_no_longer_overpromote_negative_gamma_puts():
    cfg = get_trade_runtime_thresholds()

    assert float(cfg["regime_strength_relief_negative_gamma"]) == 0.0
    assert float(cfg["regime_composite_relief_negative_gamma"]) == 0.0
    assert float(cfg["negative_gamma_size_multiplier"]) <= 1.0
    assert float(cfg["positive_gamma_size_multiplier"]) >= 0.95
