from __future__ import annotations

from config.signal_policy import get_trade_runtime_thresholds
from engine.signal_engine import (
    _evaluate_high_composite_soft_override_eligibility,
    _evaluate_provider_health_override_eligibility,
    _runtime_composite_observation_policy,
)


def _runtime_thresholds(**overrides):
    base = {
        "enable_provider_health_degraded_override": 0,
        "provider_health_override_dte_max": 1.0,
        "provider_health_override_min_strength_buffer": 18,
        "provider_health_override_min_composite_buffer": 10,
        "provider_health_override_min_effective_priced_ratio": 0.70,
        "provider_health_override_max_proxy_ratio": 0.25,
        "provider_health_override_size_cap": 0.20,
        "provider_health_override_hold_cap_minutes": 20,
        "provider_health_override_require_strong_confirmation": 1,
        "provider_health_override_one_sided_quote_ratio_max": 0.20,
        "provider_health_override_allow_block_status": 0,
        "provider_health_override_allowed_summary_statuses": ["CAUTION"],
        "provider_health_override_allowed_data_quality_statuses": ["GOOD", "STRONG"],
        "provider_health_override_allowed_block_reasons": ["core_iv_weak"],
    }
    base.update(overrides)
    return base


def test_runtime_composite_observation_policy_keeps_80_to_85_observational():
    policy = _runtime_composite_observation_policy(
        84.9,
        {
            "runtime_composite_observation_threshold": 80,
            "high_composite_soft_override_threshold": 85,
        },
    )

    assert policy["runtime_composite_observation_tier"] == "OBSERVE_80_85"
    assert policy["runtime_composite_score"] == 84
    assert policy["eligible_for_soft_override"] is False


def test_high_composite_soft_override_allows_only_live_safe_soft_blockers():
    payload = {
        "runtime_composite_score": 86,
        "confirmation_status": "CONFIRMED",
        "data_quality_status": "GOOD",
        "analytics_usable": True,
        "execution_suggestion_usable": True,
        "provider_health_blocking_status": "PASS",
        "market_data_trade_blocking_status": "PASS",
    }

    allowed, details = _evaluate_high_composite_soft_override_eligibility(
        payload=payload,
        runtime_thresholds=_runtime_thresholds(enable_high_composite_soft_block_override=1),
        blocker="GLOBAL_RISK_WATCHLIST",
    )

    assert allowed is True
    assert details["runtime_composite_observation_tier"] == "OVERRIDE_85_PLUS"
    assert details["fail_reasons"] == []


def test_high_composite_soft_override_rejects_hard_data_blockers():
    payload = {
        "runtime_composite_score": 90,
        "confirmation_status": "STRONG_CONFIRMATION",
        "data_quality_status": "GOOD",
        "analytics_usable": True,
        "execution_suggestion_usable": False,
        "provider_health_blocking_status": "PASS",
        "market_data_trade_blocking_status": "PASS",
    }

    allowed, details = _evaluate_high_composite_soft_override_eligibility(
        payload=payload,
        runtime_thresholds=_runtime_thresholds(enable_high_composite_soft_block_override=1),
        blocker="GLOBAL_RISK_WATCHLIST",
    )

    assert allowed is False
    assert "execution_data_unusable" in details["fail_reasons"]


def test_provider_override_disabled_by_default():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=_runtime_thresholds(),
        provider_health_blocking_reasons=[],
        provider_health_summary="CAUTION",
        data_quality_status="GOOD",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=95,
        min_trade_strength=62,
        runtime_composite_score=80,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.95, "row_count": 20, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.0},
        ranked_strikes=[{"iv_is_proxy": False, "delta_is_proxy": False} for _ in range(5)],
        days_to_expiry=0.5,
        blocked=False,
    )

    assert allowed is False
    assert details["reason"] == "override_disabled"


def test_provider_override_rejects_weak_data_quality_even_if_reenabled():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=_runtime_thresholds(enable_provider_health_degraded_override=1),
        provider_health_blocking_reasons=[],
        provider_health_summary="CAUTION",
        data_quality_status="CAUTION",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=95,
        min_trade_strength=62,
        runtime_composite_score=80,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.95, "row_count": 20, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.0},
        ranked_strikes=[{"iv_is_proxy": False, "delta_is_proxy": False} for _ in range(5)],
        days_to_expiry=0.5,
        blocked=False,
    )

    assert allowed is False
    assert "data_quality_not_allowlisted" in details["fail_reasons"]


def test_provider_override_rejects_block_status_unless_explicitly_allowed():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=_runtime_thresholds(enable_provider_health_degraded_override=1),
        provider_health_blocking_reasons=["core_iv_weak"],
        provider_health_summary="CAUTION",
        data_quality_status="GOOD",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=95,
        min_trade_strength=62,
        runtime_composite_score=80,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.95, "row_count": 20, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.0},
        ranked_strikes=[{"iv_is_proxy": False, "delta_is_proxy": False} for _ in range(5)],
        days_to_expiry=0.5,
        blocked=True,
    )

    assert allowed is False
    assert "block_status_not_allowed" in details["fail_reasons"]


def test_provider_override_only_allows_tightly_clean_exception_when_explicitly_enabled():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=_runtime_thresholds(
            enable_provider_health_degraded_override=1,
            provider_health_override_allow_block_status=1,
        ),
        provider_health_blocking_reasons=["core_iv_weak"],
        provider_health_summary="CAUTION",
        data_quality_status="GOOD",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=95,
        min_trade_strength=62,
        runtime_composite_score=80,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.95, "row_count": 20, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.0},
        ranked_strikes=[
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
        ],
        days_to_expiry=0.25,
        blocked=True,
    )

    assert allowed is True
    assert details["eligible"] is True
    assert details["fail_reasons"] == []


def test_runtime_defaults_allow_isolated_iv_weak_override_under_guardrails():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=get_trade_runtime_thresholds(),
        provider_health_blocking_reasons=["core_iv_weak", "atm_iv_weak"],
        provider_health_summary="WEAK",
        data_quality_status="CAUTION",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=92,
        min_trade_strength=62,
        runtime_composite_score=79,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.72, "row_count": 40, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.0},
        ranked_strikes=[
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": True, "delta_is_proxy": False},
        ],
        days_to_expiry=5.0,
        blocked=True,
    )

    assert allowed is True
    assert details["eligible"] is True
    assert details["fail_reasons"] == []


def test_runtime_defaults_allow_moderate_buffer_iv_only_override():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=get_trade_runtime_thresholds(),
        provider_health_blocking_reasons=["core_iv_weak", "atm_iv_weak"],
        provider_health_summary="WEAK",
        data_quality_status="CAUTION",
        confirmation_status="CONFIRMED",
        adjusted_trade_strength=76,
        min_trade_strength=62,
        runtime_composite_score=66,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.68, "row_count": 40, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.0},
        ranked_strikes=[
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": True, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
        ],
        days_to_expiry=3.0,
        blocked=True,
    )

    assert allowed is True
    assert details["eligible"] is True
    assert details["fail_reasons"] == []


def test_runtime_defaults_allow_clean_probe_override_near_base_threshold_when_efficiency_is_high():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=get_trade_runtime_thresholds(),
        provider_health_blocking_reasons=[],
        provider_health_summary="WEAK",
        data_quality_status="GOOD",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=62,
        min_trade_strength=62,
        runtime_composite_score=61,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.82, "row_count": 40, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.02},
        ranked_strikes=[
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
        ],
        days_to_expiry=0.4,
        blocked=False,
        option_efficiency_score=84,
        premium_efficiency_score=76,
    )

    assert allowed is True
    assert details["eligible"] is True
    assert details["fail_reasons"] == []


def test_probe_override_still_rejects_low_efficiency_setup():
    allowed, details = _evaluate_provider_health_override_eligibility(
        runtime_thresholds=get_trade_runtime_thresholds(),
        provider_health_blocking_reasons=[],
        provider_health_summary="WEAK",
        data_quality_status="GOOD",
        confirmation_status="STRONG_CONFIRMATION",
        adjusted_trade_strength=62,
        min_trade_strength=62,
        runtime_composite_score=61,
        min_composite_score=58,
        option_chain_validation={"effective_priced_ratio": 0.82, "row_count": 40, "one_sided_quote_rows": 0},
        provider_health={"core_one_sided_quote_ratio": 0.02},
        ranked_strikes=[
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
            {"iv_is_proxy": False, "delta_is_proxy": False},
        ],
        days_to_expiry=0.4,
        blocked=False,
        option_efficiency_score=56,
        premium_efficiency_score=52,
    )

    assert allowed is False
    assert "option_efficiency_below_floor" in details["fail_reasons"]
