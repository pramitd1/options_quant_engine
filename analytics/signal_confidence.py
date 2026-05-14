"""
Module: signal_confidence.py

Purpose:
    Compute a single reliability score representing how trustworthy a
    generated trade signal is.

Role in the System:
    Post-signal analysis layer.  Operates on the already-built trade payload
    without modifying signal generation, scoring, or risk logic.

Key Outputs:
    A dict containing ``confidence_score`` (0–100), ``confidence_level``,
    and the five component sub-scores that feed into the final number.

Downstream Usage:
    Consumed by terminal_output and streamlit_app for display only.
"""

from __future__ import annotations

from config.signal_policy import get_trade_runtime_thresholds
from strategy.score_calibration import (
    create_calibration_segment_key,
    normalize_calibration_context,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _threshold(thresholds: dict, key: str, default: float) -> float:
    return _safe_float((thresholds or {}).get(key), default)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _resolve_feature_reliability_weights(trade: dict) -> dict[str, float]:
    weights = trade.get("feature_reliability_weights") if isinstance(trade.get("feature_reliability_weights"), dict) else {}
    return {
        "flow": _clip(_safe_float(weights.get("flow"), 1.0), 0.0, 1.0),
        "vol_surface": _clip(_safe_float(weights.get("vol_surface"), 1.0), 0.0, 1.0),
        "greeks": _clip(_safe_float(weights.get("greeks"), 1.0), 0.0, 1.0),
        "liquidity": _clip(_safe_float(weights.get("liquidity"), 1.0), 0.0, 1.0),
        "macro": _clip(_safe_float(weights.get("macro"), 1.0), 0.0, 1.0),
    }


def _blend_feature_reliability(trade: dict, **components: float) -> float:
    weights = _resolve_feature_reliability_weights(trade)
    total_weight = 0.0
    blended = 0.0
    for key, component_weight in components.items():
        weight = max(0.0, _safe_float(component_weight, 0.0))
        if weight <= 0.0:
            continue
        blended += weights.get(key, 0.85) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 1.0
    return _clip(blended / total_weight, 0.0, 1.0)


def _resolve_runtime_thresholds(trade: dict) -> dict[str, float]:
    if not isinstance(trade, dict):
        return get_trade_runtime_thresholds()
    thresholds = trade.get("runtime_thresholds")
    if isinstance(thresholds, dict) and thresholds:
        return thresholds
    try:
        return get_trade_runtime_thresholds()
    except Exception:
        return {}


def _resolve_put_bias_boosts(trade: dict) -> tuple[float, float]:
    thresholds = _resolve_runtime_thresholds(trade)
    if not bool(int(_safe_float(thresholds.get("enable_directional_bias_correction"), 1.0))):
        return 0.0, 0.0
    return (
        _safe_float(thresholds.get("put_signal_confidence_boost"), 4.0),
        _safe_float(thresholds.get("put_signal_strength_boost"), 3.0),
    )


def _apply_rolling_strength_normalization(trade: dict, strength_norm: float) -> tuple[float, float]:
    mean = trade.get("rolling_signal_strength_mean")
    std = trade.get("rolling_signal_strength_std")
    try:
        mean = float(mean)
        std = float(std)
    except (TypeError, ValueError):
        return strength_norm, 0.0

    if std <= 0.0:
        return strength_norm, 0.0

    zscore = _clip((strength_norm - mean) / std, -3.0, 3.0)
    adjustment = _clip(1.0 + 0.04 * zscore, 0.88, 1.12)
    return _clip(strength_norm * adjustment, 0, 100), round(zscore, 3)


_TRADE_STRENGTH_LABELS = {
    "VERY_STRONG": 95,
    "STRONG": 80,
    "MODERATE": 60,
    "WEAK": 35,
    "VERY_WEAK": 15,
}


# ---------------------------------------------------------------------------
# Component scorers (each returns 0–100)
# ---------------------------------------------------------------------------

def _signal_strength_component(trade: dict) -> float:
    """Derived from trade_strength and hybrid_move_probability."""
    raw_strength = _safe_float(trade.get("trade_strength"), 0)
    # trade_strength is maintained on a 0-100 scale by the signal engine.
    strength_norm = _clip(raw_strength, 0, 100)
    strength_norm, rolling_zscore = _apply_rolling_strength_normalization(trade, strength_norm)

    put_confidence_boost, put_strength_boost = _resolve_put_bias_boosts(trade)
    if _as_upper(trade.get("direction")) == "PUT":
        strength_norm = min(100.0, strength_norm + put_strength_boost)

    prob = _safe_float(trade.get("hybrid_move_probability"), 0)
    prob_norm = _clip(prob * 100.0, 0, 100)

    component = _clip(0.60 * strength_norm + 0.40 * prob_norm, 0, 100)
    if _as_upper(trade.get("direction")) == "PUT":
        component = _clip(component + put_confidence_boost, 0, 100)

    reliability = _blend_feature_reliability(trade, flow=0.55, liquidity=0.25, greeks=0.20)
    bounded = _clip(component * reliability, 0, 100)
    trade["rolling_strength_zscore"] = rolling_zscore
    return bounded


def _confirmation_component(trade: dict) -> float:
    """Derived from confirmation_status and confirmation_breakdown."""
    status = str(trade.get("confirmation_status") or "").upper().strip()
    status_map = {
        "STRONG_CONFIRMATION": 100,
        "CONFIRMED": 90,
        "MIXED": 55,
        "CONFLICT": 20,
        "NO_DIRECTION": 10,
    }
    status_score = status_map.get(status, 30)

    breakdown = trade.get("confirmation_breakdown")
    if isinstance(breakdown, dict) and breakdown:
        positive = sum(1 for v in breakdown.values() if _safe_float(v) > 0)
        total = max(len(breakdown), 1)
        ratio_bonus = _clip((positive / total) * 100, 0, 100)
        component = _clip(0.70 * status_score + 0.30 * ratio_bonus, 0, 100)
        reliability = _blend_feature_reliability(trade, flow=0.75, liquidity=0.15, greeks=0.10)
        return _clip(component * reliability, 0, 100)

    reliability = _blend_feature_reliability(trade, flow=0.75, liquidity=0.15, greeks=0.10)
    return _clip(float(status_score) * reliability, 0, 100)


def _market_stability_component(trade: dict) -> float:
    """Derived from macro_regime, global_risk_state, volatility_shock_score."""
    regime = str(trade.get("macro_regime") or "").upper().strip()
    regime_map = {
        "RISK_ON": 90,
        "MACRO_NEUTRAL": 65,
        "RISK_OFF": 25,
        "EVENT_LOCKDOWN": 10,
    }
    regime_score = regime_map.get(regime, 50)

    risk_state = str(trade.get("global_risk_state") or "").upper().strip()
    risk_map = {
        "RISK_ON": 90,
        "GLOBAL_NEUTRAL": 70,
        "RISK_OFF": 25,
        "EVENT_LOCKDOWN": 10,
        "LOW_RISK": 95,
        "MODERATE_RISK": 70,
        "ELEVATED_RISK": 40,
        "HIGH_RISK": 15,
        "EXTREME_RISK": 5,
    }
    risk_score = risk_map.get(risk_state, 50)

    vol_shock = _safe_float(trade.get("market_volatility_shock_score"), 0)
    # Higher vol shock → lower stability; invert and normalise (0–100 scale)
    vol_stability = _clip(100 - vol_shock, 0, 100)

    gamma_vol = _safe_float(
        trade.get("gamma_vol_acceleration_score_normalized", trade.get("gamma_vol_acceleration_score")),
        0,
    )
    gamma_stability = _clip(100 - _clip(gamma_vol, 0, 100), 0, 100)

    component = _clip(
        0.30 * regime_score + 0.30 * risk_score + 0.20 * vol_stability + 0.20 * gamma_stability,
        0, 100,
    )
    reliability = _blend_feature_reliability(trade, vol_surface=0.75, macro=0.25)
    return _clip(component * reliability, 0, 100)


def _data_integrity_component(trade: dict) -> float:
    """Derived from data_quality_score and provider_health."""
    dq_status = str(trade.get("data_quality_status") or "").upper().strip()
    dq_status_map = {
        "STRONG": 100,
        "GOOD": 85,
        "CAUTION": 60,
        "WEAK": 35,
    }
    dq_norm = dq_status_map.get(dq_status)
    if dq_norm is None:
        dq_norm = _clip(_safe_float(trade.get("data_quality_score"), 50), 0, 100)

    ph = trade.get("provider_health")
    ocv = trade.get("option_chain_validation") if isinstance(trade.get("option_chain_validation"), dict) else {}

    if isinstance(ph, dict) and isinstance(ocv, dict) and ocv:
        eff_ratio = _clip(_safe_float(ocv.get("effective_priced_ratio"), _safe_float(ocv.get("priced_ratio"), 0.0)), 0.0, 1.0)
        pair_ratio = _clip(_safe_float(ocv.get("paired_strike_ratio"), 0.0), 0.0, 1.0)
        iv_ratio = _clip(_safe_float(ocv.get("iv_ratio"), 0.0), 0.0, 1.0)
        dup_ratio = _clip(_safe_float(ocv.get("duplicate_ratio"), 0.0), 0.0, 1.0)

        ph_score = _clip(
            100.0 * (
                0.45 * eff_ratio
                + 0.25 * pair_ratio
                + 0.20 * iv_ratio
                + 0.10 * (1.0 - dup_ratio)
            ),
            0,
            100,
        )
    elif isinstance(ph, dict):
        explicit_score = ph.get("market_data_readiness_score")
        if explicit_score is not None:
            ph_score = _clip(_safe_float(explicit_score, 50.0), 0, 100)
        else:
            status = str(ph.get("summary_status") or "").upper().strip()
            ph_map = {
                "HEALTHY": 100,
                "GOOD": 90,
                "DEGRADED": 55,
                "CAUTION": 45,
                "WEAK": 20,
                "UNHEALTHY": 15,
            }
            ph_score = ph_map.get(status, 50)
    else:
        ph_score = 50.0

    component = _clip(0.60 * dq_norm + 0.40 * ph_score, 0, 100)
    reliability = _blend_feature_reliability(trade, flow=0.20, vol_surface=0.30, greeks=0.25, liquidity=0.25)
    return _clip(component * reliability, 0, 100)


def _option_efficiency_component(trade: dict) -> float:
    """Derived from option_efficiency_score and premium_efficiency_score."""
    oe = _safe_float(trade.get("option_efficiency_score"), 50)
    pe = _safe_float(trade.get("premium_efficiency_score"), 50)
    component = _clip(0.55 * _clip(oe, 0, 100) + 0.45 * _clip(pe, 0, 100), 0, 100)
    reliability = _blend_feature_reliability(trade, liquidity=0.45, greeks=0.30, vol_surface=0.25)
    return _clip(component * reliability, 0, 100)


def _ta_component(trade: dict) -> float:
    """Derived from technical analysis features."""
    ta_direction = str(trade.get("ta_direction") or "").upper().strip()
    ta_confidence = _safe_float(trade.get("ta_confidence"), 0.0)

    # Base score from TA confidence
    base_score = ta_confidence * 100.0

    # Boost for strong directional alignment
    if ta_direction in {"CALL", "PUT"}:
        direction_boost = 10  # Small boost for having a clear TA signal
        base_score = min(base_score + direction_boost, 100.0)

    # Penalize conflicting TA regimes
    ta_regime = str(trade.get("ta_regime") or "").lower().strip()
    if ta_regime in {"mixed_signals", "error", "insufficient_data"}:
        base_score *= 0.7  # Reduce confidence for unreliable TA

    component = _clip(base_score, 0, 100)
    # TA reliability is lower since it's supplementary
    reliability = _blend_feature_reliability(trade, flow=0.1, vol_surface=0.3, macro=0.3, liquidity=0.3)
    return _clip(component * reliability, 0, 100)


def _directional_bias_component(trade: dict) -> float:
    """Estimate directional bias alignment for the current trade direction."""
    direction = _as_upper(trade.get("direction"))
    if direction not in {"CALL", "PUT"}:
        return 50.0

    oi_change_bias = _safe_float(trade.get("net_oi_change_bias"), 0.0)
    hedging_bias = _as_upper(trade.get("dealer_hedging_bias"))
    ta_direction = _as_upper(trade.get("ta_direction"))

    support = 0
    conflict = 0

    if direction == "CALL":
        support += int(oi_change_bias < 0)
        conflict += int(oi_change_bias > 0)
        support += int(hedging_bias in {"UPSIDE_ACCELERATION", "UPSIDE_PINNING"})
        conflict += int(hedging_bias in {"DOWNSIDE_ACCELERATION", "DOWNSIDE_PINNING", "PINNING"})
        support += int(ta_direction == "CALL")
        conflict += int(ta_direction == "PUT")
    else:
        support += int(oi_change_bias > 0)
        conflict += int(oi_change_bias < 0)
        support += int(hedging_bias in {"DOWNSIDE_ACCELERATION", "DOWNSIDE_PINNING"})
        conflict += int(hedging_bias in {"UPSIDE_ACCELERATION", "UPSIDE_PINNING", "PINNING"})
        support += int(ta_direction == "PUT")
        conflict += int(ta_direction == "CALL")

    component = 55.0 + 15.0 * support - 18.0 * conflict
    return _clip(component, 0, 100)


def _directional_bias_multiplier(bias_component: float) -> float:
    return float(_clip(0.80 + 0.20 * (bias_component / 100.0), 0.70, 1.0))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_LEVEL_THRESHOLDS = [
    (85, "VERY_HIGH"),
    (70, "HIGH"),
    (55, "MODERATE"),
    (40, "LOW"),
]


def _classify(score: float) -> str:
    for threshold, label in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "UNRELIABLE"


def _as_upper(value) -> str:
    return str(value or "").upper().strip()


def _runtime_calibration_context(trade: dict) -> dict[str, str]:
    return normalize_calibration_context(
        {
            "direction": trade.get("direction"),
            "gamma_regime": trade.get("gamma_regime"),
            "vol_regime": trade.get("vol_regime") or trade.get("volatility_regime"),
        }
    )


def _score_calibration_regime_match(trade: dict) -> tuple[str, dict[str, str], str | None]:
    runtime_context = _runtime_calibration_context(trade)
    expected_key = create_calibration_segment_key(runtime_context) if runtime_context else None
    selected_key = trade.get("score_calibration_segment_key")
    selected_context = normalize_calibration_context(
        trade.get("score_calibration_segment_context")
        if isinstance(trade.get("score_calibration_segment_context"), dict)
        else {}
    )

    if not runtime_context:
        return "UNKNOWN", runtime_context, expected_key
    if selected_key and expected_key and str(selected_key) == expected_key:
        return "FULL", runtime_context, expected_key
    if not selected_context or str(selected_key or "").lower() == "default":
        return "FALLBACK", runtime_context, expected_key

    for key, value in selected_context.items():
        if runtime_context.get(key) != value:
            return "MISMATCH", runtime_context, expected_key

    if set(selected_context.keys()) == set(runtime_context.keys()):
        return "FULL", runtime_context, expected_key
    return "PARTIAL", runtime_context, expected_key


def _calibration_guardrail(trade: dict) -> tuple[float, list[str], dict]:
    thresholds = _resolve_runtime_thresholds(trade)
    cap = 100.0
    reasons: list[str] = []

    active = any(
        key in trade
        for key in (
            "live_calibration_gate",
            "score_calibration_enabled",
            "score_calibration_applied",
            "score_calibration_segment_key",
            "regime_segment_guard",
            "regime_segment_samples",
            "direction_head_calibration_applied",
        )
    )

    live_gate = trade.get("live_calibration_gate") if isinstance(trade.get("live_calibration_gate"), dict) else {}
    regime_guard = trade.get("regime_segment_guard") if isinstance(trade.get("regime_segment_guard"), dict) else {}
    sample_size = None
    recency_days = None
    live_verdict = _as_upper(live_gate.get("verdict"))
    regime_verdict = _as_upper(regime_guard.get("verdict"))

    min_completed = int(max(0.0, _threshold(thresholds, "confidence_guard_min_calibration_trades", 80.0)))
    max_stale_days = _threshold(thresholds, "confidence_guard_max_calibration_staleness_days", 5.0)

    if live_gate:
        sample_size = int(_safe_float(live_gate.get("completed_trades"), 0.0))
        recency_days = _safe_float(live_gate.get("days_since_last_completed_trade"), None)
        live_reason = str(live_gate.get("reason") or "").strip()
        if live_verdict == "BLOCK":
            cap = min(cap, _threshold(thresholds, "confidence_guard_live_calibration_block_cap", 52.0))
            reasons.append("live_calibration_block")
        elif live_verdict == "CAUTION":
            cap = min(cap, _threshold(thresholds, "confidence_guard_live_calibration_caution_cap", 68.0))
            if live_reason in {"insufficient_completed_trades", "insufficient_recent_trades"}:
                reasons.append("calibration_sample_insufficient")
            elif live_reason == "stale_completed_trade_history":
                reasons.append("calibration_history_stale")
            else:
                reasons.append("live_calibration_caution")
        elif live_verdict in {"UNAVAILABLE", ""} and live_gate.get("ok") is False:
            cap = min(cap, _threshold(thresholds, "confidence_guard_calibration_unavailable_cap", 72.0))
            reasons.append("live_calibration_unavailable")

        if min_completed > 0 and sample_size < min_completed:
            cap = min(cap, _threshold(thresholds, "confidence_guard_insufficient_sample_cap", 66.0))
            reasons.append("calibration_sample_insufficient")
        if recency_days is not None and max_stale_days > 0 and recency_days > max_stale_days:
            cap = min(cap, _threshold(thresholds, "confidence_guard_stale_calibration_cap", 64.0))
            reasons.append("calibration_history_stale")

    regime_sample = regime_guard.get("sample_size") if regime_guard else trade.get("regime_segment_samples")
    if regime_sample is not None:
        regime_sample_int = int(_safe_float(regime_sample, 0.0))
        sample_size = regime_sample_int if sample_size is None else min(sample_size, regime_sample_int)
    if regime_guard:
        if regime_verdict == "BLOCK":
            cap = min(cap, _threshold(thresholds, "confidence_guard_regime_segment_block_cap", 55.0))
            reasons.append("regime_segment_block")
        elif regime_verdict == "CAUTION":
            cap = min(cap, _threshold(thresholds, "confidence_guard_regime_segment_caution_cap", 70.0))
            reasons.append("regime_segment_caution")
        elif regime_verdict == "UNAVAILABLE":
            reason = str(regime_guard.get("reason") or "").strip()
            if reason in {"segment_sample_too_small", "no_matching_segment", "segment_context_missing"}:
                cap = min(cap, _threshold(thresholds, "confidence_guard_regime_segment_unavailable_cap", 72.0))
                reasons.append("regime_segment_sample_insufficient")

    regime_match, runtime_context, expected_segment_key = _score_calibration_regime_match(trade)
    score_cal_enabled_present = "score_calibration_enabled" in trade
    score_cal_enabled = _as_bool(trade.get("score_calibration_enabled"), default=False)
    score_cal_applied = _as_bool(trade.get("score_calibration_applied"), default=False)
    score_cal_attempted = (
        trade.get("runtime_composite_score") is not None
        or trade.get("score_calibration_segment_key") is not None
        or score_cal_applied
    )
    if score_cal_enabled_present and score_cal_enabled and score_cal_attempted and not score_cal_applied:
        cap = min(cap, _threshold(thresholds, "confidence_guard_score_calibration_unavailable_cap", 72.0))
        reasons.append("score_calibration_unavailable")
    elif score_cal_applied:
        if regime_match == "MISMATCH":
            cap = min(cap, _threshold(thresholds, "confidence_guard_score_calibration_mismatch_cap", 60.0))
            reasons.append("score_calibration_segment_mismatch")
        elif regime_match == "FALLBACK":
            cap = min(cap, _threshold(thresholds, "confidence_guard_score_calibration_fallback_cap", 76.0))
            reasons.append("score_calibration_segment_fallback")
        elif regime_match == "PARTIAL":
            cap = min(cap, _threshold(thresholds, "confidence_guard_score_calibration_partial_cap", 82.0))
            reasons.append("score_calibration_segment_partial")

    direction_head_used = _as_bool(trade.get("direction_head_used_for_final"), default=False)
    direction_head_calibrated = _as_bool(trade.get("direction_head_calibration_applied"), default=False)
    if direction_head_used and not direction_head_calibrated:
        cap = min(cap, _threshold(thresholds, "confidence_guard_direction_head_uncalibrated_cap", 78.0))
        reasons.append("direction_head_calibration_unavailable")

    reasons = _dedupe_keep_order(reasons)
    severe = {
        "live_calibration_block",
        "calibration_history_stale",
        "score_calibration_segment_mismatch",
        "regime_segment_block",
    }
    has_calibration_evidence = bool(live_gate or regime_guard or score_cal_attempted or direction_head_used)
    if not active or not has_calibration_evidence:
        status = "UNKNOWN"
    elif any(reason in severe for reason in reasons):
        status = "WEAK"
    elif reasons:
        status = "CAUTION"
    else:
        status = "PASS"

    diagnostics = {
        "status": status,
        "reasons": reasons,
        "sample_size": sample_size,
        "min_sample_size": min_completed,
        "recency_days": round(float(recency_days), 2) if recency_days is not None else None,
        "max_recency_days": max_stale_days,
        "regime_match": regime_match,
        "runtime_context": runtime_context,
        "expected_segment_key": expected_segment_key,
        "selected_segment_key": trade.get("score_calibration_segment_key"),
        "selected_segment_context": (
            normalize_calibration_context(trade.get("score_calibration_segment_context"))
            if isinstance(trade.get("score_calibration_segment_context"), dict)
            else {}
        ),
        "score_calibration_applied": score_cal_applied,
        "score_calibration_enabled": score_cal_enabled if score_cal_enabled_present else None,
        "live_calibration_verdict": live_verdict or None,
        "regime_segment_verdict": regime_verdict or None,
        "confidence_cap": round(float(cap), 2) if cap < 100.0 else None,
    }
    return cap, reasons, diagnostics


def _apply_recalibration_guards(trade: dict, score: float) -> tuple[float, list[str], bool, dict]:
    """Apply conservative caps when execution preconditions are not met.

    Returns
    -------
    (capped_score, guard_caps, cap_was_applied, calibration_guardrail)
        ``cap_was_applied`` is True only when the cap chain actually reduced the
        score below its raw weighted value.  When False the guards are still
        relevant context but none of them was the binding constraint.
    """
    guard_caps = []

    trade_status = _as_upper(trade.get("trade_status"))
    data_quality = _as_upper(trade.get("data_quality_status"))
    confirmation_status = _as_upper(trade.get("confirmation_status"))
    direction = _as_upper(trade.get("direction"))
    no_trade_reason_code = _as_upper(trade.get("no_trade_reason_code"))

    provider_health_summary = _as_upper(trade.get("provider_health_summary"))
    if not provider_health_summary and isinstance(trade.get("provider_health"), dict):
        provider_health_summary = _as_upper(trade["provider_health"].get("summary_status"))

    cap = 100.0

    if trade_status in {"WATCHLIST", "NO_SIGNAL", "NO_TRADE", "DATA_INVALID", "BUDGET_FAIL"}:
        cap = min(cap, 72.0)
        guard_caps.append("status_watchlist_or_blocked")

    if data_quality == "CAUTION":
        cap = min(cap, 68.0)
        guard_caps.append("data_quality_caution")
    elif data_quality == "WEAK":
        cap = min(cap, 45.0)
        guard_caps.append("data_quality_weak")

    if provider_health_summary in {"CAUTION", "DEGRADED"}:
        cap = min(cap, 62.0)
        guard_caps.append("provider_health_caution")
    elif provider_health_summary in {"WEAK", "UNHEALTHY"}:
        cap = min(cap, 42.0)
        guard_caps.append("provider_health_weak")

    if confirmation_status == "NO_DIRECTION":
        cap = min(cap, 52.0)
        guard_caps.append("confirmation_no_direction")
    elif confirmation_status == "CONFLICT":
        cap = min(cap, 52.0)
        guard_caps.append("confirmation_conflict")

    if no_trade_reason_code and trade_status != "TRADE":
        cap = min(cap, 60.0)
        guard_caps.append("explicit_no_trade_reason")

    if direction not in {"CALL", "PUT"} and trade_status != "TRADE":
        cap = min(cap, 55.0)
        guard_caps.append("direction_unresolved")

    calibration_cap, calibration_guards, calibration_diagnostics = _calibration_guardrail(trade)
    cap = min(cap, calibration_cap)
    guard_caps.extend(calibration_guards)

    capped_score = round(min(score, cap), 2)
    return (capped_score, _dedupe_keep_order(guard_caps), capped_score < score, calibration_diagnostics)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "signal_strength": 0.24,
    "confirmation": 0.19,
    "market_stability": 0.16,
    "data_integrity": 0.12,
    "option_efficiency": 0.09,
    "ta": 0.14,  # Technical analysis component
    "directional_bias": 0.06,
}


def compute_signal_confidence(trade: dict) -> dict:
    """Compute the signal confidence meter for a trade payload.

    Parameters
    ----------
    trade : dict
        The trade payload produced by the signal engine.  If *None* or
        empty, returns baseline UNRELIABLE result.

    Returns
    -------
    dict
        Keys: ``confidence_score``, ``confidence_level``, and one key
        per component (``signal_strength_component``, …).
    """
    if not trade:
        return {
            "confidence_score": 0,
            "confidence_level": "UNRELIABLE",
            "signal_strength_component": 0,
            "confirmation_component": 0,
            "market_stability_component": 0,
            "data_integrity_component": 0,
            "option_efficiency_component": 0,
            "ta_component": 0,
            "confidence_recalibration_guards": [],
            "calibration_status": "UNKNOWN",
            "calibration_sample_size": None,
            "calibration_regime_match": "UNKNOWN",
            "calibration_guardrail": {
                "status": "UNKNOWN",
                "reasons": [],
                "sample_size": None,
                "regime_match": "UNKNOWN",
            },
        }

    components = {
        "signal_strength_component": round(_signal_strength_component(trade), 2),
        "confirmation_component": round(_confirmation_component(trade), 2),
        "market_stability_component": round(_market_stability_component(trade), 2),
        "data_integrity_component": round(_data_integrity_component(trade), 2),
        "option_efficiency_component": round(_option_efficiency_component(trade), 2),
        "ta_component": round(_ta_component(trade), 2),
        "directional_bias_component": round(_directional_bias_component(trade), 2),
    }

    raw = (
        _WEIGHTS["signal_strength"] * components["signal_strength_component"]
        + _WEIGHTS["confirmation"] * components["confirmation_component"]
        + _WEIGHTS["market_stability"] * components["market_stability_component"]
        + _WEIGHTS["data_integrity"] * components["data_integrity_component"]
        + _WEIGHTS["option_efficiency"] * components["option_efficiency_component"]
        + _WEIGHTS["ta"] * components["ta_component"]
        + _WEIGHTS["directional_bias"] * components["directional_bias_component"]
    )
    score = round(_clip(raw, 0, 100), 2)
    score, applied_guards, cap_was_applied, calibration_guardrail = _apply_recalibration_guards(trade, score)

    bias_component = components["directional_bias_component"]
    bias_mult = _directional_bias_multiplier(bias_component)
    if bias_mult < 1.0:
        score = round(_clip(score * bias_mult, 0, 100), 2)
        applied_guards.append("directional_bias_correction")

    result = {
        "confidence_score": score,
        "confidence_level": _classify(score),
        "confidence_recalibration_guards": applied_guards,
        "confidence_cap_applied": cap_was_applied,
        "directional_bias_multiplier": round(bias_mult, 4),
        "calibration_status": calibration_guardrail.get("status", "UNKNOWN"),
        "calibration_sample_size": calibration_guardrail.get("sample_size"),
        "calibration_regime_match": calibration_guardrail.get("regime_match", "UNKNOWN"),
        "calibration_guardrail": calibration_guardrail,
        **components,
    }
    if "rolling_strength_zscore" in trade:
        result["rolling_strength_zscore"] = trade["rolling_strength_zscore"]
    return result
