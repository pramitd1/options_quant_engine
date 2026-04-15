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

    prob = _safe_float(trade.get("hybrid_move_probability"), 0)
    prob_norm = _clip(prob * 100.0, 0, 100)

    component = _clip(0.60 * strength_norm + 0.40 * prob_norm, 0, 100)
    reliability = _blend_feature_reliability(trade, flow=0.55, liquidity=0.25, greeks=0.20)
    return _clip(component * reliability, 0, 100)


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


def _apply_recalibration_guards(trade: dict, score: float) -> tuple[float, list[str]]:
    """Apply conservative caps when execution preconditions are not met.

    Returns
    -------
    (capped_score, guard_caps, cap_was_applied)
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

    capped_score = round(min(score, cap), 2)
    return (capped_score, guard_caps, capped_score < score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "signal_strength": 0.30,
    "confirmation": 0.25,
    "market_stability": 0.20,
    "data_integrity": 0.15,
    "option_efficiency": 0.10,
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
            "confidence_recalibration_guards": [],
        }

    components = {
        "signal_strength_component": round(_signal_strength_component(trade), 2),
        "confirmation_component": round(_confirmation_component(trade), 2),
        "market_stability_component": round(_market_stability_component(trade), 2),
        "data_integrity_component": round(_data_integrity_component(trade), 2),
        "option_efficiency_component": round(_option_efficiency_component(trade), 2),
    }

    raw = (
        _WEIGHTS["signal_strength"] * components["signal_strength_component"]
        + _WEIGHTS["confirmation"] * components["confirmation_component"]
        + _WEIGHTS["market_stability"] * components["market_stability_component"]
        + _WEIGHTS["data_integrity"] * components["data_integrity_component"]
        + _WEIGHTS["option_efficiency"] * components["option_efficiency_component"]
    )
    score = round(_clip(raw, 0, 100), 2)
    score, applied_guards, cap_was_applied = _apply_recalibration_guards(trade, score)

    return {
        "confidence_score": score,
        "confidence_level": _classify(score),
        "confidence_recalibration_guards": applied_guards,
            "confidence_cap_applied": cap_was_applied,
        **components,
    }
