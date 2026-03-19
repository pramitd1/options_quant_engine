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
    # trade_strength is typically 0..~50+ integer; normalise to 0–100
    strength_norm = _clip(raw_strength * 2.0, 0, 100)

    prob = _safe_float(trade.get("hybrid_move_probability"), 0)
    prob_norm = _clip(prob * 100.0, 0, 100)

    return _clip(0.60 * strength_norm + 0.40 * prob_norm, 0, 100)


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
        return _clip(0.70 * status_score + 0.30 * ratio_bonus, 0, 100)

    return float(status_score)


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

    return _clip(
        0.30 * regime_score + 0.30 * risk_score + 0.20 * vol_stability + 0.20 * gamma_stability,
        0, 100,
    )


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
    if isinstance(ph, dict):
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

    return _clip(0.60 * dq_norm + 0.40 * ph_score, 0, 100)


def _option_efficiency_component(trade: dict) -> float:
    """Derived from option_efficiency_score and premium_efficiency_score."""
    oe = _safe_float(trade.get("option_efficiency_score"), 50)
    pe = _safe_float(trade.get("premium_efficiency_score"), 50)
    return _clip(0.55 * _clip(oe, 0, 100) + 0.45 * _clip(pe, 0, 100), 0, 100)


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

    return {
        "confidence_score": score,
        "confidence_level": _classify(score),
        **components,
    }
