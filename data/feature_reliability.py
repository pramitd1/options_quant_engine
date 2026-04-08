"""Feature reliability weights derived from tradable data quality diagnostics."""

from __future__ import annotations

from typing import Any


def compute_feature_reliability_weights(validation_payload: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(validation_payload, dict):
        return {
            "flow": 0.5,
            "vol_surface": 0.5,
            "greeks": 0.5,
            "liquidity": 0.5,
            "macro": 0.7,
        }

    tradable = validation_payload.get("tradable_data") if isinstance(validation_payload.get("tradable_data"), dict) else {}
    crossed_ratio = float((tradable.get("crossed_locked") or {}).get("crossed_or_locked_ratio") or 0.0)
    outlier_ratio = float((tradable.get("outlier_rejection") or {}).get("outlier_ratio") or 0.0)
    strike_conf = float((tradable.get("per_strike_confidence") or {}).get("mean") or 0.0)

    # Pull the three structural IV quality checks added in the upgrade.
    provider_health = (
        (validation_payload.get("provider_health") or {})
        if isinstance(validation_payload.get("provider_health"), dict)
        else {}
    )
    _HEALTH_SCORE = {"GOOD": 1.0, "CAUTION": 0.6, "WEAK": 0.2, "N/A": 1.0}
    atm_iv_score = _HEALTH_SCORE.get(str(provider_health.get("atm_iv_health") or ""), 1.0)
    iv_parity_score = _HEALTH_SCORE.get(str(provider_health.get("iv_parity_health") or ""), 1.0)
    iv_staleness_score = _HEALTH_SCORE.get(str(provider_health.get("iv_staleness_health") or ""), 1.0)
    # Blended IV structural quality: ATM presence is the most critical gate (50%),
    # parity consistency is next (30%), staleness is a softer signal (20%).
    iv_structural_quality = (
        0.50 * atm_iv_score
        + 0.30 * iv_parity_score
        + 0.20 * iv_staleness_score
    )

    # Reliability decays from a strong baseline as structural quote quality degrades.
    liquidity = max(0.0, min(1.0, 1.0 - crossed_ratio - (0.5 * outlier_ratio)))
    # vol_surface now incorporates the three industry-grade IV structural checks.
    vol_surface = max(0.0, min(1.0, (0.2 + strike_conf - outlier_ratio) * iv_structural_quality))
    greeks = max(0.0, min(1.0, 0.2 + strike_conf - (0.5 * crossed_ratio)))
    flow = max(0.0, min(1.0, 0.3 + strike_conf - crossed_ratio - outlier_ratio))

    return {
        "flow": round(flow, 4),
        "vol_surface": round(vol_surface, 4),
        "greeks": round(greeks, 4),
        "liquidity": round(liquidity, 4),
        # Macro feed often comes separately, so start high and avoid over-penalizing from chain-only issues.
        "macro": 0.8,
    }
