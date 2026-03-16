"""
Module: gamma_vol_acceleration_regime.py

Purpose:
    Classify gamma vol acceleration states and actions from risk features.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

from config.gamma_vol_acceleration_policy import get_gamma_vol_acceleration_policy_config
from risk.gamma_vol_acceleration_models import GammaVolAccelerationState


def _clip(value, lo, hi):
    """
    Purpose:
        Clamp a numeric value to the configured bounds.

    Context:
        Function inside the `gamma vol acceleration regime` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

    Inputs:
        value (Any): Raw value supplied by the caller.
        lo (Any): Inclusive lower bound for the returned value.
        hi (Any): Inclusive upper bound for the returned value.

    Returns:
        float | int: Bounded value returned by the helper.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    return max(lo, min(hi, value))


def _safe_float(value, default=0.0):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `gamma vol acceleration regime` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (Any): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _classify_squeeze_risk_state(score):
    """
    Purpose:
        Classify squeeze risk state into the appropriate regime or label.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        score (Any): Input associated with score.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_gamma_vol_acceleration_policy_config()
    if score >= cfg.extreme_risk_threshold:
        return "EXTREME_ACCELERATION_RISK"
    if score >= cfg.high_risk_threshold:
        return "HIGH_ACCELERATION_RISK"
    if score >= cfg.moderate_risk_threshold:
        return "MODERATE_ACCELERATION_RISK"
    return "LOW_ACCELERATION_RISK"


def _directional_state(upside_squeeze_risk, downside_airpocket_risk, gamma_vol_acceleration_score):
    """
    Purpose:
        Process directional state for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        upside_squeeze_risk (Any): Input associated with upside squeeze risk.
        downside_airpocket_risk (Any): Input associated with downside airpocket risk.
        gamma_vol_acceleration_score (Any): Score value for gamma vol acceleration.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_gamma_vol_acceleration_policy_config()
    upside = _clip(_safe_float(upside_squeeze_risk, 0.0), 0.0, 1.0)
    downside = _clip(_safe_float(downside_airpocket_risk, 0.0), 0.0, 1.0)

    if gamma_vol_acceleration_score < cfg.low_risk_threshold and max(upside, downside) < 0.25:
        return "NO_CONVEXITY_EDGE"

    if (
        upside >= cfg.two_sided_edge_threshold
        and downside >= cfg.two_sided_edge_threshold
        and abs(upside - downside) <= cfg.two_sided_balance_tolerance
    ):
        return "TWO_SIDED_VOLATILITY_RISK"

    if upside >= cfg.directional_edge_threshold and upside > downside:
        return "UPSIDE_SQUEEZE_RISK"

    if downside >= cfg.directional_edge_threshold and downside > upside:
        return "DOWNSIDE_AIRPOCKET_RISK"

    if max(upside, downside) >= cfg.two_sided_edge_threshold:
        return "TWO_SIDED_VOLATILITY_RISK"

    return "NO_CONVEXITY_EDGE"


def _adjustment_score(squeeze_risk_state, directional_convexity_state, gamma_regime_score):
    """
    Purpose:
        Compute the adjustment score used by the surrounding model.

    Context:
        Function inside the `gamma vol acceleration regime` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

    Inputs:
        squeeze_risk_state (Any): State payload for squeeze risk.
        directional_convexity_state (Any): State payload for directional convexity.
        gamma_regime_score (Any): Normalized score for gamma regime.

    Returns:
        float | int: Score produced by the current heuristic.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    cfg = get_gamma_vol_acceleration_policy_config()

    if gamma_regime_score < 0 and squeeze_risk_state == "LOW_ACCELERATION_RISK":
        return cfg.score_dampen_long_gamma

    if squeeze_risk_state == "EXTREME_ACCELERATION_RISK":
        return cfg.score_boost_extreme
    if squeeze_risk_state == "HIGH_ACCELERATION_RISK":
        return cfg.score_boost_high
    if squeeze_risk_state == "MODERATE_ACCELERATION_RISK" and directional_convexity_state != "NO_CONVEXITY_EDGE":
        return cfg.score_boost_moderate

    return 0


def _evaluate_overnight_convexity(
    *,
    squeeze_risk_state,
    directional_convexity_state,
    gamma_vol_acceleration_score,
    upside_squeeze_risk,
    downside_airpocket_risk,
    overnight_convexity_risk,
    global_risk_state,
    macro_event_risk_score,
    overnight_relevant,
):
    """
    Purpose:
        Process evaluate overnight convexity for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        squeeze_risk_state (Any): Structured state payload for squeeze risk.
        directional_convexity_state (Any): Structured state payload for directional convexity.
        gamma_vol_acceleration_score (Any): Score value for gamma vol acceleration.
        upside_squeeze_risk (Any): Input associated with upside squeeze risk.
        downside_airpocket_risk (Any): Input associated with downside airpocket risk.
        overnight_convexity_risk (Any): Input associated with overnight convexity risk.
        global_risk_state (Any): Structured state payload for global risk.
        macro_event_risk_score (Any): Macro-event risk score used by fallback or overlay logic.
        overnight_relevant (Any): Input associated with overnight relevant.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_gamma_vol_acceleration_policy_config()

    if not overnight_relevant:
        return True, "overnight_convexity_contained", 0, 0

    penalty = 0
    boost = 0
    reasons = []

    if squeeze_risk_state == "EXTREME_ACCELERATION_RISK":
        penalty += 4
        reasons.append("extreme_acceleration_risk")
    elif squeeze_risk_state == "HIGH_ACCELERATION_RISK":
        penalty += 2
        reasons.append("high_acceleration_risk")

    if _safe_float(overnight_convexity_risk, 0.0) >= 0.75:
        penalty += 4
        reasons.append("overnight_convexity_risk_high")
    elif _safe_float(overnight_convexity_risk, 0.0) >= 0.55:
        penalty += 2
        reasons.append("overnight_convexity_risk_elevated")

    state = str(global_risk_state or "").upper().strip()
    if state in {"VOL_SHOCK", "EVENT_LOCKDOWN"}:
        penalty += 4
        reasons.append("unstable_global_risk_regime")
    elif state == "RISK_OFF":
        penalty += 2
        reasons.append("risk_off_global_regime")

    event_score = _safe_float(macro_event_risk_score, 0.0)
    if event_score >= 70.0:
        penalty += 3
        reasons.append("macro_event_risk_high")
    elif event_score >= 45.0:
        penalty += 1
        reasons.append("macro_event_risk_elevated")

    if directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK":
        penalty += 1
        reasons.append("two_sided_convexity_risk")
    elif (
        gamma_vol_acceleration_score >= cfg.high_risk_threshold
        and max(_safe_float(upside_squeeze_risk, 0.0), _safe_float(downside_airpocket_risk, 0.0)) >= 0.62
        and state in {"GLOBAL_NEUTRAL", "RISK_ON"}
        and event_score < 35.0
    ):
        boost = 1
        reasons.append("contained_directional_convexity")

    penalty = int(_clip(penalty, 0, 10))
    if penalty >= cfg.overnight_block_threshold:
        return False, reasons[0] if reasons else "overnight_convexity_block", penalty, boost

    if penalty >= cfg.overnight_watch_threshold:
        return True, reasons[0] if reasons else "overnight_convexity_watch", penalty, boost

    if boost > 0:
        return True, reasons[0], penalty, boost

    return True, "overnight_convexity_contained", 0, 0


def classify_gamma_vol_acceleration_state(features: dict | None) -> GammaVolAccelerationState:
    """
    Purpose:
        Classify gamma vol acceleration state into the appropriate regime or label.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        features (dict | None): Input associated with features.
    
    Returns:
        GammaVolAccelerationState: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    features = features if isinstance(features, dict) else {}

    gamma_vol_acceleration_score = int(round(_clip(_safe_float(features.get("normalized_acceleration"), 0.0) * 100.0, 0.0, 100.0)))
    upside_squeeze_risk = round(_clip(_safe_float(features.get("upside_squeeze_risk"), 0.0), 0.0, 1.0), 4)
    downside_airpocket_risk = round(_clip(_safe_float(features.get("downside_airpocket_risk"), 0.0), 0.0, 1.0), 4)
    overnight_convexity_risk = round(_clip(_safe_float(features.get("overnight_convexity_risk"), 0.0), 0.0, 1.0), 4)

    squeeze_risk_state = _classify_squeeze_risk_state(gamma_vol_acceleration_score)
    directional_convexity_state = _directional_state(
        upside_squeeze_risk,
        downside_airpocket_risk,
        gamma_vol_acceleration_score,
    )
    gamma_regime_score = _safe_float(features.get("gamma_regime_score"), 0.0)
    gamma_vol_adjustment_score = _adjustment_score(
        squeeze_risk_state,
        directional_convexity_state,
        gamma_regime_score,
    )

    holding_context = features.get("holding_context", {})
    holding_context = holding_context if isinstance(holding_context, dict) else {}
    overnight_hold_allowed, overnight_hold_reason, overnight_convexity_penalty, overnight_convexity_boost = _evaluate_overnight_convexity(
        squeeze_risk_state=squeeze_risk_state,
        directional_convexity_state=directional_convexity_state,
        gamma_vol_acceleration_score=gamma_vol_acceleration_score,
        upside_squeeze_risk=upside_squeeze_risk,
        downside_airpocket_risk=downside_airpocket_risk,
        overnight_convexity_risk=overnight_convexity_risk,
        global_risk_state=features.get("global_risk_state"),
        macro_event_risk_score=features.get("macro_event_risk_score"),
        overnight_relevant=bool(holding_context.get("overnight_relevant", False)),
    )

    reasons = []
    if features.get("neutral_fallback"):
        reasons.append("gamma_vol_neutral_fallback")
    if squeeze_risk_state != "LOW_ACCELERATION_RISK":
        reasons.append(squeeze_risk_state.lower())
    if directional_convexity_state != "NO_CONVEXITY_EDGE":
        reasons.append(directional_convexity_state.lower())
    if not reasons:
        reasons.append("convexity_risk_contained")

    diagnostics = {
        "feature_confidence": round(_safe_float(features.get("feature_confidence"), 0.0), 4),
        "input_availability": dict(features.get("input_availability", {})),
        "gamma_regime_score": round(gamma_regime_score, 4),
        "flip_proximity_score": round(_safe_float(features.get("flip_proximity_score"), 0.0), 4),
        "volatility_transition_score": round(_safe_float(features.get("volatility_transition_score"), 0.0), 4),
        "liquidity_vacuum_score": round(_safe_float(features.get("liquidity_vacuum_score"), 0.0), 4),
        "hedging_bias_score": round(_safe_float(features.get("hedging_bias_score"), 0.0), 4),
        "pinning_dampener": round(_safe_float(features.get("pinning_dampener"), 0.0), 4),
        "intraday_extension_score": round(_safe_float(features.get("intraday_extension_score"), 0.0), 4),
        "macro_global_boost": round(_safe_float(features.get("macro_global_boost"), 0.0), 4),
        "acceleration_core": round(_safe_float(features.get("acceleration_core"), 0.0), 4),
        "dampening_core": round(_safe_float(features.get("dampening_core"), 0.0), 4),
        "warnings": list(features.get("warnings", [])),
    }

    return GammaVolAccelerationState(
        gamma_vol_acceleration_score=gamma_vol_acceleration_score,
        squeeze_risk_state=squeeze_risk_state,
        directional_convexity_state=directional_convexity_state,
        upside_squeeze_risk=upside_squeeze_risk,
        downside_airpocket_risk=downside_airpocket_risk,
        overnight_convexity_risk=overnight_convexity_risk,
        overnight_hold_allowed=overnight_hold_allowed,
        overnight_hold_reason=overnight_hold_reason,
        overnight_convexity_penalty=overnight_convexity_penalty,
        overnight_convexity_boost=overnight_convexity_boost,
        gamma_vol_adjustment_score=gamma_vol_adjustment_score,
        neutral_fallback=bool(features.get("neutral_fallback", False)),
        gamma_vol_reasons=reasons,
        gamma_vol_features=features,
        gamma_vol_diagnostics=diagnostics,
    )
