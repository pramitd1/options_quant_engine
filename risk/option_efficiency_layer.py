"""
Module: option_efficiency_layer.py

Purpose:
    Assemble the option efficiency overlay decision from features and policy rules.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

from config.option_efficiency_policy import get_option_efficiency_policy_config
from risk.option_efficiency_features import build_option_efficiency_features
from risk.option_efficiency_models import OptionEfficiencyState


def _clip(value, lo, hi):
    """
    Purpose:
        Clamp a numeric value to the configured bounds.

    Context:
        Function inside the `option efficiency layer` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

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
        Function inside the `option efficiency layer` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

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


def _score_target_reachability(features):
    """
    Purpose:
        Process score target reachability for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        features (Any): Input associated with features.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    ratio = features.get("expected_move_coverage_ratio")
    if ratio is None:
        return cfg.neutral_score
    ratio = _safe_float(ratio, 0.0)
    if ratio >= 1.25:
        return 90
    if ratio >= 1.0:
        return 78
    if ratio >= 0.75:
        return 62
    if ratio >= 0.50:
        return 38
    return 15


def _score_premium_efficiency(features):
    """
    Purpose:
        Process score premium efficiency for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        features (Any): Input associated with features.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    ratio = features.get("premium_coverage_ratio")
    if ratio is None:
        return cfg.neutral_score
    ratio = _safe_float(ratio, 0.0)
    if ratio >= 1.40:
        return 88
    if ratio >= 1.0:
        return 74
    if ratio >= 0.75:
        return 58
    if ratio >= 0.50:
        return 34
    return 18


def _score_strike_efficiency(features):
    """
    Purpose:
        Process score strike efficiency for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        features (Any): Input associated with features.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    ratio = features.get("strike_distance_ratio")
    bucket = str(features.get("strike_moneyness_bucket", "UNKNOWN")).upper().strip()
    premium_ratio = _safe_float(features.get("premium_coverage_ratio"), None)
    if ratio is None:
        return cfg.neutral_score
    ratio = _safe_float(ratio, 0.0)

    if bucket == "ATM":
        base = 78
    elif bucket == "OTM":
        if ratio <= 0.55:
            base = 68
        elif ratio <= 0.90:
            base = 52
        else:
            base = 24
    elif bucket == "ITM":
        if premium_ratio is not None and premium_ratio < 0.65:
            base = 34
        else:
            base = 58
    else:
        base = cfg.neutral_score

    if ratio > 1.20:
        base -= 12
    elif ratio > 0.90:
        base -= 6

    return int(_clip(base, 0, 100))


def _adjustment_score(option_efficiency_score, premium_efficiency_score, strike_efficiency_score, target_reachability_score):
    """
    Purpose:
        Compute the adjustment score used by the surrounding model.

    Context:
        Function inside the `option efficiency layer` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

    Inputs:
        option_efficiency_score (Any): Normalized score for option efficiency.
        premium_efficiency_score (Any): Normalized score for premium efficiency.
        strike_efficiency_score (Any): Normalized score for strike efficiency.
        target_reachability_score (Any): Normalized score for target reachability.

    Returns:
        float | int: Score produced by the current heuristic.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    cfg = get_option_efficiency_policy_config()

    if option_efficiency_score <= cfg.poor_efficiency_threshold:
        return cfg.poor_efficiency_penalty
    if premium_efficiency_score <= cfg.poor_efficiency_threshold:
        return cfg.premium_penalty
    if strike_efficiency_score <= cfg.weak_efficiency_threshold:
        return cfg.strike_penalty
    if option_efficiency_score >= cfg.high_efficiency_threshold and target_reachability_score >= cfg.good_efficiency_threshold:
        return cfg.target_reachability_boost
    if option_efficiency_score >= cfg.good_efficiency_threshold and target_reachability_score >= cfg.good_efficiency_threshold:
        return cfg.target_reachability_moderate_boost
    return 0


def _overnight_evaluation(
    features,
    option_efficiency_score,
    target_reachability_score,
    premium_efficiency_score,
    strike_efficiency_score,
):
    """
    Purpose:
        Process overnight evaluation for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        features (Any): Input associated with features.
        option_efficiency_score (Any): Score value for option efficiency.
        target_reachability_score (Any): Score value for target reachability.
        premium_efficiency_score (Any): Score value for premium efficiency.
        strike_efficiency_score (Any): Score value for strike efficiency.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    holding_context = features.get("holding_context", {})
    overnight_relevant = bool((holding_context or {}).get("overnight_relevant", False))
    if not overnight_relevant:
        return True, "overnight_option_efficiency_contained", 0

    penalty = 0
    reasons = []

    if option_efficiency_score <= 32:
        penalty += 4
        reasons.append("option_efficiency_poor")
    elif option_efficiency_score <= 45:
        penalty += 2
        reasons.append("option_efficiency_weak")

    if target_reachability_score <= 32:
        penalty += 3
        reasons.append("target_reachability_weak")

    if premium_efficiency_score <= 28:
        penalty += 2
        reasons.append("premium_efficiency_poor")

    if strike_efficiency_score <= 20:
        penalty += 3
        reasons.append("strike_efficiency_poor")
    elif strike_efficiency_score <= 35:
        penalty += 1
        reasons.append("strike_efficiency_weak")

    penalty = int(_clip(penalty, 0, 10))
    if penalty >= cfg.overnight_block_threshold:
        return False, reasons[0] if reasons else "overnight_option_efficiency_block", penalty
    if penalty >= cfg.overnight_watch_threshold:
        return True, reasons[0] if reasons else "overnight_option_efficiency_watch", penalty
    return True, "overnight_option_efficiency_contained", 0


def classify_option_efficiency_state(features: dict | None) -> OptionEfficiencyState:
    """
    Purpose:
        Classify option efficiency state into the appropriate regime or label.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        features (dict | None): Input associated with features.
    
    Returns:
        OptionEfficiencyState: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    features = features if isinstance(features, dict) else {}

    target_reachability_score = _score_target_reachability(features)
    premium_efficiency_score = _score_premium_efficiency(features)
    strike_efficiency_score = _score_strike_efficiency(features)

    if features.get("neutral_fallback"):
        option_efficiency_score = cfg.neutral_score
    else:
        option_efficiency_score = int(round(_clip(
            (0.38 * premium_efficiency_score)
            + (0.34 * target_reachability_score)
            + (0.28 * strike_efficiency_score),
            0.0,
            100.0,
        )))

    option_efficiency_adjustment_score = _adjustment_score(
        option_efficiency_score,
        premium_efficiency_score,
        strike_efficiency_score,
        target_reachability_score,
    )
    overnight_hold_allowed, overnight_hold_reason, overnight_option_efficiency_penalty = _overnight_evaluation(
        features,
        option_efficiency_score,
        target_reachability_score,
        premium_efficiency_score,
        strike_efficiency_score,
    )

    reasons = []
    if features.get("neutral_fallback"):
        reasons.append("option_efficiency_neutral_fallback")
    if option_efficiency_score >= cfg.high_efficiency_threshold:
        reasons.append("high_option_efficiency")
    elif option_efficiency_score <= cfg.weak_efficiency_threshold:
        reasons.append("weak_option_efficiency")
    if target_reachability_score <= cfg.weak_efficiency_threshold:
        reasons.append("target_reachability_weak")
    if premium_efficiency_score <= cfg.weak_efficiency_threshold:
        reasons.append("premium_efficiency_weak")
    if not reasons:
        reasons.append("option_efficiency_balanced")

    diagnostics = {
        "iv_source": features.get("iv_source"),
        "iv_unit": features.get("iv_unit"),
        "dte_source": features.get("dte_source"),
        "effective_delta": features.get("effective_delta"),
        "premium_coverage_ratio": features.get("premium_coverage_ratio"),
        "strike_distance_ratio": features.get("strike_distance_ratio"),
        "convexity_multiplier": features.get("convexity_multiplier"),
        "warnings": list(features.get("warnings", [])),
    }

    return OptionEfficiencyState(
        expected_move_points=features.get("expected_move_points"),
        expected_move_pct=features.get("expected_move_pct"),
        expected_move_quality=str(features.get("expected_move_quality", "UNAVAILABLE")),
        target_distance_points=features.get("target_distance_points"),
        target_distance_pct=features.get("target_distance_pct"),
        expected_move_coverage_ratio=features.get("expected_move_coverage_ratio"),
        target_reachability_score=target_reachability_score,
        premium_efficiency_score=premium_efficiency_score,
        strike_efficiency_score=strike_efficiency_score,
        option_efficiency_score=option_efficiency_score,
        option_efficiency_adjustment_score=option_efficiency_adjustment_score,
        overnight_hold_allowed=overnight_hold_allowed,
        overnight_hold_reason=overnight_hold_reason,
        overnight_option_efficiency_penalty=overnight_option_efficiency_penalty,
        strike_moneyness_bucket=str(features.get("strike_moneyness_bucket", "UNKNOWN")),
        strike_distance_from_spot=features.get("strike_distance_from_spot"),
        payoff_efficiency_hint=str(features.get("payoff_efficiency_hint", "unknown")),
        neutral_fallback=bool(features.get("neutral_fallback", False)),
        option_efficiency_reasons=reasons,
        option_efficiency_features=features,
        option_efficiency_diagnostics=diagnostics,
    )


def build_option_efficiency_state(**kwargs):
    """
    Purpose:
        Build the option efficiency state used by downstream trading logic.

    Context:
        Function inside the `option efficiency layer` module. The module sits in the risk overlay layer that can resize, downgrade, or block trade ideas.

    Inputs:
        **kwargs (Any): Additional keyword inputs forwarded by the caller.

    Returns:
        dict: State payload produced for downstream consumption.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    features = build_option_efficiency_features(**kwargs)
    return classify_option_efficiency_state(features).to_dict()


def score_option_efficiency_candidate(
    row,
    *,
    spot,
    direction,
    atm_iv=None,
    selected_expiry=None,
    valuation_time=None,
    hybrid_move_probability=None,
    gamma_regime=None,
    volatility_regime=None,
    volatility_shock_score=None,
    volatility_compression_score=None,
    macro_event_risk_score=None,
    global_risk_state=None,
    gamma_vol_acceleration_score=None,
    dealer_hedging_pressure_score=None,
    liquidity_vacuum_state=None,
    support_wall=None,
    resistance_wall=None,
):
    """
    Purpose:
        Process score option efficiency candidate for downstream use.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        row (Any): Input associated with row.
        spot (Any): Input associated with spot.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        atm_iv (Any): Input associated with ATM IV.
        selected_expiry (Any): Expiry associated with the contract referenced by the signal.
        valuation_time (Any): Input associated with valuation time.
        hybrid_move_probability (Any): Input associated with hybrid move probability.
        gamma_regime (Any): Input associated with gamma regime.
        volatility_regime (Any): Input associated with volatility regime.
        volatility_shock_score (Any): Score value for volatility shock.
        volatility_compression_score (Any): Score value for volatility compression.
        macro_event_risk_score (Any): Macro-event risk score used by fallback or overlay logic.
        global_risk_state (Any): Structured state payload for global risk.
        gamma_vol_acceleration_score (Any): Score value for gamma vol acceleration.
        dealer_hedging_pressure_score (Any): Score value for dealer hedging pressure.
        liquidity_vacuum_state (Any): Structured state payload for liquidity vacuum.
        support_wall (Any): Input associated with support wall.
        resistance_wall (Any): Input associated with resistance wall.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    row = row if hasattr(row, "get") else {}
    state = build_option_efficiency_state(
        spot=spot,
        atm_iv=atm_iv,
        fallback_iv=row.get("impliedVolatility", row.get("IV")),
        expiry_value=row.get("EXPIRY_DT", selected_expiry),
        valuation_time=valuation_time,
        time_to_expiry_years=row.get("TTE"),
        direction=direction,
        strike=row.get("strikePrice"),
        option_type=row.get("OPTION_TYP"),
        entry_price=row.get("lastPrice", row.get("LAST_PRICE")),
        target=None,
        stop_loss=None,
        hybrid_move_probability=hybrid_move_probability,
        gamma_regime=gamma_regime,
        volatility_regime=volatility_regime,
        volatility_shock_score=volatility_shock_score,
        volatility_compression_score=volatility_compression_score,
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        gamma_vol_acceleration_score=gamma_vol_acceleration_score,
        dealer_hedging_pressure_score=dealer_hedging_pressure_score,
        liquidity_vacuum_state=liquidity_vacuum_state,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        delta=row.get("DELTA"),
    )

    score_adjustment = 0
    cfg = get_option_efficiency_policy_config()
    if state["option_efficiency_score"] >= cfg.high_efficiency_threshold:
        score_adjustment = 3
    elif state["option_efficiency_score"] >= cfg.good_efficiency_threshold:
        score_adjustment = 1
    elif state["option_efficiency_score"] <= cfg.poor_efficiency_threshold:
        score_adjustment = -4
    elif state["strike_efficiency_score"] <= cfg.weak_efficiency_threshold:
        score_adjustment = -2

    return {
        "score_adjustment": score_adjustment,
        "option_efficiency_score": state["option_efficiency_score"],
        "strike_efficiency_score": state["strike_efficiency_score"],
        "premium_efficiency_score": state["premium_efficiency_score"],
        "expected_move_points": state["expected_move_points"],
        "expected_move_quality": state["expected_move_quality"],
        "strike_moneyness_bucket": state["strike_moneyness_bucket"],
    }
