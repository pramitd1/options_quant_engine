"""
Module: exit_model.py

Purpose:
    Implement exit model logic used during trade construction.

Role in the System:
    Part of the strategy layer that converts directional intent into executable option trades.

Key Outputs:
    Strike rankings, trade-construction inputs, and exit or sizing recommendations.

Downstream Usage:
    Consumed by the signal engine and by research tooling that inspects trade construction choices.
"""
from config.settings import TARGET_PROFIT_PERCENT, STOP_LOSS_PERCENT


def calculate_exit(
    entry_price,
    target_profit_percent=TARGET_PROFIT_PERCENT,
    stop_loss_percent=STOP_LOSS_PERCENT,
):

    """
    Purpose:
        Compute the initial target and stop-loss prices for a selected option
        contract.

    Context:
        Called after strike selection so the engine can package a complete trade
        plan with entry, target, and risk limits.

    Inputs:
        entry_price (Any): Option premium used as the trade entry.
        target_profit_percent (Any): Percentage profit target applied to the
        entry premium.
        stop_loss_percent (Any): Percentage stop-loss applied to the entry
        premium.

    Returns:
        tuple[float, float]: Target price and stop-loss price for the trade.

    Notes:
        The function applies percentage-based exits only; it does not model path
        dependence or dynamic trailing behavior.
    """
    target = entry_price * (1 + target_profit_percent / 100)

    stop = entry_price * (1 - stop_loss_percent / 100)

    return target, stop


def compute_exit_timing(
    *,
    trade_strength=0,
    gamma_regime=None,
    vol_regime=None,
    minutes_since_open=None,
    minutes_to_close=None,
):
    """
    Purpose:
        Recommend a time-based exit horizon and urgency classification based
        on empirical alpha-decay observations (peak alpha ~120 min, significant
        decay by session close).

    Inputs:
        trade_strength (int): Composite trade-strength score (0-100).
        gamma_regime (str): Current gamma regime label.
        vol_regime (str): Current volatility regime label.
        minutes_since_open (float|None): Minutes elapsed since market open.
        minutes_to_close (float|None): Minutes remaining until market close.

    Returns:
        dict: Exit timing recommendation with keys:
            recommended_hold_minutes, max_hold_minutes, exit_urgency,
            exit_timing_reason.
    """
    from config.signal_policy import get_exit_timing_policy_config

    cfg = get_exit_timing_policy_config()

    recommended = cfg.peak_alpha_minutes
    max_hold = cfg.max_hold_minutes
    reasons = []

    # Early-session entries get a longer runway
    if minutes_since_open is not None and minutes_since_open <= cfg.early_session_cutoff_minutes_from_open:
        recommended = cfg.early_session_peak_alpha_minutes
        reasons.append("early_session_extended_runway")

    # Late-session entries must exit faster
    if minutes_to_close is not None and minutes_to_close <= cfg.late_session_cutoff_minutes_to_close:
        max_hold = min(max_hold, cfg.late_session_max_hold_minutes)
        recommended = min(recommended, cfg.late_session_max_hold_minutes)
        reasons.append("late_session_compressed_hold")

    # Strong signals can hold longer
    if trade_strength >= cfg.strong_signal_threshold:
        recommended += cfg.strong_signal_hold_extension_minutes
        reasons.append("strong_signal_extension")

    # Volatile environments favor faster exits
    if vol_regime == "VOL_EXPANSION":
        recommended -= cfg.vol_expansion_hold_reduction_minutes
        reasons.append("vol_expansion_reduction")

    # Negative gamma = faster mean reversion, shorter hold
    if gamma_regime in ("NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"):
        recommended -= cfg.negative_gamma_hold_reduction_minutes
        reasons.append("negative_gamma_reduction")

    recommended = max(recommended, 15)
    max_hold = max(max_hold, recommended)

    # Determine exit urgency based on remaining session time
    if minutes_to_close is not None:
        if minutes_to_close <= cfg.urgency_critical_minutes:
            urgency = "CRITICAL"
        elif minutes_to_close <= cfg.urgency_high_minutes:
            urgency = "HIGH"
        elif minutes_to_close <= cfg.urgency_moderate_minutes:
            urgency = "MODERATE"
        else:
            urgency = "LOW"
    else:
        urgency = "LOW"

    return {
        "recommended_hold_minutes": recommended,
        "max_hold_minutes": max_hold,
        "exit_urgency": urgency,
        "exit_timing_reasons": reasons if reasons else ["default_alpha_window"],
    }
