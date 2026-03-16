"""
Module: runtime_metadata.py

Purpose:
    Define operator-facing metadata keys used by runtime, CLI, and research views.

Role in the System:
    Part of the signal engine that turns analytics, probability estimates, and overlays into final trade decisions.

Key Outputs:
    Trade decisions, intermediate state bundles, and signal diagnostics.

Downstream Usage:
    Consumed by the live runtime loop, backtests, shadow mode, and signal-evaluation logging.
"""

from __future__ import annotations

from typing import Any


EXECUTION_TRADE_KEYS = [
    "symbol",
    "spot",
    "direction",
    "direction_source",
    "selected_expiry",
    "strike",
    "option_type",
    "entry_price",
    "target",
    "stop_loss",
    "trade_strength",
    "signal_quality",
    "signal_regime",
    "execution_regime",
    "trade_status",
    "message",
    "budget_constraint_applied",
    "lot_size",
    "requested_lots",
    "number_of_lots",
    "optimized_lots",
    "capital_per_lot",
    "capital_required",
    "max_affordable_lots",
    "budget_ok",
    "macro_position_size_multiplier",
    "macro_suggested_lots",
    "macro_size_applied",
    "overnight_hold_allowed",
    "overnight_hold_reason",
    "parameter_pack_name",
    "signal_id",
]

TRADER_VIEW_KEYS = list(dict.fromkeys([
    *EXECUTION_TRADE_KEYS,
    "hybrid_move_probability",
    "large_move_probability",
    "ml_move_probability",
    "macro_event_risk_score",
    "event_window_status",
    "event_lockdown_flag",
    "minutes_to_next_event",
    "next_event_name",
    "macro_regime",
    "macro_sentiment_score",
    "volatility_shock_score",
    "news_confidence_score",
    "macro_adjustment_score",
    "macro_position_size_multiplier",
    "macro_suggested_lots",
    "global_risk_state",
    "global_risk_score",
    "overnight_gap_risk_score",
    "volatility_expansion_risk_score",
    "overnight_hold_allowed",
    "overnight_hold_reason",
    "overnight_risk_penalty",
    "overnight_trade_block",
    "global_risk_adjustment_score",
    "gamma_vol_acceleration_score",
    "squeeze_risk_state",
    "directional_convexity_state",
    "upside_squeeze_risk",
    "downside_airpocket_risk",
    "overnight_convexity_risk",
    "overnight_convexity_penalty",
    "overnight_convexity_boost",
    "gamma_vol_adjustment_score",
    "dealer_hedging_pressure_score",
    "dealer_flow_state",
    "upside_hedging_pressure",
    "downside_hedging_pressure",
    "pinning_pressure_score",
    "overnight_hedging_risk",
    "overnight_dealer_pressure_penalty",
    "overnight_dealer_pressure_boost",
    "dealer_pressure_adjustment_score",
    "expected_move_points",
    "expected_move_pct",
    "expected_move_quality",
    "target_reachability_score",
    "premium_efficiency_score",
    "strike_efficiency_score",
    "option_efficiency_score",
    "option_efficiency_adjustment_score",
    "overnight_option_efficiency_penalty",
    "oil_shock_score",
    "market_volatility_shock_score",
    "commodity_risk_score",
    "risk_off_intensity",
    "volatility_compression_score",
    "volatility_explosion_probability",
    "global_risk_level",
    "global_risk_action",
    "global_risk_size_cap",
    "data_quality_score",
    "data_quality_status",
]))

_STRUCTURAL_TRADE_KEYS = {"execution_trade", "trade_audit"}


def split_trade_payload(trade: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Purpose:
        Split a merged trade payload into execution-facing and audit-facing
        views.

    Context:
        Used by the runtime layer when the signal engine's rich payload needs to
        be presented differently to execution, operator, and research
        consumers.

    Inputs:
        trade (dict[str, Any] | None): Flat trade payload produced by the signal
            engine or a compatibility test double.

    Returns:
        tuple[dict[str, Any] | None, dict[str, Any] | None]: Execution trade
        view followed by the audit payload.

    Notes:
        The execution view contains the minimum fields required to route,
        display, size, and monitor a trade, while the audit view keeps the
        explanatory diagnostics used by research, replay, and shadow analysis.
    """

    if not isinstance(trade, dict):
        return None, None

    execution_trade: dict[str, Any] = {}
    trade_audit: dict[str, Any] = {}

    for key, value in trade.items():
        if key in _STRUCTURAL_TRADE_KEYS:
            continue
        if key in EXECUTION_TRADE_KEYS:
            execution_trade[key] = value
        else:
            trade_audit[key] = value

    return execution_trade, trade_audit


def attach_trade_views(trade: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Purpose:
        Ensure a trade payload exposes explicit execution and audit subviews.

    Context:
        Used by the signal engine and runtime layer so downstream callers can
        read the compact execution contract without losing access to the richer
        diagnostic record kept for research and governance.

    Inputs:
        trade (dict[str, Any] | None): Trade payload to augment in place.

    Returns:
        dict[str, Any] | None: Original trade payload with `execution_trade` and
        `trade_audit` keys populated when a trade payload is present.

    Notes:
        The legacy flat payload is preserved for backward compatibility while
        new orchestration code can adopt the clearer split contract.
    """

    if not isinstance(trade, dict):
        return trade

    execution_trade, trade_audit = split_trade_payload(trade)
    trade["execution_trade"] = execution_trade
    trade["trade_audit"] = trade_audit
    return trade


def empty_scoring_breakdown():
    """
    Purpose:
        Process empty scoring breakdown for downstream use.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    return {
        "flow_signal_score": 0,
        "smart_money_flow_score": 0,
        "gamma_event_score": 0,
        "dealer_position_score": 0,
        "volatility_regime_score": 0,
        "liquidity_void_score": 0,
        "liquidity_vacuum_score": 0,
        "spot_vs_flip_score": 0,
        "hedging_bias_score": 0,
        "gamma_regime_score": 0,
        "intraday_gamma_shift_score": 0,
        "wall_proximity_score": 0,
        "liquidity_map_score": 0,
        "move_model_score": 0,
        "directional_consensus_score": 0,
        "global_risk_adjustment_score": 0,
        "gamma_vol_adjustment_score": 0,
        "dealer_pressure_adjustment_score": 0,
        "option_efficiency_adjustment_score": 0,
    }


def empty_confirmation_state():
    """
    Purpose:
        Return an empty confirmation state that matches the expected schema.

    Context:
        Function inside the `runtime metadata` module. The module sits in the signal engine layer that assembles analytics, strategy logic, and overlays into trade decisions.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        dict: State payload produced for downstream consumption.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    return {
        "score_adjustment": 0,
        "status": "NO_DIRECTION",
        "veto": False,
        "reasons": ["no_direction"],
        "breakdown": {
            "open_alignment_score": 0,
            "prev_close_alignment_score": 0,
            "range_expansion_score": 0,
            "flow_confirmation_score": 0,
            "hedging_confirmation_score": 0,
            "gamma_event_confirmation_score": 0,
            "move_probability_confirmation_score": 0,
            "flip_alignment_score": 0,
        },
    }
