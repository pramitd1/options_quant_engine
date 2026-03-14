"""
Shared runtime metadata used by multiple live entry points.
"""

TRADER_VIEW_KEYS = [
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
    "budget_constraint_applied",
    "lot_size",
    "requested_lots",
    "number_of_lots",
    "optimized_lots",
    "capital_per_lot",
    "capital_required",
    "max_affordable_lots",
    "budget_ok",
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
    "data_quality_score",
    "data_quality_status",
]


def empty_scoring_breakdown():
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
    }


def empty_confirmation_state():
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
