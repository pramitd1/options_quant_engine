"""
Module: pre_market_policy.py

Purpose:
    Define pre-market policy configuration for component initialization and readiness checks.

Role in the System:
    Part of the configuration layer that centralizes pre-market tuning, thresholds, and behavior flags.

Key Outputs:
    Policy configuration objects used by pre-market engine components.

Downstream Usage:
    Consumed by pre_market_engine, engine_runner, and signal_engine for pre-open signal readiness.
"""


class PreMarketPolicyConfig:
    """Configuration for pre-market operations and readiness checks."""
    
    # Time window boundaries (IST)
    # Pre-market window: 08:00 IST to 09:15 IST
    pre_market_start_hour = 8
    pre_market_start_minute = 0
    pre_market_end_hour = 9
    pre_market_end_minute = 15
    
    # Intraday starts at 09:15 IST
    market_open_hour = 9
    market_open_minute = 15
    market_close_hour = 15
    market_close_minute = 30
    
    # Dealer setup: carryover from previous session
    # How many hours to lookback for dealer positioning data
    dealer_lookback_hours = 24
    
    # Use previous close dealer position if current session data unavailable
    use_previous_session_dealer_position = True
    
    # Dealer position confidence thresholds
    dealer_position_confidence_threshold = 0.5
    
    # Volatility initialization
    # IV surface lookback days
    volatility_lookback_days = 5
    
    # Minimum IV observations needed for regime initialization
    min_iv_observations = 10
    
    # Realized vol anchor lookback days
    realized_vol_lookback_days = 30
    
    # IV regime initialization: should we use previous session's regime?
    use_previous_iv_regime = True
    
    # Signal readiness: pre-market validation checks
    # Minimum data quality score (0-100) required to enable pre-open signals
    min_data_quality_score = 70
    
    # Minimum IV observations in option chain
    min_option_chain_iv_count = 20
    
    # Maximum staleness allowed for global market snapshot (minutes)
    max_global_market_staleness_minutes = 60
    
    # Enable pre-market signal generation?
    enable_pre_market_signals = False
    
    # Pre-market signal requires higher quality thresholds?
    pre_market_signal_quality_boost = 1.25  # Multiply confidence score by this factor
    
    # Warming up: should we accept lower trade-strength signals pre-market?
    pre_market_min_trade_strength = 40  # vs normal 50
    
    # Market readiness state: once market opens, how long to retain pre-market state?
    intraday_readiness_retention_minutes = 10
    
    # Logging policy
    log_pre_market_debug = True
    log_readiness_checks = True
    

def get_pre_market_policy_config() -> PreMarketPolicyConfig:
    """
    Purpose:
        Retrieve the active pre-market policy configuration.
    
    Context:
        Configuration loader used throughout the codebase.
    
    Returns:
        PreMarketPolicyConfig: Active policy configuration object.
    
    Notes:
        Future: extend to support environment overrides, A/B testing configs, or dynamic reloading.
    """
    return PreMarketPolicyConfig()
