"""
Module: news_category_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by news category.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

from __future__ import annotations


CATEGORY_SENTIMENT_MULTIPLIERS = {
    "india_macro": 1.0,
    "policy": 1.0,
    "global_macro": 1.0,
    "geopolitics": 1.0,
    "commodity_oil": 1.0,
    "banking_financial_stress": 1.0,
    "earnings_company_specific": 1.0,
}

CATEGORY_VOL_MULTIPLIERS = {
    "india_macro": 1.0,
    "policy": 1.0,
    "global_macro": 1.0,
    "geopolitics": 1.0,
    "commodity_oil": 1.0,
    "banking_financial_stress": 1.0,
    "earnings_company_specific": 1.0,
}

CATEGORY_IMPACT_MULTIPLIERS = {
    "india_macro": 1.0,
    "policy": 1.0,
    "global_macro": 1.0,
    "geopolitics": 1.0,
    "commodity_oil": 1.0,
    "banking_financial_stress": 1.0,
    "earnings_company_specific": 1.0,
}

CATEGORY_INDIA_BIAS_MULTIPLIERS = {
    "india_macro": 1.0,
    "policy": 1.0,
    "global_macro": 1.0,
    "geopolitics": 1.0,
    "commodity_oil": 1.0,
    "banking_financial_stress": 1.0,
    "earnings_company_specific": 1.0,
}

CATEGORY_GLOBAL_BIAS_MULTIPLIERS = {
    "india_macro": 1.0,
    "policy": 1.0,
    "global_macro": 1.0,
    "geopolitics": 1.0,
    "commodity_oil": 1.0,
    "banking_financial_stress": 1.0,
    "earnings_company_specific": 1.0,
}


def get_category_sentiment_multipliers():
    """
    Purpose:
        Return category sentiment multipliers for downstream use.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("keyword_category.sentiment", CATEGORY_SENTIMENT_MULTIPLIERS)


def get_category_vol_multipliers():
    """
    Purpose:
        Return category vol multipliers for downstream use.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("keyword_category.volatility", CATEGORY_VOL_MULTIPLIERS)


def get_category_impact_multipliers():
    """
    Purpose:
        Return category impact multipliers for downstream use.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("keyword_category.impact", CATEGORY_IMPACT_MULTIPLIERS)


def get_category_india_bias_multipliers():
    """
    Purpose:
        Return category india bias multipliers for downstream use.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("keyword_category.india_bias", CATEGORY_INDIA_BIAS_MULTIPLIERS)


def get_category_global_bias_multipliers():
    """
    Purpose:
        Return category global bias multipliers for downstream use.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("keyword_category.global_bias", CATEGORY_GLOBAL_BIAS_MULTIPLIERS)
