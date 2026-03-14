"""
Category-level multipliers for headline rule tuning.

This deliberately tunes category behavior rather than individual keywords to
reduce overfitting and preserve interpretability.
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
    from tuning.runtime import resolve_mapping

    return resolve_mapping("keyword_category.sentiment", CATEGORY_SENTIMENT_MULTIPLIERS)


def get_category_vol_multipliers():
    from tuning.runtime import resolve_mapping

    return resolve_mapping("keyword_category.volatility", CATEGORY_VOL_MULTIPLIERS)


def get_category_impact_multipliers():
    from tuning.runtime import resolve_mapping

    return resolve_mapping("keyword_category.impact", CATEGORY_IMPACT_MULTIPLIERS)


def get_category_india_bias_multipliers():
    from tuning.runtime import resolve_mapping

    return resolve_mapping("keyword_category.india_bias", CATEGORY_INDIA_BIAS_MULTIPLIERS)


def get_category_global_bias_multipliers():
    from tuning.runtime import resolve_mapping

    return resolve_mapping("keyword_category.global_bias", CATEGORY_GLOBAL_BIAS_MULTIPLIERS)
