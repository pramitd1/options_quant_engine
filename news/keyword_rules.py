"""
Module: keyword_rules.py

Purpose:
    Implement keyword rules logic used to classify headlines and derive news context.

Role in the System:
    Part of the news context layer that scores headline risk and directional news pressure.

Key Outputs:
    Headline state, news sentiment features, and risk flags.

Downstream Usage:
    Consumed by macro/news overlays, the signal engine, and research logging.
"""

from __future__ import annotations

from config.news_keyword_policy import (
    HEADLINE_RULES,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    get_headline_rules,
    get_negative_keywords,
    get_positive_keywords,
)

__all__ = [
    "HEADLINE_RULES",
    "POSITIVE_KEYWORDS",
    "NEGATIVE_KEYWORDS",
    "get_headline_rules",
    "get_positive_keywords",
    "get_negative_keywords",
]
