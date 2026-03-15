"""
Deterministic keyword dictionaries for headline classification.
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
