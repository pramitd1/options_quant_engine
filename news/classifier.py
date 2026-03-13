"""
Deterministic headline classification and scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from macro.macro_news_config import HEADLINE_CLASSIFICATION_CONFIG
from news.keyword_rules import HEADLINE_RULES, NEGATIVE_KEYWORDS, POSITIVE_KEYWORDS
from news.models import HeadlineRecord


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class HeadlineClassification:
    timestamp: str
    source: str
    headline: str
    url_or_identifier: str
    primary_category: str
    matched_categories: list[str]
    matched_rules: list[str]
    matched_keywords: list[str]
    macro_sentiment_score: float
    volatility_shock_score: float
    headline_impact_score: float
    india_macro_bias: float
    global_risk_bias: float
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_headline(record: HeadlineRecord) -> HeadlineClassification:
    cfg = HEADLINE_CLASSIFICATION_CONFIG
    text = record.headline.lower()
    matched_rules = []
    matched_categories = []
    matched_keywords = []
    matched_vol_weights = []

    sentiment = 0.0
    vol = 0.0
    impact = 0.0
    india_bias = 0.0
    global_bias = 0.0

    for rule in HEADLINE_RULES:
        keywords = [keyword for keyword in rule["keywords"] if keyword in text]
        if not keywords:
            continue

        matched_rules.append(rule["name"])
        matched_categories.append(rule["category"])
        matched_keywords.extend(keywords)
        sentiment += float(rule["sentiment_weight"])
        matched_vol_weights.append(float(rule["vol_weight"]))
        impact = max(impact, float(rule["impact_score"]))
        india_bias += float(rule["india_macro_bias"])
        global_bias += float(rule["global_risk_bias"])

    positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
    negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
    sentiment += cfg.positive_hit_weight * positive_hits
    sentiment -= cfg.negative_hit_weight * negative_hits

    if not matched_categories:
        primary_category = "uncategorized"
        impact = cfg.uncategorized_keyword_impact if positive_hits or negative_hits else 0.0
    else:
        category_order = {rule["category"]: idx for idx, rule in enumerate(HEADLINE_RULES)}
        primary_category = sorted(set(matched_categories), key=lambda cat: category_order.get(cat, 999))[0]

    sentiment = _clip(sentiment, -1.0, 1.0)
    base_vol = max(matched_vol_weights) if matched_vol_weights else 0.0
    vol = _clip(base_vol + cfg.volatility_keyword_bonus * max(positive_hits, negative_hits), 0.0, 1.0)
    india_bias = _clip(
        india_bias + cfg.india_bias_hit_weight * positive_hits - cfg.india_bias_hit_weight * negative_hits,
        -1.0,
        1.0,
    )
    global_bias = _clip(
        global_bias
        + cfg.global_bias_positive_hit_weight * positive_hits
        - cfg.global_bias_negative_hit_weight * negative_hits,
        -1.0,
        1.0,
    )
    impact = _clip(impact, 0.0, 100.0)

    return HeadlineClassification(
        timestamp=record.timestamp.isoformat(),
        source=record.source,
        headline=record.headline,
        url_or_identifier=record.url_or_identifier,
        primary_category=primary_category,
        matched_categories=sorted(set(matched_categories)),
        matched_rules=matched_rules,
        matched_keywords=sorted(set(matched_keywords)),
        macro_sentiment_score=round(sentiment * 100.0, 2),
        volatility_shock_score=round(vol * 100.0, 2),
        headline_impact_score=round(impact, 2),
        india_macro_bias=round(india_bias, 4),
        global_risk_bias=round(global_bias, 4),
        debug={
            "positive_keyword_hits": positive_hits,
            "negative_keyword_hits": negative_hits,
        },
    )


def classify_headlines(records: list[HeadlineRecord]) -> list[HeadlineClassification]:
    return [classify_headline(record) for record in records]
