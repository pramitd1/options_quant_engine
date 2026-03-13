"""
Deterministic keyword dictionaries for headline classification.
"""

from __future__ import annotations


HEADLINE_RULES = [
    {
        "name": "india_macro_inflation",
        "category": "india_macro",
        "keywords": [
            "india cpi", "cpi inflation", "wpi inflation", "retail inflation",
            "inflation cools", "inflation eases", "inflation rises",
            "inflation surprise", "inflation shock", "core inflation",
        ],
        "sentiment_weight": 0.0,
        "vol_weight": 0.45,
        "impact_score": 70,
        "india_macro_bias": 0.0,
        "global_risk_bias": 0.0,
    },
    {
        "name": "india_growth",
        "category": "india_macro",
        "keywords": [
            "gdp", "iip", "industrial production", "manufacturing output",
            "economic growth", "growth slows", "growth picks up", "services activity",
            "pmi", "factory output",
        ],
        "sentiment_weight": 0.1,
        "vol_weight": 0.35,
        "impact_score": 65,
        "india_macro_bias": 0.2,
        "global_risk_bias": 0.0,
    },
    {
        "name": "policy_central_bank",
        "category": "policy",
        "keywords": [
            "rbi", "repo rate", "monetary policy", "fomc", "fed", "ecb", "boj",
            "rate hike", "rate cut", "policy decision", "governor", "hawkish",
            "dovish", "liquidity measures",
        ],
        "sentiment_weight": 0.0,
        "vol_weight": 0.7,
        "impact_score": 85,
        "india_macro_bias": 0.0,
        "global_risk_bias": 0.0,
    },
    {
        "name": "global_macro",
        "category": "global_macro",
        "keywords": [
            "us cpi", "us inflation", "payrolls", "nonfarm payrolls", "treasury yields",
            "bond yields", "global growth", "recession fears", "soft landing",
            "hard landing", "yield curve", "risk appetite",
        ],
        "sentiment_weight": 0.0,
        "vol_weight": 0.55,
        "impact_score": 75,
        "india_macro_bias": 0.0,
        "global_risk_bias": 0.0,
    },
    {
        "name": "geopolitics",
        "category": "geopolitics",
        "keywords": [
            "war", "missile", "sanction", "border tension", "conflict", "attack",
            "ceasefire", "tariff", "trade war", "military strike", "troop buildup",
            "shipping disruption",
        ],
        "sentiment_weight": -0.25,
        "vol_weight": 0.8,
        "impact_score": 90,
        "india_macro_bias": -0.1,
        "global_risk_bias": -0.75,
    },
    {
        "name": "commodity_oil",
        "category": "commodity_oil",
        "keywords": [
            "oil", "crude", "brent", "wti", "opec", "gas prices", "commodity prices",
            "energy prices", "oil spikes", "crude jumps",
        ],
        "sentiment_weight": 0.0,
        "vol_weight": 0.5,
        "impact_score": 70,
        "india_macro_bias": -0.15,
        "global_risk_bias": -0.1,
    },
    {
        "name": "banking_financial_stress",
        "category": "banking_financial_stress",
        "keywords": [
            "bank stress", "bank run", "liquidity crunch", "default", "credit event",
            "solvency", "financial stability", "npa", "funding stress", "credit spreads",
            "contagion", "counterparty risk",
        ],
        "sentiment_weight": -0.35,
        "vol_weight": 0.85,
        "impact_score": 95,
        "india_macro_bias": -0.2,
        "global_risk_bias": -0.85,
    },
    {
        "name": "earnings_company_specific",
        "category": "earnings_company_specific",
        "keywords": [
            "earnings", "guidance", "results", "revenue", "ebitda", "profit warning",
            "buyback", "merger", "acquisition", "board meeting", "order book",
            "margin outlook", "capex", "management commentary",
        ],
        "sentiment_weight": 0.0,
        "vol_weight": 0.35,
        "impact_score": 55,
        "india_macro_bias": 0.0,
        "global_risk_bias": 0.0,
    },
]


POSITIVE_KEYWORDS = [
    "eases", "cools", "beats", "upgrade", "surplus", "strong growth", "soft landing",
    "stimulus", "rate cut", "ceasefire", "supportive", "improves", "recovers",
    "liquidity support", "contained", "disinflation", "stable outlook", "resilient",
]


NEGATIVE_KEYWORDS = [
    "surges", "jumps", "misses", "downgrade", "stress", "crunch", "default",
    "war", "attack", "hike", "cuts outlook", "profit warning", "recession",
    "inflation rises", "yields spike", "risk-off", "shock", "selloff",
    "panic", "funding stress", "conflict escalates", "volatility surges",
]
