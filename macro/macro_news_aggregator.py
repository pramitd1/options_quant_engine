"""
Aggregate scheduled event risk and classified headlines into a compact macro regime state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from macro.macro_news_config import (
    MACRO_NEWS_AGGREGATION_CONFIG,
    MACRO_NEWS_REGIME_CONFIG,
)
from news.classifier import HeadlineClassification, classify_headlines
from news.models import HeadlineIngestionState


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


@dataclass(frozen=True)
class MacroNewsState:
    macro_regime: str
    macro_event_risk_score: int
    macro_sentiment_score: float
    volatility_shock_score: float
    event_lockdown_flag: bool
    news_confidence_score: float
    headline_impact_score: float
    india_macro_bias: float
    global_risk_bias: float
    headline_velocity: float
    headline_count: int
    classified_headline_count: int
    event_window_status: str
    next_event_name: str | None
    neutral_fallback: bool
    macro_regime_reasons: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    classification_preview: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _neutral_macro_news_state(event_state: dict | None = None, issues=None, warnings=None) -> MacroNewsState:
    event_state = event_state or {}
    return MacroNewsState(
        macro_regime="MACRO_NEUTRAL" if not event_state.get("event_lockdown_flag") else "EVENT_LOCKDOWN",
        macro_event_risk_score=int(_safe_float(event_state.get("macro_event_risk_score"), 0)),
        macro_sentiment_score=0.0,
        volatility_shock_score=0.0,
        event_lockdown_flag=bool(event_state.get("event_lockdown_flag", False)),
        news_confidence_score=0.0,
        headline_impact_score=0.0,
        india_macro_bias=0.0,
        global_risk_bias=0.0,
        headline_velocity=0.0,
        headline_count=0,
        classified_headline_count=0,
        event_window_status=event_state.get("event_window_status", "NO_EVENT_DATA"),
        next_event_name=event_state.get("next_event_name"),
        neutral_fallback=True,
        macro_regime_reasons=["neutral_fallback"],
        issues=issues or [],
        warnings=warnings or [],
        classification_preview=[],
    )


def _weighted_headline_aggregates(classified: list[HeadlineClassification], as_of=None):
    cfg = MACRO_NEWS_AGGREGATION_CONFIG
    if not classified:
        return {
            "macro_sentiment_score": 0.0,
            "volatility_shock_score": 0.0,
            "headline_impact_score": 0.0,
            "india_macro_bias": 0.0,
            "global_risk_bias": 0.0,
            "headline_velocity": 0.0,
            "news_confidence_score": 0.0,
            "classification_preview": [],
        }

    now_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Kolkata")
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("Asia/Kolkata")

    weighted_sentiment = 0.0
    weighted_vol = 0.0
    weighted_impact = 0.0
    weighted_india_bias = 0.0
    weighted_global_bias = 0.0
    total_weight = 0.0
    recent_count = 0

    preview = []

    for item in classified:
        ts = pd.Timestamp(item.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")

        age_minutes = max((now_ts - ts).total_seconds() / 60.0, 0.0)
        decay = 1.0 / (1.0 + age_minutes / max(cfg.decay_half_life_minutes, 1.0))
        impact_weight = max(item.headline_impact_score / 100.0, 0.1)
        weight = decay * impact_weight

        weighted_sentiment += item.macro_sentiment_score * weight
        weighted_vol += item.volatility_shock_score * weight
        weighted_impact += item.headline_impact_score * weight
        weighted_india_bias += item.india_macro_bias * weight
        weighted_global_bias += item.global_risk_bias * weight
        total_weight += weight

        if age_minutes <= cfg.headline_burst_lookback_minutes:
            recent_count += 1

        if len(preview) < 5:
            preview.append(item.to_dict())

    if total_weight <= 0:
        return {
            "macro_sentiment_score": 0.0,
            "volatility_shock_score": 0.0,
            "headline_impact_score": 0.0,
            "india_macro_bias": 0.0,
            "global_risk_bias": 0.0,
            "headline_velocity": 0.0,
            "news_confidence_score": 0.0,
            "classification_preview": preview,
        }

    headline_velocity = _clip(recent_count / max(cfg.headline_velocity_base_count, 1), 0.0, 1.0)
    news_confidence = _clip(
        (len(classified) / max(cfg.confidence_count_divisor, 1.0)) * cfg.confidence_count_weight
        + headline_velocity * cfg.confidence_velocity_weight
        + min(total_weight, 1.0) * cfg.confidence_total_weight_weight,
        0.0,
        1.0,
    )

    return {
        "macro_sentiment_score": round(weighted_sentiment / total_weight, 2),
        "volatility_shock_score": round(weighted_vol / total_weight, 2),
        "headline_impact_score": round(weighted_impact / total_weight, 2),
        "india_macro_bias": round(weighted_india_bias / total_weight, 4),
        "global_risk_bias": round(weighted_global_bias / total_weight, 4),
        "headline_velocity": round(headline_velocity, 4),
        "news_confidence_score": round(news_confidence * 100.0, 2),
        "classification_preview": preview,
    }


def _derive_macro_regime(*, event_lockdown_flag: bool, macro_sentiment_score: float, global_risk_bias: float, india_macro_bias: float, volatility_shock_score: float):
    cfg = MACRO_NEWS_REGIME_CONFIG
    reasons = []
    if event_lockdown_flag:
        return "EVENT_LOCKDOWN", ["event_lockdown"]

    risk_off = (
        macro_sentiment_score <= cfg.sentiment_off_threshold
        or global_risk_bias <= -cfg.risk_bias_threshold
        or india_macro_bias <= -cfg.risk_bias_threshold
        or volatility_shock_score >= cfg.risk_off_vol_shock_min
    )
    if risk_off:
        if macro_sentiment_score <= cfg.sentiment_off_threshold:
            reasons.append("sentiment_risk_off")
        if global_risk_bias <= -cfg.risk_bias_threshold:
            reasons.append("global_risk_bias_negative")
        if india_macro_bias <= -cfg.risk_bias_threshold:
            reasons.append("india_macro_bias_negative")
        if volatility_shock_score >= cfg.risk_off_vol_shock_min:
            reasons.append("volatility_shock_high")
        return "RISK_OFF", reasons

    risk_on = (
        macro_sentiment_score >= cfg.sentiment_on_threshold
        and global_risk_bias >= cfg.risk_bias_threshold
        and volatility_shock_score < cfg.risk_on_vol_shock_max
    )
    if risk_on:
        return "RISK_ON", ["sentiment_risk_on", "global_risk_bias_positive", "volatility_shock_contained"]

    return "MACRO_NEUTRAL", ["mixed_or_weak_macro_news"]


def build_macro_news_state(
    *,
    event_state: dict | None,
    headline_state: HeadlineIngestionState | None,
    as_of=None,
) -> MacroNewsState:
    event_state = event_state or {}

    if headline_state is None:
        return _neutral_macro_news_state(
            event_state=event_state,
            warnings=["headline_state_missing"],
        )

    if headline_state.neutral_fallback or not headline_state.data_available:
        return _neutral_macro_news_state(
            event_state=event_state,
            warnings=list(headline_state.warnings),
            issues=list(headline_state.issues),
        )

    classified = classify_headlines(headline_state.records)
    aggregates = _weighted_headline_aggregates(classified, as_of=as_of or headline_state.fetched_at)

    macro_event_risk_score = int(_safe_float(event_state.get("macro_event_risk_score"), 0))
    event_lockdown_flag = bool(event_state.get("event_lockdown_flag", False))
    # Keep event risk explicit and separate. The engine already handles
    # scheduled-event penalties and lockdown logic independently.
    volatility_shock_score = _clip(aggregates["volatility_shock_score"], 0.0, 100.0)

    macro_regime, macro_regime_reasons = _derive_macro_regime(
        event_lockdown_flag=event_lockdown_flag,
        macro_sentiment_score=aggregates["macro_sentiment_score"],
        global_risk_bias=aggregates["global_risk_bias"],
        india_macro_bias=aggregates["india_macro_bias"],
        volatility_shock_score=volatility_shock_score,
    )

    return MacroNewsState(
        macro_regime=macro_regime,
        macro_event_risk_score=macro_event_risk_score,
        macro_sentiment_score=aggregates["macro_sentiment_score"],
        volatility_shock_score=round(volatility_shock_score, 2),
        event_lockdown_flag=event_lockdown_flag,
        news_confidence_score=aggregates["news_confidence_score"],
        headline_impact_score=aggregates["headline_impact_score"],
        india_macro_bias=aggregates["india_macro_bias"],
        global_risk_bias=aggregates["global_risk_bias"],
        headline_velocity=aggregates["headline_velocity"],
        headline_count=len(headline_state.records),
        classified_headline_count=len(classified),
        event_window_status=event_state.get("event_window_status", "NO_EVENT_DATA"),
        next_event_name=event_state.get("next_event_name"),
        neutral_fallback=False,
        macro_regime_reasons=macro_regime_reasons,
        issues=list(headline_state.issues),
        warnings=list(headline_state.warnings),
        classification_preview=aggregates["classification_preview"],
    )
