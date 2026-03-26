"""
Module: macro_news_aggregator.py

Purpose:
    Implement macro news aggregator logic used to score scheduled events and macro catalysts.

Role in the System:
    Part of the macro context layer that scores scheduled events and broad market catalysts.

Key Outputs:
    Macro-event state, catalyst scores, and gating diagnostics.

Downstream Usage:
    Consumed by the signal engine, risk overlays, and research diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from config.settings import EVENT_INTELLIGENCE_ENABLED, EVENT_INTELLIGENCE_LLM_ENABLED
from features.event_features import aggregate_event_features
from macro.macro_news_config import (
    get_macro_news_aggregation_config,
    get_macro_news_regime_config,
)
from nlp.extraction.structured_extractor import extract_structured_event
from nlp.ingestion.event_ingestion import build_raw_event_payloads
from news.classifier import HeadlineClassification, classify_headlines
from news.models import HeadlineIngestionState
from utils.numerics import clip as _clip, safe_float as _safe_float  # noqa: F401


@dataclass(frozen=True)
class MacroNewsState:
    """
    Purpose:
        Dataclass representing MacroNewsState within the repository.
    
    Context:
        Used within the macro context layer that scores scheduled events and broad catalysts. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        macro_regime (str): Regime label for macro.
        macro_event_risk_score (int): Score value for macro event risk.
        macro_sentiment_score (float): Score value for macro sentiment.
        volatility_shock_score (float): Score value for volatility shock.
        event_lockdown_flag (bool): Boolean flag controlling whether event lockdown flag is active.
        news_confidence_score (float): Score value for news confidence.
        headline_impact_score (float): Score value for headline impact.
        india_macro_bias (float): Value supplied for india macro bias.
        global_risk_bias (float): Value supplied for global risk bias.
        headline_velocity (float): Value supplied for headline velocity.
        headline_count (int): Count recorded for headline.
        classified_headline_count (int): Count recorded for classified headline.
        event_window_status (str): Macro-event window state used by the current heuristic.
        next_event_name (str | None): Value supplied for next event name.
        neutral_fallback (bool): Boolean flag controlling whether neutral fallback is active.
        macro_regime_reasons (list[str]): Human-readable explanations for macro regime.
        issues (list[str]): Collection of issues.
        warnings (list[str]): Collection of warnings.
        classification_preview (list[dict[str, Any]]): Collection of classification preview.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
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
    event_intelligence_enabled: bool = False
    event_features: dict[str, Any] = field(default_factory=dict)
    event_explanations: list[str] = field(default_factory=list)
    structured_event_preview: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `MacroNewsState` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return asdict(self)


def _neutral_macro_news_state(event_state: dict | None = None, issues=None, warnings=None) -> MacroNewsState:
    """
    Purpose:
        Process neutral macro news state for downstream use.
    
    Context:
        Internal helper within the macro context layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        event_state (dict | None): Structured state payload for event.
        issues (Any): Input associated with issues.
        warnings (Any): Input associated with warnings.
    
    Returns:
        MacroNewsState: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
        event_intelligence_enabled=False,
        event_features={},
        event_explanations=["event_intelligence_neutral_fallback"],
        structured_event_preview=[],
    )


def _build_event_intelligence_state(
    *,
    headline_state: HeadlineIngestionState,
    direction_hint: str | None = None,
    underlying_symbol: str | None = None,
    as_of=None,
) -> dict[str, Any]:
    if not EVENT_INTELLIGENCE_ENABLED:
        return {
            "event_intelligence_enabled": False,
            "event_features": {},
            "event_explanations": ["event_intelligence_disabled"],
            "structured_event_preview": [],
        }

    raw_events = build_raw_event_payloads(headline_state.records)
    structured = []
    for item in raw_events:
        record = extract_structured_event(
            text=item.get("text"),
            timestamp=item.get("timestamp"),
            source=str(item.get("source") or "headline"),
            llm_enabled=EVENT_INTELLIGENCE_LLM_ENABLED,
        )
        if record is not None:
            structured.append(record)

    feature_state = aggregate_event_features(
        structured,
        direction_hint=direction_hint,
        underlying_symbol=underlying_symbol,
        as_of=as_of,
    )
    return {
        "event_intelligence_enabled": True,
        "event_features": feature_state.to_dict(),
        "event_explanations": list(feature_state.explanation_lines),
        "structured_event_preview": feature_state.structured_events[:5],
    }


def _weighted_headline_aggregates(classified: list[HeadlineClassification], as_of=None):
    """
    Purpose:
        Process weighted headline aggregates for downstream use.
    
    Context:
        Internal helper within the macro context layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        classified (list[HeadlineClassification]): Input associated with classified.
        as_of (Any): Input associated with as of.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    cfg = get_macro_news_aggregation_config()
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
    """
    Purpose:
        Derive macro regime from the supplied inputs.
    
    Context:
        Internal helper within the macro context layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        event_lockdown_flag (bool): Boolean flag indicating whether scheduled-event rules require a hard lockdown.
        macro_sentiment_score (float): Score value for macro sentiment.
        global_risk_bias (float): Input associated with global risk bias.
        india_macro_bias (float): Input associated with india macro bias.
        volatility_shock_score (float): Score value for volatility shock.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    cfg = get_macro_news_regime_config()
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
    symbol: str | None = None,
) -> MacroNewsState:
    """
    Purpose:
        Build the macro news state used by downstream components.
    
    Context:
        Public function within the macro context layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        event_state (dict | None): Structured state payload for event.
        headline_state (HeadlineIngestionState | None): Structured state payload for headline.
        as_of (Any): Input associated with as of.
    
    Returns:
        MacroNewsState: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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

    event_intel_state = _build_event_intelligence_state(
        headline_state=headline_state,
        direction_hint=None,
        underlying_symbol=symbol,
        as_of=as_of or headline_state.fetched_at,
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
        event_intelligence_enabled=bool(event_intel_state["event_intelligence_enabled"]),
        event_features=event_intel_state["event_features"],
        event_explanations=event_intel_state["event_explanations"],
        structured_event_preview=event_intel_state["structured_event_preview"],
    )
