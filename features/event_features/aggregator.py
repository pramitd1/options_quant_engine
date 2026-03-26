from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from nlp.impact_model.event_impact_model import apply_event_impact_model
from nlp.schemas.event_schema import EventIntelligenceRecord


def _clip01(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return max(0.0, min(1.0, out))


def _time_decay_multiplier(horizon: str) -> float:
    if horizon == "intraday":
        return 0.85
    if horizon == "1_3_sessions":
        return 1.0
    if horizon == "1_2_weeks":
        return 1.1
    return 1.05


@dataclass(frozen=True)
class EventFeatureState:
    bullish_event_score: float
    bearish_event_score: float
    vol_expansion_score: float
    vol_compression_score: float
    event_uncertainty_score: float
    gap_risk_score: float
    catalyst_alignment_score: float
    contradictory_event_penalty: float
    recent_event_cluster_score: float
    decayed_event_signal: float
    routed_event_relevance_score: float
    routed_event_count: int
    event_count: int
    explanation_lines: list[str] = field(default_factory=list)
    structured_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_event_features(
    events: list[EventIntelligenceRecord],
    *,
    direction_hint: str | None = None,
    underlying_symbol: str | None = None,
    as_of=None,
) -> EventFeatureState:
    if not events:
        return EventFeatureState(
            bullish_event_score=0.0,
            bearish_event_score=0.0,
            vol_expansion_score=0.0,
            vol_compression_score=0.0,
            event_uncertainty_score=0.0,
            gap_risk_score=0.0,
            catalyst_alignment_score=0.0,
            contradictory_event_penalty=0.0,
            recent_event_cluster_score=0.0,
            decayed_event_signal=0.0,
            routed_event_relevance_score=0.0,
            routed_event_count=0,
            event_count=0,
            explanation_lines=["No usable event text; event overlay neutralized."],
            structured_events=[],
        )

    adjusted = [apply_event_impact_model(item) for item in events]
    bullish = 0.0
    bearish = 0.0
    vol_expand = 0.0
    vol_compress = 0.0
    uncertainty = 0.0
    gap = 0.0
    strength_weight_sum = 0.0
    routing_weight_sum = 0.0
    routed_event_count = 0
    parsed_timestamp_cache: dict[str, pd.Timestamp | None] = {}
    age_minutes_cache: dict[str, float] = {}

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Kolkata")
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("Asia/Kolkata")

    symbol_norm = str(underlying_symbol or "").upper().strip()
    is_index_symbol = symbol_norm in {"NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "NIFTY50", "MIDCPNIFTY"}

    def _trading_minutes_between(start_ts: pd.Timestamp | None, end_ts: pd.Timestamp) -> float:
        if start_ts is None:
            return 0.0
        cur = start_ts
        if cur.tzinfo is None:
            cur = cur.tz_localize("Asia/Kolkata")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("Asia/Kolkata")
        if cur >= end_ts:
            return 0.0

        total = 0.0
        day = cur.normalize()
        end_day = end_ts.normalize()
        while day <= end_day:
            weekday = day.weekday()
            if weekday >= 5:
                day += pd.Timedelta(days=1)
                continue

            session_start = day + pd.Timedelta(hours=9, minutes=15)
            session_end = day + pd.Timedelta(hours=15, minutes=30)
            seg_start = max(cur, session_start)
            seg_end = min(end_ts, session_end)
            if seg_end > seg_start:
                total += (seg_end - seg_start).total_seconds() / 60.0
            day += pd.Timedelta(days=1)
        return max(0.0, total)

    def _half_life_m(item: EventIntelligenceRecord) -> float:
        if item.time_horizon == "intraday":
            return 90.0
        if item.time_horizon == "1_3_sessions":
            return 330.0
        if item.time_horizon == "1_2_weeks":
            return 1320.0
        return 2640.0

    def _routing_weight(item: EventIntelligenceRecord) -> float:
        if not symbol_norm:
            return 1.0
        if item.instrument_scope == "single_stock":
            symbol_match = symbol_norm in {s.upper() for s in item.symbols}
            if symbol_match:
                return 1.0
            return 0.25 if is_index_symbol else 0.15
        if item.instrument_scope == "sector":
            return 0.75 if is_index_symbol else 0.6
        if item.instrument_scope == "index":
            return 1.0 if is_index_symbol else 0.35
        return 0.7

    for item in adjusted:
        ts_key = str(item.event_timestamp or "")
        if ts_key not in age_minutes_cache:
            if ts_key not in parsed_timestamp_cache:
                event_ts = None
                if item.event_timestamp:
                    try:
                        event_ts = pd.Timestamp(item.event_timestamp)
                    except Exception:
                        event_ts = None
                parsed_timestamp_cache[ts_key] = event_ts
            age_minutes_cache[ts_key] = _trading_minutes_between(parsed_timestamp_cache[ts_key], as_of_ts)
        age_minutes = age_minutes_cache[ts_key]
        recency_decay = 0.5 ** (age_minutes / max(_half_life_m(item), 1.0))
        routing_w = _routing_weight(item)
        w = _clip01(item.event_strength) * _time_decay_multiplier(item.time_horizon) * recency_decay * routing_w
        strength_weight_sum += w
        routing_weight_sum += routing_w
        if routing_w >= 0.5:
            routed_event_count += 1

        if item.expected_direction == "bullish":
            bullish += w * item.directional_confidence
        elif item.expected_direction == "bearish":
            bearish += w * item.directional_confidence
        elif item.expected_direction == "mixed":
            bullish += w * 0.5 * item.directional_confidence
            bearish += w * 0.5 * item.directional_confidence

        if item.vol_impact == "expansion":
            vol_expand += w * item.vol_confidence
        elif item.vol_impact == "compression":
            vol_compress += w * item.vol_confidence
        elif item.vol_impact == "mixed":
            vol_expand += w * 0.5 * item.vol_confidence
            vol_compress += w * 0.5 * item.vol_confidence

        uncertainty += w * item.uncertainty_score
        gap += w * item.gap_risk_score

    denom = max(strength_weight_sum, 1e-6)
    bullish_score = _clip01(bullish / denom)
    bearish_score = _clip01(bearish / denom)
    vol_expand_score = _clip01(vol_expand / denom)
    vol_compress_score = _clip01(vol_compress / denom)
    uncertainty_score = _clip01(uncertainty / denom)
    gap_score = _clip01(gap / denom)

    cluster_score = _clip01(len(adjusted) / 5.0)
    directional_net = bullish_score - bearish_score
    decayed_event_signal = _clip01(abs(directional_net) * (1.0 - 0.35 * uncertainty_score))

    hint = str(direction_hint or "").upper().strip()
    alignment = 0.0
    contradiction = 0.0
    if hint == "CALL":
        alignment = bullish_score
        contradiction = bearish_score
    elif hint == "PUT":
        alignment = bearish_score
        contradiction = bullish_score

    alignment = _clip01(alignment * (1.0 - uncertainty_score * 0.25))
    contradiction = _clip01(contradiction * (0.6 + 0.4 * gap_score))
    routed_relevance_score = _clip01(routing_weight_sum / max(len(adjusted), 1))

    explanation_lines = [
        f"Bullish catalyst score={bullish_score:.2f}, bearish catalyst score={bearish_score:.2f}.",
        f"Volatility impact separated: expansion={vol_expand_score:.2f}, compression={vol_compress_score:.2f}.",
        f"Uncertainty={uncertainty_score:.2f}, gap risk={gap_score:.2f}, cluster={cluster_score:.2f}.",
        f"Routed relevance={routed_relevance_score:.2f} for symbol={symbol_norm or 'N/A'}.",
    ]

    return EventFeatureState(
        bullish_event_score=round(bullish_score, 4),
        bearish_event_score=round(bearish_score, 4),
        vol_expansion_score=round(vol_expand_score, 4),
        vol_compression_score=round(vol_compress_score, 4),
        event_uncertainty_score=round(uncertainty_score, 4),
        gap_risk_score=round(gap_score, 4),
        catalyst_alignment_score=round(alignment, 4),
        contradictory_event_penalty=round(contradiction, 4),
        recent_event_cluster_score=round(cluster_score, 4),
        decayed_event_signal=round(decayed_event_signal, 4),
        routed_event_relevance_score=round(routed_relevance_score, 4),
        routed_event_count=routed_event_count,
        event_count=len(adjusted),
        explanation_lines=explanation_lines,
        structured_events=[item.to_dict() for item in adjusted],
    )
