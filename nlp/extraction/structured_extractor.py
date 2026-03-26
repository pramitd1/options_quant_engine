from __future__ import annotations

import re
from typing import Any

import pandas as pd

from config.settings import EVENT_INTELLIGENCE_LLM_MIN_CONFIDENCE
from llm.extraction.event_llm_adapter import llm_extract_event_payload
from llm.validators.event_validator import validate_llm_event_payload
from nlp.classification.event_classifier import classify_event_type
from nlp.preprocessing.text_normalizer import preprocess_event_text
from nlp.schemas.event_schema import EventIntelligenceRecord, validate_event_record

_INDEX_TOKENS = {"nifty", "banknifty", "finnifty", "sensex", "nifty50"}
_SECTOR_TOKENS = {
    "bank",
    "banking",
    "it",
    "pharma",
    "auto",
    "metal",
    "energy",
    "fmcg",
    "realty",
    "psu",
}

_POSITIVE_WORDS = ("beat", "strong", "surge", "upgrade", "wins", "upside", "raises")
_NEGATIVE_WORDS = ("miss", "weak", "fall", "downgrade", "probe", "penalty", "cuts", "resigns")
_VOL_EXPAND_WORDS = ("volatility", "uncertain", "probe", "litigation", "penalty", "shock", "rumor")
_VOL_COMPRESS_WORDS = ("stable", "clarity", "resolved", "visibility")


def _extract_symbols(raw_text: str) -> list[str]:
    # NSE style tickers in uppercase, bounded length, e.g. RELIANCE, TCS, HDFCBANK.
    matches = re.findall(r"\b[A-Z]{2,15}\b", raw_text)
    unique: list[str] = []
    seen = set()
    for token in matches:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique[:5]


def _instrument_scope_from_text(text: str) -> str:
    words = set(text.split())
    if words & _INDEX_TOKENS:
        return "index"
    if words & _SECTOR_TOKENS:
        return "sector"
    return "single_stock"


def _direction_from_text(text: str) -> tuple[str, float]:
    pos = sum(1 for w in _POSITIVE_WORDS if w in text)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text)
    total = max(pos + neg, 1)
    conf = min(1.0, abs(pos - neg) / total + 0.35)
    if pos > neg:
        return "bullish", conf
    if neg > pos:
        return "bearish", conf
    if pos == 0 and neg == 0:
        return "neutral", 0.35
    return "mixed", 0.45


def _vol_from_text(text: str) -> tuple[str, float]:
    expand = sum(1 for w in _VOL_EXPAND_WORDS if w in text)
    compress = sum(1 for w in _VOL_COMPRESS_WORDS if w in text)
    total = max(expand + compress, 1)
    conf = min(1.0, abs(expand - compress) / total + 0.35)
    if expand > compress:
        return "expansion", conf
    if compress > expand:
        return "compression", conf
    if expand == 0 and compress == 0:
        return "neutral", 0.3
    return "mixed", 0.45


def _safe_iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return None


def extract_structured_event(
    *,
    text: str | None,
    timestamp: Any = None,
    source: str = "headline",
    llm_enabled: bool = False,
) -> EventIntelligenceRecord | None:
    normalized = preprocess_event_text(text)
    if not normalized:
        return None

    llm_payload = llm_extract_event_payload(text=str(text or ""), llm_enabled=llm_enabled)
    if isinstance(llm_payload, dict):
        try:
            validated = validate_llm_event_payload(llm_payload)
            llm_conf = max(
                float(validated.directional_confidence),
                float(validated.vol_confidence),
                float(validated.event_strength),
            )
            if llm_conf >= float(EVENT_INTELLIGENCE_LLM_MIN_CONFIDENCE):
                return EventIntelligenceRecord(
                    event_type=validated.event_type,
                    instrument_scope=validated.instrument_scope,
                    expected_direction=validated.expected_direction,
                    directional_confidence=validated.directional_confidence,
                    vol_impact=validated.vol_impact,
                    vol_confidence=validated.vol_confidence,
                    event_strength=validated.event_strength,
                    uncertainty_score=validated.uncertainty_score,
                    gap_risk_score=validated.gap_risk_score,
                    time_horizon=validated.time_horizon,
                    catalyst_quality=validated.catalyst_quality,
                    risk_flag=validated.risk_flag,
                    summary=validated.summary,
                    source="llm_openai",
                    symbols=validated.symbols,
                    event_timestamp=validated.event_timestamp,
                    event_age_minutes=validated.event_age_minutes,
                )
        except Exception:
            pass

    event_type = classify_event_type(normalized)
    direction, direction_conf = _direction_from_text(normalized)
    vol_impact, vol_conf = _vol_from_text(normalized)

    uncertainty_score = 0.35
    gap_risk_score = 0.25
    if event_type in {"regulatory_action", "litigation_adverse_order", "rumor_unconfirmed_report"}:
        uncertainty_score = 0.8
        gap_risk_score = 0.75
    elif event_type in {"earnings_result", "guidance_revision", "merger_acquisition"}:
        uncertainty_score = 0.55
        gap_risk_score = 0.6

    if vol_impact == "expansion":
        uncertainty_score = min(1.0, uncertainty_score + 0.1)
        gap_risk_score = min(1.0, gap_risk_score + 0.1)

    payload = {
        "event_type": event_type,
        "instrument_scope": _instrument_scope_from_text(normalized),
        "expected_direction": direction,
        "directional_confidence": direction_conf,
        "vol_impact": vol_impact,
        "vol_confidence": vol_conf,
        "event_strength": min(1.0, (direction_conf + vol_conf) * 0.5),
        "uncertainty_score": uncertainty_score,
        "gap_risk_score": gap_risk_score,
        "time_horizon": "1_3_sessions",
        "catalyst_quality": "medium",
        "risk_flag": uncertainty_score >= 0.7 or gap_risk_score >= 0.7,
        "summary": str(text or "").strip()[:280],
        "source": source,
        "symbols": _extract_symbols(str(text or "")),
        "event_timestamp": _safe_iso_timestamp(timestamp),
        "event_age_minutes": 0.0,
    }
    return validate_event_record(payload)
