from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EVENT_TYPES = {
    "earnings_result",
    "guidance_revision",
    "large_order_win",
    "regulatory_action",
    "litigation_adverse_order",
    "management_change",
    "promoter_insider_activity",
    "merger_acquisition",
    "rating_action",
    "macro_event_sector_index",
    "government_policy_event",
    "block_bulk_deal",
    "rumor_unconfirmed_report",
    "unknown",
}

INSTRUMENT_SCOPES = {"single_stock", "sector", "index", "mixed"}
DIRECTION_LABELS = {"bullish", "bearish", "neutral", "mixed"}
VOL_IMPACT_LABELS = {"expansion", "compression", "neutral", "mixed"}
TIME_HORIZONS = {"intraday", "1_3_sessions", "1_2_weeks", "longer"}
CATALYST_QUALITY = {"low", "medium", "high"}


class EventSchemaValidationError(ValueError):
    """Raised when extracted event payload fails schema checks."""


@dataclass(frozen=True)
class EventIntelligenceRecord:
    event_type: str
    instrument_scope: str
    expected_direction: str
    directional_confidence: float
    vol_impact: str
    vol_confidence: float
    event_strength: float
    uncertainty_score: float
    gap_risk_score: float
    time_horizon: str
    catalyst_quality: str
    risk_flag: bool
    summary: str
    source: str = "rule_based"
    symbols: list[str] = field(default_factory=list)
    event_timestamp: str | None = None
    event_age_minutes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return max(0.0, min(1.0, out))


def validate_event_record(payload: dict[str, Any]) -> EventIntelligenceRecord:
    if not isinstance(payload, dict):
        raise EventSchemaValidationError("event payload must be a dict")

    event_type = str(payload.get("event_type") or "unknown").strip().lower()
    if event_type not in EVENT_TYPES:
        event_type = "unknown"

    instrument_scope = str(payload.get("instrument_scope") or "single_stock").strip().lower()
    if instrument_scope not in INSTRUMENT_SCOPES:
        instrument_scope = "mixed"

    expected_direction = str(payload.get("expected_direction") or "neutral").strip().lower()
    if expected_direction not in DIRECTION_LABELS:
        expected_direction = "mixed"

    vol_impact = str(payload.get("vol_impact") or "neutral").strip().lower()
    if vol_impact not in VOL_IMPACT_LABELS:
        vol_impact = "mixed"

    time_horizon = str(payload.get("time_horizon") or "1_3_sessions").strip().lower()
    if time_horizon not in TIME_HORIZONS:
        time_horizon = "1_3_sessions"

    catalyst_quality = str(payload.get("catalyst_quality") or "medium").strip().lower()
    if catalyst_quality not in CATALYST_QUALITY:
        catalyst_quality = "medium"

    summary = str(payload.get("summary") or "").strip()[:500]
    if not summary:
        summary = "No summary available"

    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        symbols = []
    symbols = [str(item).upper().strip() for item in symbols if str(item).strip()]

    source = str(payload.get("source") or "rule_based").strip().lower()
    event_timestamp = payload.get("event_timestamp")
    event_timestamp = str(event_timestamp).strip() if event_timestamp else None

    return EventIntelligenceRecord(
        event_type=event_type,
        instrument_scope=instrument_scope,
        expected_direction=expected_direction,
        directional_confidence=_clip01(payload.get("directional_confidence")),
        vol_impact=vol_impact,
        vol_confidence=_clip01(payload.get("vol_confidence")),
        event_strength=_clip01(payload.get("event_strength")),
        uncertainty_score=_clip01(payload.get("uncertainty_score")),
        gap_risk_score=_clip01(payload.get("gap_risk_score")),
        time_horizon=time_horizon,
        catalyst_quality=catalyst_quality,
        risk_flag=bool(payload.get("risk_flag", False)),
        summary=summary,
        source=source,
        symbols=symbols,
        event_timestamp=event_timestamp,
        event_age_minutes=max(0.0, float(payload.get("event_age_minutes") or 0.0)),
    )
