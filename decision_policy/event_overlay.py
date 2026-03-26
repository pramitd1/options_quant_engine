from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clip(value: Any, low: float, high: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = low
    return max(low, min(high, out))


@dataclass(frozen=True)
class EventOverlayDecision:
    probability_multiplier: float
    size_multiplier: float
    score_adjustment: int
    suppress_signal: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_event_overlay(
    *,
    direction: str | None,
    event_features: dict[str, Any] | None,
    enabled: bool = True,
    uncertainty_block_threshold: float = 0.88,
) -> EventOverlayDecision:
    if not enabled:
        return EventOverlayDecision(
            probability_multiplier=1.0,
            size_multiplier=1.0,
            score_adjustment=0,
            suppress_signal=False,
            reasons=["event_overlay_disabled"],
        )

    features = event_features if isinstance(event_features, dict) else {}
    if not features:
        return EventOverlayDecision(
            probability_multiplier=1.0,
            size_multiplier=1.0,
            score_adjustment=0,
            suppress_signal=False,
            reasons=["event_overlay_no_features"],
        )

    bullish = _clip(features.get("bullish_event_score", 0.0), 0.0, 1.0)
    bearish = _clip(features.get("bearish_event_score", 0.0), 0.0, 1.0)
    vol_expand = _clip(features.get("vol_expansion_score", 0.0), 0.0, 1.0)
    uncertainty = _clip(features.get("event_uncertainty_score", 0.0), 0.0, 1.0)
    gap_risk = _clip(features.get("gap_risk_score", 0.0), 0.0, 1.0)
    contradiction = _clip(features.get("contradictory_event_penalty", 0.0), 0.0, 1.0)
    alignment = _clip(features.get("catalyst_alignment_score", 0.0), 0.0, 1.0)
    routed_relevance = _clip(features.get("routed_event_relevance_score", 1.0), 0.0, 1.0)

    if routed_relevance < 0.2:
        return EventOverlayDecision(
            probability_multiplier=1.0,
            size_multiplier=1.0,
            score_adjustment=0,
            suppress_signal=False,
            reasons=["event_overlay_low_symbol_relevance"],
        )

    direction_norm = str(direction or "").upper().strip()
    directional_edge = 0.0
    if direction_norm == "CALL":
        directional_edge = bullish - bearish
    elif direction_norm == "PUT":
        directional_edge = bearish - bullish

    score_adjustment = int(round(directional_edge * 6.0 * routed_relevance))
    probability_multiplier = _clip(
        1.0 + (directional_edge * 0.22 * routed_relevance) - (contradiction * 0.30 * routed_relevance),
        0.0,
        1.25,
    )

    uncertainty_penalty = _clip(1.0 - (0.35 * uncertainty * routed_relevance) - (0.25 * gap_risk * routed_relevance), 0.4, 1.0)
    size_multiplier = _clip((1.0 + alignment * 0.12) * uncertainty_penalty, 0.35, 1.1)

    suppress_signal = uncertainty >= uncertainty_block_threshold and gap_risk >= 0.75
    reasons: list[str] = []
    if directional_edge > 0.15:
        reasons.append("event_catalyst_aligns_with_direction")
    elif directional_edge < -0.15:
        reasons.append("event_catalyst_contradicts_direction")
    if vol_expand >= 0.65:
        reasons.append("event_volatility_expansion_risk")
    if uncertainty >= 0.65:
        reasons.append("event_uncertainty_elevated")
    if gap_risk >= 0.65:
        reasons.append("event_gap_risk_elevated")
    if suppress_signal:
        size_multiplier = min(size_multiplier, 0.35)
        reasons.append("event_overlay_signal_suppressed")

    return EventOverlayDecision(
        probability_multiplier=round(probability_multiplier, 4),
        size_multiplier=round(size_multiplier, 4),
        score_adjustment=score_adjustment,
        suppress_signal=suppress_signal,
        reasons=reasons,
    )
