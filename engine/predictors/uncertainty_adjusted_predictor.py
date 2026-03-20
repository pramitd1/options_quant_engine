"""
ResearchUncertaintyAdjustedPredictor
====================================
Research-only predictor that penalizes probability when uncertainty is high.

Uncertainty is modeled from:
  - disagreement between engine-hybrid and research confidence, and
  - confidence proximity to 0.50 (higher ambiguity near the midpoint).
"""
from __future__ import annotations

import logging
from typing import Any

from engine.predictors.inference_row import build_inference_row
from engine.predictors.protocol import MovePredictor, PredictionResult

logger = logging.getLogger(__name__)


class ResearchUncertaintyAdjustedPredictor:
    """Apply a deterministic uncertainty discount over research confidence."""

    disagreement_weight: float = 0.80
    ambiguity_weight: float = 0.35
    block_floor: float = 0.35

    @property
    def name(self) -> str:
        return "research_uncertainty_adjusted"

    def predict(self, market_ctx: dict[str, Any]) -> PredictionResult:
        from engine.trading_support.probability import _compute_probability_state_impl
        from research.ml_models.ml_inference import infer_single

        raw = _compute_probability_state_impl(**market_ctx)

        rule_prob = raw.get("rule_move_probability")
        ml_prob = raw.get("ml_move_probability")
        model_features = raw.get("model_features")
        components = raw.get("components", {})
        hybrid_prob = raw.get("hybrid_move_probability")

        rank_score = None
        confidence_score = None
        try:
            result = infer_single(build_inference_row(market_ctx, raw))
            if result is not None:
                rank_score = result.ml_rank_score
                confidence_score = result.ml_confidence_score
        except Exception:
            logger.debug("Uncertainty-adjusted inference unavailable; using engine blend", exc_info=True)

        if confidence_score is None:
            return PredictionResult(
                rule_move_probability=rule_prob,
                ml_move_probability=ml_prob,
                hybrid_move_probability=hybrid_prob,
                model_features=model_features,
                components={
                    **components,
                    "research_rank_score": rank_score,
                    "research_confidence_score": confidence_score,
                    "engine_hybrid_probability": hybrid_prob,
                    "uncertainty_multiplier": 1.0,
                    "uncertainty_blocked": False,
                },
                predictor_name=self.name,
            )

        baseline = float(hybrid_prob if hybrid_prob is not None else confidence_score)
        disagreement = abs(float(confidence_score) - baseline)
        ambiguity = 1.0 - abs((2.0 * float(confidence_score)) - 1.0)

        multiplier = 1.0 - (self.disagreement_weight * disagreement) - (self.ambiguity_weight * ambiguity)
        multiplier = max(0.0, min(1.0, multiplier))

        effective_prob = float(confidence_score) * multiplier
        blocked = effective_prob < self.block_floor
        if blocked:
            effective_prob = 0.0

        components["research_rank_score"] = rank_score
        components["research_confidence_score"] = confidence_score
        components["engine_hybrid_probability"] = hybrid_prob
        components["uncertainty_disagreement"] = round(disagreement, 6)
        components["uncertainty_ambiguity"] = round(ambiguity, 6)
        components["uncertainty_multiplier"] = round(multiplier, 6)
        components["uncertainty_block_floor"] = self.block_floor
        components["uncertainty_blocked"] = bool(blocked)

        return PredictionResult(
            rule_move_probability=rule_prob,
            ml_move_probability=ml_prob,
            hybrid_move_probability=effective_prob,
            model_features=model_features,
            components=components,
            predictor_name=self.name,
        )
