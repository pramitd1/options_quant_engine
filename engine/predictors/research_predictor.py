"""
ResearchDualModelPredictor — uses GBT (ranking) + LogReg (calibration).

Runs the standard engine pipeline to get features and components, then
replaces the probability estimate with the research dual-model inference.

RESEARCH ONLY — set PREDICTION_METHOD=research_dual_model to activate.
"""
from __future__ import annotations

import logging
from typing import Any

from engine.predictors.protocol import MovePredictor, PredictionResult

logger = logging.getLogger(__name__)


class ResearchDualModelPredictor:
    """
    Dual-model predictor using GBT for ranking and LogReg for calibration.

    Computes both the standard engine probabilities (for components/diagnostics)
    AND the research model scores, using the research score as hybrid output.
    """

    @property
    def name(self) -> str:
        return "research_dual_model"

    def predict(self, market_ctx: dict[str, Any]) -> PredictionResult:
        from engine.trading_support.probability import (
            _compute_probability_state_impl,
        )

        # Run the standard pipeline to get features and components.
        raw = _compute_probability_state_impl(**market_ctx)

        rule_prob = raw.get("rule_move_probability")
        ml_prob = raw.get("ml_move_probability")
        model_features = raw.get("model_features")
        components = raw.get("components", {})

        # Overlay research dual-model inference.
        research_prob = None
        rank_score = None
        confidence_score = None
        try:
            from research.ml_models.ml_inference import infer_single

            result = infer_single(model_features)
            if result is not None:
                rank_score = result.ml_rank_score
                confidence_score = result.ml_confidence_score
                # Use the calibrated confidence as hybrid probability.
                if confidence_score is not None:
                    research_prob = confidence_score
                elif rank_score is not None:
                    research_prob = rank_score
        except Exception:
            logger.debug("Research dual-model inference unavailable, falling back to engine blend", exc_info=True)

        # Enrich components with research scores.
        components["research_rank_score"] = rank_score
        components["research_confidence_score"] = confidence_score
        components["engine_hybrid_probability"] = raw.get("hybrid_move_probability")

        return PredictionResult(
            rule_move_probability=rule_prob,
            ml_move_probability=ml_prob,
            hybrid_move_probability=research_prob if research_prob is not None else raw.get("hybrid_move_probability"),
            model_features=model_features,
            components=components,
            predictor_name=self.name,
        )
