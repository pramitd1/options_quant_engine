"""
ResearchRankGatePredictor
=========================
Research-only predictor that applies a rank-score gate on top of the
dual-model inference path.

Behavior:
  1. Run standard engine probability pipeline.
  2. Run research dual-model inference.
  3. BLOCK low-rank signals by clamping probability to 0.0.
  4. Use calibrated confidence (when available) for allowed signals.
"""
from __future__ import annotations

import logging
from typing import Any

from engine.predictors.inference_row import build_inference_row
from engine.predictors.protocol import MovePredictor, PredictionResult

logger = logging.getLogger(__name__)


class ResearchRankGatePredictor:
    """Research predictor that blocks signals below a rank threshold."""

    rank_threshold: float = 0.55

    @property
    def name(self) -> str:
        return "research_rank_gate"

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
        policy_decision = "MISSING"
        policy_reason = "rank_score_unavailable"

        try:
            result = infer_single(build_inference_row(market_ctx, raw))
            if result is not None:
                rank_score = result.ml_rank_score
                confidence_score = result.ml_confidence_score
        except Exception:
            logger.debug("Research rank-gate inference failed; using engine probability", exc_info=True)

        effective_prob = confidence_score if confidence_score is not None else hybrid_prob

        if rank_score is not None:
            if rank_score < self.rank_threshold:
                policy_decision = "BLOCK"
                policy_reason = f"rank_below_threshold({rank_score:.4f}<{self.rank_threshold:.2f})"
                effective_prob = 0.0
            else:
                policy_decision = "ALLOW"
                policy_reason = f"rank_pass({rank_score:.4f}>={self.rank_threshold:.2f})"

        components["research_rank_score"] = rank_score
        components["research_confidence_score"] = confidence_score
        components["engine_hybrid_probability"] = hybrid_prob
        components["rank_gate_threshold"] = self.rank_threshold
        components["rank_gate_decision"] = policy_decision
        components["rank_gate_reason"] = policy_reason

        return PredictionResult(
            rule_move_probability=rule_prob,
            ml_move_probability=ml_prob,
            hybrid_move_probability=effective_prob,
            model_features=model_features,
            components=components,
            predictor_name=self.name,
        )
