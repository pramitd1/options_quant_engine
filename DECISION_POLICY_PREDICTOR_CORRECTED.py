"""
ResearchDecisionPolicyPredictor — CORRECTED VERSION
=====================================================
Fixes the probability replacement bug. Now properly overlays policy
decisions on the ENGINE's probability, rather than replacing it with
ML model scores.

Key Changes:
  1. Use engine's hybrid_move_probability as THE base probability
  2. ML models provide decision-policy context (ALLOW/DOWNGRADE/BLOCK)
  3. Policy modifies the engine probability via multipliers, not replacement
  4. Add validation to keep probability in [0, 1]

RESEARCH ONLY — set PREDICTION_METHOD=research_decision_policy to activate.
"""
from __future__ import annotations

import logging
from typing import Any

from engine.predictors.inference_row import build_inference_row
from engine.predictors.protocol import MovePredictor, PredictionResult

logger = logging.getLogger(__name__)


class ResearchDecisionPolicyPredictor:
    """
    Decision-policy-aware predictor built on top of the dual-model layer
    (CORRECTED: uses engine probability as base, not ML replacement).
    """

    @property
    def name(self) -> str:
        return "research_decision_policy"

    def predict(self, market_ctx: dict[str, Any]) -> PredictionResult:
        from engine.trading_support.probability import (
            _compute_probability_state_impl,
        )

        # 1. Standard engine pipeline
        raw = _compute_probability_state_impl(**market_ctx)

        rule_prob = raw.get("rule_move_probability")
        ml_prob = raw.get("ml_move_probability")
        model_features = raw.get("model_features")
        components = raw.get("components", {})
        
        # ★ CRITICAL FIX: Keep engine probability as the base
        engine_hybrid_prob = raw.get("hybrid_move_probability")

        # 2. Research dual-model overlay (for policy context only)
        rank_score = None
        confidence_score = None
        try:
            from research.ml_models.ml_inference import infer_single

            result = infer_single(build_inference_row(market_ctx, raw))
            if result is not None:
                rank_score = result.ml_rank_score
                confidence_score = result.ml_confidence_score
        except Exception:
            logger.debug(
                "Research ML inference unavailable — falling back to engine probability",
                exc_info=True,
            )

        # 3. Apply the dual-threshold decision policy
        policy_decision = "ALLOW"
        policy_reason = "Policy not evaluated"
        size_multiplier = 1.0
        try:
            from research.decision_policy.policy_definitions import (
                dual_threshold_policy,
            )

            # Build a synthetic row for the policy function
            policy_row: dict[str, Any] = {
                "hybrid_move_probability": engine_hybrid_prob,
                "ml_rank_score": rank_score,
                "ml_confidence_score": confidence_score,
            }
            dec = dual_threshold_policy(policy_row)
            policy_decision = dec.decision
            policy_reason = dec.reason
            size_multiplier = dec.size_multiplier
        except Exception:
            logger.debug("Decision policy evaluation failed — defaulting to ALLOW", exc_info=True)

        # 4. ★ CORRECTED: Apply policy as OVERLAY on engine probability
        # Use engine probability as base, modify via policy decision
        effective_prob = engine_hybrid_prob
        
        if policy_decision == "BLOCK":
            effective_prob = 0.0
        elif policy_decision == "DOWNGRADE" and effective_prob is not None:
            # Policy applies sizing penalty, not ML re-penalization
            effective_prob = effective_prob * size_multiplier
        # ALLOW: leave engine probability unchanged

        # 5. Validate effective_prob stays in valid range
        if effective_prob is not None:
            effective_prob = max(0.0, min(1.0, effective_prob))

        # 6. Enrich components with full diagnostic detail
        components["research_rank_score"] = rank_score
        components["research_confidence_score"] = confidence_score
        components["engine_hybrid_probability"] = engine_hybrid_prob
        components["policy_decision"] = policy_decision
        components["policy_reason"] = policy_reason
        components["policy_size_multiplier"] = size_multiplier

        return PredictionResult(
            rule_move_probability=rule_prob,
            ml_move_probability=ml_prob,
            hybrid_move_probability=effective_prob,
            model_features=model_features,
            components=components,
            predictor_name=self.name,
        )
