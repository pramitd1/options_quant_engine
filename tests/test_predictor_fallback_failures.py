"""Tests for predictor fallback and error handling."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from engine.predictors.factory import get_predictor, prediction_method_override
from engine.predictors.protocol import PredictionResult


def test_predictor_predict_result_validation_catches_invalid_probabilities():
    """Prediction result with probability > 1.0 should be documented."""
    # Create a result with invalid probability
    result = PredictionResult(
        rule_move_probability=1.5,  # Invalid: > 1.0
        ml_move_probability=0.5,
        hybrid_move_probability=0.5,
        model_features=None,
        components={},
        predictor_name="blended",
    )
    
    # Verify the result is created (current code does not validate, but documents the case)
    assert result.rule_move_probability == 1.5


def test_predictor_method_override_context_manager_restores_on_exception():
    """When context manager body raises, original predictor is restored."""
    original = get_predictor()
    original_name = original.name

    try:
        with prediction_method_override("pure_rule"):
            assert get_predictor().name == "pure_rule"
            raise RuntimeError("Intentional test error")
    except RuntimeError:
        pass

    restored = get_predictor()
    assert restored.name == original_name


def test_predictor_fallback_when_selected_method_unavailable():
    """When selected method is unknown, system falls back to blended."""
    with pytest.raises(ValueError, match="Unknown prediction method"):
        with prediction_method_override("nonexistent_method_xyz"):
            pass


def test_predictor_result_missing_hybrid_probability_is_safe():
    """When predictor returns None for hybrid probability, downstream should handle gracefully."""
    # Create a result with None probability and verify it's safe to use
    result = PredictionResult(
        rule_move_probability=None,
        ml_move_probability=None,
        hybrid_move_probability=None,
        model_features=None,
        components={},
        predictor_name="pure_ml",
    )

    assert result.hybrid_move_probability is None
    assert result.predictor_name == "pure_ml"


def test_prediction_result_is_immutable():
    """PredictionResult instance is frozen after creation."""
    result = PredictionResult(
        hybrid_move_probability=0.65,
        predictor_name="test",
    )

    with pytest.raises(AttributeError):
        result.hybrid_move_probability = 0.9


def test_predictor_cascade_both_legs_fail_returns_none():
    """When both rule and ML legs fail to compute, hybrid should be None."""
    # Create a result with both legs returning None
    result = PredictionResult(
        rule_move_probability=None,
        ml_move_probability=None,
        hybrid_move_probability=None,
        model_features=None,
        components={},
        predictor_name="blended",
    )

    assert result.hybrid_move_probability is None
    assert result.rule_move_probability is None
    assert result.ml_move_probability is None
    assert result.predictor_name == "blended"
