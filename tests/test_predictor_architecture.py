"""Tests for the pluggable predictor architecture."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.predictors.protocol import MovePredictor, PredictionResult
from engine.predictors.factory import (
    get_predictor,
    reset_predictor,
    prediction_method_override,
    _ensure_registry,
)
from engine.predictors.builtin_predictors import (
    DefaultBlendedPredictor,
    PureMLPredictor,
    PureRulePredictor,
)
from engine.predictors.research_predictor import ResearchDualModelPredictor
from engine.predictors.ev_sizing_predictor import EVSizingPredictor
from engine.predictors.rank_gate_predictor import ResearchRankGatePredictor
from engine.predictors.uncertainty_adjusted_predictor import ResearchUncertaintyAdjustedPredictor


def test_registry_contains_all_methods():
    registry = _ensure_registry()
    assert "blended" in registry
    assert "pure_ml" in registry
    assert "pure_rule" in registry
    assert "research_dual_model" in registry
    assert "decision_policy" in registry
    assert "research_decision_policy" in registry
    assert "ev_sizing" in registry
    assert "research_rank_gate" in registry
    assert "research_uncertainty_adjusted" in registry


def test_default_predictor_is_blended():
    """Default production method is decision policy, backed by strict leaderboard cuts."""
    reset_predictor()
    p = get_predictor()
    assert p.name == "research_decision_policy"
    assert isinstance(p, MovePredictor)


def test_prediction_method_override_swaps_and_restores():
    reset_predictor()
    original = get_predictor()
    assert original.name == "research_decision_policy"

    with prediction_method_override("pure_rule") as pred:
        assert pred.name == "pure_rule"
        # get_predictor inside the context should return the override
        assert get_predictor().name == "pure_rule"

    # After exit, the original (decision policy) is restored
    assert get_predictor().name == "research_decision_policy"


def test_prediction_method_override_invalid_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown prediction method"):
        with prediction_method_override("nonexistent"):
            pass


def test_prediction_result_is_frozen():
    r = PredictionResult(hybrid_move_probability=0.65, predictor_name="test")
    assert r.hybrid_move_probability == 0.65
    import pytest
    with pytest.raises(AttributeError):
        r.hybrid_move_probability = 0.9


def test_all_predictors_satisfy_protocol():
    for cls in [
        DefaultBlendedPredictor,
        PureMLPredictor,
        PureRulePredictor,
        ResearchDualModelPredictor,
        EVSizingPredictor,
        ResearchRankGatePredictor,
        ResearchUncertaintyAdjustedPredictor,
    ]:
        inst = cls()
        assert isinstance(inst, MovePredictor), f"{cls.__name__} fails MovePredictor protocol"
        assert isinstance(inst.name, str)


def test_blended_does_not_mimic_pure_rule_when_ml_available(monkeypatch):
    """Blended should carry ML influence; pure_rule must not."""
    from engine.predictors.builtin_predictors import DefaultBlendedPredictor, PureRulePredictor
    import engine.trading_support.probability as prob

    calls = []

    def fake_impl(*args, **kwargs):
        calls.append(dict(kwargs))
        rule = 0.62
        ml = None if kwargs.get("_force_rule_only") else 0.20
        return {
            "rule_move_probability": rule,
            "ml_move_probability": ml,
            "hybrid_move_probability": 0.62 if kwargs.get("_force_rule_only") else 0.41,
            "model_features": [1.0, 2.0],
            "components": {"test": True},
        }

    monkeypatch.setattr(prob, "_compute_probability_state_impl", fake_impl)

    market_ctx = {
        "df": None,
        "spot": 1.0,
        "symbol": "NIFTY",
        "market_state": {
            "gamma_regime": "NEGATIVE_GAMMA",
            "final_flow_signal": "NEUTRAL",
            "vol_regime": "NORMAL_VOL",
            "hedging_bias": "PINNING",
            "spot_vs_flip": "ABOVE_FLIP",
            "vacuum_state": "NONE",
            "atm_iv": 0.2,
            "vacuum_zones": [],
            "hedging_flow": {},
            "flow_signal_value": "NEUTRAL",
            "smart_money_signal_value": "NEUTRAL",
            "flip": None,
            "voids": [],
        },
    }

    blended = DefaultBlendedPredictor().predict(market_ctx)
    pure_rule = PureRulePredictor().predict(market_ctx)

    assert blended.ml_move_probability == 0.20
    assert blended.hybrid_move_probability == 0.41
    assert pure_rule.ml_move_probability is None
    assert pure_rule.hybrid_move_probability == pure_rule.rule_move_probability == 0.62
    assert len(calls) == 2
    assert calls[0].get("_force_rule_only") is None
    assert calls[1].get("_force_rule_only") is True


def test_holistic_backtest_accepts_prediction_method_param():
    """Verify the backtester signature accepts prediction_method."""
    import inspect
    from backtest.holistic_backtest_runner import run_holistic_backtest
    sig = inspect.signature(run_holistic_backtest)
    assert "prediction_method" in sig.parameters
    param = sig.parameters["prediction_method"]
    assert param.default is None
