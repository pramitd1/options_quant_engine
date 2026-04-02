from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _base_market_ctx() -> dict:
    return {
        "spot": 22500.0,
        "day_high": 22600.0,
        "day_low": 22400.0,
        "day_open": 22480.0,
        "prev_close": 22450.0,
        "lookback_avg_range_pct": 0.9,
        "market_state": {
            "gamma_regime": "POSITIVE_GAMMA",
            "final_flow_signal": "BULLISH_FLOW",
            "vol_regime": "VOL_EXPANSION",
            "hedging_bias": "UPSIDE_ACCELERATION",
            "spot_vs_flip": "ABOVE_FLIP",
            "vacuum_state": "BREAKOUT_ZONE",
            "dealer_pos": "LONG GAMMA",
            "greek_exposures": {
                "vanna_regime": "VANNA_BULLISH",
                "charm_regime": "CHARM_BULLISH",
            },
        },
        "global_context": {
            "macro_event_risk_score": 0.2,
            "macro_regime": "NO_EVENT",
            "india_vix_level": 14.0,
            "india_vix_change_24h": -0.5,
            "oil_shock_score": 0.1,
            "commodity_risk_score": 0.2,
            "volatility_shock_score": 0.1,
            "days_to_expiry": 2,
        },
    }


def _stub_raw_probability_state(**overrides):
    state = {
        "rule_move_probability": 0.41,
        "ml_move_probability": 0.57,
        "hybrid_move_probability": 0.52,
        "model_features": [0.1, 0.2, 0.3],
        "components": {
            "gamma_flip_distance_pct": 0.4,
            "vacuum_strength": 0.8,
            "hedging_flow_ratio": 0.5,
            "smart_money_flow_score": 0.6,
            "atm_iv_percentile": 0.45,
            "intraday_range_pct": 1.1,
            "lookback_avg_range_pct": 0.9,
            "day_high": 22600.0,
            "day_low": 22400.0,
            "day_open": 22480.0,
            "prev_close": 22450.0,
        },
    }
    state.update(overrides)
    return state


def test_research_predictor_passes_row_dict_to_infer_single(monkeypatch):
    from engine.predictors.research_predictor import ResearchDualModelPredictor

    captured = {"arg_type": None}

    class _Result:
        ml_rank_score = 0.62
        ml_confidence_score = 0.71

    def _fake_infer_single(arg):
        captured["arg_type"] = type(arg)
        assert isinstance(arg, dict)
        assert arg.get("gamma_regime") == "POSITIVE_GAMMA"
        return _Result()

    def _fake_impl(**kwargs):
        return _stub_raw_probability_state()

    import research.ml_models.ml_inference as ml_inf
    import engine.trading_support.probability as prob

    monkeypatch.setattr(ml_inf, "infer_single", _fake_infer_single)
    monkeypatch.setattr(prob, "_compute_probability_state_impl", _fake_impl)

    predictor = ResearchDualModelPredictor()
    out = predictor.predict(_base_market_ctx())

    assert captured["arg_type"] is dict
    assert out.hybrid_move_probability == 0.71
    assert out.components.get("research_rank_score") == 0.62
    assert out.components.get("research_confidence_score") == 0.71


def test_decision_policy_predictor_uses_ml_score_fields(monkeypatch):
    from engine.predictors.decision_policy_predictor import ResearchDecisionPolicyPredictor

    class _Result:
        ml_rank_score = 0.68
        ml_confidence_score = 0.74

    def _fake_infer_single(_arg):
        return _Result()

    def _fake_impl(**kwargs):
        return _stub_raw_probability_state(hybrid_move_probability=0.49)

    import research.ml_models.ml_inference as ml_inf
    import engine.trading_support.probability as prob

    monkeypatch.setattr(ml_inf, "infer_single", _fake_infer_single)
    monkeypatch.setattr(prob, "_compute_probability_state_impl", _fake_impl)

    predictor = ResearchDecisionPolicyPredictor()
    out = predictor.predict(_base_market_ctx())

    assert out.components.get("research_rank_score") == 0.68
    assert out.components.get("research_confidence_score") == 0.74
    assert out.components.get("policy_decision") == "ALLOW"
    # Decision-policy predictor now preserves engine hybrid probability and
    # applies policy as an overlay rather than replacing it with ML confidence.
    assert out.components.get("engine_hybrid_probability") == 0.49
    assert out.hybrid_move_probability == 0.49


def test_decision_policy_predictor_preserves_engine_probability_when_policy_blocks(monkeypatch):
    from engine.predictors.decision_policy_predictor import ResearchDecisionPolicyPredictor
    import research.decision_policy.policy_definitions as policy_defs
    import research.ml_models.ml_inference as ml_inf
    import engine.trading_support.probability as prob

    class _Result:
        ml_rank_score = 0.12
        ml_confidence_score = 0.18

    class _BlockedDecision:
        decision = "BLOCK"
        reason = "below_dual_thresholds"
        size_multiplier = 0.0

    def _fake_infer_single(_arg):
        return _Result()

    def _fake_impl(**kwargs):
        return _stub_raw_probability_state(hybrid_move_probability=0.52)

    monkeypatch.setattr(ml_inf, "infer_single", _fake_infer_single)
    monkeypatch.setattr(prob, "_compute_probability_state_impl", _fake_impl)
    monkeypatch.setattr(policy_defs, "dual_threshold_policy", lambda _row: _BlockedDecision())

    predictor = ResearchDecisionPolicyPredictor()
    out = predictor.predict(_base_market_ctx())

    assert out.components.get("policy_decision") == "BLOCK"
    assert out.components.get("policy_reason") == "below_dual_thresholds"
    assert out.components.get("engine_hybrid_probability") == 0.52
    assert out.hybrid_move_probability == 0.52


def test_ev_sizing_predictor_passes_row_dict_to_infer_single(monkeypatch):
    from engine.predictors.ev_sizing_predictor import EVSizingPredictor

    captured = {"seen": False}

    class _Result:
        ml_rank_score = 0.58
        ml_confidence_score = 0.63
        ml_rank_bucket = None
        ml_confidence_bucket = None

    def _fake_infer_single(arg):
        captured["seen"] = True
        assert isinstance(arg, dict)
        assert "macro_regime" in arg
        return _Result()

    def _fake_impl(**kwargs):
        return _stub_raw_probability_state(hybrid_move_probability=0.51)

    import research.ml_models.ml_inference as ml_inf
    import engine.trading_support.probability as prob

    monkeypatch.setattr(ml_inf, "infer_single", _fake_infer_single)
    monkeypatch.setattr(prob, "_compute_probability_state_impl", _fake_impl)

    predictor = EVSizingPredictor()
    out = predictor.predict(_base_market_ctx())

    assert captured["seen"] is True
    assert out.components.get("research_rank_score") == 0.58
    assert out.components.get("research_confidence_score") == 0.63
    assert out.hybrid_move_probability == 0.63


def test_ev_sizing_predictor_uses_market_state_gamma_regime_for_ev_lookup(monkeypatch):
    from engine.predictors.ev_sizing_predictor import EVSizingPredictor
    import engine.predictors.ev_sizing_predictor as ev_pred

    captured = {"regime": None}

    class _Result:
        ml_rank_score = 0.58
        ml_confidence_score = 0.63
        ml_rank_bucket = "Q4_high"
        ml_confidence_bucket = "Q5_highest"

    class _Cell:
        backed_off = False
        hit_rate = 0.61

    def _fake_infer_single(_arg):
        return _Result()

    def _fake_impl(**kwargs):
        return _stub_raw_probability_state(hybrid_move_probability=0.51)

    def _fake_lookup(_table, rank_bucket, confidence_bucket, regime):
        assert rank_bucket == "Q4_high"
        assert confidence_bucket == "Q5_highest"
        captured["regime"] = regime
        return _Cell()

    import research.ml_models.ml_inference as ml_inf
    import engine.trading_support.probability as prob
    from research.ml_evaluation.ev_and_regime_policy import conditional_return_tables as crt
    from research.ml_evaluation.ev_and_regime_policy import ev_sizing_model as ev_model

    monkeypatch.setattr(ml_inf, "infer_single", _fake_infer_single)
    monkeypatch.setattr(prob, "_compute_probability_state_impl", _fake_impl)
    monkeypatch.setattr(crt, "lookup", _fake_lookup)
    monkeypatch.setattr(ev_model, "compute_ev", lambda p_win, cell: (0.2, 12.0, -6.0))
    monkeypatch.setattr(ev_model, "normalize_ev", lambda ev_raw, lo, hi: 0.7)
    monkeypatch.setattr(ev_model, "classify_ev_bucket", lambda value: "POSITIVE")
    monkeypatch.setattr(ev_model, "ev_to_size_multiplier", lambda value: 1.0)
    monkeypatch.setattr(ev_model, "compute_ev_reliability", lambda cell: 0.85)
    monkeypatch.setattr(EVSizingPredictor, "_ensure_crt_loaded", staticmethod(lambda: None))

    ev_pred._CRT_CACHE = object()
    ev_pred._EV_BOUNDS = (0.0, 1.0)
    ev_pred._CRT_LOAD_ATTEMPTED = True

    predictor = EVSizingPredictor()
    out = predictor.predict(_base_market_ctx())

    assert captured["regime"] == "POSITIVE_GAMMA"
    assert out.components.get("gamma_regime") == "POSITIVE_GAMMA"
    assert out.components.get("ev_bucket") == "POSITIVE"
