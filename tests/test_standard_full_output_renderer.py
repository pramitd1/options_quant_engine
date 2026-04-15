from contextlib import redirect_stdout
from io import StringIO

from app.terminal_output import render_snapshot


def _base_payloads():
    trade = {
        "symbol": "NIFTY",
        "trade_status": "WATCHLIST",
        "direction": None,
        "confirmation_status": "NO_DIRECTION",
        "trade_strength": 0,
        "hybrid_move_probability": 0.60,
        "data_quality_status": "CAUTION",
        "provider_health_summary": "WEAK",
        "final_flow_signal": "BULLISH_FLOW",
        "macro_regime": "RISK_OFF",
        "global_risk_state": "RISK_OFF",
        "analytics_usable": True,
        "execution_suggestion_usable": False,
        "tradable_data": {
            "status": "ANALYTICS_ONLY",
            "score": 0.44,
            "reasons": ["crossed_quotes_high"],
        },
        "feature_reliability_weights": {"gamma": 0.92, "surface": 0.38},
        "feature_reliability_status": "CAUTION",
        "feature_reliability_score": 72.5,
        "provider_health": {
            "summary_status": "CAUTION",
            "atm_iv_health": "CAUTION",
            "atm_iv_midpoint": 0.185,
            "atm_iv_vs_vix_consistent": False,
            "iv_parity_health": "GOOD",
            "iv_parity_breach_ratio": 0.05,
            "iv_staleness_health": "GOOD",
            "iv_stale_ratio": 0.08,
        },
        "scoring_breakdown": {
            "feature_reliability_penalty": -3,
            "chain_confirmation_reliability_weight": 0.62,
            "chain_confirmation_reliability_delta": -2,
            "gamma_vol_reliability_weight": 0.55,
            "gamma_vol_reliability_delta": -3,
            "dealer_pressure_reliability_weight": 0.58,
            "dealer_pressure_reliability_delta": -2,
            "option_efficiency_reliability_weight": 0.60,
            "option_efficiency_reliability_delta": -1,
        },
        "iv_surface_residual_status": "DEGRADED",
        "iv_surface_residual_penalty_score": 7,
        "option_efficiency_status": "UNAVAILABLE_NEUTRALIZED",
        "option_efficiency_reason": "option_efficiency_features_missing",
    }

    result = {
        "symbol": "NIFTY",
        "mode": "LIVE",
        "source": "ZERODHA",
        "option_chain_rows": 0,
        "option_chain_frame": None,
        "previous_chain_frame": None,
        "premium_baseline_chain_frames": None,
        "premium_baseline_labels": None,
        "premium_baseline_chain_frame": None,
        "zerodha_oi_baseline_chain_frame": None,
    }

    spot_summary = {
        "spot": 23940.0,
        "day_open": 23850.0,
        "day_high": 24000.0,
        "day_low": 23800.0,
        "prev_close": 23123.0,
        "timestamp": "2026-04-08T12:47:40+05:30",
        "lookback_avg_range_pct": 1.7,
    }

    spot_validation = {
        "validation_mode": "LIVE",
        "is_valid": True,
        "live_trading_valid": True,
        "replay_analysis_valid": True,
        "is_stale": False,
        "age_minutes": 0,
        "issues": [],
        "warnings": [],
    }
    option_chain_validation = {
        "validation_mode": "LIVE",
        "is_valid": True,
        "live_trading_valid": True,
        "replay_analysis_valid": True,
        "is_stale": False,
        "age_minutes": 0,
        "issues": [],
        "warnings": [],
    }
    macro_event_state = {
        "macro_event_risk_score": 0,
        "event_window_status": "NONE",
        "event_lockdown_flag": False,
        "minutes_to_next_event": None,
        "next_event_name": None,
    }
    macro_news_state = {
        "macro_regime": "RISK_OFF",
        "macro_sentiment_score": -0.2,
        "volatility_shock_score": 0.1,
        "news_confidence_score": 0.7,
        "macro_regime_reasons": ["test"],
        "headline_velocity": 0.0,
        "headline_count": 0,
        "classified_headline_count": 0,
        "next_event_name": None,
        "neutral_fallback": False,
    }
    global_risk_state = {
        "global_risk_state": "RISK_OFF",
        "global_risk_score": 16,
        "global_risk_adjustment_score": -8,
        "overnight_hold_allowed": False,
        "overnight_gap_risk_score": 72,
        "volatility_expansion_risk_score": 61,
        "overnight_hold_reason": "risk_off",
        "overnight_risk_penalty": -5,
        "global_risk_reasons": ["test"],
    }
    global_market_snapshot = {
        "provider": "TEST",
        "data_available": True,
        "stale": False,
        "warnings": [],
        "market_inputs": {
            "oil_change_24h": 0.0,
            "vix_change_24h": 0.0,
            "india_vix_level": 0.0,
            "india_vix_change_24h": 0.0,
            "sp500_change_24h": 0.0,
            "us10y_change_bp": 0.0,
            "usdinr_change_24h": 0.0,
        },
    }
    headline_state = {
        "provider_name": "TEST",
        "data_available": True,
        "is_stale": False,
        "warnings": [],
        "issues": [],
        "provider_metadata": {},
    }

    return {
        "trade": trade,
        "result": result,
        "spot_summary": spot_summary,
        "spot_validation": spot_validation,
        "option_chain_validation": option_chain_validation,
        "macro_event_state": macro_event_state,
        "macro_news_state": macro_news_state,
        "global_risk_state": global_risk_state,
        "global_market_snapshot": global_market_snapshot,
        "headline_state": headline_state,
    }


def test_standard_mode_renders_confidence_note_and_consistency_check() -> None:
    payload = _base_payloads()
    with StringIO() as buffer, redirect_stdout(buffer):
        render_snapshot(
            "STANDARD",
            result=payload["result"],
            spot_summary=payload["spot_summary"],
            spot_validation=payload["spot_validation"],
            option_chain_validation=payload["option_chain_validation"],
            macro_event_state=payload["macro_event_state"],
            macro_news_state=payload["macro_news_state"],
            global_risk_state=payload["global_risk_state"],
            global_market_snapshot=payload["global_market_snapshot"],
            headline_state=payload["headline_state"],
            trade=payload["trade"],
            execution_trade=None,
        )
        output = buffer.getvalue()

    assert "SIGNAL CONFIDENCE" in output
    assert "DATA USABILITY" in output
    assert "RELIABILITY DAMPING" in output
    assert "chain_confirm_delta" in output
    assert "atm_iv_health" in output
    assert "iv_parity_health" in output
    assert "execution_suggestion_usable" in output
    assert "confidence_note" in output
    assert "CONSISTENCY CHECK" in output
    assert "FLOW_MACRO_REGIME_CONTRADICTION" not in output
    assert "bullish flow signal (BULLISH_FLOW) conflicts with RISK_OFF macro/global regime" in output


def test_full_debug_mode_renders_confidence_note_and_consistency_check() -> None:
    payload = _base_payloads()
    with StringIO() as buffer, redirect_stdout(buffer):
        render_snapshot(
            "FULL_DEBUG",
            result=payload["result"],
            spot_summary=payload["spot_summary"],
            spot_validation=payload["spot_validation"],
            option_chain_validation=payload["option_chain_validation"],
            macro_event_state=payload["macro_event_state"],
            macro_news_state=payload["macro_news_state"],
            global_risk_state=payload["global_risk_state"],
            global_market_snapshot=payload["global_market_snapshot"],
            headline_state=payload["headline_state"],
            trade=payload["trade"],
            execution_trade=None,
        )
        output = buffer.getvalue()

    assert "SIGNAL CONFIDENCE" in output
    assert "DATA USABILITY" in output
    assert "RELIABILITY DAMPING" in output
    assert "gamma_vol_delta" in output
    assert "atm_iv_health" in output
    assert "iv_staleness_health" in output
    assert "iv_surface_residual_penalty_score" in output
    assert "confidence_note" in output
    assert "CONSISTENCY CHECK" in output
    assert "bullish flow signal (BULLISH_FLOW) conflicts with RISK_OFF macro/global regime" in output


def test_compact_mode_uses_bias_and_execution_suggestion_wording() -> None:
    payload = _base_payloads()
    payload["trade"].update(
        {
            "decision_classification": "BLOCKED_SETUP",
            "trade_status": "BLOCKED_SETUP",
            "direction": "PUT",
            "direction_source": "FLOW+MICROSTRUCTURE_FRICTION",
            "confirmation_status": "NO_DIRECTION",
            "no_trade_reason": "Provider health is blocking execution",
            "blocked_by": ["provider_health"],
            "iv_hv_regime": "IV_RICH",
        }
    )

    with StringIO() as buffer, redirect_stdout(buffer):
        render_snapshot(
            "COMPACT",
            result=payload["result"],
            spot_summary=payload["spot_summary"],
            spot_validation=payload["spot_validation"],
            option_chain_validation=payload["option_chain_validation"],
            macro_event_state=payload["macro_event_state"],
            macro_news_state=payload["macro_news_state"],
            global_risk_state=payload["global_risk_state"],
            global_market_snapshot=payload["global_market_snapshot"],
            headline_state=payload["headline_state"],
            trade=payload["trade"],
            execution_trade=None,
        )
        output = buffer.getvalue()

    assert "direction_bias" in output
    assert "execution_suggestion_usable" in output
    assert "iv_hv_regime" in output
    assert "MICROSTRUCTURE_FRICTION" not in output