from __future__ import annotations

from utils.consistency_checks import (
    collect_trade_consistency_findings,
    resolve_trade_consistency_escalation_policy,
    select_trade_escalation_findings,
)


def test_consistency_checks_return_empty_for_aligned_payload():
    trade = {
        "volatility_regime": "VOL_EXPANSION",
        "intraday_gamma_state": "VOL_EXPANSION",
        "option_efficiency_status": "AVAILABLE",
        "expected_move_quality": "DIRECT",
        "option_efficiency_features": {
            "expected_move_quality": "DIRECT",
            "expected_move_points": 160.0,
        },
        "atm_straddle_price": 185.0,
        "expected_move_pct": 0.8,
    }

    findings = collect_trade_consistency_findings(trade)

    assert findings == []


def test_consistency_checks_emit_high_severity_option_efficiency_divergence():
    trade = {
        "volatility_regime": "LOW_VOL",
        "intraday_gamma_state": "VOL_EXPANSION",
        "option_efficiency_status": "UNAVAILABLE_NEUTRALIZED",
        "expected_move_quality": "DIRECT",
        "option_efficiency_features": {
            "expected_move_quality": "DIRECT",
            "expected_move_points": 92.0,
        },
        "atm_straddle_price": 172.0,
        "expected_move_pct": 0.71,
    }

    findings = collect_trade_consistency_findings(trade)
    codes = {item["code"] for item in findings}
    severities = {item["severity"] for item in findings}

    assert "OPTION_EFFICIENCY_STATUS_DIVERGENCE" in codes
    assert "VOL_GAMMA_SHIFT_DIVERGENCE" in codes
    assert "HIGH" in severities


def test_escalation_policy_is_stricter_in_negative_gamma_risk_off():
    trade = {
        "gamma_regime": "NEGATIVE_GAMMA",
        "global_risk_state": "RISK_OFF",
        "volatility_regime": "VOL_EXPANSION",
        "confirmation_status": "NO_DIRECTION",
    }

    policy = resolve_trade_consistency_escalation_policy(trade)

    assert policy["min_severity"] == "MEDIUM"
    assert policy["matched_rule"] == "gamma=NEGATIVE_GAMMA;global_risk=RISK_OFF"


def test_escalation_policy_is_softer_in_normal_vol_confirmed_context():
    trade = {
        "gamma_regime": "POSITIVE_GAMMA",
        "global_risk_state": "RISK_ON",
        "volatility_regime": "NORMAL_VOL",
        "confirmation_status": "CONFIRMED",
    }

    policy = resolve_trade_consistency_escalation_policy(trade)

    assert policy["min_severity"] == "HIGH"
    assert policy["matched_rule"] == "vol=NORMAL_VOL;confirmation=CONFIRMED"


def test_trade_escalation_findings_respect_resolved_min_severity():
    trade = {
        "gamma_regime": "NEGATIVE_GAMMA",
        "global_risk_state": "RISK_OFF",
        "volatility_regime": "VOL_EXPANSION",
        "confirmation_status": "CONFLICT",
    }
    findings = [
        {"code": "LOW_CASE", "severity": "LOW", "message": "low"},
        {"code": "MED_CASE", "severity": "MEDIUM", "message": "medium"},
        {"code": "HIGH_CASE", "severity": "HIGH", "message": "high"},
    ]

    state = select_trade_escalation_findings(trade, findings)
    codes = [item["code"] for item in state["matched_findings"]]

    assert state["policy"]["min_severity"] == "MEDIUM"
    assert "LOW_CASE" not in codes
    assert "MED_CASE" in codes
    assert "HIGH_CASE" in codes


def test_flow_macro_contradiction_bullish_flow_with_risk_off_is_flagged():
    trade = {
        "final_flow_signal": "BULLISH_FLOW",
        "macro_regime": "RISK_OFF",
        "global_risk_state": "RISK_OFF",
        "volatility_regime": "NORMAL_VOL",
        "option_efficiency_status": "AVAILABLE",
    }

    findings = collect_trade_consistency_findings(trade)
    codes = {item["code"] for item in findings}

    assert "FLOW_MACRO_REGIME_CONTRADICTION" in codes
    contradiction = next(f for f in findings if f["code"] == "FLOW_MACRO_REGIME_CONTRADICTION")
    assert contradiction["severity"] == "MEDIUM"
    assert "bullish" in contradiction["message"]
    assert "RISK_OFF" in contradiction["message"]


def test_flow_macro_contradiction_aligned_flow_is_not_flagged():
    trade = {
        "final_flow_signal": "BEARISH_FLOW",
        "macro_regime": "RISK_OFF",
        "global_risk_state": "RISK_OFF",
        "volatility_regime": "NORMAL_VOL",
        "option_efficiency_status": "AVAILABLE",
    }

    findings = collect_trade_consistency_findings(trade)
    codes = {item["code"] for item in findings}

    assert "FLOW_MACRO_REGIME_CONTRADICTION" not in codes


def test_flow_macro_contradiction_not_flagged_when_no_flow_signal():
    trade = {
        "macro_regime": "RISK_OFF",
        "global_risk_state": "RISK_OFF",
        "volatility_regime": "NORMAL_VOL",
        "option_efficiency_status": "AVAILABLE",
    }

    findings = collect_trade_consistency_findings(trade)
    codes = {item["code"] for item in findings}

    assert "FLOW_MACRO_REGIME_CONTRADICTION" not in codes
