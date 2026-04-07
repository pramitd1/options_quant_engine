"""
Module: consistency_checks.py

Purpose:
    Compute cross-field consistency diagnostics for runtime and presentation layers.

Role in the System:
    Shared utility that allows engine gating, terminal rendering, and research capture
    to evaluate the same consistency rules.
"""

from __future__ import annotations

from config.signal_consistency_policy import get_signal_consistency_policy_config


_SEVERITY_RANK = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def _normalize_severity(value):
    normalized = str(value or "").upper().strip()
    return normalized if normalized in _SEVERITY_RANK else "HIGH"


def _canonical_vol_bucket(value):
    normalized = str(value or "").upper().strip()
    if normalized in {"VOL_EXPANSION", "HIGH_VOL", "SHOCK_VOL", "VOLATILE"}:
        return "EXPANSION"
    if normalized in {"VOL_CONTRACTION", "LOW_VOL", "COMPRESSED_VOL", "VOL_SUPPRESSION"}:
        return "CONTRACTION"
    if normalized in {"NORMAL_VOL", "NORMAL", "MID_VOL", "BALANCED_VOL"}:
        return "NORMAL"
    return "UNKNOWN"


def collect_trade_consistency_findings(trade):
    """Return structured consistency findings for a trade payload.

    Each finding is a dict with: code, severity, message.
    Severity levels: HIGH, MEDIUM, LOW.
    """
    if not isinstance(trade, dict):
        return []

    findings = []

    vol_bucket = _canonical_vol_bucket(trade.get("volatility_regime") or trade.get("vol_surface_regime"))
    gamma_shift = str(trade.get("intraday_gamma_state") or "").upper().strip()
    if gamma_shift == "VOL_EXPANSION" and vol_bucket == "CONTRACTION":
        findings.append(
            {
                "code": "VOL_GAMMA_SHIFT_DIVERGENCE",
                "severity": "LOW",
                "message": "volatility regime is contraction/low-vol while intraday gamma shift signals expansion",
            }
        )
    elif gamma_shift in {"VOL_SUPPRESSION", "GAMMA_INCREASE"} and vol_bucket == "EXPANSION":
        findings.append(
            {
                "code": "VOL_GAMMA_SHIFT_DIVERGENCE",
                "severity": "LOW",
                "message": "volatility regime is expansion/high-vol while intraday gamma shift signals suppression",
            }
        )

    oe_status = str(trade.get("option_efficiency_status") or "").upper().strip()
    oe_features_raw = trade.get("option_efficiency_features")
    oe_features = oe_features_raw if isinstance(oe_features_raw, dict) else {}
    oe_quality = str(oe_features.get("expected_move_quality") or trade.get("expected_move_quality") or "").upper().strip()
    oe_move_points = oe_features.get("expected_move_points")
    market_expected_available = any(
        trade.get(key) is not None
        for key in (
            "atm_straddle_price",
            "expected_move_up",
            "expected_move_down",
            "expected_move_pct",
            "expected_move_pct_model",
        )
    )

    if "UNAVAILABLE" in oe_status and market_expected_available and oe_quality != "UNAVAILABLE":
        findings.append(
            {
                "code": "OPTION_EFFICIENCY_STATUS_DIVERGENCE",
                "severity": "HIGH",
                "message": "option efficiency is marked unavailable while market expected-move fields are populated",
            }
        )
    if oe_quality == "UNAVAILABLE" and oe_move_points is not None:
        findings.append(
            {
                "code": "OPTION_EFFICIENCY_QUALITY_VALUE_MISMATCH",
                "severity": "HIGH",
                "message": "option efficiency quality is unavailable but expected_move_points is populated",
            }
        )
    if oe_quality in {"DIRECT", "FALLBACK"} and oe_move_points is None:
        findings.append(
            {
                "code": "OPTION_EFFICIENCY_QUALITY_POINTS_MISSING",
                "severity": "HIGH",
                "message": "option efficiency quality implies availability but expected_move_points is missing",
            }
        )

    return findings


def resolve_trade_consistency_escalation_policy(trade):
    """Resolve regime-aware escalation threshold for consistency findings."""
    cfg = get_signal_consistency_policy_config()
    default_severity = _normalize_severity(cfg.default_trade_escalation_min_severity)

    if not isinstance(trade, dict):
        return {
            "min_severity": default_severity,
            "matched_rule": None,
            "matched_conditions": {},
        }

    context = {
        "gamma": str(trade.get("gamma_regime") or "").upper().strip(),
        "global_risk": str(trade.get("global_risk_state") or "").upper().strip(),
        "vol": str(trade.get("volatility_regime") or trade.get("vol_surface_regime") or "").upper().strip(),
        "confirmation": str(trade.get("confirmation_status") or "").upper().strip(),
    }

    matched_rule = None
    matched_conditions = {}
    best_specificity = -1
    rule_map = cfg.trade_escalation_regime_map if isinstance(cfg.trade_escalation_regime_map, dict) else {}

    for raw_rule, raw_severity in rule_map.items():
        rule = str(raw_rule or "").strip()
        if not rule:
            continue

        rule_conditions = {}
        rule_match = True
        for token in rule.split(";"):
            piece = str(token).strip()
            if not piece or "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            key = str(key).strip().lower()
            value = str(value).strip().upper()
            if not key:
                continue
            rule_conditions[key] = value
            if context.get(key, "") != value:
                rule_match = False
                break

        if not rule_match or not rule_conditions:
            continue

        specificity = len(rule_conditions)
        if specificity > best_specificity:
            best_specificity = specificity
            matched_rule = rule
            matched_conditions = rule_conditions
            default_severity = _normalize_severity(raw_severity)

    return {
        "min_severity": default_severity,
        "matched_rule": matched_rule,
        "matched_conditions": matched_conditions,
    }


def select_trade_escalation_findings(trade, findings):
    """Select findings that meet the resolved regime-aware escalation threshold."""
    policy = resolve_trade_consistency_escalation_policy(trade)
    min_severity = policy["min_severity"]
    threshold = _SEVERITY_RANK[min_severity]

    selected = []
    for finding in findings or []:
        if not isinstance(finding, dict):
            continue
        severity = _normalize_severity(finding.get("severity"))
        if _SEVERITY_RANK[severity] >= threshold:
            selected.append(finding)

    return {
        "policy": policy,
        "matched_findings": selected,
    }
