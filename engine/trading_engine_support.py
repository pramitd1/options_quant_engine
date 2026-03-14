"""
Internal support functions for the trading engine.

These helpers are intentionally kept separate from the public
`generate_trade()` entrypoint so the engine orchestration stays easier to
navigate while preserving backward compatibility for callers.
"""

from __future__ import annotations

import pandas as pd

from analytics import dealer_gamma_path as dealer_gamma_path_mod
from analytics import dealer_hedging_flow as dealer_hedging_flow_mod
from analytics import dealer_hedging_simulator as dealer_hedging_simulator_mod
from analytics import dealer_inventory as dealer_inventory_mod
from analytics import gamma_exposure as gamma_exposure_mod
from analytics import gamma_flip as gamma_flip_mod
from analytics import gamma_walls as gamma_walls_mod
from analytics import intraday_gamma_shift as intraday_gamma_shift_mod
from analytics import liquidity_heatmap as liquidity_heatmap_mod
from analytics import liquidity_vacuum as liquidity_vacuum_mod
from analytics import liquidity_void as liquidity_void_mod
from analytics import market_gamma_map as market_gamma_map_mod
from analytics import options_flow_imbalance as options_flow_imbalance_mod
from analytics import smart_money_flow as smart_money_flow_mod
from analytics import volatility_regime as volatility_regime_mod
from analytics import volatility_surface as volatility_surface_mod
from analytics.dealer_liquidity_map import build_dealer_liquidity_map
from analytics.greeks_engine import enrich_chain_with_greeks, summarize_greek_exposures
from config.global_risk_policy import get_global_risk_policy_config
from config.signal_policy import (
    get_direction_thresholds,
    get_direction_vote_weights,
    get_trade_runtime_thresholds,
)
from config.symbol_microstructure import get_microstructure_config
from engine.runtime_metadata import empty_confirmation_state, empty_scoring_breakdown
import models.feature_builder as feature_builder_mod
import models.large_move_probability as large_move_probability_mod
import models.ml_move_predictor as ml_move_predictor_mod
from strategy.confirmation_filters import compute_confirmation_filters
from strategy.trade_strength import compute_trade_strength


_MOVE_PREDICTOR = None


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_div(a, b, default=0.0):
    try:
        a = float(a)
        b = float(b)
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def _map_vacuum_strength(vacuum_state, liquidity_voids=None, nearest_vacuum_gap_pct=None):
    base = {
        "BREAKOUT_ZONE": 0.85,
        "NEAR_VACUUM": 0.60,
        "VACUUM_WATCH": 0.40,
    }.get(vacuum_state, 0.15)

    if nearest_vacuum_gap_pct is not None:
        gap = _clip(_safe_float(nearest_vacuum_gap_pct), 0.0, 1.5)
        proximity_boost = 1.0 - (gap / 1.5)
        base = 0.6 * base + 0.4 * proximity_boost

    if liquidity_voids is not None:
        try:
            void_count = len(liquidity_voids)
            base += min(void_count, 5) * 0.03
        except Exception:
            pass

    return round(_clip(base, 0.0, 1.0), 3)


def _map_hedging_flow_ratio(hedging_bias, hedge_flow_value=None):
    if hedge_flow_value is not None:
        return round(_clip(_safe_float(hedge_flow_value), -1.0, 1.0), 3)

    mapping = {
        "UPSIDE_ACCELERATION": 0.75,
        "DOWNSIDE_ACCELERATION": -0.75,
        "UPSIDE_PINNING": 0.20,
        "DOWNSIDE_PINNING": -0.20,
        "PINNING": 0.0,
    }
    return round(mapping.get(hedging_bias, 0.0), 3)


def _map_smart_money_flow_score(smart_money_flow, flow_imbalance=None):
    base = {
        "BULLISH_FLOW": 0.70,
        "BEARISH_FLOW": -0.70,
        "MIXED_FLOW": 0.0,
        "NEUTRAL_FLOW": 0.0,
    }.get(smart_money_flow, 0.0)

    if flow_imbalance is not None:
        base = 0.5 * base + 0.5 * _clip(_safe_float(flow_imbalance), -1.0, 1.0)

    return round(_clip(base, -1.0, 1.0), 3)


def _compute_gamma_flip_distance_pct(spot_price, gamma_flip):
    if gamma_flip is None:
        return None

    spot = _safe_float(spot_price, None)
    flip = _safe_float(gamma_flip, None)

    if spot in (None, 0) or flip is None:
        return None

    return round(abs(spot - flip) / spot * 100.0, 4)


def _compute_intraday_range_pct(
    symbol=None,
    spot_price=None,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None,
):
    micro_cfg = get_microstructure_config(symbol)
    spot = _safe_float(spot_price, None)
    if spot in (None, 0):
        return None

    high = _safe_float(day_high, None)
    low = _safe_float(day_low, None)
    open_px = _safe_float(day_open, None)
    prev_close_px = _safe_float(prev_close, None)
    avg_range = _safe_float(lookback_avg_range_pct, None)

    realized_range_pct = None

    if high is not None and low is not None and high >= low:
        realized_range_pct = ((high - low) / spot) * 100.0
    else:
        anchor_moves = []

        if open_px not in (None, 0):
            anchor_moves.append(abs(spot - open_px) / spot * 100.0)

        if prev_close_px not in (None, 0):
            anchor_moves.append(abs(spot - prev_close_px) / spot * 100.0)

        if anchor_moves:
            realized_range_pct = max(anchor_moves) * 1.5

    if realized_range_pct is None:
        return None

    baseline_floor = _safe_float(micro_cfg.get("range_baseline_floor_pct"), 0.9)
    baseline = avg_range if avg_range not in (None, 0) else baseline_floor
    baseline = max(baseline, baseline_floor)
    normalized = realized_range_pct / max(baseline, 0.25)

    return round(_clip(normalized, 0.0, 1.5), 4)


def _compute_atm_iv_percentile(atm_iv, low_iv=8.0, high_iv=28.0):
    iv = _safe_float(atm_iv, None)
    if iv is None:
        return None

    pct = (iv - low_iv) / max(high_iv - low_iv, 1e-6)
    return round(_clip(pct, 0.0, 1.0), 4)


def _blend_move_probability(rule_prob, ml_prob):
    rule_prob = _safe_float(rule_prob, 0.22)

    if ml_prob is None:
        return round(_clip(rule_prob, 0.05, 0.95), 2)

    ml_prob = _safe_float(ml_prob, rule_prob)
    hybrid = 0.35 * rule_prob + 0.65 * ml_prob
    hybrid = 0.10 + 0.80 * hybrid

    return round(_clip(hybrid, 0.05, 0.95), 2)


def _get_move_predictor():
    global _MOVE_PREDICTOR

    predictor_class = getattr(ml_move_predictor_mod, "MovePredictor", None)
    if predictor_class is None:
        return None

    if _MOVE_PREDICTOR is None:
        try:
            _MOVE_PREDICTOR = predictor_class()
        except Exception:
            _MOVE_PREDICTOR = False

    if _MOVE_PREDICTOR is False:
        return None

    return _MOVE_PREDICTOR


def normalize_option_chain(option_chain, spot=None, valuation_time=None):
    df = option_chain.copy()

    rename_map = {
        "strikePrice": "STRIKE_PR",
        "openInterest": "OPEN_INT",
        "impliedVolatility": "IV",
        "totalTradedVolume": "VOLUME",
        "lastPrice": "LAST_PRICE",
        "changeinOI": "CHG_IN_OI",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "STRIKE_PR" in df.columns and "strikePrice" not in df.columns:
        df["strikePrice"] = df["STRIKE_PR"]

    if "OPEN_INT" in df.columns and "openInterest" not in df.columns:
        df["openInterest"] = df["OPEN_INT"]

    if "IV" in df.columns and "impliedVolatility" not in df.columns:
        df["impliedVolatility"] = df["IV"]

    if "VOLUME" in df.columns and "totalTradedVolume" not in df.columns:
        df["totalTradedVolume"] = df["VOLUME"]

    if "LAST_PRICE" in df.columns and "lastPrice" not in df.columns:
        df["lastPrice"] = df["LAST_PRICE"]

    if "EXPIRY_DT" not in df.columns:
        df["EXPIRY_DT"] = None

    if spot is None:
        spot = df["strikePrice"].median() if "strikePrice" in df.columns else None

    greek_cols = ["DELTA", "GAMMA", "THETA", "VEGA", "RHO", "TTE"]
    has_usable_greeks = all(col in df.columns for col in greek_cols)
    if has_usable_greeks:
        gamma_valid = pd.to_numeric(df["GAMMA"], errors="coerce").notna().any()
        delta_valid = pd.to_numeric(df["DELTA"], errors="coerce").notna().any()
        tte_valid = pd.to_numeric(df["TTE"], errors="coerce").notna().any()
        has_usable_greeks = gamma_valid and delta_valid and tte_valid

    if not has_usable_greeks:
        df = enrich_chain_with_greeks(df, spot=spot, valuation_time=valuation_time)

    return df


def _call_first(module, candidate_names, *args, default=None, **kwargs):
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
            except Exception:
                continue
    return default


def _to_python_number(x):
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass

    try:
        if isinstance(x, float) and x.is_integer():
            return int(x)
    except Exception:
        pass

    return x


def _clean_zone_list(zones):
    if not zones:
        return []

    cleaned = []

    for zone in zones:
        try:
            low, high = zone
            low = _to_python_number(low)
            high = _to_python_number(high)
            cleaned.append((low, high))
        except Exception:
            continue

    return cleaned


def derive_global_risk_trade_modifiers(global_risk_state):
    cfg = get_global_risk_policy_config()
    global_risk_state = global_risk_state if isinstance(global_risk_state, dict) else {}
    features = global_risk_state.get("global_risk_features", {})
    features = features if isinstance(features, dict) else {}

    base_adjustment_score = int(_safe_float(global_risk_state.get("global_risk_adjustment_score"), 0.0))
    feature_adjustment_score = 0
    adjustment_reasons = []

    volatility_explosion_probability = _safe_float(features.get("volatility_explosion_probability"), 0.0)
    oil_shock_score = _safe_float(features.get("oil_shock_score"), 0.0)

    if volatility_explosion_probability > cfg.volatility_explosion_penalty_threshold:
        feature_adjustment_score += int(cfg.volatility_explosion_penalty_score)
        adjustment_reasons.append("volatility_explosion_probability_high")

    if oil_shock_score >= cfg.oil_shock_penalty_threshold:
        feature_adjustment_score += int(cfg.oil_shock_penalty_score)
        adjustment_reasons.append("oil_shock_score_high")

    effective_adjustment_score = base_adjustment_score + feature_adjustment_score
    overnight_hold_allowed = bool(global_risk_state.get("overnight_hold_allowed", True))

    return {
        "base_adjustment_score": base_adjustment_score,
        "feature_adjustment_score": feature_adjustment_score,
        "effective_adjustment_score": effective_adjustment_score,
        "adjustment_reasons": adjustment_reasons,
        "oil_shock_score": oil_shock_score,
        "commodity_risk_score": _safe_float(features.get("commodity_risk_score"), 0.0),
        "volatility_shock_score": _safe_float(features.get("volatility_shock_score"), 0.0),
        "volatility_explosion_probability": volatility_explosion_probability,
        "overnight_hold_allowed": overnight_hold_allowed,
        "overnight_hold_reason": str(global_risk_state.get("overnight_hold_reason", "overnight_risk_contained")),
        "overnight_risk_penalty": int(_safe_float(global_risk_state.get("overnight_risk_penalty"), 0.0)),
        "overnight_trade_block": not overnight_hold_allowed,
        "force_no_trade": str(global_risk_state.get("global_risk_state", "")).upper().strip() == "EVENT_LOCKDOWN",
    }


def derive_gamma_vol_trade_modifiers(gamma_vol_state, direction=None):
    gamma_vol_state = gamma_vol_state if isinstance(gamma_vol_state, dict) else {}
    base_adjustment_score = int(_safe_float(gamma_vol_state.get("gamma_vol_adjustment_score"), 0.0))
    alignment_adjustment_score = 0
    adjustment_reasons = []

    directional_convexity_state = str(gamma_vol_state.get("directional_convexity_state", "NO_CONVEXITY_EDGE")).upper().strip()
    squeeze_risk_state = str(gamma_vol_state.get("squeeze_risk_state", "LOW_ACCELERATION_RISK")).upper().strip()
    direction = str(direction or "").upper().strip()

    if direction == "CALL":
        if directional_convexity_state == "UPSIDE_SQUEEZE_RISK":
            alignment_adjustment_score += 2
            adjustment_reasons.append("upside_squeeze_alignment")
        elif directional_convexity_state == "DOWNSIDE_AIRPOCKET_RISK":
            alignment_adjustment_score -= 6
            adjustment_reasons.append("downside_convexity_conflict")
        elif directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK" and squeeze_risk_state in {"HIGH_ACCELERATION_RISK", "EXTREME_ACCELERATION_RISK"}:
            alignment_adjustment_score += 1
            adjustment_reasons.append("two_sided_volatility_convexity")
    elif direction == "PUT":
        if directional_convexity_state == "DOWNSIDE_AIRPOCKET_RISK":
            alignment_adjustment_score += 2
            adjustment_reasons.append("downside_airpocket_alignment")
        elif directional_convexity_state == "UPSIDE_SQUEEZE_RISK":
            alignment_adjustment_score -= 6
            adjustment_reasons.append("upside_convexity_conflict")
        elif directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK" and squeeze_risk_state in {"HIGH_ACCELERATION_RISK", "EXTREME_ACCELERATION_RISK"}:
            alignment_adjustment_score += 1
            adjustment_reasons.append("two_sided_volatility_convexity")

    effective_adjustment_score = int(_clip(base_adjustment_score + alignment_adjustment_score, -6, 8))
    overnight_hold_allowed = bool(gamma_vol_state.get("overnight_hold_allowed", True))

    return {
        "base_adjustment_score": base_adjustment_score,
        "alignment_adjustment_score": alignment_adjustment_score,
        "effective_adjustment_score": effective_adjustment_score,
        "adjustment_reasons": adjustment_reasons,
        "gamma_vol_acceleration_score": int(_safe_float(gamma_vol_state.get("gamma_vol_acceleration_score"), 0.0)),
        "squeeze_risk_state": str(gamma_vol_state.get("squeeze_risk_state", "LOW_ACCELERATION_RISK")),
        "directional_convexity_state": str(gamma_vol_state.get("directional_convexity_state", "NO_CONVEXITY_EDGE")),
        "upside_squeeze_risk": _safe_float(gamma_vol_state.get("upside_squeeze_risk"), 0.0),
        "downside_airpocket_risk": _safe_float(gamma_vol_state.get("downside_airpocket_risk"), 0.0),
        "overnight_convexity_risk": _safe_float(gamma_vol_state.get("overnight_convexity_risk"), 0.0),
        "overnight_hold_allowed": overnight_hold_allowed,
        "overnight_hold_reason": str(gamma_vol_state.get("overnight_hold_reason", "overnight_convexity_contained")),
        "overnight_convexity_penalty": int(_safe_float(gamma_vol_state.get("overnight_convexity_penalty"), 0.0)),
        "overnight_convexity_boost": int(_safe_float(gamma_vol_state.get("overnight_convexity_boost"), 0.0)),
    }


def derive_dealer_pressure_trade_modifiers(dealer_pressure_state, direction=None):
    dealer_pressure_state = dealer_pressure_state if isinstance(dealer_pressure_state, dict) else {}
    base_adjustment_score = int(_safe_float(dealer_pressure_state.get("dealer_pressure_adjustment_score"), 0.0))
    alignment_adjustment_score = 0
    adjustment_reasons = []

    dealer_flow_state = str(dealer_pressure_state.get("dealer_flow_state", "HEDGING_NEUTRAL")).upper().strip()
    direction = str(direction or "").upper().strip()

    if dealer_flow_state == "PINNING_DOMINANT":
        alignment_adjustment_score -= 2
        adjustment_reasons.append("pinning_dampens_option_buying")
    elif dealer_flow_state == "TWO_SIDED_INSTABILITY":
        alignment_adjustment_score -= 1
        adjustment_reasons.append("two_sided_hedging_instability")
    elif direction == "CALL" and dealer_flow_state == "UPSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score += 2
        adjustment_reasons.append("upside_hedging_alignment")
    elif direction == "PUT" and dealer_flow_state == "DOWNSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score += 2
        adjustment_reasons.append("downside_hedging_alignment")
    elif direction == "CALL" and dealer_flow_state == "DOWNSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score -= 3
        adjustment_reasons.append("downside_hedging_conflict")
    elif direction == "PUT" and dealer_flow_state == "UPSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score -= 3
        adjustment_reasons.append("upside_hedging_conflict")

    effective_adjustment_score = int(_clip(base_adjustment_score + alignment_adjustment_score, -6, 8))

    return {
        "base_adjustment_score": base_adjustment_score,
        "alignment_adjustment_score": alignment_adjustment_score,
        "effective_adjustment_score": effective_adjustment_score,
        "adjustment_reasons": adjustment_reasons,
        "dealer_hedging_pressure_score": int(_safe_float(dealer_pressure_state.get("dealer_hedging_pressure_score"), 0.0)),
        "dealer_flow_state": str(dealer_pressure_state.get("dealer_flow_state", "HEDGING_NEUTRAL")),
        "upside_hedging_pressure": _safe_float(dealer_pressure_state.get("upside_hedging_pressure"), 0.0),
        "downside_hedging_pressure": _safe_float(dealer_pressure_state.get("downside_hedging_pressure"), 0.0),
        "pinning_pressure_score": _safe_float(dealer_pressure_state.get("pinning_pressure_score"), 0.0),
        "overnight_hedging_risk": _safe_float(dealer_pressure_state.get("overnight_hedging_risk"), 0.0),
        "overnight_hold_allowed": bool(dealer_pressure_state.get("overnight_hold_allowed", True)),
        "overnight_hold_reason": str(dealer_pressure_state.get("overnight_hold_reason", "overnight_hedging_contained")),
        "overnight_dealer_pressure_penalty": int(_safe_float(dealer_pressure_state.get("overnight_dealer_pressure_penalty"), 0.0)),
        "overnight_dealer_pressure_boost": int(_safe_float(dealer_pressure_state.get("overnight_dealer_pressure_boost"), 0.0)),
    }


def derive_option_efficiency_trade_modifiers(option_efficiency_state):
    option_efficiency_state = option_efficiency_state if isinstance(option_efficiency_state, dict) else {}
    adjustment_score = int(_safe_float(option_efficiency_state.get("option_efficiency_adjustment_score"), 0.0))
    return {
        "effective_adjustment_score": adjustment_score,
        "expected_move_points": option_efficiency_state.get("expected_move_points"),
        "expected_move_pct": option_efficiency_state.get("expected_move_pct"),
        "expected_move_quality": str(option_efficiency_state.get("expected_move_quality", "UNAVAILABLE")),
        "target_reachability_score": int(_safe_float(option_efficiency_state.get("target_reachability_score"), 50.0)),
        "premium_efficiency_score": int(_safe_float(option_efficiency_state.get("premium_efficiency_score"), 50.0)),
        "strike_efficiency_score": int(_safe_float(option_efficiency_state.get("strike_efficiency_score"), 50.0)),
        "option_efficiency_score": int(_safe_float(option_efficiency_state.get("option_efficiency_score"), 50.0)),
        "option_efficiency_adjustment_score": adjustment_score,
        "overnight_hold_allowed": bool(option_efficiency_state.get("overnight_hold_allowed", True)),
        "overnight_hold_reason": str(option_efficiency_state.get("overnight_hold_reason", "overnight_option_efficiency_contained")),
        "overnight_option_efficiency_penalty": int(_safe_float(option_efficiency_state.get("overnight_option_efficiency_penalty"), 0.0)),
        "strike_moneyness_bucket": str(option_efficiency_state.get("strike_moneyness_bucket", "UNKNOWN")),
        "strike_distance_from_spot": option_efficiency_state.get("strike_distance_from_spot"),
        "payoff_efficiency_hint": str(option_efficiency_state.get("payoff_efficiency_hint", "unknown")),
    }


def _extract_nearest_vacuum_gap_pct(spot, vacuum_zones):
    if not vacuum_zones:
        return None

    best_gap = None
    for zone in vacuum_zones:
        try:
            low, high = zone
            if low <= spot <= high:
                gap = 0.0
            elif spot < low:
                gap = ((low - spot) / max(spot, 1e-6)) * 100.0
            else:
                gap = ((spot - high) / max(spot, 1e-6)) * 100.0
        except Exception:
            continue

        if best_gap is None or gap < best_gap:
            best_gap = gap

    if best_gap is None:
        return None

    return round(_clip(best_gap, 0.0, 1.5), 4)


def _extract_hedge_flow_value(hedging_flow):
    if hedging_flow is None:
        return None

    if isinstance(hedging_flow, dict):
        for key in ("hedging_flow", "net_flow", "flow_ratio", "bias_score"):
            value = hedging_flow.get(key)
            if value is not None:
                return _clip(_safe_float(value), -1.0, 1.0)
        return None

    if isinstance(hedging_flow, (int, float)):
        return _clip(float(hedging_flow), -1.0, 1.0)

    return None


def _categorical_flow_score(value):
    return {
        "BULLISH_FLOW": 1.0,
        "BEARISH_FLOW": -1.0,
        "MIXED_FLOW": 0.0,
        "NEUTRAL_FLOW": 0.0,
    }.get(value, 0.0)


def _normalize_validation_dict(validation):
    return validation if isinstance(validation, dict) else {}


def _compute_data_quality(*, spot_validation, option_chain_validation, analytics_state, probability_state):
    spot_validation = _normalize_validation_dict(spot_validation)
    option_chain_validation = _normalize_validation_dict(option_chain_validation)
    analytics_state = analytics_state if isinstance(analytics_state, dict) else {}
    probability_state = probability_state if isinstance(probability_state, dict) else {}

    score = 100
    reasons = []
    analytics_missing = []

    if not spot_validation.get("is_valid", True):
        score -= 45
        reasons.append("invalid_spot_snapshot")
    elif spot_validation.get("is_stale"):
        score -= 10
        reasons.append("stale_spot_snapshot")

    if not option_chain_validation.get("is_valid", True):
        score -= 45
        reasons.append("invalid_option_chain")
    elif option_chain_validation.get("is_stale"):
        score -= 10
        reasons.append("stale_option_chain")

    provider_health = option_chain_validation.get("provider_health") or {}
    provider_summary = provider_health.get("summary_status")
    if provider_summary == "WEAK":
        score -= 18
        reasons.append("weak_provider_health")
    elif provider_summary == "CAUTION":
        score -= 8
        reasons.append("provider_health_caution")

    critical_analytics = {
        "flip": analytics_state.get("flip"),
        "gamma_regime": analytics_state.get("gamma_regime"),
        "final_flow_signal": analytics_state.get("final_flow_signal"),
        "dealer_pos": analytics_state.get("dealer_pos"),
        "hedging_bias": analytics_state.get("hedging_bias"),
        "vol_regime": analytics_state.get("vol_regime"),
    }

    for name, value in critical_analytics.items():
        if value in (None, "", "UNKNOWN"):
            analytics_missing.append(name)

    if analytics_missing:
        score -= min(len(analytics_missing) * 6, 24)
        reasons.append(f"missing_critical_analytics:{','.join(sorted(analytics_missing))}")

    if probability_state.get("rule_move_probability") is None and probability_state.get("ml_move_probability") is None:
        score -= 10
        reasons.append("missing_all_move_probabilities")
    elif probability_state.get("hybrid_move_probability") is None:
        score -= 5
        reasons.append("missing_hybrid_move_probability")

    score = int(_clip(score, 0, 100))

    if score >= 85:
        status = "STRONG"
    elif score >= 70:
        status = "GOOD"
    elif score >= 55:
        status = "CAUTION"
    else:
        status = "WEAK"

    analytics_quality = {
        "missing_critical": analytics_missing,
        "critical_missing_count": len(analytics_missing),
    }

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "analytics_quality": analytics_quality,
        "fatal": (not spot_validation.get("is_valid", True)) or (not option_chain_validation.get("is_valid", True)),
    }


def classify_spot_vs_flip(spot, flip):
    return classify_spot_vs_flip_for_symbol(None, spot, flip)


def classify_spot_vs_flip_for_symbol(symbol, spot, flip):
    if flip is None:
        return "UNKNOWN"

    flip_buffer = _safe_float(get_microstructure_config(symbol).get("flip_buffer_points"), 25.0)

    if abs(spot - flip) <= flip_buffer:
        return "AT_FLIP"

    if spot > flip:
        return "ABOVE_FLIP"

    return "BELOW_FLIP"


def classify_signal_quality(trade_strength):
    thresholds = get_trade_runtime_thresholds()
    if trade_strength >= thresholds["strong_signal_threshold"]:
        return "STRONG"
    if trade_strength >= thresholds["medium_signal_threshold"]:
        return "MEDIUM"
    if trade_strength >= thresholds["weak_signal_threshold"]:
        return "WEAK"
    return "VERY_WEAK"


def classify_signal_regime(
    *,
    direction,
    adjusted_trade_strength,
    final_flow_signal,
    gamma_regime,
    confirmation_status,
    event_lockdown_flag,
    data_quality_status,
):
    thresholds = get_trade_runtime_thresholds()
    if event_lockdown_flag:
        return "LOCKDOWN"
    if direction is None:
        return "NEUTRAL"

    directional_flow = final_flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"}
    unstable_gamma = gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}

    if (
        adjusted_trade_strength >= thresholds["expansion_bias_threshold"]
        and directional_flow
        and unstable_gamma
        and confirmation_status in {"STRONG_CONFIRMATION", "CONFIRMED"}
    ):
        return "EXPANSION_BIAS"
    if adjusted_trade_strength >= thresholds["directional_bias_threshold"] and directional_flow:
        return "DIRECTIONAL_BIAS"
    if data_quality_status in {"CAUTION", "WEAK"} or confirmation_status == "CONFLICT":
        return "CONFLICTED"
    return "BALANCED"


def classify_execution_regime(*, trade_status, signal_regime, data_quality_score, macro_position_size_multiplier):
    if trade_status in {"DATA_INVALID", "EVENT_LOCKDOWN", "NO_TRADE", "BUDGET_FAIL"}:
        return "BLOCKED"
    if trade_status == "TRADE" and macro_position_size_multiplier < 1.0:
        return "RISK_REDUCED"
    if trade_status == "TRADE":
        return "ACTIVE"
    if signal_regime == "CONFLICTED" or data_quality_score < 70:
        return "OBSERVE"
    return "SETUP"


def normalize_flow_signal(flow_signal_value, smart_money_signal_value):
    bullish_votes = 0
    bearish_votes = 0

    if flow_signal_value == "BULLISH_FLOW":
        bullish_votes += 1
    elif flow_signal_value == "BEARISH_FLOW":
        bearish_votes += 1

    if smart_money_signal_value == "BULLISH_FLOW":
        bullish_votes += 1
    elif smart_money_signal_value == "BEARISH_FLOW":
        bearish_votes += 1

    if bullish_votes > bearish_votes:
        return "BULLISH_FLOW"

    if bearish_votes > bullish_votes:
        return "BEARISH_FLOW"

    return "NEUTRAL_FLOW"


def decide_direction(
    final_flow_signal,
    dealer_pos,
    vol_regime,
    spot_vs_flip,
    gamma_regime,
    hedging_bias,
    gamma_event,
    vanna_regime=None,
    charm_regime=None,
    backtest_mode=False,
):
    direction_weights = get_direction_vote_weights()
    direction_thresholds = get_direction_thresholds()
    bullish_votes = []
    bearish_votes = []

    def add_vote(side, reason):
        weight = float(direction_weights.get(reason, 1.0))
        entry = (reason, round(weight, 2))
        if side == "BULLISH":
            bullish_votes.append(entry)
        elif side == "BEARISH":
            bearish_votes.append(entry)

    if final_flow_signal == "BULLISH_FLOW":
        add_vote("BULLISH", "FLOW")
    elif final_flow_signal == "BEARISH_FLOW":
        add_vote("BEARISH", "FLOW")

    if gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}:
        if hedging_bias == "UPSIDE_ACCELERATION":
            add_vote("BULLISH", "HEDGING_BIAS")
        elif hedging_bias == "DOWNSIDE_ACCELERATION":
            add_vote("BEARISH", "HEDGING_BIAS")

    if gamma_event == "GAMMA_SQUEEZE":
        if spot_vs_flip == "ABOVE_FLIP":
            add_vote("BULLISH", "GAMMA_SQUEEZE")
        elif spot_vs_flip == "BELOW_FLIP":
            add_vote("BEARISH", "GAMMA_SQUEEZE")

    if spot_vs_flip == "ABOVE_FLIP":
        if dealer_pos in {"Long Gamma", "Short Gamma"}:
            add_vote("BULLISH", "GAMMA_FLIP")
    elif spot_vs_flip == "BELOW_FLIP" and dealer_pos == "Short Gamma":
        add_vote("BEARISH", "GAMMA_FLIP")

    if dealer_pos == "Short Gamma" and vol_regime == "VOL_EXPANSION":
        if spot_vs_flip == "ABOVE_FLIP":
            add_vote("BULLISH", "DEALER_VOL")
        elif spot_vs_flip == "BELOW_FLIP":
            add_vote("BEARISH", "DEALER_VOL")

    if vanna_regime == "POSITIVE_VANNA" and spot_vs_flip == "ABOVE_FLIP":
        add_vote("BULLISH", "VANNA")
    elif vanna_regime == "NEGATIVE_VANNA" and spot_vs_flip == "BELOW_FLIP":
        add_vote("BEARISH", "VANNA")

    if charm_regime == "POSITIVE_CHARM" and spot_vs_flip == "ABOVE_FLIP":
        add_vote("BULLISH", "CHARM")
    elif charm_regime == "NEGATIVE_CHARM" and spot_vs_flip == "BELOW_FLIP":
        add_vote("BEARISH", "CHARM")

    if backtest_mode and not bullish_votes and not bearish_votes:
        if spot_vs_flip == "ABOVE_FLIP" and dealer_pos == "Long Gamma":
            add_vote("BULLISH", "BACKTEST_FALLBACK")
        elif spot_vs_flip == "BELOW_FLIP" and dealer_pos == "Short Gamma":
            add_vote("BEARISH", "BACKTEST_FALLBACK")

    bullish_score = round(sum(weight for _, weight in bullish_votes), 2)
    bearish_score = round(sum(weight for _, weight in bearish_votes), 2)
    score_margin = round(abs(bullish_score - bearish_score), 2)

    def build_source(votes):
        return "+".join(reason for reason, _ in votes)

    if (
        bullish_score >= direction_thresholds["min_score"]
        and bullish_score > bearish_score
        and score_margin >= direction_thresholds["min_margin"]
    ):
        return "CALL", build_source(bullish_votes)

    if (
        bearish_score >= direction_thresholds["min_score"]
        and bearish_score > bullish_score
        and score_margin >= direction_thresholds["min_margin"]
    ):
        return "PUT", build_source(bearish_votes)

    return None, None


def _summarize_market_gamma(market_gex):
    if isinstance(market_gex, dict):
        summary = {}
        for key in ("total_gamma", "call_gamma", "put_gamma", "net_gamma"):
            value = market_gex.get(key)
            if value is not None:
                summary[key] = round(_safe_float(value), 2)
        return summary or market_gex
    return market_gex


def _extract_probability(result):
    if result is None:
        return None
    if isinstance(result, dict):
        for key in ("probability", "move_probability", "large_move_probability", "score"):
            if key in result:
                value = _safe_float(result.get(key), None)
                if value is not None:
                    return round(_clip(value, 0.05, 0.95), 2)
        return None
    value = _safe_float(result, None)
    if value is None:
        return None
    return round(_clip(value, 0.05, 0.95), 2)


def _collect_market_state(df, spot, symbol=None, prev_df=None):
    gamma = _call_first(
        gamma_exposure_mod,
        ["calculate_gamma_exposure", "calculate_gex"],
        df,
        spot,
        default=0,
    )

    flip = _call_first(
        gamma_flip_mod,
        ["gamma_flip_level", "find_gamma_flip"],
        df,
        spot=spot,
        default=None,
    )

    dealer_metrics = _call_first(
        dealer_inventory_mod,
        ["dealer_inventory_metrics"],
        df,
        default={},
    ) or {}

    dealer_pos = dealer_metrics.get("position") or _call_first(
        dealer_inventory_mod,
        ["dealer_inventory_position", "dealer_inventory"],
        df,
        default="Unknown",
    )

    vol_regime = _call_first(
        volatility_regime_mod,
        ["detect_volatility_regime", "volatility_regime"],
        df,
        default="UNKNOWN",
    )

    gamma_path_result = _call_first(
        dealer_gamma_path_mod,
        ["simulate_gamma_path"],
        df,
        spot,
        default=([], []),
    )
    if isinstance(gamma_path_result, tuple) and len(gamma_path_result) == 2:
        prices, gamma_curve = gamma_path_result
    else:
        prices, gamma_curve = [], []

    gamma_event = _call_first(
        dealer_gamma_path_mod,
        ["detect_gamma_squeeze"],
        prices,
        gamma_curve,
        default="NORMAL",
    )

    flow_signal_value = _call_first(
        options_flow_imbalance_mod,
        ["flow_signal", "calculate_flow_signal"],
        df,
        spot=spot,
        default="NEUTRAL_FLOW",
    )

    smart_money_signal_value = _call_first(
        smart_money_flow_mod,
        ["smart_money_signal", "classify_flow"],
        df,
        spot=spot,
        default="NEUTRAL_FLOW",
    )

    final_flow_signal = normalize_flow_signal(flow_signal_value, smart_money_signal_value)

    liquidity_levels = _call_first(
        liquidity_heatmap_mod,
        ["strongest_liquidity_levels", "build_liquidity_heatmap"],
        df,
        default=[],
    )
    if isinstance(liquidity_levels, pd.Series):
        liquidity_levels = list(liquidity_levels.index[:5])
    elif isinstance(liquidity_levels, pd.Index):
        liquidity_levels = list(liquidity_levels[:5])
    elif liquidity_levels is None:
        liquidity_levels = []
    liquidity_levels = [_to_python_number(x) for x in liquidity_levels]

    voids = _call_first(
        liquidity_void_mod,
        ["detect_liquidity_voids", "detect_liquidity_void"],
        df,
        default=[],
    )
    void_signal = _call_first(
        liquidity_void_mod,
        ["liquidity_void_signal"],
        spot,
        voids,
        default=None,
    )

    vacuum_zones = _call_first(
        liquidity_vacuum_mod,
        ["detect_liquidity_vacuum"],
        df,
        default=[],
    )
    vacuum_zones = _clean_zone_list(vacuum_zones)
    vacuum_state = _call_first(
        liquidity_vacuum_mod,
        ["vacuum_direction"],
        spot,
        vacuum_zones,
        default="NORMAL",
    )

    walls = _call_first(
        gamma_walls_mod,
        ["classify_walls"],
        df,
        default={},
    ) or {}
    support_wall = _to_python_number(walls.get("support_wall") if isinstance(walls, dict) else None)
    resistance_wall = _to_python_number(walls.get("resistance_wall") if isinstance(walls, dict) else None)

    market_gex = _call_first(
        market_gamma_map_mod,
        ["calculate_market_gamma"],
        df,
        default=None,
    )
    market_gamma_summary = _summarize_market_gamma(market_gex)
    gamma_regime = _call_first(
        market_gamma_map_mod,
        ["market_gamma_regime"],
        market_gex,
        default=None,
    )
    gamma_clusters = _call_first(
        market_gamma_map_mod,
        ["largest_gamma_strikes"],
        market_gex,
        default=[],
    )
    gamma_clusters = [_to_python_number(x) for x in gamma_clusters] if gamma_clusters else []

    greek_exposures = summarize_greek_exposures(df)
    if gamma_regime is None:
        if flip is None:
            gamma_regime = "UNKNOWN"
        elif spot > flip:
            gamma_regime = "LONG_GAMMA_ZONE"
        else:
            gamma_regime = "SHORT_GAMMA_ZONE"

    hedging_flow = _call_first(
        dealer_hedging_flow_mod,
        ["dealer_hedging_flow"],
        df,
        default=None,
    )
    hedging_sim = _call_first(
        dealer_hedging_simulator_mod,
        ["simulate_dealer_hedging"],
        df,
        default={},
    )
    hedging_bias = _call_first(
        dealer_hedging_simulator_mod,
        ["hedging_bias"],
        hedging_sim,
        default=None,
    )

    intraday_gamma_state = None
    if prev_df is not None:
        intraday_gamma_state = _call_first(
            intraday_gamma_shift_mod,
            ["gamma_shift_signal", "detect_gamma_shift"],
            prev_df,
            df,
            spot,
            default=None,
        )

    atm_iv = _call_first(
        volatility_surface_mod,
        ["atm_vol"],
        df,
        spot,
        default=None,
    )
    surface_regime = None
    if atm_iv is not None:
        surface_regime = _call_first(
            volatility_surface_mod,
            ["vol_regime"],
            atm_iv,
            default=None,
        )

    spot_vs_flip = classify_spot_vs_flip_for_symbol(symbol, spot, flip)
    dealer_liquidity_map = build_dealer_liquidity_map(
        spot=spot,
        gamma_flip=flip,
        liquidity_levels=liquidity_levels,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        gamma_clusters=gamma_clusters,
        vacuum_zones=vacuum_zones,
    )

    return {
        "gamma": gamma,
        "flip": flip,
        "dealer_metrics": dealer_metrics,
        "dealer_pos": dealer_pos,
        "vol_regime": vol_regime,
        "gamma_event": gamma_event,
        "flow_signal_value": flow_signal_value,
        "smart_money_signal_value": smart_money_signal_value,
        "final_flow_signal": final_flow_signal,
        "liquidity_levels": liquidity_levels,
        "voids": voids,
        "void_signal": void_signal,
        "vacuum_zones": vacuum_zones,
        "vacuum_state": vacuum_state,
        "support_wall": support_wall,
        "resistance_wall": resistance_wall,
        "market_gamma_summary": market_gamma_summary,
        "gamma_regime": gamma_regime,
        "gamma_clusters": gamma_clusters,
        "greek_exposures": greek_exposures,
        "hedging_flow": hedging_flow,
        "hedging_bias": hedging_bias,
        "intraday_gamma_state": intraday_gamma_state,
        "atm_iv": atm_iv,
        "surface_regime": surface_regime,
        "spot_vs_flip": spot_vs_flip,
        "dealer_liquidity_map": dealer_liquidity_map,
    }


def _compute_probability_state(
    df,
    *,
    spot,
    symbol,
    market_state,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None,
):
    model_features = _call_first(
        feature_builder_mod,
        ["build_features"],
        df,
        spot=spot,
        gamma_regime=market_state["gamma_regime"],
        final_flow_signal=market_state["final_flow_signal"],
        vol_regime=market_state["vol_regime"],
        hedging_bias=market_state["hedging_bias"],
        spot_vs_flip=market_state["spot_vs_flip"],
        vacuum_state=market_state["vacuum_state"],
        atm_iv=market_state["atm_iv"],
        default=None,
    )

    nearest_vacuum_gap_pct = _extract_nearest_vacuum_gap_pct(
        spot=spot,
        vacuum_zones=market_state["vacuum_zones"],
    )
    hedge_flow_value = _extract_hedge_flow_value(market_state["hedging_flow"])
    flow_imbalance = (
        0.5 * _categorical_flow_score(market_state["flow_signal_value"])
        + 0.5 * _categorical_flow_score(market_state["smart_money_signal_value"])
    )
    gamma_flip_distance_pct = _compute_gamma_flip_distance_pct(
        spot_price=spot,
        gamma_flip=market_state["flip"],
    )
    vacuum_strength = _map_vacuum_strength(
        vacuum_state=market_state["vacuum_state"],
        liquidity_voids=market_state["voids"],
        nearest_vacuum_gap_pct=nearest_vacuum_gap_pct,
    )
    hedging_flow_ratio = _map_hedging_flow_ratio(
        hedging_bias=market_state["hedging_bias"],
        hedge_flow_value=hedge_flow_value,
    )
    smart_money_flow_score = _map_smart_money_flow_score(
        smart_money_flow=market_state["smart_money_signal_value"],
        flow_imbalance=flow_imbalance,
    )
    atm_iv_percentile = _compute_atm_iv_percentile(atm_iv=market_state["atm_iv"])
    intraday_range_pct = _compute_intraday_range_pct(
        symbol=symbol,
        spot_price=spot,
        day_high=day_high,
        day_low=day_low,
        day_open=day_open,
        prev_close=prev_close,
        lookback_avg_range_pct=lookback_avg_range_pct,
    )

    rule_move_probability = _call_first(
        large_move_probability_mod,
        ["large_move_probability", "predict_large_move_probability"],
        market_state["gamma_regime"],
        market_state["vacuum_state"],
        market_state["hedging_bias"],
        market_state["final_flow_signal"],
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        vacuum_strength=vacuum_strength,
        hedging_flow_ratio=hedging_flow_ratio,
        smart_money_flow_score=smart_money_flow_score,
        atm_iv_percentile=atm_iv_percentile,
        intraday_range_pct=intraday_range_pct,
        default=None,
    )
    rule_move_probability = _extract_probability(rule_move_probability)

    ml_move_probability = None
    predictor = _get_move_predictor()
    if predictor is not None:
        try:
            if model_features is not None:
                ml_move_probability = predictor.predict_probability(model_features)
            ml_move_probability = _extract_probability(ml_move_probability)
            if ml_move_probability is not None:
                ml_move_probability = round(_clip(float(ml_move_probability), 0.05, 0.95), 2)
        except Exception:
            ml_move_probability = None

    hybrid_move_probability = _blend_move_probability(
        rule_prob=rule_move_probability,
        ml_prob=ml_move_probability,
    )

    return {
        "rule_move_probability": rule_move_probability,
        "ml_move_probability": ml_move_probability,
        "hybrid_move_probability": hybrid_move_probability,
        "model_features": model_features,
        "components": {
            "gamma_flip_distance_pct": gamma_flip_distance_pct,
            "nearest_vacuum_gap_pct": nearest_vacuum_gap_pct,
            "vacuum_strength": vacuum_strength,
            "hedging_flow_ratio": hedging_flow_ratio,
            "smart_money_flow_score": smart_money_flow_score,
            "atm_iv_percentile": atm_iv_percentile,
            "intraday_range_pct": intraday_range_pct,
            "flow_imbalance": round(flow_imbalance, 3),
            "hedge_flow_value": hedge_flow_value,
            "day_high": day_high,
            "day_low": day_low,
            "day_open": day_open,
            "prev_close": prev_close,
            "lookback_avg_range_pct": lookback_avg_range_pct,
        },
    }


def _compute_signal_state(
    *,
    spot,
    symbol,
    day_open,
    prev_close,
    intraday_range_pct,
    backtest_mode,
    market_state,
    probability_state,
):
    direction, direction_source = decide_direction(
        final_flow_signal=market_state["final_flow_signal"],
        dealer_pos=market_state["dealer_pos"],
        vol_regime=market_state["vol_regime"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_regime=market_state["gamma_regime"],
        hedging_bias=market_state["hedging_bias"],
        gamma_event=market_state["gamma_event"],
        vanna_regime=market_state["greek_exposures"].get("vanna_regime"),
        charm_regime=market_state["greek_exposures"].get("charm_regime"),
        backtest_mode=backtest_mode,
    )

    if direction is None:
        return {
            "direction": None,
            "direction_source": None,
            "trade_strength": 0,
            "scoring_breakdown": empty_scoring_breakdown(),
            "confirmation": empty_confirmation_state(),
        }

    trade_strength, scoring_breakdown = compute_trade_strength(
        direction=direction,
        flow_signal_value=market_state["flow_signal_value"],
        smart_money_signal_value=market_state["smart_money_signal_value"],
        gamma_event=market_state["gamma_event"],
        dealer_pos=market_state["dealer_pos"],
        vol_regime=market_state["vol_regime"],
        void_signal=market_state["void_signal"],
        vacuum_state=market_state["vacuum_state"],
        spot_vs_flip=market_state["spot_vs_flip"],
        hedging_bias=market_state["hedging_bias"],
        gamma_regime=market_state["gamma_regime"],
        intraday_gamma_state=market_state["intraday_gamma_state"],
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        spot=spot,
        next_support=market_state["dealer_liquidity_map"].get("next_support"),
        next_resistance=market_state["dealer_liquidity_map"].get("next_resistance"),
        squeeze_zone=market_state["dealer_liquidity_map"].get("gamma_squeeze_zone"),
        large_move_probability=probability_state["hybrid_move_probability"],
        ml_move_probability=probability_state["ml_move_probability"],
        proximity_buffer=get_microstructure_config(symbol).get("wall_proximity_points", 50.0),
    )

    confirmation = compute_confirmation_filters(
        direction=direction,
        spot=spot,
        symbol=symbol,
        day_open=day_open,
        prev_close=prev_close,
        intraday_range_pct=intraday_range_pct,
        final_flow_signal=market_state["final_flow_signal"],
        hedging_bias=market_state["hedging_bias"],
        gamma_event=market_state["gamma_event"],
        hybrid_move_probability=probability_state["hybrid_move_probability"],
        spot_vs_flip=market_state["spot_vs_flip"],
    )

    return {
        "direction": direction,
        "direction_source": direction_source,
        "trade_strength": trade_strength,
        "scoring_breakdown": scoring_breakdown,
        "confirmation": confirmation,
    }
