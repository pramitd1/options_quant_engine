from __future__ import annotations

from config.probability_feature_policy import get_probability_feature_policy_config
from config.symbol_microstructure import get_microstructure_config
import models.feature_builder as feature_builder_mod
import models.large_move_probability as large_move_probability_mod
import models.ml_move_predictor as ml_move_predictor_mod

from .common import _call_first, _clip, _safe_float


_MOVE_PREDICTOR = None


def _map_vacuum_strength(vacuum_state, liquidity_voids=None, nearest_vacuum_gap_pct=None):
    cfg = get_probability_feature_policy_config()
    base = {
        "BREAKOUT_ZONE": cfg.vacuum_breakout_strength,
        "NEAR_VACUUM": cfg.vacuum_near_strength,
        "VACUUM_WATCH": cfg.vacuum_watch_strength,
    }.get(vacuum_state, cfg.vacuum_default_strength)

    if nearest_vacuum_gap_pct is not None:
        gap = _clip(_safe_float(nearest_vacuum_gap_pct), 0.0, cfg.vacuum_gap_pct_cap)
        proximity_boost = 1.0 - (gap / max(cfg.vacuum_gap_pct_cap, 1e-6))
        base = (cfg.vacuum_gap_base_weight * base) + (cfg.vacuum_gap_proximity_weight * proximity_boost)

    if liquidity_voids is not None:
        try:
            base += min(len(liquidity_voids), int(cfg.vacuum_void_count_cap)) * cfg.vacuum_void_increment
        except Exception:
            pass

    return round(_clip(base, 0.0, 1.0), 3)


def _map_hedging_flow_ratio(hedging_bias, hedge_flow_value=None):
    cfg = get_probability_feature_policy_config()
    if hedge_flow_value is not None:
        return round(_clip(_safe_float(hedge_flow_value), -1.0, 1.0), 3)

    mapping = {
        "UPSIDE_ACCELERATION": cfg.hedging_bias_upside_acceleration_score,
        "DOWNSIDE_ACCELERATION": cfg.hedging_bias_downside_acceleration_score,
        "UPSIDE_PINNING": cfg.hedging_bias_upside_pinning_score,
        "DOWNSIDE_PINNING": cfg.hedging_bias_downside_pinning_score,
        "PINNING": cfg.hedging_bias_pinning_score,
    }
    return round(mapping.get(hedging_bias, 0.0), 3)


def _map_smart_money_flow_score(smart_money_flow, flow_imbalance=None):
    cfg = get_probability_feature_policy_config()
    base = {
        "BULLISH_FLOW": cfg.smart_money_bullish_score,
        "BEARISH_FLOW": cfg.smart_money_bearish_score,
        "MIXED_FLOW": cfg.smart_money_neutral_score,
        "NEUTRAL_FLOW": cfg.smart_money_neutral_score,
    }.get(smart_money_flow, cfg.smart_money_neutral_score)

    if flow_imbalance is not None:
        base = (
            cfg.smart_money_categorical_weight * base
            + cfg.smart_money_flow_imbalance_weight * _clip(_safe_float(flow_imbalance), -1.0, 1.0)
        )

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
    cfg = get_probability_feature_policy_config()
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
            realized_range_pct = max(anchor_moves) * cfg.intraday_range_anchor_multiplier

    if realized_range_pct is None:
        return None

    baseline_floor = _safe_float(
        micro_cfg.get("range_baseline_floor_pct"),
        cfg.intraday_range_baseline_floor_pct,
    )
    baseline = avg_range if avg_range not in (None, 0) else baseline_floor
    baseline = max(baseline, baseline_floor)
    normalized = realized_range_pct / max(baseline, cfg.intraday_range_denominator_floor_pct)
    return round(_clip(normalized, 0.0, cfg.intraday_range_clip_cap), 4)


def _compute_atm_iv_percentile(atm_iv):
    cfg = get_probability_feature_policy_config()
    iv = _safe_float(atm_iv, None)
    if iv is None:
        return None

    pct = (iv - cfg.atm_iv_low) / max(cfg.atm_iv_high - cfg.atm_iv_low, 1e-6)
    return round(_clip(pct, 0.0, 1.0), 4)


def _blend_move_probability(rule_prob, ml_prob):
    cfg = get_probability_feature_policy_config()
    rule_prob = _safe_float(rule_prob, cfg.probability_default_rule)
    if ml_prob is None:
        return round(_clip(rule_prob, cfg.probability_floor, cfg.probability_ceiling), 2)

    ml_prob = _safe_float(ml_prob, rule_prob)
    hybrid = (cfg.probability_rule_weight * rule_prob) + (cfg.probability_ml_weight * ml_prob)
    hybrid = cfg.probability_intercept + (cfg.probability_scale * hybrid)
    return round(_clip(hybrid, cfg.probability_floor, cfg.probability_ceiling), 2)


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


def _extract_nearest_vacuum_gap_pct(spot, vacuum_zones):
    cfg = get_probability_feature_policy_config()
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
    return round(_clip(best_gap, 0.0, cfg.vacuum_gap_pct_cap), 4)


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


def _extract_probability(result):
    cfg = get_probability_feature_policy_config()
    if result is None:
        return None
    if isinstance(result, dict):
        for key in ("probability", "move_probability", "large_move_probability", "score"):
            if key in result:
                value = _safe_float(result.get(key), None)
                if value is not None:
                    return round(_clip(value, cfg.probability_floor, cfg.probability_ceiling), 2)
        return None
    value = _safe_float(result, None)
    if value is None:
        return None
    return round(_clip(value, cfg.probability_floor, cfg.probability_ceiling), 2)


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
    cfg = get_probability_feature_policy_config()
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
        cfg.categorical_flow_weight * _categorical_flow_score(market_state["flow_signal_value"])
        + cfg.smart_money_flow_weight * _categorical_flow_score(market_state["smart_money_signal_value"])
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
                ml_move_probability = round(
                    _clip(float(ml_move_probability), cfg.probability_floor, cfg.probability_ceiling),
                    2,
                )
        except Exception:
            ml_move_probability = None

    return {
        "rule_move_probability": rule_move_probability,
        "ml_move_probability": ml_move_probability,
        "hybrid_move_probability": _blend_move_probability(
            rule_prob=rule_move_probability,
            ml_prob=ml_move_probability,
        ),
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
