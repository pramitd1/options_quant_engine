"""
Refactored Trading Engine

Adds:
- Dealer Liquidity Map Predictor
- Model integration from models/
- Hybrid move probability
- Clean liquidity vacuum zone formatting
- Proper ML move probability wiring
- Optional budget-aware trade sizing
- Separate backtest mode with lower trade threshold

Enhancements:
- Continuous large-move probability methodology
- Rule/ML probability diagnostics
- Better calibrated hybrid move probability
- Real intraday range feature using spot session data
"""

import pandas as pd

from config.settings import (
    LOT_SIZE,
    NUMBER_OF_LOTS,
    MAX_CAPITAL_PER_TRADE,
    BACKTEST_MIN_TRADE_STRENGTH
)

from strategy.exit_model import calculate_exit
from strategy.trade_strength import compute_trade_strength
from strategy.budget_optimizer import optimize_lots

from analytics import gamma_exposure as gamma_exposure_mod
from analytics import gamma_flip as gamma_flip_mod
from analytics import dealer_inventory as dealer_inventory_mod
from analytics import volatility_regime as volatility_regime_mod

from analytics import dealer_gamma_path as dealer_gamma_path_mod
from analytics import options_flow_imbalance as options_flow_imbalance_mod
from analytics import liquidity_heatmap as liquidity_heatmap_mod
from analytics import liquidity_void as liquidity_void_mod

from analytics import dealer_hedging_flow as dealer_hedging_flow_mod
from analytics import dealer_hedging_simulator as dealer_hedging_simulator_mod
from analytics import market_gamma_map as market_gamma_map_mod
from analytics import gamma_walls as gamma_walls_mod
from analytics import intraday_gamma_shift as intraday_gamma_shift_mod
from analytics import smart_money_flow as smart_money_flow_mod
from analytics import liquidity_vacuum as liquidity_vacuum_mod
from analytics import volatility_surface as volatility_surface_mod
from analytics.dealer_liquidity_map import build_dealer_liquidity_map

import models.feature_builder as feature_builder_mod
import models.large_move_probability as large_move_probability_mod
import models.ml_move_predictor as ml_move_predictor_mod


MIN_TRADE_STRENGTH = 45
STRONG_SIGNAL_THRESHOLD = 75
MEDIUM_SIGNAL_THRESHOLD = 60
WEAK_SIGNAL_THRESHOLD = 40


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
    spot_price=None,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None
):
    """
    Returns a normalized 0..1.5 measure of how expanded today's underlying move is.

    Priority:
    1. Use day_high/day_low if available
    2. Otherwise use max distance from open / prev_close as fallback
    3. Normalize by recent average range if available, else by fixed baseline
    """
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

    baseline = avg_range if avg_range not in (None, 0) else 0.9
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


def normalize_option_chain(option_chain):
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

    if "DELTA" not in df.columns:
        deltas = []
        spot_guess = df["strikePrice"].median() if "strikePrice" in df.columns else 0

        for _, row in df.iterrows():
            strike = row["strikePrice"]
            opt_type = row["OPTION_TYP"]

            if strike == spot_guess:
                delta = 0.5 if opt_type == "CE" else -0.5
            elif strike < spot_guess:
                delta = 0.7 if opt_type == "CE" else -0.3
            else:
                delta = 0.3 if opt_type == "CE" else -0.7

            deltas.append(delta)

        df["DELTA"] = deltas

    if "GAMMA" not in df.columns:
        gammas = []
        spot_guess = df["strikePrice"].median() if "strikePrice" in df.columns else 0

        for _, row in df.iterrows():
            strike = row["strikePrice"]
            distance = abs(strike - spot_guess)
            gammas.append(1 / (1 + distance))

        df["GAMMA"] = gammas

    if "EXPIRY_DT" not in df.columns:
        df["EXPIRY_DT"] = "NEAR"

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


def _extract_nearest_vacuum_gap_pct(spot, vacuum_zones):
    if not vacuum_zones:
        return None

    spot_val = _safe_float(spot, None)
    if spot_val in (None, 0):
        return None

    min_gap = None

    for zone in vacuum_zones:
        try:
            low, high = zone
            low = float(low)
            high = float(high)

            if low <= spot_val <= high:
                gap = 0.0
            elif spot_val < low:
                gap = low - spot_val
            else:
                gap = spot_val - high

            if min_gap is None or gap < min_gap:
                min_gap = gap
        except Exception:
            continue

    if min_gap is None:
        return None

    return round((min_gap / spot_val) * 100.0, 4)


def _extract_hedge_flow_value(hedging_flow):
    if hedging_flow is None:
        return None

    if isinstance(hedging_flow, (float, int)):
        return round(_clip(float(hedging_flow), -1.0, 1.0), 3)

    if isinstance(hedging_flow, dict):
        candidate_keys = [
            "flow_ratio",
            "net_flow_ratio",
            "hedging_flow_ratio",
            "normalized_flow",
            "normalized_value",
            "value",
            "score",
            "bias_score",
        ]
        for key in candidate_keys:
            if key in hedging_flow:
                try:
                    return round(_clip(float(hedging_flow[key]), -1.0, 1.0), 3)
                except Exception:
                    continue

    return None


def _categorical_flow_score(value):
    mapping = {
        "BULLISH_FLOW": 0.75,
        "BEARISH_FLOW": -0.75,
        "MIXED_FLOW": 0.0,
        "NEUTRAL_FLOW": 0.0,
    }
    return mapping.get(value, 0.0)


def choose_strike(option_chain, spot, direction):
    strikes = sorted(option_chain["strikePrice"].dropna().unique().tolist())

    if not strikes:
        return None

    if direction == "CALL":
        candidates = [s for s in strikes if s >= spot]
        return min(candidates) if candidates else max(strikes)

    if direction == "PUT":
        candidates = [s for s in strikes if s <= spot]
        return max(candidates) if candidates else min(strikes)

    return None


def classify_spot_vs_flip(spot, flip):
    if flip is None:
        return "UNKNOWN"

    if abs(spot - flip) <= 25:
        return "AT_FLIP"

    if spot > flip:
        return "ABOVE_FLIP"

    return "BELOW_FLIP"


def classify_signal_quality(trade_strength):
    if trade_strength >= STRONG_SIGNAL_THRESHOLD:
        return "STRONG"
    if trade_strength >= MEDIUM_SIGNAL_THRESHOLD:
        return "MEDIUM"
    if trade_strength >= WEAK_SIGNAL_THRESHOLD:
        return "WEAK"
    return "VERY_WEAK"


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
    backtest_mode=False
):
    if final_flow_signal == "BULLISH_FLOW":
        return "CALL", "FLOW"

    if final_flow_signal == "BEARISH_FLOW":
        return "PUT", "FLOW"

    if gamma_regime in ["NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"]:
        if hedging_bias == "UPSIDE_ACCELERATION":
            return "CALL", "HEDGING_BIAS"
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            return "PUT", "HEDGING_BIAS"

    if gamma_event == "GAMMA_SQUEEZE":
        if spot_vs_flip == "ABOVE_FLIP":
            return "CALL", "GAMMA_SQUEEZE"
        if spot_vs_flip == "BELOW_FLIP":
            return "PUT", "GAMMA_SQUEEZE"

    if spot_vs_flip == "BELOW_FLIP" and vol_regime == "VOL_EXPANSION":
        if dealer_pos == "Short Gamma":
            return "PUT", "GAMMA_FLIP"
        if dealer_pos == "Long Gamma":
            return "CALL", "GAMMA_FLIP"

    if spot_vs_flip == "ABOVE_FLIP":
        if dealer_pos == "Long Gamma":
            return "CALL", "GAMMA_FLIP"
        if dealer_pos == "Short Gamma":
            return "PUT", "GAMMA_FLIP"

    if dealer_pos == "Short Gamma" and vol_regime == "VOL_EXPANSION":
        return "PUT", "DEALER_VOL"

    if dealer_pos == "Long Gamma" and vol_regime in ["NORMAL_VOL", "VOL_EXPANSION"]:
        return "CALL", "DEALER_VOL"

    if backtest_mode:
        if smart_money_signal_value := final_flow_signal:
            if smart_money_signal_value == "NEUTRAL_FLOW":
                if dealer_pos == "Long Gamma":
                    return "CALL", "BACKTEST_FALLBACK"
                if dealer_pos == "Short Gamma":
                    return "PUT", "BACKTEST_FALLBACK"

        if spot_vs_flip == "BELOW_FLIP":
            return "CALL", "BACKTEST_FALLBACK"
        if spot_vs_flip == "ABOVE_FLIP":
            return "PUT", "BACKTEST_FALLBACK"

    return None, None


def _summarize_market_gamma(market_gex):
    if market_gex is None:
        return None

    if hasattr(market_gex, "sum"):
        try:
            return float(market_gex.sum())
        except Exception:
            return None

    try:
        return float(market_gex)
    except Exception:
        return None


def _extract_probability(result):
    if result is None:
        return None

    try:
        if isinstance(result, (float, int)):
            return float(result)
    except Exception:
        pass

    if hasattr(result, "__len__") and not isinstance(result, str):
        try:
            if len(result) > 0:
                return float(result[0])
        except Exception:
            pass

    return None


def generate_trade(
    symbol,
    spot,
    option_chain,
    previous_chain=None,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None,
    apply_budget_constraint=False,
    requested_lots=NUMBER_OF_LOTS,
    lot_size=LOT_SIZE,
    max_capital=MAX_CAPITAL_PER_TRADE,
    backtest_mode=False
):
    if option_chain is None or option_chain.empty:
        return None

    df = normalize_option_chain(option_chain)
    prev_df = normalize_option_chain(previous_chain) if previous_chain is not None else None

    gamma = _call_first(
        gamma_exposure_mod,
        ["calculate_gamma_exposure", "calculate_gex"],
        df,
        spot,
        default=0
    )

    flip = _call_first(
        gamma_flip_mod,
        ["gamma_flip_level", "find_gamma_flip"],
        df,
        default=None
    )

    dealer_pos = _call_first(
        dealer_inventory_mod,
        ["dealer_inventory_position", "dealer_inventory"],
        df,
        default="Unknown"
    )

    vol_regime = _call_first(
        volatility_regime_mod,
        ["detect_volatility_regime", "volatility_regime"],
        df,
        default="UNKNOWN"
    )

    gamma_path_result = _call_first(
        dealer_gamma_path_mod,
        ["simulate_gamma_path"],
        df,
        spot,
        default=([], [])
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
        default="NORMAL"
    )

    flow_signal_value = _call_first(
        options_flow_imbalance_mod,
        ["flow_signal", "calculate_flow_signal"],
        df,
        default="NEUTRAL_FLOW"
    )

    smart_money_signal_value = _call_first(
        smart_money_flow_mod,
        ["smart_money_signal", "classify_flow"],
        df,
        default="NEUTRAL_FLOW"
    )

    final_flow_signal = normalize_flow_signal(
        flow_signal_value,
        smart_money_signal_value
    )

    liquidity_levels = _call_first(
        liquidity_heatmap_mod,
        ["strongest_liquidity_levels", "build_liquidity_heatmap"],
        df,
        default=[]
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
        default=[]
    )

    void_signal = _call_first(
        liquidity_void_mod,
        ["liquidity_void_signal"],
        spot,
        voids,
        default=None
    )

    vacuum_zones = _call_first(
        liquidity_vacuum_mod,
        ["detect_liquidity_vacuum"],
        df,
        default=[]
    )

    vacuum_zones = _clean_zone_list(vacuum_zones)

    vacuum_state = _call_first(
        liquidity_vacuum_mod,
        ["vacuum_direction"],
        spot,
        vacuum_zones,
        default="NORMAL"
    )

    walls = _call_first(
        gamma_walls_mod,
        ["classify_walls"],
        df,
        default={}
    ) or {}

    support_wall = walls.get("support_wall") if isinstance(walls, dict) else None
    resistance_wall = walls.get("resistance_wall") if isinstance(walls, dict) else None

    support_wall = _to_python_number(support_wall)
    resistance_wall = _to_python_number(resistance_wall)

    market_gex = _call_first(
        market_gamma_map_mod,
        ["calculate_market_gamma"],
        df,
        default=None
    )

    market_gamma_summary = _summarize_market_gamma(market_gex)

    gamma_regime = _call_first(
        market_gamma_map_mod,
        ["market_gamma_regime"],
        market_gex,
        default=None
    )

    gamma_clusters = _call_first(
        market_gamma_map_mod,
        ["largest_gamma_strikes"],
        market_gex,
        default=[]
    )

    gamma_clusters = [_to_python_number(x) for x in gamma_clusters] if gamma_clusters else []

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
        default=None
    )

    hedging_sim = _call_first(
        dealer_hedging_simulator_mod,
        ["simulate_dealer_hedging"],
        df,
        default={}
    )

    hedging_bias = _call_first(
        dealer_hedging_simulator_mod,
        ["hedging_bias"],
        hedging_sim,
        default=None
    )

    intraday_gamma_state = None
    if prev_df is not None:
        intraday_gamma_state = _call_first(
            intraday_gamma_shift_mod,
            ["gamma_shift_signal", "detect_gamma_shift"],
            prev_df,
            df,
            spot,
            default=None
        )

    atm_iv = _call_first(
        volatility_surface_mod,
        ["atm_vol"],
        df,
        spot,
        default=None
    )

    surface_regime = None
    if atm_iv is not None:
        surface_regime = _call_first(
            volatility_surface_mod,
            ["vol_regime"],
            atm_iv,
            default=None
        )

    spot_vs_flip = classify_spot_vs_flip(spot, flip)

    dealer_liquidity_map = build_dealer_liquidity_map(
        spot=spot,
        gamma_flip=flip,
        liquidity_levels=liquidity_levels,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        gamma_clusters=gamma_clusters,
        vacuum_zones=vacuum_zones
    )

    model_features = _call_first(
        feature_builder_mod,
        ["build_features"],
        df,
        spot=spot,
        gamma_regime=gamma_regime,
        final_flow_signal=final_flow_signal,
        vol_regime=vol_regime,
        hedging_bias=hedging_bias,
        spot_vs_flip=spot_vs_flip,
        vacuum_state=vacuum_state,
        atm_iv=atm_iv,
        default=None
    )

    nearest_vacuum_gap_pct = _extract_nearest_vacuum_gap_pct(
        spot=spot,
        vacuum_zones=vacuum_zones
    )

    hedge_flow_value = _extract_hedge_flow_value(hedging_flow)

    flow_imbalance = (
        0.5 * _categorical_flow_score(flow_signal_value)
        + 0.5 * _categorical_flow_score(smart_money_signal_value)
    )

    gamma_flip_distance_pct = _compute_gamma_flip_distance_pct(
        spot_price=spot,
        gamma_flip=flip,
    )

    vacuum_strength = _map_vacuum_strength(
        vacuum_state=vacuum_state,
        liquidity_voids=voids,
        nearest_vacuum_gap_pct=nearest_vacuum_gap_pct,
    )

    hedging_flow_ratio = _map_hedging_flow_ratio(
        hedging_bias=hedging_bias,
        hedge_flow_value=hedge_flow_value,
    )

    smart_money_flow_score = _map_smart_money_flow_score(
        smart_money_flow=final_flow_signal,
        flow_imbalance=flow_imbalance,
    )

    atm_iv_percentile = _compute_atm_iv_percentile(
        atm_iv=atm_iv,
    )

    intraday_range_pct = _compute_intraday_range_pct(
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
        gamma_regime,
        vacuum_state,
        hedging_bias,
        final_flow_signal,
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        vacuum_strength=vacuum_strength,
        hedging_flow_ratio=hedging_flow_ratio,
        smart_money_flow_score=smart_money_flow_score,
        atm_iv_percentile=atm_iv_percentile,
        intraday_range_pct=intraday_range_pct,
        default=None
    )

    rule_move_probability = _extract_probability(rule_move_probability)

    ml_move_probability = None
    predictor_class = getattr(ml_move_predictor_mod, "MovePredictor", None)
    if predictor_class is not None:
        try:
            predictor = predictor_class()
            if model_features is not None:
                ml_move_probability = predictor.predict_probability(model_features)
            ml_move_probability = _extract_probability(ml_move_probability)
            if ml_move_probability is not None:
                ml_move_probability = round(_clip(float(ml_move_probability), 0.05, 0.95), 2)
        except Exception:
            ml_move_probability = None

    hybrid_move_probability = _blend_move_probability(
        rule_prob=rule_move_probability,
        ml_prob=ml_move_probability
    )

    direction, direction_source = decide_direction(
        final_flow_signal=final_flow_signal,
        dealer_pos=dealer_pos,
        vol_regime=vol_regime,
        spot_vs_flip=spot_vs_flip,
        gamma_regime=gamma_regime,
        hedging_bias=hedging_bias,
        gamma_event=gamma_event,
        backtest_mode=backtest_mode
    )

    score_direction = direction if direction is not None else "CALL"

    trade_strength, scoring_breakdown = compute_trade_strength(
        direction=score_direction,
        flow_signal_value=flow_signal_value,
        smart_money_signal_value=smart_money_signal_value,
        gamma_event=gamma_event,
        dealer_pos=dealer_pos,
        vol_regime=vol_regime,
        void_signal=void_signal,
        vacuum_state=vacuum_state,
        spot_vs_flip=spot_vs_flip,
        hedging_bias=hedging_bias,
        gamma_regime=gamma_regime,
        intraday_gamma_state=intraday_gamma_state,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        spot=spot,
        next_support=dealer_liquidity_map.get("next_support"),
        next_resistance=dealer_liquidity_map.get("next_resistance"),
        squeeze_zone=dealer_liquidity_map.get("gamma_squeeze_zone"),
        large_move_probability=hybrid_move_probability,
        ml_move_probability=ml_move_probability
    )

    min_trade_strength = BACKTEST_MIN_TRADE_STRENGTH if backtest_mode else MIN_TRADE_STRENGTH

    base_payload = {
        "symbol": symbol,
        "spot": round(spot, 2),
        "gamma_exposure": round(gamma, 2) if gamma is not None else None,
        "market_gamma": market_gamma_summary,
        "gamma_flip": _to_python_number(flip),
        "spot_vs_flip": spot_vs_flip,
        "gamma_regime": gamma_regime,
        "gamma_clusters": gamma_clusters,
        "dealer_position": dealer_pos,
        "dealer_hedging_flow": hedging_flow,
        "dealer_hedging_bias": hedging_bias,
        "intraday_gamma_state": intraday_gamma_state,
        "volatility_regime": vol_regime,
        "vol_surface_regime": surface_regime,
        "atm_iv": round(float(atm_iv), 2) if atm_iv is not None else None,
        "flow_signal": flow_signal_value,
        "smart_money_flow": smart_money_signal_value,
        "final_flow_signal": final_flow_signal,
        "gamma_event": gamma_event,
        "support_wall": support_wall,
        "resistance_wall": resistance_wall,
        "liquidity_levels": liquidity_levels,
        "liquidity_voids": voids,
        "liquidity_void_signal": void_signal,
        "liquidity_vacuum_zones": vacuum_zones,
        "liquidity_vacuum_state": vacuum_state,
        "dealer_liquidity_map": dealer_liquidity_map,
        "rule_move_probability": rule_move_probability,
        "ml_move_probability": ml_move_probability,
        "hybrid_move_probability": hybrid_move_probability,
        "large_move_probability": hybrid_move_probability,
        "move_probability_components": {
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
        "direction_source": direction_source,
        "trade_strength": trade_strength,
        "signal_quality": classify_signal_quality(trade_strength),
        "scoring_breakdown": scoring_breakdown,
        "budget_constraint_applied": apply_budget_constraint,
        "lot_size": lot_size,
        "requested_lots": requested_lots,
        "max_capital_per_trade": max_capital,
        "backtest_mode": backtest_mode
    }

    if direction is None:
        base_payload["message"] = "No trade signal"
        base_payload["trade_status"] = "NO_SIGNAL"
        return base_payload

    strike = choose_strike(df, spot, direction)
    if strike is None:
        base_payload["direction"] = direction
        base_payload["message"] = "No valid strike found"
        base_payload["trade_status"] = "NO_SIGNAL"
        return base_payload

    option_type = "CE" if direction == "CALL" else "PE"

    option_row = df[
        (df["strikePrice"] == strike) &
        (df["OPTION_TYP"] == option_type)
    ]

    if option_row.empty:
        base_payload["direction"] = direction
        base_payload["message"] = "Selected strike/option type not available"
        base_payload["trade_status"] = "NO_SIGNAL"
        return base_payload

    entry_price = float(option_row.iloc[0]["lastPrice"])
    target, stop_loss = calculate_exit(entry_price)

    base_payload.update({
        "direction": direction,
        "strike": _to_python_number(strike),
        "option_type": option_type,
        "entry_price": round(entry_price, 2),
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
    })

    if apply_budget_constraint:
        budget_info = optimize_lots(
            entry_price=entry_price,
            lot_size=lot_size,
            max_capital=max_capital,
            requested_lots=requested_lots
        )

        base_payload.update(budget_info)

        if not budget_info["budget_ok"]:
            base_payload["message"] = "Trade filtered out due to budget constraint"
            base_payload["trade_status"] = "BUDGET_FAIL"
            return base_payload

        base_payload["number_of_lots"] = budget_info.get("optimized_lots", requested_lots)
    else:
        base_payload["number_of_lots"] = requested_lots
        base_payload["capital_per_lot"] = round(entry_price * lot_size, 2)
        base_payload["capital_required"] = round(entry_price * lot_size * requested_lots, 2)

    if trade_strength < min_trade_strength:
        base_payload["message"] = "Trade filtered out due to low strength"
        base_payload["trade_status"] = "WATCHLIST"
        return base_payload

    if apply_budget_constraint:
        base_payload["message"] = "Tradable signal generated with budget optimization"
    else:
        base_payload["message"] = "Tradable signal generated"

    base_payload["trade_status"] = "TRADE"
    return base_payload