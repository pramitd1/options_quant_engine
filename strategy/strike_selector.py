"""
Strike Selector

Ranks candidate option strikes for the engine instead of choosing only the
nearest strike mechanically.
"""

from config.strike_selection_policy import get_strike_selection_score_config


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


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


def _score_moneyness(direction, strike, spot):
    cfg = get_strike_selection_score_config()
    distance_pct = abs(strike - spot) / max(spot, 1e-6) * 100.0

    # Prefer near-ATM / slight ITM options for buying.
    if distance_pct <= cfg["atm_distance_pct"]:
        return int(cfg["moneyness_atm_score"])
    if distance_pct <= cfg["near_distance_pct"]:
        return int(cfg["moneyness_near_score"])
    if distance_pct <= cfg["mid_distance_pct"]:
        return int(cfg["moneyness_mid_score"])
    if distance_pct <= cfg["far_distance_pct"]:
        return int(cfg["moneyness_far_score"])
    return int(cfg["moneyness_deep_penalty"])


def _score_directional_side(direction, strike, spot):
    """
    Slight preference for:
    - CALL: ATM to slightly OTM/ITM but not too far
    - PUT: ATM to slightly OTM/ITM but not too far
    """
    cfg = get_strike_selection_score_config()
    if direction == "CALL":
        if strike >= spot:
            return int(cfg["call_above_spot_score"])
        return int(cfg["call_below_spot_score"])

    if direction == "PUT":
        if strike <= spot:
            return int(cfg["put_below_spot_score"])
        return int(cfg["put_above_spot_score"])

    return 0


def _score_premium(last_price, max_capital=None, lot_size=None):
    cfg = get_strike_selection_score_config()
    premium = _safe_float(last_price, 0.0)

    if premium <= 0:
        return -10

    # Raw premium preference band for option buying.
    if cfg["premium_optimal_min"] <= premium <= cfg["premium_optimal_max"]:
        score = int(cfg["premium_optimal_score"])
    elif cfg["premium_secondary_min"] <= premium < cfg["premium_optimal_min"]:
        score = int(cfg["premium_secondary_score"])
    elif cfg["premium_optimal_max"] < premium <= cfg["premium_secondary_max"]:
        score = int(cfg["premium_upper_mid_score"])
    elif cfg["premium_lower_tail_min"] <= premium < cfg["premium_secondary_min"]:
        score = int(cfg["premium_lower_tail_score"])
    else:
        score = int(cfg["premium_default_score"])

    if max_capital is not None and lot_size is not None:
        capital_per_lot = premium * lot_size
        if capital_per_lot > max_capital:
            score += int(cfg["premium_over_budget_penalty"])
        elif capital_per_lot > cfg["premium_near_budget_ratio"] * max_capital:
            score += int(cfg["premium_near_budget_penalty"])

    return score


def _score_liquidity(volume, open_interest):
    cfg = get_strike_selection_score_config()
    vol = _safe_float(volume, 0.0)
    oi = _safe_float(open_interest, 0.0)

    score = 0

    if vol >= cfg["volume_high_threshold"]:
        score += int(cfg["volume_high_score"])
    elif vol >= cfg["volume_medium_threshold"]:
        score += int(cfg["volume_medium_score"])
    elif vol >= cfg["volume_low_threshold"]:
        score += int(cfg["volume_low_score"])

    if oi >= cfg["oi_high_threshold"]:
        score += int(cfg["oi_high_score"])
    elif oi >= cfg["oi_medium_threshold"]:
        score += int(cfg["oi_medium_score"])
    elif oi >= cfg["oi_low_threshold"]:
        score += int(cfg["oi_low_score"])

    return score


def _score_wall_distance(direction, strike, support_wall=None, resistance_wall=None):
    cfg = get_strike_selection_score_config()
    score = 0

    support = _safe_float(support_wall, None)
    resistance = _safe_float(resistance_wall, None)

    if direction == "CALL" and resistance is not None:
        dist = abs(strike - resistance)
        if dist <= cfg["wall_near_distance_points"]:
            score += int(cfg["wall_near_penalty"])
        elif dist <= cfg["wall_medium_distance_points"]:
            score += int(cfg["wall_medium_penalty"])

    if direction == "PUT" and support is not None:
        dist = abs(strike - support)
        if dist <= cfg["wall_near_distance_points"]:
            score += int(cfg["wall_near_penalty"])
        elif dist <= cfg["wall_medium_distance_points"]:
            score += int(cfg["wall_medium_penalty"])

    return score


def _score_gamma_cluster_distance(strike, gamma_clusters):
    cfg = get_strike_selection_score_config()
    if not gamma_clusters:
        return 0

    clean_clusters = []
    for g in gamma_clusters:
        try:
            clean_clusters.append(float(g))
        except Exception:
            continue

    if not clean_clusters:
        return 0

    nearest = min(abs(strike - g) for g in clean_clusters)

    if nearest <= cfg["gamma_cluster_near_distance_points"]:
        return int(cfg["gamma_cluster_near_penalty"])
    if nearest <= cfg["gamma_cluster_medium_distance_points"]:
        return int(cfg["gamma_cluster_medium_penalty"])
    return int(cfg["gamma_cluster_far_bonus"])


def _score_iv(iv):
    cfg = get_strike_selection_score_config()
    iv_val = _safe_float(iv, 0.0)

    if iv_val <= 0:
        return 0

    if cfg["iv_low_min"] <= iv_val <= cfg["iv_low_max"]:
        return int(cfg["iv_low_score"])
    if cfg["iv_low_max"] < iv_val <= cfg["iv_mid_max"]:
        return int(cfg["iv_mid_score"])
    if iv_val > cfg["iv_high_threshold"]:
        return int(cfg["iv_high_penalty"])
    return 0


def _infer_strike_step(rows):
    if rows is None or rows.empty or "strikePrice" not in rows.columns:
        return None

    strikes = []
    for value in rows["strikePrice"].tolist():
        strike = _safe_float(value, None)
        if strike is not None:
            strikes.append(float(strike))

    strikes = sorted(set(strikes))
    if len(strikes) < 2:
        return None

    diffs = [round(strikes[idx] - strikes[idx - 1], 6) for idx in range(1, len(strikes))]
    diffs = [diff for diff in diffs if diff > 0]

    if not diffs:
        return None

    return min(diffs)


def _apply_strike_window(rows, spot, window_steps):
    if rows is None or rows.empty:
        return rows

    if window_steps is None or window_steps <= 0:
        return rows

    strike_step = _infer_strike_step(rows)
    if strike_step in (None, 0):
        return rows

    lower_bound = float(spot) - (window_steps * strike_step)
    upper_bound = float(spot) + (window_steps * strike_step)

    filtered = rows[
        (rows["strikePrice"].astype(float) >= lower_bound) &
        (rows["strikePrice"].astype(float) <= upper_bound)
    ].copy()

    return filtered if not filtered.empty else rows


def _build_candidate_record(
    row,
    direction,
    spot,
    support_wall=None,
    resistance_wall=None,
    gamma_clusters=None,
    max_capital=None,
    lot_size=None,
    candidate_score_hook=None,
):
    strike = _safe_float(row.get("strikePrice"), None)
    if strike is None:
        return None

    premium = _safe_float(row.get("lastPrice"), 0.0)
    volume = _safe_float(row.get("totalTradedVolume"), row.get("VOLUME", 0.0))
    oi = _safe_float(row.get("openInterest"), row.get("OPEN_INT", 0.0))
    iv = _safe_float(row.get("impliedVolatility"), row.get("IV", 0.0))

    score_breakdown = {
        "moneyness_score": _score_moneyness(direction, strike, spot),
        "directional_side_score": _score_directional_side(direction, strike, spot),
        "premium_score": _score_premium(premium, max_capital=max_capital, lot_size=lot_size),
        "liquidity_score": _score_liquidity(volume, oi),
        "wall_distance_score": _score_wall_distance(
            direction,
            strike,
            support_wall=support_wall,
            resistance_wall=resistance_wall,
        ),
        "gamma_cluster_score": _score_gamma_cluster_distance(strike, gamma_clusters),
        "iv_score": _score_iv(iv),
    }

    hook_payload = {}
    if callable(candidate_score_hook):
        try:
            hook_payload = candidate_score_hook(row, {
                "strike": _to_python_number(strike),
                "last_price": round(premium, 2),
                "volume": int(volume),
                "open_interest": int(oi),
                "iv": round(iv, 2) if iv else 0,
            }) or {}
        except Exception:
            hook_payload = {}

    efficiency_score_adjustment = int(_safe_float(hook_payload.get("score_adjustment"), 0.0))
    if efficiency_score_adjustment:
        score_breakdown["option_efficiency_score_adjustment"] = efficiency_score_adjustment

    total_score = int(sum(score_breakdown.values()))

    record = {
        "strike": _to_python_number(strike),
        "last_price": round(premium, 2),
        "volume": int(volume),
        "open_interest": int(oi),
        "iv": round(iv, 2) if iv else 0,
        "capital_per_lot": round(premium * lot_size, 2) if lot_size is not None else None,
        "score": total_score,
        "score_breakdown": score_breakdown,
    }
    if hook_payload:
        record.update({key: value for key, value in hook_payload.items() if key != "score_adjustment"})
    return record


def rank_strike_candidates(
    option_chain,
    direction,
    spot,
    support_wall=None,
    resistance_wall=None,
    gamma_clusters=None,
    lot_size=None,
    max_capital=None,
    top_n=5,
    strike_window_steps=None,
    candidate_score_hook=None,
):
    if option_chain is None or len(option_chain) == 0:
        return []

    option_type = "CE" if direction == "CALL" else "PE"

    rows = option_chain[option_chain["OPTION_TYP"] == option_type].copy()
    if rows.empty:
        return []

    effective_window_steps = (
        get_strike_selection_score_config()["strike_window_steps"]
        if strike_window_steps is None
        else strike_window_steps
    )
    rows = _apply_strike_window(rows, spot=spot, window_steps=effective_window_steps)

    candidates = []

    for _, row in rows.iterrows():
        record = _build_candidate_record(
            row=row,
            direction=direction,
            spot=spot,
            support_wall=support_wall,
            resistance_wall=resistance_wall,
            gamma_clusters=gamma_clusters,
            max_capital=max_capital,
            lot_size=lot_size,
            candidate_score_hook=candidate_score_hook,
        )
        if record is not None:
            candidates.append(record)

    candidates.sort(
        key=lambda x: (
            -x["score"],
            abs(float(x["strike"]) - float(spot)),
            x["last_price"] if x["last_price"] > 0 else 10**9,
        )
    )

    return candidates[:top_n]


def select_best_strike(
    option_chain,
    direction,
    spot,
    support_wall=None,
    resistance_wall=None,
    gamma_clusters=None,
    lot_size=None,
    max_capital=None,
    strike_window_steps=None,
    candidate_score_hook=None,
):
    ranked = rank_strike_candidates(
        option_chain=option_chain,
        direction=direction,
        spot=spot,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        gamma_clusters=gamma_clusters,
        lot_size=lot_size,
        max_capital=max_capital,
        top_n=5,
        strike_window_steps=strike_window_steps,
        candidate_score_hook=candidate_score_hook,
    )

    if not ranked:
        return None, []

    return ranked[0]["strike"], ranked
