"""
Strike Selector

Ranks candidate option strikes for the engine instead of choosing only the
nearest strike mechanically.
"""

import numpy as np
import pandas as pd

from config.strike_selection_policy import get_strike_selection_score_config


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
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


def _normalize_candidate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    normalized = rows.copy()
    normalized["_normalized_strike"] = pd.to_numeric(normalized.get("strikePrice"), errors="coerce")
    normalized["_normalized_last_price"] = pd.to_numeric(
        normalized.get("lastPrice", normalized.get("LAST_PRICE")),
        errors="coerce",
    ).fillna(0.0)
    normalized["_normalized_volume"] = pd.to_numeric(
        normalized.get("totalTradedVolume", normalized.get("VOLUME")),
        errors="coerce",
    ).fillna(0.0)
    normalized["_normalized_open_interest"] = pd.to_numeric(
        normalized.get("openInterest", normalized.get("OPEN_INT")),
        errors="coerce",
    ).fillna(0.0)
    normalized["_normalized_iv"] = pd.to_numeric(
        normalized.get("impliedVolatility", normalized.get("IV")),
        errors="coerce",
    ).fillna(0.0)
    return normalized.dropna(subset=["_normalized_strike"]).copy()


def _score_moneyness_series(strikes: pd.Series, *, spot, cfg) -> pd.Series:
    distance_pct = (strikes.astype(float) - float(spot)).abs() / max(float(spot), 1e-6) * 100.0
    scores = np.select(
        [
            distance_pct <= cfg["atm_distance_pct"],
            distance_pct <= cfg["near_distance_pct"],
            distance_pct <= cfg["mid_distance_pct"],
            distance_pct <= cfg["far_distance_pct"],
        ],
        [
            int(cfg["moneyness_atm_score"]),
            int(cfg["moneyness_near_score"]),
            int(cfg["moneyness_mid_score"]),
            int(cfg["moneyness_far_score"]),
        ],
        default=int(cfg["moneyness_deep_penalty"]),
    )
    return pd.Series(scores, index=strikes.index, dtype="int64")


def _score_directional_side_series(strikes: pd.Series, *, direction, spot, cfg) -> pd.Series:
    strikes = strikes.astype(float)
    if direction == "CALL":
        scores = np.where(
            strikes >= float(spot),
            int(cfg["call_above_spot_score"]),
            int(cfg["call_below_spot_score"]),
        )
        return pd.Series(scores, index=strikes.index, dtype="int64")
    if direction == "PUT":
        scores = np.where(
            strikes <= float(spot),
            int(cfg["put_below_spot_score"]),
            int(cfg["put_above_spot_score"]),
        )
        return pd.Series(scores, index=strikes.index, dtype="int64")
    return pd.Series(0, index=strikes.index, dtype="int64")


def _score_premium_series(
    premiums: pd.Series,
    *,
    cfg,
    max_capital=None,
    lot_size=None,
) -> pd.Series:
    premium_series = premiums.astype(float)
    scores = np.select(
        [
            premium_series <= 0,
            premium_series.between(cfg["premium_optimal_min"], cfg["premium_optimal_max"], inclusive="both"),
            premium_series.between(cfg["premium_secondary_min"], cfg["premium_optimal_min"], inclusive="left"),
            premium_series.gt(cfg["premium_optimal_max"]) & premium_series.le(cfg["premium_secondary_max"]),
            premium_series.between(cfg["premium_lower_tail_min"], cfg["premium_secondary_min"], inclusive="left"),
        ],
        [
            int(cfg["premium_invalid_penalty"]),
            int(cfg["premium_optimal_score"]),
            int(cfg["premium_secondary_score"]),
            int(cfg["premium_upper_mid_score"]),
            int(cfg["premium_lower_tail_score"]),
        ],
        default=int(cfg["premium_default_score"]),
    )
    premium_scores = pd.Series(scores, index=premium_series.index, dtype="int64")

    if max_capital is not None and lot_size is not None:
        capital_per_lot = premium_series * float(lot_size)
        premium_scores = premium_scores.mask(
            capital_per_lot > float(max_capital),
            premium_scores + int(cfg["premium_over_budget_penalty"]),
        )
        premium_scores = premium_scores.mask(
            (capital_per_lot <= float(max_capital))
            & (capital_per_lot > (float(cfg["premium_near_budget_ratio"]) * float(max_capital))),
            premium_scores + int(cfg["premium_near_budget_penalty"]),
        )

    return premium_scores.astype("int64")


def _score_liquidity_series(volume: pd.Series, open_interest: pd.Series, *, cfg) -> pd.Series:
    vol = volume.astype(float)
    oi = open_interest.astype(float)

    volume_scores = np.select(
        [
            vol >= cfg["volume_high_threshold"],
            vol >= cfg["volume_medium_threshold"],
            vol >= cfg["volume_low_threshold"],
        ],
        [
            int(cfg["volume_high_score"]),
            int(cfg["volume_medium_score"]),
            int(cfg["volume_low_score"]),
        ],
        default=0,
    )
    oi_scores = np.select(
        [
            oi >= cfg["oi_high_threshold"],
            oi >= cfg["oi_medium_threshold"],
            oi >= cfg["oi_low_threshold"],
        ],
        [
            int(cfg["oi_high_score"]),
            int(cfg["oi_medium_score"]),
            int(cfg["oi_low_score"]),
        ],
        default=0,
    )
    return pd.Series(volume_scores + oi_scores, index=vol.index, dtype="int64")


def _score_wall_distance_series(
    strikes: pd.Series,
    *,
    direction,
    cfg,
    support_wall=None,
    resistance_wall=None,
) -> pd.Series:
    strikes = strikes.astype(float)
    score = pd.Series(0, index=strikes.index, dtype="int64")
    support = _safe_float(support_wall, None)
    resistance = _safe_float(resistance_wall, None)

    if direction == "CALL" and resistance is not None:
        dist = (strikes - resistance).abs()
        score = pd.Series(
            np.select(
                [
                    dist <= cfg["wall_near_distance_points"],
                    dist <= cfg["wall_medium_distance_points"],
                ],
                [
                    int(cfg["wall_near_penalty"]),
                    int(cfg["wall_medium_penalty"]),
                ],
                default=0,
            ),
            index=strikes.index,
            dtype="int64",
        )
    elif direction == "PUT" and support is not None:
        dist = (strikes - support).abs()
        score = pd.Series(
            np.select(
                [
                    dist <= cfg["wall_near_distance_points"],
                    dist <= cfg["wall_medium_distance_points"],
                ],
                [
                    int(cfg["wall_near_penalty"]),
                    int(cfg["wall_medium_penalty"]),
                ],
                default=0,
            ),
            index=strikes.index,
            dtype="int64",
        )

    return score


def _score_gamma_cluster_distance_series(strikes: pd.Series, gamma_clusters, *, cfg) -> pd.Series:
    if not gamma_clusters:
        return pd.Series(0, index=strikes.index, dtype="int64")

    clean_clusters = []
    for cluster in gamma_clusters:
        try:
            clean_clusters.append(float(cluster))
        except Exception:
            continue

    if not clean_clusters:
        return pd.Series(0, index=strikes.index, dtype="int64")

    strike_values = strikes.astype(float).to_numpy(dtype=float)
    cluster_values = np.asarray(clean_clusters, dtype=float)
    nearest = np.min(np.abs(strike_values[:, None] - cluster_values[None, :]), axis=1)
    scores = np.select(
        [
            nearest <= cfg["gamma_cluster_near_distance_points"],
            nearest <= cfg["gamma_cluster_medium_distance_points"],
        ],
        [
            int(cfg["gamma_cluster_near_penalty"]),
            int(cfg["gamma_cluster_medium_penalty"]),
        ],
        default=int(cfg["gamma_cluster_far_bonus"]),
    )
    return pd.Series(scores, index=strikes.index, dtype="int64")


def _score_iv_series(iv: pd.Series, *, cfg) -> pd.Series:
    iv_values = iv.astype(float)
    scores = np.select(
        [
            iv_values <= 0,
            iv_values.between(cfg["iv_low_min"], cfg["iv_low_max"], inclusive="both"),
            iv_values.gt(cfg["iv_low_max"]) & iv_values.le(cfg["iv_mid_max"]),
            iv_values > cfg["iv_high_threshold"],
        ],
        [
            0,
            int(cfg["iv_low_score"]),
            int(cfg["iv_mid_score"]),
            int(cfg["iv_high_penalty"]),
        ],
        default=0,
    )
    return pd.Series(scores, index=iv.index, dtype="int64")


def _score_candidate_frame(
    rows: pd.DataFrame,
    *,
    direction,
    spot,
    cfg,
    support_wall=None,
    resistance_wall=None,
    gamma_clusters=None,
    max_capital=None,
    lot_size=None,
) -> pd.DataFrame:
    scored = rows.copy()
    scored["_moneyness_score"] = _score_moneyness_series(scored["_normalized_strike"], spot=spot, cfg=cfg)
    scored["_directional_side_score"] = _score_directional_side_series(
        scored["_normalized_strike"],
        direction=direction,
        spot=spot,
        cfg=cfg,
    )
    scored["_premium_score"] = _score_premium_series(
        scored["_normalized_last_price"],
        cfg=cfg,
        max_capital=max_capital,
        lot_size=lot_size,
    )
    scored["_liquidity_score"] = _score_liquidity_series(
        scored["_normalized_volume"],
        scored["_normalized_open_interest"],
        cfg=cfg,
    )
    scored["_wall_distance_score"] = _score_wall_distance_series(
        scored["_normalized_strike"],
        direction=direction,
        cfg=cfg,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
    )
    scored["_gamma_cluster_score"] = _score_gamma_cluster_distance_series(
        scored["_normalized_strike"],
        gamma_clusters,
        cfg=cfg,
    )
    scored["_iv_score"] = _score_iv_series(scored["_normalized_iv"], cfg=cfg)
    scored["_base_score"] = (
        scored["_moneyness_score"]
        + scored["_directional_side_score"]
        + scored["_premium_score"]
        + scored["_liquidity_score"]
        + scored["_wall_distance_score"]
        + scored["_gamma_cluster_score"]
        + scored["_iv_score"]
    ).astype("int64")
    return scored


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

    cfg = get_strike_selection_score_config()
    effective_window_steps = (
        cfg["strike_window_steps"]
        if strike_window_steps is None
        else strike_window_steps
    )
    rows = _apply_strike_window(rows, spot=spot, window_steps=effective_window_steps)
    rows = _normalize_candidate_rows(rows)
    if rows.empty:
        return []
    rows = _score_candidate_frame(
        rows,
        direction=direction,
        spot=spot,
        cfg=cfg,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        gamma_clusters=gamma_clusters,
        max_capital=max_capital,
        lot_size=lot_size,
    )

    candidates = []
    for row in rows.to_dict(orient="records"):
        strike = _safe_float(row.get("_normalized_strike", row.get("strikePrice")), None)
        if strike is None:
            continue

        premium = _safe_float(row.get("_normalized_last_price"), _safe_float(row.get("lastPrice"), 0.0))
        volume = _safe_float(row.get("_normalized_volume"), _safe_float(row.get("totalTradedVolume"), row.get("VOLUME", 0.0)))
        oi = _safe_float(row.get("_normalized_open_interest"), _safe_float(row.get("openInterest"), row.get("OPEN_INT", 0.0)))
        iv = _safe_float(row.get("_normalized_iv"), _safe_float(row.get("impliedVolatility"), row.get("IV", 0.0)))

        score_breakdown = {
            "moneyness_score": int(row.get("_moneyness_score", 0)),
            "directional_side_score": int(row.get("_directional_side_score", 0)),
            "premium_score": int(row.get("_premium_score", 0)),
            "liquidity_score": int(row.get("_liquidity_score", 0)),
            "wall_distance_score": int(row.get("_wall_distance_score", 0)),
            "gamma_cluster_score": int(row.get("_gamma_cluster_score", 0)),
            "iv_score": int(row.get("_iv_score", 0)),
        }

        hook_payload = {}
        if callable(candidate_score_hook):
            try:
                hook_payload = candidate_score_hook(
                    row,
                    {
                        "strike": _to_python_number(strike),
                        "last_price": round(premium, 2),
                        "volume": int(volume),
                        "open_interest": int(oi),
                        "iv": round(iv, 2) if iv else 0,
                    },
                ) or {}
            except Exception:
                hook_payload = {}

        efficiency_score_adjustment = int(_safe_float(hook_payload.get("score_adjustment"), 0.0))
        if efficiency_score_adjustment:
            score_breakdown["option_efficiency_score_adjustment"] = efficiency_score_adjustment

        total_score = int(row.get("_base_score", 0)) + efficiency_score_adjustment
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
        candidates.append(record)

    candidates.sort(
        key=lambda x: (
            -x["score"],
            abs(float(x["strike"]) - float(spot)),
            x["last_price"] if x["last_price"] > 0 else 10**9,
            float(x["strike"]),
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
