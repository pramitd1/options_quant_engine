"""
Module: strike_selector.py

Purpose:
    Rank option contracts and choose the strike that best expresses the current directional thesis.

Role in the System:
    Part of the strategy layer that converts directional intent into executable option trades.

Key Outputs:
    Ranked candidate strikes, factor-level score breakdowns, and the selected strike for trade construction.

Downstream Usage:
    Consumed by the signal engine during contract selection and by research tooling for diagnostics.
"""

import math
import numpy as np
import pandas as pd
import logging

from analytics.greeks_engine import estimate_iv_from_price, compute_option_greeks, _parse_expiry_years
from config.strike_selection_policy import get_strike_selection_score_config
from strategy.enhanced_strike_scoring import compute_enhanced_strike_scores
from utils.numerics import clip as _clip, safe_float as _safe_float, to_python_number as _to_python_number  # noqa: F401


_LOG = logging.getLogger(__name__)
_WARN_ONCE_KEYS: set[str] = set()


def _warn_once(key: str, message: str, *args) -> None:
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    _LOG.warning(message, *args)


def _infer_strike_step(rows):
    """
    Purpose:
        Infer the minimum strike spacing present in the option-chain candidates.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        rows (Any): Candidate option-contract rows under evaluation.
    
    Returns:
        float | None: Inferred strike interval, or `None` when the chain is too sparse to infer spacing.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
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
    """
    Purpose:
        Filter candidate rows to strikes within the configured distance from spot.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        rows (Any): Candidate option-contract rows under evaluation.
        spot (Any): Current underlying spot price.
        window_steps (Any): Number of strike intervals to retain on each side of spot.
    
    Returns:
        Any: Filtered candidate rows when a strike window can be applied, otherwise the original rows.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    if rows is None or rows.empty:
        return rows

    # Guard: validate spot is numeric and positive
    spot_validated = _safe_float(spot, None)
    if spot_validated is None or spot_validated <= 0:
        # Cannot filter by spot proximity without valid spot
        return rows
    
    if window_steps is None or window_steps <= 0:
        return rows

    strike_step = _infer_strike_step(rows)
    if strike_step in (None, 0):
        return rows

    lower_bound = float(spot_validated) - (window_steps * strike_step)
    upper_bound = float(spot_validated) + (window_steps * strike_step)

    filtered = rows[
        (rows["strikePrice"].astype(float) >= lower_bound) &
        (rows["strikePrice"].astype(float) <= upper_bound)
    ].copy()

    return filtered if not filtered.empty else rows


def _normalize_candidate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Normalize provider-specific strike columns into the common scoring schema.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        rows (pd.DataFrame): Candidate option-contract rows under evaluation.
    
    Returns:
        pd.DataFrame: Candidate rows with normalized numeric helper columns used by scoring.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    def _first_numeric(df: pd.DataFrame, primary: str, fallback=None, *, default: float = 0.0) -> pd.Series:
        if primary in df.columns:
            source = df[primary]
        elif fallback is not None and fallback in df.columns:
            source = df[fallback]
        else:
            return pd.Series(default, index=df.index, dtype="float64")
        return pd.to_numeric(source, errors="coerce").fillna(default)

    normalized = rows.copy()
    normalized["_normalized_strike"] = _first_numeric(normalized, "strikePrice", default=np.nan)
    normalized["_normalized_last_price"] = _first_numeric(normalized, "lastPrice", "LAST_PRICE", default=0.0)
    normalized["_normalized_volume"] = _first_numeric(normalized, "totalTradedVolume", "VOLUME", default=0.0)
    normalized["_normalized_open_interest"] = _first_numeric(normalized, "openInterest", "OPEN_INT", default=0.0)
    normalized["_normalized_iv"] = _first_numeric(normalized, "impliedVolatility", "IV", default=0.0)

    # Bid / ask spread columns (multiple provider aliases supported).
    def _first_bid_ask(df, *candidates, default=0.0):
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype="float64")

    normalized["_normalized_bid"] = _first_bid_ask(
        normalized, "bidPrice", "BID_PRICE", "best_bid_price", "bestBidPrice", "bid_price", "bid"
    )
    normalized["_normalized_ask"] = _first_bid_ask(
        normalized, "askPrice", "ASK_PRICE", "best_ask_price", "bestAskPrice", "ask_price", "ask"
    )
    mid = (normalized["_normalized_bid"] + normalized["_normalized_ask"]) / 2.0
    spread = (normalized["_normalized_ask"] - normalized["_normalized_bid"]).clip(lower=0.0)
    has_spread = (mid > 0) & (normalized["_normalized_bid"] > 0) & (normalized["_normalized_ask"] > 0)
    normalized["_normalized_ba_spread_ratio"] = (spread / mid.where(mid > 0, 1.0)).where(has_spread, np.nan)
    return normalized[normalized["_normalized_strike"].notna()].copy()


def _continuous_mode(cfg) -> bool:
    mode = str(cfg.get("strike_scoring_mode", "continuous") or "continuous").strip().lower()
    return mode == "continuous"


def _linear_decay(*, value: pd.Series, lower: float, upper: float, high_score: float, low_score: float) -> pd.Series:
    span = max(float(upper) - float(lower), 1e-6)
    t = ((value.astype(float) - float(lower)) / span).clip(0.0, 1.0)
    return high_score + (low_score - high_score) * t


def _score_moneyness_series(strikes: pd.Series, *, spot, cfg) -> pd.Series:
    """
    Purpose:
        Score each strike by how closely it matches the preferred moneyness bucket.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        strikes (pd.Series): Strike-price series for the candidate option contracts.
        spot (Any): Current underlying spot price.
        cfg (Any): Configuration dictionary containing thresholds, weights, and score buckets.
    
    Returns:
        pd.Series: Integer score for each strike based on distance from the preferred moneyness bucket.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    # Guard: validate spot is numeric and positive
    spot_safe = _safe_float(spot, None)
    if spot_safe is None or spot_safe <= 0:
        # Cannot compute moneyness distance without valid spot
        return pd.Series(0.5, index=strikes.index)
    
    distance_pct = (strikes.astype(float) - float(spot_safe)).abs() / max(float(spot_safe), 1e-6) * 100.0

    if _continuous_mode(cfg):
        atm = float(cfg["atm_distance_pct"])
        far = float(cfg["far_distance_pct"])
        deep = float(cfg["moneyness_deep_penalty"])
        atm_score = float(cfg["moneyness_atm_score"])
        far_score = float(cfg["moneyness_far_score"])

        near_curve = _linear_decay(
            value=distance_pct,
            lower=atm,
            upper=far,
            high_score=atm_score,
            low_score=far_score,
        )
        tail_span = max(1.5 * far, far + 1e-6)
        tail_curve = _linear_decay(
            value=distance_pct,
            lower=far,
            upper=tail_span,
            high_score=far_score,
            low_score=deep,
        )
        score = near_curve.where(distance_pct <= far, tail_curve)
        score = score.where(distance_pct <= tail_span, deep)
        return score.astype("float64")

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
    return pd.Series(scores, index=strikes.index, dtype="float64")


def _score_directional_side_series(strikes: pd.Series, *, direction, spot, cfg) -> pd.Series:
    """
    Purpose:
        Reward strikes whose relative position to spot matches the requested call or put direction.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        strikes (pd.Series): Strike-price series for the candidate option contracts.
        direction (Any): Directional side requested by the strategy, typically `CALL` or `PUT`.
        spot (Any): Current underlying spot price.
        cfg (Any): Configuration dictionary containing thresholds, weights, and score buckets.
    
    Returns:
        pd.Series: Integer score for each strike based on whether it sits on the preferred side of spot.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    strikes = strikes.astype(float)
    if _continuous_mode(cfg):
        spot_val = max(float(spot), 1e-6)
        signed = (strikes - float(spot)) / spot_val * 100.0
        width = max(float(cfg.get("near_distance_pct", 0.4)) * 0.5, 0.05)

        if direction == "CALL":
            t = ((signed + width) / (2.0 * width)).clip(0.0, 1.0)
            lo = float(cfg["call_below_spot_score"])
            hi = float(cfg["call_above_spot_score"])
            return (lo + (hi - lo) * t).astype("float64")

        if direction == "PUT":
            t = ((-signed + width) / (2.0 * width)).clip(0.0, 1.0)
            lo = float(cfg["put_above_spot_score"])
            hi = float(cfg["put_below_spot_score"])
            return (lo + (hi - lo) * t).astype("float64")

        return pd.Series(0.0, index=strikes.index, dtype="float64")

    if direction == "CALL":
        scores = np.where(
            strikes >= float(spot),
            int(cfg["call_above_spot_score"]),
            int(cfg["call_below_spot_score"]),
        )
        return pd.Series(scores, index=strikes.index, dtype="float64")
    if direction == "PUT":
        scores = np.where(
            strikes <= float(spot),
            int(cfg["put_below_spot_score"]),
            int(cfg["put_above_spot_score"]),
        )
        return pd.Series(scores, index=strikes.index, dtype="float64")
    return pd.Series(0.0, index=strikes.index, dtype="float64")


def _score_premium_series(
    premiums: pd.Series,
    *,
    cfg,
    max_capital=None,
    lot_size=None,
) -> pd.Series:
    """
    Purpose:
        Score candidate strikes by premium affordability and contract efficiency.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        premiums (pd.Series): Option premium series used to judge affordability and contract quality.
        cfg (Any): Configuration dictionary containing thresholds, weights, and score buckets.
        max_capital (Any): Maximum capital budget available for the trade.
        lot_size (Any): Lot size used to translate premium into per-trade capital usage.
    
    Returns:
        pd.Series: Integer score for each strike based on affordability and premium quality.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    premium_series = premiums.astype(float)

    if _continuous_mode(cfg):
        invalid_penalty = float(cfg["premium_invalid_penalty"])
        optimal_score = float(cfg["premium_optimal_score"])
        secondary_score = float(cfg["premium_secondary_score"])
        default_score = float(cfg["premium_default_score"])

        pmin = float(cfg["premium_optimal_min"])
        pmax = float(cfg["premium_optimal_max"])
        low_tail = float(cfg["premium_lower_tail_min"])
        high_tail = float(cfg["premium_secondary_max"])

        premium_scores = pd.Series(default_score, index=premium_series.index, dtype="float64")
        premium_scores = premium_scores.mask(premium_series <= 0, invalid_penalty)

        left = _linear_decay(
            value=premium_series,
            lower=low_tail,
            upper=pmin,
            high_score=secondary_score,
            low_score=optimal_score,
        )
        premium_scores = premium_scores.mask((premium_series > 0) & (premium_series < pmin), left)
        premium_scores = premium_scores.mask((premium_series >= pmin) & (premium_series <= pmax), optimal_score)

        right = _linear_decay(
            value=premium_series,
            lower=pmax,
            upper=high_tail,
            high_score=optimal_score,
            low_score=default_score,
        )
        premium_scores = premium_scores.mask((premium_series > pmax) & (premium_series <= high_tail), right)
        premium_scores = premium_scores.mask(premium_series > high_tail, default_score)
    else:
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
        premium_scores = pd.Series(scores, index=premium_series.index, dtype="float64")

    if max_capital is not None and lot_size is not None:
        capital_per_lot = premium_series * float(lot_size)
        premium_scores = premium_scores.mask(
            capital_per_lot > float(max_capital),
            premium_scores + float(cfg["premium_over_budget_penalty"]),
        )
        premium_scores = premium_scores.mask(
            (capital_per_lot <= float(max_capital))
            & (capital_per_lot > (float(cfg["premium_near_budget_ratio"]) * float(max_capital))),
            premium_scores + float(cfg["premium_near_budget_penalty"]),
        )

    return premium_scores.astype("float64")


def _score_liquidity_series(volume: pd.Series, open_interest: pd.Series, *, cfg) -> pd.Series:
    """
    Purpose:
        Score candidate strikes using traded volume and open-interest liquidity proxies.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        volume (pd.Series): Traded-volume series used as a liquidity proxy.
        open_interest (pd.Series): Open-interest series used as a liquidity and crowding proxy.
        cfg (Any): Configuration dictionary containing thresholds, weights, and score buckets.
    
    Returns:
        pd.Series: Integer score for each strike based on volume and open-interest quality.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    vol = volume.astype(float)
    oi = open_interest.astype(float)

    if _continuous_mode(cfg):
        v_hi = max(float(cfg["volume_high_threshold"]), 1.0)
        oi_hi = max(float(cfg["oi_high_threshold"]), 1.0)
        v_ratio = np.log1p(np.clip(vol, a_min=0.0, a_max=None)) / np.log1p(v_hi)
        oi_ratio = np.log1p(np.clip(oi, a_min=0.0, a_max=None)) / np.log1p(oi_hi)
        volume_scores = np.clip(v_ratio, 0.0, 1.0) * float(cfg["volume_high_score"])
        oi_scores = np.clip(oi_ratio, 0.0, 1.0) * float(cfg["oi_high_score"])
        return pd.Series(volume_scores + oi_scores, index=vol.index, dtype="float64")

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
    return pd.Series(volume_scores + oi_scores, index=vol.index, dtype="float64")


def _score_wall_distance_series(
    strikes: pd.Series,
    *,
    direction,
    cfg,
    support_wall=None,
    resistance_wall=None,
) -> pd.Series:
    """
    Purpose:
        Penalize strikes that sit too close to the most relevant structural wall.

    Context:
        Strike selection should avoid buying calls directly into nearby resistance or buying puts directly into nearby support. This helper converts that market-structure idea into an additive strike-ranking penalty.

    Inputs:
        strikes (pd.Series): Candidate strike prices.
        direction (Any): Requested trade direction, typically `CALL` or `PUT`.
        cfg (Any): Strike-selection configuration containing wall-distance thresholds and penalties.
        support_wall (Any): Support level inferred from positioning or liquidity structure.
        resistance_wall (Any): Resistance level inferred from positioning or liquidity structure.

    Returns:
        pd.Series: Integer wall-distance score for each strike.

    Notes:
        Only the wall that matters for the current direction is considered because that is the nearer execution obstacle.
    """
    strikes = strikes.astype(float)
    score = pd.Series(0.0, index=strikes.index, dtype="float64")
    support = _safe_float(support_wall, None)
    resistance = _safe_float(resistance_wall, None)

    if direction == "CALL" and resistance is not None:
        dist = (strikes - resistance).abs()
        if _continuous_mode(cfg):
            near = float(cfg["wall_near_distance_points"])
            medium = max(float(cfg["wall_medium_distance_points"]), near + 1e-6)
            near_pen = float(cfg["wall_near_penalty"])
            med_pen = float(cfg["wall_medium_penalty"])
            mid_curve = _linear_decay(
                value=dist,
                lower=near,
                upper=medium,
                high_score=near_pen,
                low_score=med_pen,
            )
            values = np.where(dist <= near, near_pen, np.where(dist <= medium, mid_curve.to_numpy(), 0.0))
            return pd.Series(values, index=strikes.index, dtype="float64")
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
            dtype="float64",
        )
    elif direction == "PUT" and support is not None:
        dist = (strikes - support).abs()
        if _continuous_mode(cfg):
            near = float(cfg["wall_near_distance_points"])
            medium = max(float(cfg["wall_medium_distance_points"]), near + 1e-6)
            near_pen = float(cfg["wall_near_penalty"])
            med_pen = float(cfg["wall_medium_penalty"])
            mid_curve = _linear_decay(
                value=dist,
                lower=near,
                upper=medium,
                high_score=near_pen,
                low_score=med_pen,
            )
            values = np.where(dist <= near, near_pen, np.where(dist <= medium, mid_curve.to_numpy(), 0.0))
            return pd.Series(values, index=strikes.index, dtype="float64")
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
            dtype="float64",
        )

    return score


def _score_gamma_cluster_distance_series(strikes: pd.Series, gamma_clusters, *, cfg) -> pd.Series:
    """
    Purpose:
        Score candidate strikes by their distance from known gamma concentration levels.

    Context:
        Gamma clusters can act as pinning magnets or acceleration triggers. The selector therefore penalizes strikes too close to those levels and can mildly reward contracts farther away.

    Inputs:
        strikes (pd.Series): Candidate strike prices.
        gamma_clusters (Any): Gamma concentration levels inferred from the analytics layer.
        cfg (Any): Strike-selection configuration containing cluster-distance thresholds and penalties.

    Returns:
        pd.Series: Integer gamma-cluster distance score for each strike.

    Notes:
        This is a structural heuristic meant to keep trade construction aligned with the surrounding dealer-positioning map.
    """
    if not gamma_clusters:
        return pd.Series(0.0, index=strikes.index, dtype="float64")

    clean_clusters = []
    for cluster in gamma_clusters:
        try:
            clean_clusters.append(float(cluster))
        except (TypeError, ValueError) as exc:
            _warn_once(
                "gamma_cluster_parse_failed",
                "strike_selector: ignored malformed gamma cluster value %r (%s)",
                cluster,
                exc,
            )
            continue

    if not clean_clusters:
        return pd.Series(0.0, index=strikes.index, dtype="float64")

    strike_values = strikes.astype(float).to_numpy(dtype=float)
    cluster_values = np.asarray(clean_clusters, dtype=float)
    nearest = np.min(np.abs(strike_values[:, None] - cluster_values[None, :]), axis=1)
    if _continuous_mode(cfg):
        near = float(cfg["gamma_cluster_near_distance_points"])
        medium = max(float(cfg["gamma_cluster_medium_distance_points"]), near + 1e-6)
        near_pen = float(cfg["gamma_cluster_near_penalty"])
        med_pen = float(cfg["gamma_cluster_medium_penalty"])
        far_bonus = float(cfg["gamma_cluster_far_bonus"])
        mid_curve = _linear_decay(
            value=pd.Series(nearest),
            lower=near,
            upper=medium,
            high_score=near_pen,
            low_score=med_pen,
        )
        values = np.where(nearest <= near, near_pen, np.where(nearest <= medium, mid_curve.to_numpy(), far_bonus))
        return pd.Series(values, index=strikes.index, dtype="float64")

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
    return pd.Series(scores, index=strikes.index, dtype="float64")


def _score_ba_spread_series(spread_ratio: pd.Series, *, cfg) -> pd.Series:
    """
    Score candidate strikes by bid-ask spread quality.
    Rows where bid/ask was unavailable (NaN spread ratio) receive 0 (neutral).
    Narrow spreads earn a small bonus; wide spreads earn a penalty.
    """
    threshold = float(cfg.get("ba_spread_ratio_threshold", 0.04))
    wide = float(cfg.get("ba_spread_ratio_wide", 0.10))
    narrow_bonus = float(cfg.get("ba_spread_narrow_bonus", 1))
    wide_penalty = float(cfg.get("ba_spread_wide_penalty", -3))

    score = pd.Series(0.0, index=spread_ratio.index, dtype="float64")
    valid = spread_ratio.notna()
    if _continuous_mode(cfg):
        # Below threshold → linearly interpolate from narrow_bonus down to 0
        left = _linear_decay(
            value=spread_ratio,
            lower=0.0,
            upper=threshold,
            high_score=narrow_bonus,
            low_score=0.0,
        )
        # Between threshold and wide → decrease toward penalty
        mid_decay = _linear_decay(
            value=spread_ratio,
            lower=threshold,
            upper=max(wide, threshold + 1e-6),
            high_score=0.0,
            low_score=wide_penalty,
        )
        score = score.mask(valid & (spread_ratio < threshold), left)
        score = score.mask(valid & (spread_ratio >= threshold) & (spread_ratio < wide), mid_decay)
        score = score.mask(valid & (spread_ratio >= wide), wide_penalty)
    else:
        score = score.mask(valid & (spread_ratio < threshold), narrow_bonus)
        score = score.mask(valid & (spread_ratio >= wide), wide_penalty)
    return score.astype("float64")


def _score_iv_series(iv: pd.Series, *, cfg) -> pd.Series:
    """
    Purpose:
        Score candidate strikes using implied-volatility fairness heuristics.
    
    Context:
        Internal helper in the `strike selector` module. It isolates one strike-selection factor so candidate ranking remains explainable.
    
    Inputs:
        iv (pd.Series): Implied-volatility series for the candidate contracts.
        cfg (Any): Configuration dictionary containing thresholds, weights, and score buckets.
    
    Returns:
        pd.Series: Integer score for each strike based on implied-volatility fairness.
    
    Notes:
        Factor outputs are kept separate so the final strike ranking can be audited in research and shadow-mode diagnostics.
    """
    iv_values = iv.astype(float)

    if _continuous_mode(cfg):
        iv_low_min = float(cfg["iv_low_min"])
        iv_low_max = float(cfg["iv_low_max"])
        iv_mid_max = float(cfg["iv_mid_max"])
        iv_high = float(cfg["iv_high_threshold"])
        low_score = float(cfg["iv_low_score"])
        mid_score = float(cfg["iv_mid_score"])
        high_pen = float(cfg["iv_high_penalty"])

        score = pd.Series(0.0, index=iv.index, dtype="float64")
        left = _linear_decay(
            value=iv_values,
            lower=iv_low_min,
            upper=iv_low_max,
            high_score=low_score,
            low_score=mid_score,
        )
        right = _linear_decay(
            value=iv_values,
            lower=iv_mid_max,
            upper=max(iv_high, iv_mid_max + 1e-6),
            high_score=mid_score,
            low_score=high_pen,
        )

        score = score.mask(iv_values.between(iv_low_min, iv_low_max, inclusive="both"), left)
        score = score.mask(iv_values.gt(iv_low_max) & iv_values.le(iv_mid_max), mid_score)
        score = score.mask(iv_values.gt(iv_mid_max) & iv_values.le(iv_high), right)
        score = score.mask(iv_values > iv_high, high_pen)
        score = score.mask(iv_values <= 0, 0.0)
        return score

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
    return pd.Series(scores, index=iv.index, dtype="float64")


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
    """
    Purpose:
        Compute the additive strike-ranking factors for every candidate contract.

    Context:
        The strike selector expresses trade construction as a weighted scorecard. This helper attaches each factor separately so live diagnostics and research can explain why one strike beat another.

    Inputs:
        rows (pd.DataFrame): Candidate option rows to score.
        direction (Any): Requested trade direction, typically `CALL` or `PUT`.
        spot (Any): Current underlying spot price.
        cfg (Any): Strike-selection configuration with factor thresholds and score values.
        support_wall (Any): Support level inferred from market structure.
        resistance_wall (Any): Resistance level inferred from market structure.
        gamma_clusters (Any): Gamma concentration levels from the analytics layer.
        max_capital (Any): Maximum capital budget available for the trade.
        lot_size (Any): Lot size used when translating premium into capital usage.

    Returns:
        pd.DataFrame: Candidate frame enriched with factor-level scores and total base score.

    Notes:
        The factor design is heuristic and intentionally transparent; each column becomes part of the strike-selection audit trail.
    """
    scored = rows.copy()
    # Each factor captures a different execution trade-off: directional fit,
    # affordability, liquidity, structural levels, and volatility posture.
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
    scored["_ba_spread_score"] = _score_ba_spread_series(scored["_normalized_ba_spread_ratio"], cfg=cfg)
    scored["_base_score"] = (
        scored["_moneyness_score"]
        + scored["_directional_side_score"]
        + scored["_premium_score"]
        + scored["_liquidity_score"]
        + scored["_wall_distance_score"]
        + scored["_gamma_cluster_score"]
        + scored["_iv_score"]
        + scored["_ba_spread_score"]
    ).astype("float64")
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
    gamma_regime=None,
    spot_vs_flip=None,
    dealer_hedging_bias=None,
    gamma_flip_distance_pct=None,
    dealer_gamma_exposure=None,
    atm_iv=None,
    days_to_expiry=None,
    vol_surface_regime=None,
    volatility_shock_score=None,
    directional_call_probability=None,
    directional_put_probability=None,
):
    """
    Purpose:
        Rank option contracts that could express the requested directional thesis.
    
    Context:
        Public function in the `strike selector` module. It forms part of the strategy workflow exposed by this module.
    
    Inputs:
        option_chain (Any): Option-chain snapshot used for scoring or signal generation.
        direction (Any): Directional side requested by the strategy, typically `CALL` or `PUT`.
        spot (Any): Current underlying spot price.
        support_wall (Any): Nearest support wall inferred from positioning or open-interest structure.
        resistance_wall (Any): Nearest resistance wall inferred from positioning or open-interest structure.
        gamma_clusters (Any): Mapped gamma concentration levels that can pin or accelerate price action.
        lot_size (Any): Lot size used to translate premium into per-trade capital usage.
        max_capital (Any): Maximum capital budget available for the trade.
        top_n (Any): Number of top-ranked contracts to retain after scoring.
        strike_window_steps (Any): Strike-window width, measured in strike increments around spot.
        candidate_score_hook (Any): Optional callback that adds custom candidate-level score adjustments.
    
    Returns:
        list[dict]: Ranked contract candidates, including total scores and supporting diagnostics.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    if option_chain is None or len(option_chain) == 0:
        return []

    def _resolve_candidate_tte(_row_get):
        expiry_value = (
            _row_get("EXPIRY_DT")
            or _row_get("selected_expiry")
            or _row_get("expiry")
            or _row_get("expiry_date")
        )
        tte_value = _parse_expiry_years(expiry_value) if expiry_value is not None else None
        if tte_value is None:
            # Same-day expiries are often date-only strings. They can parse as
            # midnight and look expired intraday; keep a small positive TTE so
            # IV/Greeks fallback can still run for live contracts.
            dte = _safe_float(days_to_expiry, None)
            if dte is not None:
                if dte > 0:
                    tte_value = max(dte / 365.0, 1.0 / (365.0 * 24.0))
                else:
                    tte_value = 1.0 / (365.0 * 24.0)
        if tte_value is None and expiry_value is not None:
            try:
                expiry_ts = pd.to_datetime(expiry_value, errors="coerce")
                if not pd.isna(expiry_ts):
                    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
                    expiry_date = expiry_ts.date()
                    if expiry_date >= now_ist.date():
                        tte_value = 1.0 / (365.0 * 24.0)
            except Exception:
                pass
        return tte_value

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

    # Apply volatility-aware strike distance adjustment (PRIORITY 3).
    # If the caller did not explicitly override strike_window_steps, we map
    # the runtime volatility regime to a dynamic strike window.
    vol_shock = float(volatility_shock_score or 0.0)
    if strike_window_steps is None and vol_shock > 0.0:
        adjusted_window = 4 + (min(vol_shock, 1.0) * 11)
        effective_window_steps = int(round(adjusted_window))

    effective_window_steps = max(int(effective_window_steps), 1)
    
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

    # Compute enhanced institutional-grade scoring factors.  The result is
    # index-aligned with *rows* so columns can be looked up per-candidate
    # inside the record loop below.
    enhanced = compute_enhanced_strike_scores(
        rows,
        spot=float(spot),
        direction=direction,
        gamma_clusters=gamma_clusters,
        gamma_regime=gamma_regime,
        spot_vs_flip=spot_vs_flip,
        dealer_hedging_bias=dealer_hedging_bias,
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        dealer_gamma_exposure=dealer_gamma_exposure,
        atm_iv=atm_iv,
        days_to_expiry=days_to_expiry,
        vol_surface_regime=vol_surface_regime,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
    )
    _has_enhanced = not enhanced.empty
    if _has_enhanced:
        rows = rows.join(enhanced, how="left")

    # When no external hook mutates scores, we can pre-rank and trim the
    # candidate set before expensive IV/Greeks fallback work.
    if not callable(candidate_score_hook) and len(rows) > top_n:
        pre_rank = rows.copy()
        pre_rank["_spot_distance"] = (pre_rank["_normalized_strike"].astype(float) - float(spot)).abs()
        pre_rank["_price_tiebreak"] = pre_rank["_normalized_last_price"].where(
            pre_rank["_normalized_last_price"] > 0,
            10**9,
        )
        pre_rank = pre_rank.sort_values(
            by=["_base_score", "_spot_distance", "_price_tiebreak", "_normalized_strike"],
            ascending=[False, True, True, True],
        )
        pre_limit = max(int(top_n) * 6, 25)
        rows = pre_rank.head(pre_limit).drop(columns=["_spot_distance", "_price_tiebreak"], errors="ignore")

    # Build a same-side delta anchor curve for second-tier fallback.
    delta_anchor_points = []
    if "DELTA" in rows.columns and "_normalized_strike" in rows.columns:
        anchor_df = pd.DataFrame(
            {
                "strike": pd.to_numeric(rows["_normalized_strike"], errors="coerce"),
                "delta": pd.to_numeric(rows["DELTA"], errors="coerce"),
            }
        ).dropna()
        if not anchor_df.empty:
            anchor_df = anchor_df[np.isfinite(anchor_df["strike"]) & np.isfinite(anchor_df["delta"])]
            if not anchor_df.empty:
                anchor_df = anchor_df.groupby("strike", as_index=False)["delta"].mean().sort_values("strike")
                delta_anchor_points = [
                    (float(row["strike"]), float(row["delta"]))
                    for _, row in anchor_df.iterrows()
                ]

    iv_anchor_points = []
    if "_normalized_iv" in rows.columns and "_normalized_strike" in rows.columns:
        iv_anchor_df = pd.DataFrame(
            {
                "strike": pd.to_numeric(rows["_normalized_strike"], errors="coerce"),
                "iv": pd.to_numeric(rows["_normalized_iv"], errors="coerce"),
            }
        ).dropna()
        if not iv_anchor_df.empty:
            iv_anchor_df = iv_anchor_df[np.isfinite(iv_anchor_df["strike"]) & np.isfinite(iv_anchor_df["iv"])]
            iv_anchor_df = iv_anchor_df[iv_anchor_df["iv"] > 0]
            if not iv_anchor_df.empty:
                iv_anchor_df = iv_anchor_df.groupby("strike", as_index=False)["iv"].mean().sort_values("strike")
                iv_anchor_points = [
                    (float(row["strike"]), float(row["iv"]))
                    for _, row in iv_anchor_df.iterrows()
                ]

    strike_step = _infer_strike_step(rows) or 50.0
    max_neighbor_distance = max(float(strike_step) * 4.0, 120.0)
    spot_safe = _safe_float(spot, None)

    def _delta_from_neighbors(target_strike: float):
        if not delta_anchor_points:
            return None

        strikes = [pt[0] for pt in delta_anchor_points]
        deltas = [pt[1] for pt in delta_anchor_points]

        # Exact strike match if available.
        for idx, strike_value in enumerate(strikes):
            if abs(strike_value - target_strike) <= 1e-9:
                return deltas[idx]

        left_idx = None
        right_idx = None
        for idx, strike_value in enumerate(strikes):
            if strike_value < target_strike:
                left_idx = idx
            elif strike_value > target_strike:
                right_idx = idx
                break

        if left_idx is not None and right_idx is not None:
            left_strike, left_delta = strikes[left_idx], deltas[left_idx]
            right_strike, right_delta = strikes[right_idx], deltas[right_idx]
            span = right_strike - left_strike
            if span > 1e-9:
                weight = (target_strike - left_strike) / span
                return left_delta + weight * (right_delta - left_delta)

        nearest = None
        nearest_dist = None
        if left_idx is not None:
            dist = abs(target_strike - strikes[left_idx])
            nearest = deltas[left_idx]
            nearest_dist = dist
        if right_idx is not None:
            dist = abs(strikes[right_idx] - target_strike)
            if nearest is None or dist < nearest_dist:
                nearest = deltas[right_idx]
                nearest_dist = dist

        if nearest is not None and nearest_dist is not None and nearest_dist <= max_neighbor_distance:
            return nearest
        return None

    def _iv_from_neighbors(target_strike: float):
        if not iv_anchor_points:
            return None

        strikes = [pt[0] for pt in iv_anchor_points]
        ivs = [pt[1] for pt in iv_anchor_points]

        for idx, strike_value in enumerate(strikes):
            if abs(strike_value - target_strike) <= 1e-9:
                return ivs[idx]

        left_idx = None
        right_idx = None
        for idx, strike_value in enumerate(strikes):
            if strike_value < target_strike:
                left_idx = idx
            elif strike_value > target_strike:
                right_idx = idx
                break

        if left_idx is not None and right_idx is not None:
            left_strike, left_iv = strikes[left_idx], ivs[left_idx]
            right_strike, right_iv = strikes[right_idx], ivs[right_idx]
            span = right_strike - left_strike
            if span > 1e-9:
                weight = (target_strike - left_strike) / span
                return left_iv + weight * (right_iv - left_iv)

        nearest = None
        nearest_dist = None
        if left_idx is not None:
            dist = abs(target_strike - strikes[left_idx])
            nearest = ivs[left_idx]
            nearest_dist = dist
        if right_idx is not None:
            dist = abs(strikes[right_idx] - target_strike)
            if nearest is None or dist < nearest_dist:
                nearest = ivs[right_idx]
                nearest_dist = dist

        if nearest is not None and nearest_dist is not None and nearest_dist <= max_neighbor_distance:
            return nearest
        return None

    def _normalize_iv_pct(raw_iv):
        iv_value = _safe_float(raw_iv, None)
        if iv_value is None or iv_value <= 0:
            return None
        if iv_value <= 1.5:
            return iv_value * 100.0
        return iv_value

    def _iv_from_moneyness_proxy(target_strike: float):
        if spot_safe is None or spot_safe <= 0:
            return None

        atm_iv_pct = _normalize_iv_pct(atm_iv)
        if atm_iv_pct is None and iv_anchor_points:
            atm_iv_pct = float(np.median([pt[1] for pt in iv_anchor_points]))
        if atm_iv_pct is None:
            atm_iv_pct = 80.0

        distance_ratio = abs(float(target_strike) - float(spot_safe)) / max(float(spot_safe), 1e-6)
        smile_mult = 1.0 + min(distance_ratio * 8.0, 0.35)
        skew_mult = 1.0
        if option_type == "PE" and target_strike < spot_safe:
            skew_mult = 1.05
        elif option_type == "CE" and target_strike > spot_safe:
            skew_mult = 1.05

        proxy_iv = atm_iv_pct * smile_mult * skew_mult
        return max(5.0, min(300.0, proxy_iv))

    def _delta_from_moneyness_proxy(target_strike: float):
        if spot_safe is None or spot_safe <= 0:
            return None

        scale = max(float(spot_safe) * 0.01, float(strike_step))
        x = (float(target_strike) - float(spot_safe)) / max(scale, 1e-6)
        x = max(min(x, 20.0), -20.0)
        call_delta = 1.0 / (1.0 + math.exp(x))
        call_delta = max(0.05, min(0.95, call_delta))

        if option_type == "CE":
            return call_delta

        put_delta = call_delta - 1.0
        return max(-0.95, min(-0.05, put_delta))

    def _resolve_direction_probability(option_type_local: str) -> float:
        call_prob = _safe_float(directional_call_probability, None)
        put_prob = _safe_float(directional_put_probability, None)

        if option_type_local == "CE":
            if call_prob is not None:
                return float(_clip(call_prob, 0.05, 0.95))
            if put_prob is not None:
                return float(_clip(1.0 - put_prob, 0.05, 0.95))
        else:
            if put_prob is not None:
                return float(_clip(put_prob, 0.05, 0.95))
            if call_prob is not None:
                return float(_clip(1.0 - call_prob, 0.05, 0.95))

        return 0.5

    def _expected_value_score_adjustment(*, strike_value: float, premium_value: float, option_type_local: str, hook_payload_local: dict):
        """Translate directional edge + contract geometry into a bounded EV score tweak."""
        direction_prob = _resolve_direction_probability(option_type_local)
        option_eff = _clip(_safe_float(hook_payload_local.get("option_efficiency_score"), 50.0) / 100.0, 0.0, 1.0)
        reachability = _clip(_safe_float(hook_payload_local.get("target_reachability_score"), 50.0) / 100.0, 0.0, 1.0)
        expected_move_pts = _safe_float(hook_payload_local.get("expected_move_points"), None)

        # Strike-distance alignment relative to expected move (1.0 is best, 0.0 is poor reachability).
        distance_alignment = 0.5
        premium_drag = 0.33
        if expected_move_pts is not None and expected_move_pts > 0 and spot_safe is not None:
            distance_ratio = abs(float(strike_value) - float(spot_safe)) / max(float(expected_move_pts), 1e-6)
            distance_alignment = _clip(1.0 - 0.5 * distance_ratio, 0.0, 1.0)
            premium_drag = _clip(float(premium_value) / max(float(expected_move_pts), 1e-6), 0.0, 3.0) / 3.0
        elif spot_safe is not None:
            fallback_scale = max(abs(float(strike_value) - float(spot_safe)), 1.0)
            premium_drag = _clip(float(premium_value) / fallback_scale, 0.0, 3.0) / 3.0

        upside = direction_prob * (0.35 + (0.35 * option_eff) + (0.30 * reachability)) * (0.5 + 0.5 * distance_alignment)
        downside = (1.0 - direction_prob) * (0.4 + 0.6 * premium_drag)
        ev_edge = upside - downside
        ev_adjustment = round(_clip(ev_edge * 8.0, -4.0, 4.0), 2)

        diagnostics = {
            "direction_probability": round(direction_prob, 4),
            "distance_alignment": round(float(distance_alignment), 4),
            "premium_drag": round(float(premium_drag), 4),
            "ev_edge": round(float(ev_edge), 4),
        }
        return ev_adjustment, diagnostics

    # Convert the scored dataframe into plain records because the final engine
    # payload and downstream reporting stack work with JSON-friendly structures.
    candidates = []
    row_columns = list(rows.columns)
    col_index = {col: idx for idx, col in enumerate(row_columns)}
    for row in rows.itertuples(index=False, name=None):
        def _row_get(col_name, default=None):
            idx = col_index.get(col_name)
            if idx is None:
                return default
            return row[idx]

        strike = _safe_float(_row_get("_normalized_strike", _row_get("strikePrice")), None)
        if strike is None:
            continue

        premium = _safe_float(_row_get("_normalized_last_price"), _safe_float(_row_get("lastPrice"), 0.0))
        volume = _safe_float(_row_get("_normalized_volume"), _safe_float(_row_get("totalTradedVolume"), _row_get("VOLUME", 0.0)))
        oi = _safe_float(_row_get("_normalized_open_interest"), _safe_float(_row_get("openInterest"), _row_get("OPEN_INT", 0.0)))
        iv = _safe_float(_row_get("_normalized_iv"), _safe_float(_row_get("impliedVolatility"), _row_get("IV", 0.0)))
        iv_proxy_source = None
        tte = None

        # Fallback: estimate IV from market price when upstream enrichment missed this row
        if (iv is None or iv <= 0) and premium > 0 and strike and spot:
            tte = _resolve_candidate_tte(_row_get)
            estimated_iv = estimate_iv_from_price(premium, spot, strike, tte, option_type)
            if estimated_iv and estimated_iv > 0:
                iv = estimated_iv

        if iv is None or iv <= 0:
            neighbor_iv = _iv_from_neighbors(float(strike))
            if neighbor_iv is not None and neighbor_iv > 0:
                iv = neighbor_iv
                iv_proxy_source = "NEIGHBOR_INTERPOLATION"

        if iv is None or iv <= 0:
            proxy_iv = _iv_from_moneyness_proxy(float(strike))
            if proxy_iv is not None and proxy_iv > 0:
                iv = proxy_iv
                if _normalize_iv_pct(atm_iv) is not None:
                    iv_proxy_source = "ATM_MONEYNESS_PROXY"
                else:
                    iv_proxy_source = "MONEYNESS_PROXY"

        score_breakdown = {
            "moneyness_score": round(_safe_float(_row_get("_moneyness_score", 0.0), 0.0), 2),
            "directional_side_score": round(_safe_float(_row_get("_directional_side_score", 0.0), 0.0), 2),
            "premium_score": round(_safe_float(_row_get("_premium_score", 0.0), 0.0), 2),
            "liquidity_score": round(_safe_float(_row_get("_liquidity_score", 0.0), 0.0), 2),
            "wall_distance_score": round(_safe_float(_row_get("_wall_distance_score", 0.0), 0.0), 2),
            "gamma_cluster_score": round(_safe_float(_row_get("_gamma_cluster_score", 0.0), 0.0), 2),
            "iv_score": round(_safe_float(_row_get("_iv_score", 0.0), 0.0), 2),
            "ba_spread_score": round(_safe_float(_row_get("_ba_spread_score", 0.0), 0.0), 2),
        }

        hook_payload = {}
        if callable(candidate_score_hook):
            try:
                row_payload = {col: row[idx] for col, idx in col_index.items()}
                hook_payload = candidate_score_hook(
                    row_payload,
                    {
                        "strike": _to_python_number(strike),
                        "last_price": round(premium, 2),
                        "volume": int(volume),
                        "open_interest": int(oi),
                        "iv": round(iv, 2) if iv and iv > 0 else None,
                    },
                ) or {}
            except Exception as exc:
                _warn_once(
                    "candidate_score_hook_failed",
                    "strike_selector: candidate_score_hook failed; using neutral hook payload (%s)",
                    exc,
                )
                hook_payload = {}

        # The hook is used by option-efficiency scoring to reward contracts that
        # fit the target/stop geometry better than the base heuristics alone.
        efficiency_score_adjustment = int(_safe_float(hook_payload.get("score_adjustment"), 0.0))
        if efficiency_score_adjustment:
            score_breakdown["option_efficiency_score_adjustment"] = efficiency_score_adjustment

        ev_score_adjustment, ev_diagnostics = _expected_value_score_adjustment(
            strike_value=float(strike),
            premium_value=float(premium),
            option_type_local=option_type,
            hook_payload_local=hook_payload,
        )
        if ev_score_adjustment:
            score_breakdown["expected_value_score_adjustment"] = ev_score_adjustment

        total_score = round(
            _safe_float(_row_get("_base_score", 0.0), 0.0)
            + efficiency_score_adjustment
            + ev_score_adjustment,
            2,
        )
        delta_raw = _safe_float(_row_get("DELTA"), None)
        delta_proxy_source = None
        ba_spread_ratio = _safe_float(_row_get("_normalized_ba_spread_ratio"), None)

        # Fallback: compute delta from estimated IV when upstream enrichment missed
        if (delta_raw is None or (isinstance(delta_raw, float) and not (delta_raw == delta_raw))) and iv and iv > 0 and strike and spot:
            if tte is None:
                tte = _resolve_candidate_tte(_row_get)
            if tte is not None:
                greeks = compute_option_greeks(
                    spot=spot, strike=strike, time_to_expiry_years=tte,
                    volatility_pct=iv, option_type=option_type,
                )
                if greeks is not None:
                    delta_raw = greeks["DELTA"]
                    if iv_proxy_source:
                        delta_proxy_source = "GREEKS_FROM_IV_PROXY"
                    else:
                        delta_proxy_source = "GREEKS_FROM_IV"

        if delta_raw is None or (isinstance(delta_raw, float) and not (delta_raw == delta_raw)):
            neighbor_delta = _delta_from_neighbors(float(strike))
            if neighbor_delta is not None:
                delta_raw = neighbor_delta
                delta_proxy_source = "NEIGHBOR_INTERPOLATION"

        if delta_raw is None or (isinstance(delta_raw, float) and not (delta_raw == delta_raw)):
            proxy_delta = _delta_from_moneyness_proxy(float(strike))
            if proxy_delta is not None:
                delta_raw = proxy_delta
                delta_proxy_source = "MONEYNESS_PROXY"

        record = {
            "option_type": option_type,
            "strike": _to_python_number(strike),
            "last_price": round(premium, 2),
            "volume": int(volume),
            "open_interest": int(oi),
            "iv": round(iv, 2) if iv and iv > 0 else None,
            "iv_is_proxy": bool(iv_proxy_source),
            "delta": round(delta_raw, 4) if delta_raw is not None else None,
            "delta_is_proxy": bool(delta_proxy_source),
            "capital_per_lot": round(premium * lot_size, 2) if lot_size is not None else None,
            "score": total_score,
            "score_breakdown": score_breakdown,
        }
        if iv_proxy_source:
            record["iv_proxy_source"] = iv_proxy_source
        if delta_proxy_source:
            record["delta_proxy_source"] = delta_proxy_source
        if ba_spread_ratio is not None:
            record["ba_spread_ratio"] = round(float(ba_spread_ratio), 4)
            record["ba_spread_pct"] = round(float(ba_spread_ratio) * 100.0, 2)
            record["ba_spread_score"] = round(_safe_float(_row_get("_ba_spread_score", 0.0), 0.0), 2)
        if hook_payload:
            record.update({key: value for key, value in hook_payload.items() if key != "score_adjustment"})
        record["direction_probability"] = ev_diagnostics["direction_probability"]
        record["expected_value_edge"] = ev_diagnostics["ev_edge"]
        record["expected_value_distance_alignment"] = ev_diagnostics["distance_alignment"]
        record["expected_value_premium_drag"] = ev_diagnostics["premium_drag"]

        # Attach enhanced institutional-grade scoring fields when available.
        if _has_enhanced:
            _enhanced_fields = (
                "enhanced_strike_score", "liquidity_score", "gamma_magnetism",
                "dealer_pressure", "convexity_score", "premium_efficiency",
                "payoff_efficiency_score",
                "pe_premium_eff", "pe_delta_align", "pe_liquidity",
                "pe_dist_target", "pe_iv_eff",
                "distance_from_spot_pts", "distance_from_spot_pct",
                "gamma_regime", "spot_vs_flip", "dealer_hedging_bias",
                "vol_surface_regime", "tradable_intraday", "tradable_overnight",
                "liquidity_ok", "premium_reasonable",
            )
            for field in _enhanced_fields:
                val = _row_get(field)
                if val is not None:
                    record[field] = _to_python_number(val) if isinstance(val, (np.integer, np.floating)) else val

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
    gamma_regime=None,
    spot_vs_flip=None,
    dealer_hedging_bias=None,
    gamma_flip_distance_pct=None,
    dealer_gamma_exposure=None,
    atm_iv=None,
    days_to_expiry=None,
    vol_surface_regime=None,
    volatility_shock_score=None,
    directional_call_probability=None,
    directional_put_probability=None,
):
    """
    Purpose:
        Select the highest-ranked strike candidate for trade construction.
    
    Context:
        Public function in the `strike selector` module. It forms part of the strategy workflow exposed by this module.
    
    Inputs:
        option_chain (Any): Option-chain snapshot used for scoring or signal generation.
        direction (Any): Directional side requested by the strategy, typically `CALL` or `PUT`.
        spot (Any): Current underlying spot price.
        support_wall (Any): Nearest support wall inferred from positioning or open-interest structure.
        resistance_wall (Any): Nearest resistance wall inferred from positioning or open-interest structure.
        gamma_clusters (Any): Mapped gamma concentration levels that can pin or accelerate price action.
        lot_size (Any): Lot size used to translate premium into per-trade capital usage.
        max_capital (Any): Maximum capital budget available for the trade.
        strike_window_steps (Any): Strike-window width, measured in strike increments around spot.
        candidate_score_hook (Any): Optional callback that adds custom candidate-level score adjustments.
    
    Returns:
        Any: The highest-ranked strike candidate, or the module fallback when no candidate qualifies.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
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
        gamma_regime=gamma_regime,
        spot_vs_flip=spot_vs_flip,
        dealer_hedging_bias=dealer_hedging_bias,
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        dealer_gamma_exposure=dealer_gamma_exposure,
        atm_iv=atm_iv,
        days_to_expiry=days_to_expiry,
        vol_surface_regime=vol_surface_regime,
        volatility_shock_score=volatility_shock_score,
        directional_call_probability=directional_call_probability,
        directional_put_probability=directional_put_probability,
    )

    if not ranked:
        return None, []

    return ranked[0]["strike"], ranked
