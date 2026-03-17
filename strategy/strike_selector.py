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

import numpy as np
import pandas as pd

from analytics.greeks_engine import estimate_iv_from_price, compute_option_greeks, _parse_expiry_years
from config.strike_selection_policy import get_strike_selection_score_config
from strategy.enhanced_strike_scoring import compute_enhanced_strike_scores
from utils.numerics import safe_float as _safe_float, to_python_number as _to_python_number  # noqa: F401


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
    gamma_regime=None,
    spot_vs_flip=None,
    dealer_hedging_bias=None,
    gamma_flip_distance_pct=None,
    dealer_gamma_exposure=None,
    atm_iv=None,
    days_to_expiry=None,
    vol_surface_regime=None,
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

    # Merge enhanced columns into the scored frame so they appear in each
    # per-candidate record dict without needing a secondary index lookup.
    if _has_enhanced:
        for col in enhanced.columns:
            rows[col] = enhanced[col]

    # Convert the scored dataframe into plain records because the final engine
    # payload and downstream reporting stack work with JSON-friendly structures.
    candidates = []
    for row in rows.to_dict(orient="records"):
        strike = _safe_float(row.get("_normalized_strike", row.get("strikePrice")), None)
        if strike is None:
            continue

        premium = _safe_float(row.get("_normalized_last_price"), _safe_float(row.get("lastPrice"), 0.0))
        volume = _safe_float(row.get("_normalized_volume"), _safe_float(row.get("totalTradedVolume"), row.get("VOLUME", 0.0)))
        oi = _safe_float(row.get("_normalized_open_interest"), _safe_float(row.get("openInterest"), row.get("OPEN_INT", 0.0)))
        iv = _safe_float(row.get("_normalized_iv"), _safe_float(row.get("impliedVolatility"), row.get("IV", 0.0)))

        # Fallback: estimate IV from market price when upstream enrichment missed this row
        if (iv is None or iv <= 0) and premium > 0 and strike and spot:
            tte = _parse_expiry_years(row.get("EXPIRY_DT"))
            estimated_iv = estimate_iv_from_price(premium, spot, strike, tte, option_type)
            if estimated_iv and estimated_iv > 0:
                iv = estimated_iv

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

        # The hook is used by option-efficiency scoring to reward contracts that
        # fit the target/stop geometry better than the base heuristics alone.
        efficiency_score_adjustment = int(_safe_float(hook_payload.get("score_adjustment"), 0.0))
        if efficiency_score_adjustment:
            score_breakdown["option_efficiency_score_adjustment"] = efficiency_score_adjustment

        total_score = int(row.get("_base_score", 0)) + efficiency_score_adjustment
        delta_raw = _safe_float(row.get("DELTA"), None)

        # Fallback: compute delta from estimated IV when upstream enrichment missed
        if (delta_raw is None or (isinstance(delta_raw, float) and not (delta_raw == delta_raw))) and iv and iv > 0 and strike and spot:
            tte = _parse_expiry_years(row.get("EXPIRY_DT"))
            if tte is not None:
                greeks = compute_option_greeks(
                    spot=spot, strike=strike, time_to_expiry_years=tte,
                    volatility_pct=iv, option_type=option_type,
                )
                if greeks is not None:
                    delta_raw = greeks["DELTA"]

        record = {
            "option_type": option_type,
            "strike": _to_python_number(strike),
            "last_price": round(premium, 2),
            "volume": int(volume),
            "open_interest": int(oi),
            "iv": round(iv, 2) if iv else 0,
            "delta": round(delta_raw, 4) if delta_raw is not None else None,
            "capital_per_lot": round(premium * lot_size, 2) if lot_size is not None else None,
            "score": total_score,
            "score_breakdown": score_breakdown,
        }
        if hook_payload:
            record.update({key: value for key, value in hook_payload.items() if key != "score_adjustment"})

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
                val = row.get(field)
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
    )

    if not ranked:
        return None, []

    return ranked[0]["strike"], ranked
