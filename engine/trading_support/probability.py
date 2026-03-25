"""
Module: probability.py

Purpose:
    Convert market-state features into rule-based, model-based, and blended move probabilities.

Role in the System:
    Part of the signal engine layer that assembles analytics, strategy logic, and overlays into trade decisions.

Key Outputs:
    Probability estimates plus the component features that explain those estimates.

Downstream Usage:
    Consumed by trade-strength scoring, confirmation filters, and research diagnostics.
"""
from __future__ import annotations

import logging

from config.probability_feature_policy import get_probability_feature_policy_config
from config.symbol_microstructure import get_microstructure_config
import models.feature_builder as feature_builder_mod
import models.large_move_probability as large_move_probability_mod
import models.ml_move_predictor as ml_move_predictor_mod
from utils.regime_normalization import normalize_iv_decimal

from .common import _call_first, _clip, _safe_float


_MOVE_PREDICTOR = None
_WARN_ONCE_KEYS: set[str] = set()
_LOG = logging.getLogger(__name__)


def _warn_once(key: str, message: str, *args) -> None:
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    _LOG.warning(message, *args)


def _map_vacuum_strength(vacuum_state, liquidity_voids=None, nearest_vacuum_gap_pct=None):
    """
    Purpose:
        Convert liquidity-vacuum diagnostics into a bounded feature used by the move-probability model.

    Context:
        The probability layer treats nearby liquidity voids as a heuristic for one-sided price travel. A stronger vacuum score means price may move more freely once it leaves dense resting liquidity.

    Inputs:
        vacuum_state (Any): Categorical vacuum regime from market-state assembly.
        liquidity_voids (Any): Optional collection of detected liquidity voids.
        nearest_vacuum_gap_pct (Any): Distance from spot to the nearest vacuum gap, expressed as a percentage.

    Returns:
        float: Vacuum-strength feature clipped to the configured 0-1 range.

    Notes:
        This is a market-microstructure heuristic rather than a formal probability model; it gives more weight to nearby gaps and to markets with several voids.
    """
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
        except (TypeError, ValueError) as exc:
            _warn_once(
                "vacuum_void_count_invalid",
                "probability: invalid liquidity_voids payload; using base vacuum strength (%s)",
                exc,
            )

    return round(_clip(base, 0.0, 1.0), 3)


def _map_hedging_flow_ratio(hedging_bias, hedge_flow_value=None):
    """
    Purpose:
        Convert dealer-hedging diagnostics into a signed flow feature for the probability model.

    Context:
        Probability estimation uses dealer flow as a directional accelerator or dampener. When a continuous hedge-flow value is available it takes precedence; otherwise categorical hedging bias is mapped into a signed proxy.

    Inputs:
        hedging_bias (Any): Categorical dealer-hedging state from market-state assembly.
        hedge_flow_value (Any): Optional continuous hedge-flow estimate already normalized upstream.

    Returns:
        float: Signed hedging-flow feature in the configured `[-1, 1]` range.

    Notes:
        Positive values represent upside-supportive hedging pressure and negative values represent downside-supportive pressure.
    """
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
    """
    Purpose:
        Map categorical or numeric flow signals into a bounded smart-money score.
    
    Context:
        Internal helper in the `probability` module. It supports market-state or probability assembly without cluttering the top-level signal flow.
    
    Inputs:
        smart_money_flow (Any): Categorical flow label summarizing directional institutional activity.
        flow_imbalance (Any): Signed flow imbalance used as a continuous confirmation signal.
    
    Returns:
        float | int: Signed flow score clipped to the configured smart-money range.
    
    Notes:
        The output is kept bounded so additive scoring and downstream calibration remain stable across symbols and sessions.
    """
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
    """
    Purpose:
        Measure how far spot is from the estimated gamma-flip level.
    
    Context:
        Internal helper in the `probability` module. It supports market-state or probability assembly without cluttering the top-level signal flow.
    
    Inputs:
        spot_price (Any): Current underlying spot price.
        gamma_flip (Any): Estimated gamma-flip level where dealer hedging behavior changes sign.
    
    Returns:
        float | None: Percentage distance between spot and the gamma-flip level, or `None` when unavailable.
    
    Notes:
        The output is kept bounded so additive scoring and downstream calibration remain stable across symbols and sessions.
    """
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
    """
    Purpose:
        Normalize the session range relative to the symbol's typical intraday movement.
    
    Context:
        Internal helper in the `probability` module. It supports market-state or probability assembly without cluttering the top-level signal flow.
    
    Inputs:
        symbol (Any): Underlying symbol or index identifier.
        spot_price (Any): Current underlying spot price.
        day_high (Any): Session high used to measure realized intraday range.
        day_low (Any): Session low used to measure realized intraday range.
        day_open (Any): Session open used as an early-session anchor.
        prev_close (Any): Previous session close used as a contextual anchor.
        lookback_avg_range_pct (Any): Historical average range percentage used to normalize today's move.
    
    Returns:
        float | None: Realized intraday range normalized by the symbol's typical range, or `None` when insufficient data is available.
    
    Notes:
        The output is kept bounded so additive scoring and downstream calibration remain stable across symbols and sessions.
    """
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
        # Early in the session we may not have a meaningful full range yet, so
        # anchor the estimate to the strongest observed move from open/close.
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
    """
    Purpose:
        Compute ATM IV percentile from the supplied inputs.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        atm_iv (Any): Input associated with ATM IV.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_probability_feature_policy_config()
    iv = normalize_iv_decimal(atm_iv, default=None)
    if iv is None:
        return None

    iv_low = normalize_iv_decimal(cfg.atm_iv_low, default=None)
    iv_high = normalize_iv_decimal(cfg.atm_iv_high, default=None)
    if iv_low is None or iv_high is None or iv_high <= iv_low:
        return None

    pct = (iv - iv_low) / max(iv_high - iv_low, 1e-6)
    return round(_clip(pct, 0.0, 1.0), 4)


def _blend_move_probability(rule_prob, ml_prob):
    """
    Purpose:
        Process blend move probability for downstream use.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        rule_prob (Any): Input associated with rule prob.
        ml_prob (Any): Input associated with ML prob.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    import math

    cfg = get_probability_feature_policy_config()
    rule_prob = _safe_float(rule_prob, cfg.probability_default_rule)
    used_ml_leg = ml_prob is not None
    if ml_prob is None:
        # Explicit fallback contract: with no ML leg, use rule probability
        # directly (bounded) without post-blend recalibration.
        raw = round(_clip(rule_prob, cfg.probability_floor, cfg.probability_ceiling), 2)
    else:
        ml_prob = _safe_float(ml_prob, rule_prob)
        w_rule = max(_safe_float(cfg.probability_rule_weight, 0.0), 0.0)
        w_ml = max(_safe_float(cfg.probability_ml_weight, 0.0), 0.0)
        w_total = max(w_rule + w_ml, 1e-9)
        hybrid = ((w_rule / w_total) * rule_prob) + ((w_ml / w_total) * ml_prob)
        hybrid = cfg.probability_intercept + (cfg.probability_scale * hybrid)
        raw = round(_clip(hybrid, cfg.probability_floor, cfg.probability_ceiling), 2)

    # Post-blend logistic recalibration: stretches the compressed distribution
    # so confident setups reach higher values and weak setups are pushed lower.
    if cfg.calibration_enabled and used_ml_leg:
        exponent = -cfg.calibration_steepness * (raw - cfg.calibration_midpoint)
        exponent = max(min(exponent, 500.0), -500.0)
        calibrated = 1.0 / (1.0 + math.exp(exponent))
        raw = round(_clip(calibrated, cfg.probability_floor, cfg.probability_ceiling), 2)

    return raw


def _get_move_predictor():
    """
    Purpose:
        Return the ML-leg predictor used by the blended probability path.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        If ACTIVE_MODEL points to a registry model, it is loaded as base_model.
        If ACTIVE_MODEL is empty, MLMovePredictor still runs with its internal
        heuristic implementation. Only hard init failures disable the ML leg.
    """
    global _MOVE_PREDICTOR

    predictor_class = getattr(ml_move_predictor_mod, "MLMovePredictor", None)
    if predictor_class is None:
        return None

    if _MOVE_PREDICTOR is None:
        try:
            base_model = None
            try:
                import joblib
                from pathlib import Path

                project_root = Path(__file__).resolve().parent.parent.parent

                # Check for registry model via ACTIVE_MODEL setting
                from config import settings as _settings
                active_name = getattr(_settings, "ACTIVE_MODEL", None)
                if active_name:
                    registry_path = project_root / "models_store" / "registry" / active_name / "model.joblib"
                    if registry_path.exists():
                        import sklearn as _sklearn
                        import warnings as _warnings
                        import logging as _plog

                        _version_seen = set()

                        def _sklearn_version_handler(message, category, filename, lineno, file=None, line=None):
                            key = str(message)
                            if key not in _version_seen:
                                _version_seen.add(key)
                                _plog.getLogger(__name__).warning(
                                    "sklearn version mismatch loading model from %s "
                                    "(installed: %s). Rebuild with: "
                                    "python scripts/build_model_registry.py — %s",
                                    registry_path, _sklearn.__version__, message,
                                )

                        with _warnings.catch_warnings():
                            _warnings.simplefilter("always")
                            _warnings.showwarning = _sklearn_version_handler
                            base_model = joblib.load(registry_path)



            except Exception as exc:
                _warn_once(
                    "move_predictor_model_load_failed",
                    "probability: failed to load ACTIVE_MODEL registry artifact; using heuristic ML leg (%s)",
                    exc,
                )
            _MOVE_PREDICTOR = predictor_class(base_model=base_model)
        except Exception as exc:
            _warn_once(
                "move_predictor_init_failed",
                "probability: ML predictor init failed; disabling ML leg for this process (%s)",
                exc,
            )
            _MOVE_PREDICTOR = False

    if _MOVE_PREDICTOR is False:
        return None
    return _MOVE_PREDICTOR


def _extract_nearest_vacuum_gap_pct(spot, vacuum_zones):
    """
    Purpose:
        Extract nearest vacuum gap percentage from the supplied payload.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        spot (Any): Input associated with spot.
        vacuum_zones (Any): Input associated with vacuum zones.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Extract hedge flow value from the supplied payload.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        hedging_flow (Any): Input associated with hedging flow.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Compute the categorical flow score used by the surrounding model.

    Context:
        Used within the trading support probability workflow. The module sits in the signal-engine layer that combines analytics, strategy rules, and overlays into final decisions.

    Inputs:
        value (Any): Raw value supplied by the caller.

    Returns:
        float | int: Score produced by the current heuristic.

    Notes:
        Internal helper that keeps the surrounding trading logic compact and readable.
    """
    return {
        "BULLISH_FLOW": 1.0,
        "BEARISH_FLOW": -1.0,
        "MIXED_FLOW": 0.0,
        "NEUTRAL_FLOW": 0.0,
    }.get(value, 0.0)


def _extract_probability(result):
    """
    Purpose:
        Extract a scalar probability from a model result payload.
    
    Context:
        Internal helper in the `probability` module. It supports market-state or probability assembly without cluttering the top-level signal flow.
    
    Inputs:
        result (Any): Structured result payload produced by an earlier computation or evaluation step.
    
    Returns:
        float: Scalar probability extracted from the supplied result payload.
    
    Notes:
        The output is kept bounded so additive scoring and downstream calibration remain stable across symbols and sessions.
    """
    cfg = get_probability_feature_policy_config()
    if result is None:
        return None
    if isinstance(result, dict):
        for key in ("probability", "move_probability", "score"):
            if key in result:
                value = _safe_float(result.get(key), None)
                if value is not None:
                    return round(_clip(value, cfg.probability_floor, cfg.probability_ceiling), 2)
        return None
    value = _safe_float(result, None)
    if value is None:
        return None
    return round(_clip(value, cfg.probability_floor, cfg.probability_ceiling), 2)


def _compute_probability_state_impl(
    df=None,
    *,
    spot=None,
    symbol=None,
    market_state=None,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None,
    global_context=None,
    _force_rule_only=False,
    _force_ml_only=False,
):
    """
    Purpose:
        Build the probability-state payload for the current market snapshot.

    Context:
        This helper is the feature-assembly step between market-state analytics
        and the downstream trade-strength model. It keeps both the raw features
        and the blended probability estimates available for diagnostics.

    Inputs:
        df (Any): Normalized option-chain dataframe supplied to the routine.
        spot (Any): Current underlying spot price.
        symbol (Any): Trading symbol or index identifier.
        market_state (Any): State payload for market state.
        day_high (Any): Session high used for intraday context.
        day_low (Any): Session low used for intraday context.
        day_open (Any): Session open used for intraday context.
        prev_close (Any): Previous close used as a reference anchor.
        lookback_avg_range_pct (Any): Historical average range percentage used as a baseline.

    Returns:
        dict: Rule, ML, and blended move probabilities plus the component
        features used to explain those estimates.

    Notes:
        Component values are returned alongside probabilities so research and
        promotion workflows can trace which features actually drove an edge.
    """
    cfg = get_probability_feature_policy_config()

    # Build the 7-feature vector first (backward compat for heuristic path).
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

    # Rebuild features with full context for v2 model (33 features).
    # This passes probability sub-components + market state context so the
    # expanded_feature_builder can construct the complete feature vector.
    gc = global_context or {}
    model_features_v2 = _call_first(
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
        # Extra context for 33-feature v2 model
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        vacuum_strength=vacuum_strength,
        hedging_flow_ratio=hedging_flow_ratio,
        smart_money_flow_score=smart_money_flow_score,
        atm_iv_percentile=atm_iv_percentile,
        intraday_range_pct=intraday_range_pct,
        lookback_avg_range_pct=lookback_avg_range_pct,
        gap_pct=(((day_open - prev_close) / prev_close * 100)
                 if day_open and prev_close and prev_close > 0 else 0.0),
        close_vs_prev_close_pct=(((spot - prev_close) / prev_close * 100)
                                  if spot and prev_close and prev_close > 0 else 0.0),
        spot_in_day_range=(((spot - day_low) / (day_high - day_low))
                           if day_high and day_low and day_high > day_low and spot else 0.5),
        dealer_position=market_state.get("dealer_pos"),
        vanna_regime=market_state.get("greek_exposures", {}).get("vanna_regime"),
        charm_regime=market_state.get("greek_exposures", {}).get("charm_regime"),
        confirmation_status=market_state.get("confirmation_status"),
        macro_event_risk_score=gc.get("macro_event_risk_score", 0.0),
        macro_regime=gc.get("macro_regime", "NO_EVENT"),
        india_vix_level=gc.get("india_vix_level"),
        india_vix_change_24h=gc.get("india_vix_change_24h"),
        oil_shock_score=gc.get("oil_shock_score"),
        commodity_risk_score=gc.get("commodity_risk_score"),
        volatility_shock_score=gc.get("volatility_shock_score"),
        days_to_expiry=gc.get("days_to_expiry"),
        default=None,
    )
    # Use v2 features if available, fall back to v1
    if model_features_v2 is not None:
        model_features = model_features_v2

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
    if _force_ml_only:
        rule_move_probability = None

    ml_move_probability = None
    if not _force_rule_only:
        predictor = _get_move_predictor()
        if predictor is not None:
            try:
                # The ML leg is optional by design. Any failure here should degrade
                # gracefully back to the deterministic rule-based estimate.
                if model_features is not None:
                    ml_move_probability = predictor.predict_probability(model_features)
                ml_move_probability = _extract_probability(ml_move_probability)
                if ml_move_probability is not None:
                    ml_move_probability = round(
                        _clip(float(ml_move_probability), cfg.probability_floor, cfg.probability_ceiling),
                        2,
                    )
            except Exception as exc:
                _warn_once(
                    "ml_predict_inference_failed",
                    "probability: ML inference failed; falling back to rule probability (%s)",
                    exc,
                )
                ml_move_probability = None

    components = {
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
    }

    return {
        "rule_move_probability": rule_move_probability,
        "ml_move_probability": ml_move_probability,
        "hybrid_move_probability": _blend_move_probability(
            rule_prob=rule_move_probability,
            ml_prob=ml_move_probability,
        ),
        "model_features": model_features,
        "components": components,
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
    global_context=None,
):
    """
    Public entry point for computing probability state.

    Routes through the pluggable predictor factory when a non-default
    prediction method is configured.  Falls back to the blended implementation
    when PREDICTION_METHOD is 'blended' (default) for zero-overhead backward
    compatibility.
    """
    # Fast path: if method is default "blended", call impl directly to avoid
    # any overhead from the factory/protocol layer.
    # Check factory override first (set by prediction_method_override context manager).
    from engine.predictors.factory import _METHOD_OVERRIDE
    if _METHOD_OVERRIDE is not None:
        method = _METHOD_OVERRIDE
    else:
        try:
            from config import settings as _settings
            method = getattr(_settings, "PREDICTION_METHOD", "blended") or "blended"
        except Exception:
            method = "blended"

    if method == "blended":
        return _compute_probability_state_impl(
            df,
            spot=spot,
            symbol=symbol,
            market_state=market_state,
            day_high=day_high,
            day_low=day_low,
            day_open=day_open,
            prev_close=prev_close,
            lookback_avg_range_pct=lookback_avg_range_pct,
            global_context=global_context,
        )

    # Non-default method: dispatch through the predictor factory.
    from engine.predictors.factory import get_predictor

    predictor = get_predictor()
    market_ctx = dict(
        df=df,
        spot=spot,
        symbol=symbol,
        market_state=market_state,
        day_high=day_high,
        day_low=day_low,
        day_open=day_open,
        prev_close=prev_close,
        lookback_avg_range_pct=lookback_avg_range_pct,
        global_context=global_context,
    )
    result = predictor.predict(market_ctx)

    return {
        "rule_move_probability": result.rule_move_probability,
        "ml_move_probability": result.ml_move_probability,
        "hybrid_move_probability": result.hybrid_move_probability,
        "model_features": result.model_features,
        "components": result.components,
    }
