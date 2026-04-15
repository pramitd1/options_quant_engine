"""
Module: evaluator.py

Purpose:
    Evaluate captured trade signals against realized spot paths and persist the resulting research dataset.

Role in the System:
    Part of the research layer that turns signal-engine outputs into scored signal-evaluation rows for reporting, tuning, and governance.

Key Outputs:
    Enriched signal-evaluation rows with realized outcomes, calibration buckets, and composite research scores.

Downstream Usage:
    Consumed by signal-evaluation reports, parameter-tuning workflows, and promotion reviews.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from config.market_data_policy import IST_TIMEZONE
from config.signal_evaluation_policy import (
    MOVE_PROBABILITY_BUCKETS,
    SIGNAL_EVALUATION_HORIZON_MINUTES,
    SIGNAL_EVALUATION_WINDOW_MINUTES,
    TRADE_STRENGTH_BUCKETS,
    bucket_from_thresholds,
)
from config.signal_evaluation_scoring import (
    get_signal_evaluation_direction_weights,
    get_signal_evaluation_score_weights,
    get_signal_evaluation_thresholds,
    get_signal_evaluation_timing_weights,
)
from config.settings import BASE_DIR
from data.spot_downloader import normalize_underlying_symbol
from research.signal_evaluation.dataset import SIGNAL_DATASET_PATH, load_signals_dataset, upsert_signal_rows
from research.signal_evaluation.market_data import (
    build_realized_spot_path_cache,
    coerce_market_timestamp,
    fetch_realized_spot_history,
    fetch_realized_spot_path,
    resolve_research_as_of,
)
from utils.numerics import safe_float as _safe_float  # noqa: F401
from utils.regime_normalization import normalize_iv_decimal


def _coerce_ts(value) -> pd.Timestamp:
    """
    Purpose:
        Coerce a timestamp-like input into a `pd.Timestamp`.
    
    Context:
        Internal helper in the signal-evaluation pipeline. The evaluator needs consistent timestamps before it can align captured signals with realized spot paths.
    
    Inputs:
        value (Any): Timestamp-like value coming from captured signals, replay data, or realized market history.
    
    Returns:
        pd.Timestamp: Normalized timestamp used by downstream evaluation logic.
    
    Notes:
        Keeping timestamp normalization in one place helps live capture and delayed backfills behave identically.
    """
    return coerce_market_timestamp(value)


def _signal_direction_multiplier(direction: str | None) -> int:
    """
    Purpose:
        Map direction labels such as `CALL` and `PUT` to signed multipliers.
    
    Context:
        Internal helper in the evaluation pipeline. Later return calculations are direction-aware, so the evaluator converts bullish and bearish signals into a common signed-return convention.
    
    Inputs:
        direction (str | None): Signal direction label, typically `CALL` or `PUT`.
    
    Returns:
        int: `1` for bullish direction, `-1` for bearish direction, and `0` when direction is unavailable.
    
    Notes:
        Using signed returns lets the evaluator score directional quality without modeling option payoff convexity or execution slippage.
    """
    normalized = str(direction or "").upper().strip()
    if normalized == "CALL":
        return 1
    if normalized == "PUT":
        return -1
    return 0


def _bucket_trade_strength(value) -> str | None:
    """
    Purpose:
        Bucket trade-strength scores into reporting ranges.
    
    Context:
        Internal helper used when writing evaluation rows. Research reporting groups signals by strength bucket so hit rates can be compared across different confidence levels.
    
    Inputs:
        value (Any): Trade-strength score to bucket.
    
    Returns:
        str | None: Bucket label for the supplied trade-strength score.
    
    Notes:
        The buckets are reporting-oriented and do not feed back into the live engine directly.
    """
    value = _safe_float(value, None)
    return bucket_from_thresholds(value, TRADE_STRENGTH_BUCKETS, "0_34")


def _bucket_probability(value) -> str | None:
    """
    Purpose:
        Bucket probability estimates into reporting ranges.
    
    Context:
        Internal helper used when writing evaluation rows. Probability buckets make calibration checks and report slices easier to interpret.
    
    Inputs:
        value (Any): Probability estimate to bucket.
    
    Returns:
        str | None: Bucket label for the supplied probability estimate.
    
    Notes:
        This is a research convenience helper rather than a live trading decision rule.
    """
    value = _safe_float(value, None)
    return bucket_from_thresholds(value, MOVE_PROBABILITY_BUCKETS, "0.00_0.34")


def build_signal_id(
    *,
    signal_timestamp,
    source,
    mode,
    symbol,
    selected_expiry,
    direction,
    strike,
    option_type,
) -> str:
    """
    Purpose:
        Build a stable identifier for a captured signal snapshot.
    
    Context:
        The signal-evaluation dataset needs a deterministic key so repeated saves can upsert the same signal instead of creating duplicates. The identifier is derived from timestamp, symbol, contract metadata, and direction.
    
    Inputs:
        signal_timestamp (Any): Timestamp recorded for the captured signal.
        source (Any): Data-source label associated with the snapshot.
        mode (Any): Execution mode label such as live, replay, or backtest.
        symbol (Any): Underlying symbol or index identifier.
        selected_expiry (Any): Expiry associated with the signaled contract.
        direction (Any): Signal direction label, typically `CALL` or `PUT`.
        strike (Any): Strike price associated with the signaled contract.
        option_type (Any): Option side associated with the contract, typically `CE` or `PE`.
    
    Returns:
        str: Stable short hash used as the primary identifier in the signal-evaluation dataset.
    
    Notes:
        The identifier tracks the signal setup, not the realized outcome, so it remains stable across later backfills of market data.
    """
    parts = [
        str(_coerce_ts(signal_timestamp).isoformat()),
        str(source or "").upper().strip(),
        str(mode or "").upper().strip(),
        normalize_underlying_symbol(symbol),
        str(selected_expiry or "").strip(),
        str(direction or "").upper().strip(),
        str(option_type or "").upper().strip(),
        str(strike or "").strip(),
    ]
    raw_key = "|".join(parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]


def build_regime_fingerprint(trade: dict, provider_health: dict | None = None) -> tuple[str, str]:
    """
    Purpose:
        Build regime-fingerprint keys used to group similar signal environments.
    
    Context:
        Research reporting often cares less about a single signal and more about recurring market environments. This helper collapses the signal's regime labels into a deterministic fingerprint for later aggregation.
    
    Inputs:
        trade (dict): Trade or no-trade payload produced by the signal engine.
        provider_health (dict | None): Optional provider-health diagnostics to include in the fingerprint.
    
    Returns:
        tuple[str, str]: Full fingerprint string and a shorter hashed identifier suitable for grouping or display.
    
    Notes:
        The fingerprint is descriptive rather than predictive; it is meant for slicing evaluation results by environment, not for making live trading decisions.
    """
    provider_health = provider_health or {}
    components = {
        "signal_regime": trade.get("signal_regime") or "UNKNOWN",
        "macro_regime": trade.get("macro_regime") or "UNKNOWN",
        "gamma_regime": trade.get("gamma_regime") or "UNKNOWN",
        "spot_vs_flip": trade.get("spot_vs_flip") or "UNKNOWN",
        "flow": trade.get("final_flow_signal") or "UNKNOWN",
        "dealer_pos": trade.get("dealer_position") or "UNKNOWN",
        "hedging": trade.get("dealer_hedging_bias") or "UNKNOWN",
        "vol": trade.get("volatility_regime") or "UNKNOWN",
        "vacuum": trade.get("liquidity_vacuum_state") or "UNKNOWN",
        "confirm": trade.get("confirmation_status") or "UNKNOWN",
        "dataq": trade.get("data_quality_status") or "UNKNOWN",
        "provider": provider_health.get("summary_status") or "UNKNOWN",
    }
    fingerprint = "|".join(f"{key}={value}" for key, value in components.items())
    fingerprint_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return fingerprint, fingerprint_id


def build_signal_evaluation_row(
    result: dict,
    *,
    notes: str | None = None,
    captured_at=None,
) -> dict:
    """
    Purpose:
        Convert one engine result payload into the canonical signal-evaluation dataset row.

    Context:
        This is the capture-side entry point for the research pipeline. It lifts the trade payload, regime diagnostics, provider-health fields, and calibration metadata into a flat row that can later be backfilled with realized outcomes.

    Inputs:
        result (dict): Engine result payload containing the captured trade and market snapshot metadata.
        notes (str | None): Optional research note stored alongside the signal row.
        captured_at (Any): Timestamp used for `created_at` and `updated_at` when persisting the row.

    Returns:
        dict: Serializable signal-evaluation row with pending outcome fields initialized.

    Notes:
        The row schema intentionally mirrors the live signal contract closely so researchers can connect ex-ante signal diagnostics with ex-post realized performance.
    """
    if not isinstance(result, dict):
        raise ValueError("Result payload must be a dictionary")

    trade_obj = result.get("trade")
    if not isinstance(trade_obj, dict) or not trade_obj:
        raise ValueError("Result payload must include a trade object")

    trade = trade_obj
    spot_summary_obj = result.get("spot_summary")
    spot_summary = spot_summary_obj if isinstance(spot_summary_obj, dict) else {}
    signal_timestamp = spot_summary.get("timestamp") or trade.get("valuation_time")
    if signal_timestamp is None:
        raise ValueError("Signal row requires a stable signal timestamp")

    ranked_strikes_obj = result.get("ranked_strikes")
    ranked_strikes = ranked_strikes_obj if isinstance(ranked_strikes_obj, list) else []
    option_chain_validation_obj = result.get("option_chain_validation")
    option_chain_validation = option_chain_validation_obj if isinstance(option_chain_validation_obj, dict) else {}

    def _normalize_option_type(value):
        token = str(value or "").upper().strip()
        aliases = {
            "CE": "CE",
            "CALL": "CE",
            "C": "CE",
            "PE": "PE",
            "PUT": "PE",
            "P": "PE",
        }
        return aliases.get(token)

    inferred_option_type = _normalize_option_type(trade.get("option_type"))
    if inferred_option_type is None:
        inferred_option_type = _normalize_option_type(trade.get("direction"))

    inferred_selected_expiry = trade.get("selected_expiry") or option_chain_validation.get("selected_expiry")
    inferred_strike = trade.get("strike")
    if inferred_strike in (None, "") and isinstance(ranked_strikes, list) and ranked_strikes:
        preferred_candidates = ranked_strikes
        if inferred_option_type is not None:
            option_filtered = [
                candidate for candidate in ranked_strikes
                if _normalize_option_type((candidate or {}).get("option_type")) == inferred_option_type
            ]
            if option_filtered:
                preferred_candidates = option_filtered
        if inferred_selected_expiry not in (None, ""):
            expiry_filtered = [
                candidate for candidate in preferred_candidates
                if str((candidate or {}).get("selected_expiry") or (candidate or {}).get("expiry") or "").strip() == str(inferred_selected_expiry).strip()
            ]
            if expiry_filtered:
                preferred_candidates = expiry_filtered
        top_candidate = preferred_candidates[0] if preferred_candidates else None
        if isinstance(top_candidate, dict):
            inferred_strike = top_candidate.get("strike")
            if inferred_option_type is None:
                inferred_option_type = _normalize_option_type(top_candidate.get("option_type"))
            if inferred_selected_expiry in (None, ""):
                inferred_selected_expiry = top_candidate.get("selected_expiry") or top_candidate.get("expiry")

    signal_id = build_signal_id(
        signal_timestamp=signal_timestamp,
        source=result.get("source"),
        mode=result.get("mode"),
        symbol=result.get("symbol"),
        selected_expiry=inferred_selected_expiry,
        direction=trade.get("direction"),
        strike=inferred_strike,
        option_type=inferred_option_type,
    )

    provider_health_obj = trade.get("provider_health")
    if not isinstance(provider_health_obj, dict):
        provider_health_obj = option_chain_validation.get("provider_health")
    provider_health = provider_health_obj if isinstance(provider_health_obj, dict) else {}
    regime_fingerprint, regime_fingerprint_id = build_regime_fingerprint(trade, provider_health)
    saved_paths = result.get("saved_paths") or {}
    captured_ts = resolve_research_as_of(captured_at, default=signal_timestamp).isoformat()

    # Extract probability sub-components for ML feature extraction
    prob_components = trade.get("move_probability_components") or {}

    # Compute weekday from signal timestamp (0=Mon .. 4=Fri)
    sig_dt = _coerce_ts(signal_timestamp)
    weekday = sig_dt.weekday() if sig_dt else None

    # Compute atm_iv_scaled from raw atm_iv if available
    raw_atm_iv = trade.get("atm_iv")
    atm_iv_scaled = normalize_iv_decimal(raw_atm_iv, default=None)
    entry_price = _safe_float(trade.get("entry_price"), None)
    target_price = _safe_float(trade.get("target"), None)
    stop_loss_price = _safe_float(trade.get("stop_loss"), None)

    target_premium_return_pct = None
    if entry_price not in (None, 0.0) and target_price is not None:
        target_premium_return_pct = round(((target_price - entry_price) / entry_price) * 100.0, 4)

    stop_loss_premium_return_pct = None
    if entry_price not in (None, 0.0) and stop_loss_price is not None:
        stop_loss_premium_return_pct = round(((stop_loss_price - entry_price) / entry_price) * 100.0, 4)

    row = {
        "signal_id": signal_id,
        "signal_timestamp": _coerce_ts(signal_timestamp).isoformat(),
        "source": str(result.get("source") or "").upper().strip(),
        "mode": str(result.get("mode") or "").upper().strip(),
        "symbol": normalize_underlying_symbol(result.get("symbol")),
        "ticker": spot_summary.get("ticker") or result.get("spot_snapshot", {}).get("ticker"),
        "selected_expiry": inferred_selected_expiry,
        "direction": trade.get("direction"),
        "option_type": inferred_option_type,
        "strike": inferred_strike,
        "entry_price": trade.get("entry_price"),
        "target_premium_return_pct": target_premium_return_pct,
        "stop_loss_premium_return_pct": stop_loss_premium_return_pct,
        "selected_option_last_price": trade.get("selected_option_last_price", trade.get("entry_price")),
        "selected_option_volume": trade.get("selected_option_volume"),
        "selected_option_open_interest": trade.get("selected_option_open_interest"),
        "selected_option_iv": trade.get("selected_option_iv"),
        "selected_option_iv_is_proxy": trade.get("selected_option_iv_is_proxy"),
        "selected_option_iv_proxy_source": trade.get("selected_option_iv_proxy_source"),
        "selected_option_delta": trade.get("selected_option_delta"),
        "selected_option_delta_is_proxy": trade.get("selected_option_delta_is_proxy"),
        "selected_option_delta_proxy_source": trade.get("selected_option_delta_proxy_source"),
        "selected_option_gamma": trade.get("selected_option_gamma"),
        "selected_option_theta": trade.get("selected_option_theta"),
        "selected_option_vega": trade.get("selected_option_vega"),
        "selected_option_vanna": trade.get("selected_option_vanna"),
        "selected_option_charm": trade.get("selected_option_charm"),
        "selected_option_capital_per_lot": trade.get("selected_option_capital_per_lot"),
        "selected_option_ba_spread_ratio": trade.get("selected_option_ba_spread_ratio"),
        "selected_option_ba_spread_pct": trade.get("selected_option_ba_spread_pct"),
        "selected_option_score": trade.get("selected_option_score"),
        "target": trade.get("target"),
        "stop_loss": trade.get("stop_loss"),
        "recommended_hold_minutes": trade.get("recommended_hold_minutes"),
        "max_hold_minutes": trade.get("max_hold_minutes"),
        "exit_urgency": trade.get("exit_urgency"),
        "spot_at_signal": spot_summary.get("spot"),
        "day_open": spot_summary.get("day_open"),
        "day_high": spot_summary.get("day_high"),
        "day_low": spot_summary.get("day_low"),
        "prev_close": spot_summary.get("prev_close"),
        "lookback_avg_range_pct": spot_summary.get("lookback_avg_range_pct"),
        "trade_strength": trade.get("trade_strength"),
        "signal_quality": trade.get("signal_quality"),
        "signal_regime": trade.get("signal_regime"),
        "execution_regime": trade.get("execution_regime"),
        "regime_fingerprint": regime_fingerprint,
        "regime_fingerprint_id": regime_fingerprint_id,
        "trade_status": trade.get("trade_status"),
        "direction_source": trade.get("direction_source"),
        "final_flow_signal": trade.get("final_flow_signal"),
        "gamma_regime": trade.get("gamma_regime"),
        "spot_vs_flip": trade.get("spot_vs_flip"),
        "macro_regime": trade.get("macro_regime"),
        "event_intelligence_enabled": trade.get("event_intelligence_enabled"),
        "event_bullish_score": trade.get("event_bullish_score"),
        "event_bearish_score": trade.get("event_bearish_score"),
        "event_vol_expansion_score": trade.get("event_vol_expansion_score"),
        "event_vol_compression_score": trade.get("event_vol_compression_score"),
        "event_uncertainty_score": trade.get("event_uncertainty_score"),
        "event_gap_risk_score": trade.get("event_gap_risk_score"),
        "event_catalyst_alignment_score": trade.get("event_catalyst_alignment_score"),
        "event_contradictory_penalty": trade.get("event_contradictory_penalty"),
        "event_cluster_score": trade.get("event_cluster_score"),
        "event_decayed_signal": trade.get("event_decayed_signal"),
        "event_relevance_score": trade.get("event_relevance_score"),
        "event_count": trade.get("event_count"),
        "event_routed_count": trade.get("event_routed_count"),
        "event_overlay_probability_multiplier": trade.get("event_overlay_probability_multiplier"),
        "event_overlay_size_multiplier": trade.get("event_overlay_size_multiplier"),
        "event_overlay_score_adjustment": trade.get("event_overlay_score_adjustment"),
        "event_overlay_suppress_signal": trade.get("event_overlay_suppress_signal"),
        "event_overlay_reasons": "|".join(str(item) for item in (trade.get("event_overlay_reasons") or [])),
        "event_explanations": "|".join(str(item) for item in (trade.get("event_explanations") or [])),
        "global_risk_state": trade.get("global_risk_state"),
        "global_risk_score": trade.get("global_risk_score"),
        "oil_shock_score": trade.get("oil_shock_score"),
        "commodity_risk_score": trade.get("commodity_risk_score"),
        "volatility_shock_score": trade.get("market_volatility_shock_score"),
        "macro_news_volatility_shock_score": trade.get("macro_news_volatility_shock_score"),
        "volatility_explosion_probability": trade.get("volatility_explosion_probability"),
        "overnight_gap_risk_score": trade.get("overnight_gap_risk_score"),
        "volatility_expansion_risk_score": trade.get("volatility_expansion_risk_score"),
        "overnight_hold_allowed": trade.get("overnight_hold_allowed"),
        "overnight_hold_reason": trade.get("overnight_hold_reason"),
        "overnight_risk_penalty": trade.get("overnight_risk_penalty"),
        "global_risk_adjustment_score": trade.get("global_risk_adjustment_score"),
        "gamma_vol_acceleration_score": trade.get("gamma_vol_acceleration_score"),
        "squeeze_risk_state": trade.get("squeeze_risk_state"),
        "directional_convexity_state": trade.get("directional_convexity_state"),
        "upside_squeeze_risk": trade.get("upside_squeeze_risk"),
        "downside_airpocket_risk": trade.get("downside_airpocket_risk"),
        "overnight_convexity_risk": trade.get("overnight_convexity_risk"),
        "gamma_vol_adjustment_score": trade.get("gamma_vol_adjustment_score"),
        "dealer_hedging_pressure_score": trade.get("dealer_hedging_pressure_score"),
        "dealer_flow_state": trade.get("dealer_flow_state"),
        "upside_hedging_pressure": trade.get("upside_hedging_pressure"),
        "downside_hedging_pressure": trade.get("downside_hedging_pressure"),
        "pinning_pressure_score": trade.get("pinning_pressure_score"),
        "dealer_pressure_adjustment_score": trade.get("dealer_pressure_adjustment_score"),
        "expected_move_points": trade.get("expected_move_points"),
        "expected_move_pct": trade.get("expected_move_pct"),
        "target_reachability_score": trade.get("target_reachability_score"),
        "premium_efficiency_score": trade.get("premium_efficiency_score"),
        "strike_efficiency_score": trade.get("strike_efficiency_score"),
        "option_efficiency_score": trade.get("option_efficiency_score"),
        "option_efficiency_adjustment_score": trade.get("option_efficiency_adjustment_score"),
        "consistency_check_status": trade.get("consistency_check_status"),
        "consistency_check_issue_count": trade.get("consistency_check_issue_count"),
        "consistency_check_critical_issue_count": trade.get("consistency_check_critical_issue_count"),
        "consistency_check_escalated": trade.get("consistency_check_escalated"),
        "consistency_check_findings": json.dumps(trade.get("consistency_check_findings") or [], sort_keys=True),
        "dealer_position": trade.get("dealer_position"),
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
        "dealer_hedging_flow": trade.get("dealer_hedging_flow"),
        "market_delta_exposure": trade.get("delta_exposure"),
        "market_gamma_exposure": trade.get("gamma_exposure_greeks"),
        "market_theta_exposure": trade.get("theta_exposure"),
        "market_vega_exposure": trade.get("vega_exposure"),
        "market_vanna_exposure": trade.get("vanna_exposure"),
        "market_charm_exposure": trade.get("charm_exposure"),
        "volatility_regime": trade.get("volatility_regime"),
        "liquidity_vacuum_state": trade.get("liquidity_vacuum_state"),
        "confirmation_status": trade.get("confirmation_status"),
        "macro_event_risk_score": trade.get("macro_event_risk_score"),
        "data_quality_score": trade.get("data_quality_score"),
        "data_quality_status": trade.get("data_quality_status"),
        "provider_health_status": provider_health.get("summary_status"),
        "provider_health_row": provider_health.get("row_health"),
        "provider_health_pricing": provider_health.get("pricing_health"),
        "provider_health_pairing": provider_health.get("pairing_health"),
        "provider_health_iv": provider_health.get("iv_health"),
        "provider_health_duplicate": provider_health.get("duplicate_health"),
        "move_probability": trade.get("hybrid_move_probability"),
        "rule_move_probability": trade.get("rule_move_probability"),
        "hybrid_move_probability": trade.get("hybrid_move_probability"),
        "ml_move_probability": trade.get("ml_move_probability"),
        "large_move_probability": trade.get("large_move_probability"),
        "saved_spot_snapshot_path": saved_paths.get("spot"),
        "saved_chain_snapshot_path": saved_paths.get("chain"),

        # ML expanded features — probability sub-components
        "gamma_flip_distance_pct": prob_components.get("gamma_flip_distance_pct"),
        "vacuum_strength": prob_components.get("vacuum_strength"),
        "hedging_flow_ratio": prob_components.get("hedging_flow_ratio"),
        "smart_money_flow_score": prob_components.get("smart_money_flow_score"),
        "atm_iv_percentile": prob_components.get("atm_iv_percentile"),

        # ML expanded features — greek regimes
        "vanna_regime": trade.get("vanna_regime"),
        "charm_regime": trade.get("charm_regime"),

        # ML expanded features — global market
        "india_vix_level": trade.get("india_vix_level"),
        "india_vix_change_24h": trade.get("india_vix_change_24h"),

        # ML expanded features — derived
        "atm_iv_scaled": atm_iv_scaled,
        "weekday": weekday,

        "created_at": captured_ts,
        "updated_at": captured_ts,
        "outcome_last_updated_at": pd.NA,
        "outcome_status": "PENDING",
        "observed_minutes": 0.0,
        "evaluation_window_minutes": SIGNAL_EVALUATION_WINDOW_MINUTES,
        "directional_consistency_score": pd.NA,
        "signal_calibration_bucket": _bucket_trade_strength(trade.get("trade_strength")),
        "probability_calibration_bucket": _bucket_probability(trade.get("hybrid_move_probability")),
        "notes": notes,
    }

    # ── ML Research Layer (observational only) ──────────────────────
    # Run dual-model inference if ML research is enabled.
    # This NEVER affects trade decisions — purely for research logging.
    try:
        from research.ml_models.ml_config import ML_RESEARCH_ENABLED
        if ML_RESEARCH_ENABLED:
            from research.ml_models.ml_inference import infer_single
            ml_result = infer_single(row)
            row["ml_rank_score"] = ml_result.ml_rank_score
            row["ml_confidence_score"] = ml_result.ml_confidence_score
            row["ml_rank_bucket"] = ml_result.ml_rank_bucket
            row["ml_confidence_bucket"] = ml_result.ml_confidence_bucket
            row["ml_agreement_with_engine"] = ml_result.ml_agreement_with_engine
    except Exception:
        pass  # ML failure must never break signal capture

    for horizon in SIGNAL_EVALUATION_HORIZON_MINUTES:
        row[f"spot_{horizon}m"] = pd.NA
        row[f"signed_return_{horizon}m_bps"] = pd.NA
        row[f"correct_{horizon}m"] = pd.NA

    row["spot_session_close"] = pd.NA
    row["signed_return_session_close_bps"] = pd.NA
    row["correct_session_close"] = pd.NA
    row["spot_1d"] = pd.NA
    row["spot_2d"] = pd.NA
    row["spot_3d"] = pd.NA
    row["spot_5d"] = pd.NA
    row["spot_at_expiry"] = pd.NA
    row["return_1d_bps"] = pd.NA
    row["return_2d_bps"] = pd.NA
    row["return_3d_bps"] = pd.NA
    row["return_5d_bps"] = pd.NA
    row["return_at_expiry_bps"] = pd.NA
    row["correct_1d"] = pd.NA
    row["correct_2d"] = pd.NA
    row["correct_3d"] = pd.NA
    row["correct_5d"] = pd.NA
    row["correct_at_expiry"] = pd.NA
    row["mfe_60m_bps"] = pd.NA
    row["mae_60m_bps"] = pd.NA
    row["mfe_120m_bps"] = pd.NA
    row["mae_120m_bps"] = pd.NA
    row["realized_range_60m_bps"] = pd.NA
    row["realized_range_120m_bps"] = pd.NA
    row["eod_mfe_bps"] = pd.NA
    row["eod_mae_bps"] = pd.NA
    row["target_hit"] = False
    row["target_hit_date"] = pd.NA
    row["stop_loss_hit"] = False
    row["stop_loss_hit_date"] = pd.NA
    row["target_stop_same_bar_ambiguous"] = False
    row["target_sl_delta_used"] = pd.NA
    row["target_sl_delta_source"] = pd.NA
    row["best_outcome_horizon"] = pd.NA
    row["best_outcome_bps"] = pd.NA
    row["peak_to_close_decay_bps"] = pd.NA
    row["exit_efficiency_score"] = pd.NA
    row["horizon_edge_label"] = pd.NA
    row["tradeability_tier"] = pd.NA
    row["exit_quality_label"] = pd.NA
    return row


def _nearest_spot_at_or_after(path: pd.DataFrame, target_ts: pd.Timestamp):
    """
    Purpose:
        Find the first realized spot observation at or after a target timestamp.
    
    Context:
        Internal helper in the outcome-evaluation path. Captured signals are evaluated on discrete realized spot samples, so the evaluator needs a consistent rule for mapping horizons to observed prices.
    
    Inputs:
        path (pd.DataFrame): Realized spot path sorted by timestamp.
        target_ts (pd.Timestamp): Target timestamp for the requested evaluation horizon.
    
    Returns:
        float | None: Spot value at the first observation on or after `target_ts`, or `None` when the path has not reached that horizon.
    
    Notes:
        Using the first observation at or after the target avoids forward-filling prices into horizons the market has not actually reached yet.
    """
    candidates = path[path["timestamp"] >= target_ts]
    if candidates.empty:
        return None
    return _safe_float(candidates.iloc[0]["spot"], None)


def _window_stats(path: pd.DataFrame, entry_ts: pd.Timestamp, end_ts: pd.Timestamp, direction_mult: int, entry_spot: float):
    """
    Purpose:
        Compute MFE, MAE, and realized range statistics over a fixed evaluation window.
    
    Context:
        Internal helper used for signal evaluation research. It summarizes how favorable, adverse, and volatile the realized spot path was after the signal fired.
    
    Inputs:
        path (pd.DataFrame): Realized spot path sorted by timestamp.
        entry_ts (pd.Timestamp): Timestamp at which the signal is considered active.
        end_ts (pd.Timestamp): End of the evaluation window.
        direction_mult (int): Signed direction multiplier used to measure movement from the trade's perspective.
        entry_spot (float): Underlying spot price at signal entry.
    
    Returns:
        dict | None: Window statistics containing MFE, MAE, and realized range in basis points, or `None` when no observations fall inside the window.
    
    Notes:
        MFE and MAE are direction-signed, so positive values always mean movement favorable to the original signal thesis.
    """
    window = path[(path["timestamp"] >= entry_ts) & (path["timestamp"] <= end_ts)].copy()
    if window.empty:
        return None

    # Guard: validate entry_spot is numeric and positive
    entry_spot = _safe_float(entry_spot, None)
    if entry_spot is None or entry_spot <= 0:
        # Cannot compute window stats without valid entry spot
        return None
    
    if direction_mult == 0:
        # No direction; cannot compute signed returns
        return None

    signed_moves_bps = ((window["spot"].astype(float) - entry_spot) / entry_spot) * 10000.0 * direction_mult
    raw_moves_bps = ((window["spot"].astype(float) - entry_spot) / entry_spot) * 10000.0

    return {
        "mfe_bps": round(float(signed_moves_bps.max()), 2),
        "mae_bps": round(float(signed_moves_bps.min()), 2),
        "range_bps": round(float(raw_moves_bps.max() - raw_moves_bps.min()), 2),
    }


def _session_close_spot(path: pd.DataFrame, signal_ts: pd.Timestamp):
    """
    Purpose:
        Return the realized spot observed at the same-day session close.
    
    Context:
        Internal helper in the evaluation pipeline. Same-day close is a useful checkpoint because many signals are intraday and may not be intended for overnight holding.
    
    Inputs:
        path (pd.DataFrame): Realized spot path sorted by timestamp.
        signal_ts (pd.Timestamp): Timestamp when the signal was captured.
    
    Returns:
        float | None: Same-day closing spot, or `None` when the path does not contain that session.
    
    Notes:
        This keeps intraday evaluation separate from next-day gap behavior, which is tracked through separate checkpoints.
    """
    same_day = path[path["timestamp"].dt.date == signal_ts.date()]
    if same_day.empty:
        return None
    return _safe_float(same_day.iloc[-1]["spot"], None)


def _next_day_rows(path: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Purpose:
        Slice the realized spot path down to the first trading day after the signal date.
    
    Context:
        Internal helper used by the overnight evaluation path. Next-day open and next-day close metrics should use a single coherent post-signal session.
    
    Inputs:
        path (pd.DataFrame): Realized spot path sorted by timestamp.
        signal_ts (pd.Timestamp): Timestamp when the signal was captured.
    
    Returns:
        pd.DataFrame: Realized spot rows belonging to the first day after the signal date.
    
    Notes:
        Keeping the next-day slice explicit avoids mixing later sessions into overnight gap or follow-through metrics.
    """
    future_days = path[path["timestamp"].dt.date > signal_ts.date()].copy()
    if future_days.empty:
        return future_days
    first_next_day = future_days["timestamp"].dt.date.min()
    return future_days[future_days["timestamp"].dt.date == first_next_day].copy()


def _next_day_open_spot(path: pd.DataFrame, signal_ts: pd.Timestamp):
    """
    Purpose:
        Return the first realized spot observed on the next trading day.
    
    Context:
        Internal helper used to measure overnight gap behavior after a captured signal.
    
    Inputs:
        path (pd.DataFrame): Realized spot path sorted by timestamp.
        signal_ts (pd.Timestamp): Timestamp when the signal was captured.
    
    Returns:
        float | None: Next-day opening spot, or `None` when no later session is available.
    
    Notes:
        This isolates overnight gap behavior from intraday continuation within the same session.
    """
    next_day = _next_day_rows(path, signal_ts)
    if next_day.empty:
        return None
    return _safe_float(next_day.iloc[0]["spot"], None)


def _next_day_close_spot(path: pd.DataFrame, signal_ts: pd.Timestamp):
    """
    Purpose:
        Return the final realized spot observed on the next trading day.
    
    Context:
        Internal helper used to measure whether an overnight move persisted through the following session.
    
    Inputs:
        path (pd.DataFrame): Realized spot path sorted by timestamp.
        signal_ts (pd.Timestamp): Timestamp when the signal was captured.
    
    Returns:
        float | None: Next-day closing spot, or `None` when no later session is available.
    
    Notes:
        Pairing next-day open and next-day close helps separate pure gap behavior from next-session drift.
    """
    next_day = _next_day_rows(path, signal_ts)
    if next_day.empty:
        return None
    return _safe_float(next_day.iloc[-1]["spot"], None)


def _raw_return(entry_spot: float, horizon_spot: float):
    """
    Purpose:
        Compute the raw underlying return between entry and a later horizon.
    
    Context:
        Internal helper in the research evaluator. The signal-evaluation dataset measures realized spot movement rather than option PnL, so returns are expressed on the underlying.
    
    Inputs:
        entry_spot (float): Underlying spot at signal entry.
        horizon_spot (float): Underlying spot at the later evaluation horizon.
    
    Returns:
        float | None: Raw percentage return on the underlying, or `None` when inputs are incomplete.
    
    Notes:
        Using spot returns keeps the evaluation focused on directional quality instead of execution details such as spread, slippage, or volatility decay.
    """
    if entry_spot in (None, 0) or horizon_spot is None:
        return None
    return round(float((horizon_spot - entry_spot) / entry_spot), 6)


def _clip_score(value: float) -> float:
    """
    Purpose:
        Clip a research score into the canonical 0-100 range used by evaluation reporting.
    
    Context:
        Internal helper in the evaluation pipeline. Several component scores are combined later, so the evaluator normalizes each of them into the same bounded scale.
    
    Inputs:
        value (float): Score candidate to clamp into the reporting range.
    
    Returns:
        float: Score clipped into the inclusive 0-100 range.
    
    Notes:
        This keeps component scores comparable even when they originate from different heuristics or return horizons.
    """
    return round(max(0.0, min(100.0, float(value))), 2)


def _classify_tradeability_tier(score) -> str | pd.NA:
    score = _safe_float(score, None)
    if score is None:
        return pd.NA
    if score >= 80.0:
        return "HIGH"
    if score >= 65.0:
        return "USABLE"
    if score >= 45.0:
        return "FRAGILE"
    return "POOR"


def _resolve_best_outcome_profile(updated: dict) -> tuple[object, object, object, str]:
    candidate_fields = [
        ("5m", "signed_return_5m_bps"),
        ("15m", "signed_return_15m_bps"),
        ("30m", "signed_return_30m_bps"),
        ("60m", "signed_return_60m_bps"),
        ("120m", "signed_return_120m_bps"),
        ("session_close", "signed_return_session_close_bps"),
        ("1d", "return_1d_bps"),
        ("2d", "return_2d_bps"),
        ("3d", "return_3d_bps"),
        ("5d", "return_5d_bps"),
        ("expiry", "return_at_expiry_bps"),
    ]
    valid = []
    for label, field_name in candidate_fields:
        value = _safe_float(updated.get(field_name), None)
        if value is None:
            continue
        valid.append((label, float(value)))

    if not valid:
        return pd.NA, pd.NA, pd.NA, "PENDING"

    best_label, best_value = max(valid, key=lambda item: item[1])
    close_value = _safe_float(updated.get("signed_return_session_close_bps"), None)
    decay = None if close_value is None else round(float(close_value - best_value), 2)

    if best_value <= 0:
        profile = "NO_EDGE"
    elif best_label in {"5m", "15m"}:
        if close_value is not None and close_value < 0:
            profile = "EARLY_ALPHA_DECAY"
        elif close_value is not None and close_value >= 0.60 * best_value:
            profile = "FAST_FOLLOWTHROUGH"
        else:
            profile = "FAST_SCALP"
    elif best_label in {"30m", "60m", "120m", "session_close"}:
        if close_value is not None and close_value < 0:
            profile = "LATE_REVERSAL"
        else:
            profile = "INTRADAY_TREND"
    else:
        profile = "SWING_FOLLOWTHROUGH"

    return best_label, round(float(best_value), 2), decay, profile


def _resolve_exit_quality(updated: dict) -> tuple[object, object]:
    if bool(updated.get("target_stop_same_bar_ambiguous", False)):
        return "AMBIGUOUS", 15.0

    target_hit = bool(updated.get("target_hit", False))
    stop_loss_hit = bool(updated.get("stop_loss_hit", False))
    best_value = _safe_float(updated.get("best_outcome_bps"), None)
    close_value = _safe_float(updated.get("signed_return_session_close_bps"), None)

    if stop_loss_hit and not target_hit:
        return "STOPPED_OUT", 0.0
    if target_hit and not stop_loss_hit:
        if close_value is not None and close_value > 0:
            return "TARGET_HIT", 100.0
        return "EARLY_EXIT", 82.0
    if best_value is None:
        return pd.NA, pd.NA
    if best_value <= 0:
        return "NO_EDGE", 0.0
    if close_value is None:
        return "PENDING", 50.0

    ratio = float(close_value) / max(float(best_value), 1e-6)
    clipped_ratio = max(-1.0, min(1.0, ratio))
    exit_efficiency = _clip_score(50.0 + 50.0 * clipped_ratio)

    if close_value < 0 < best_value:
        label = "EARLY_EXIT"
    elif ratio >= 0.80:
        label = "HOLD_WINNER"
    elif ratio >= 0.40:
        label = "USABLE_EXIT"
    else:
        label = "MISSED_EXIT"

    return label, exit_efficiency


def compute_signal_evaluation_scores(row: dict) -> dict:
    """
    Purpose:
        Compute composite research scores for one evaluated signal row.
    
    Context:
        This function sits after realized-outcome calculation in the research pipeline. It converts directional correctness, move magnitude, timing, and tradeability into a comparable composite score used by reporting and governance.
    
    Inputs:
        row (dict): Signal-evaluation row containing realized outcomes and captured signal metadata.
    
    Returns:
        dict: Updated evaluation row with component scores and composite score fields added.
    
    Notes:
        These scores are research artifacts. They do not feed back into the live trading engine directly.
    """
    updated = dict(row)
    has_direction = _signal_direction_multiplier(updated.get("direction")) != 0
    direction_weights = get_signal_evaluation_direction_weights()
    thresholds = get_signal_evaluation_thresholds()
    timing_weights = get_signal_evaluation_timing_weights()
    score_weights = get_signal_evaluation_score_weights()

    direction_numerator = 0.0
    direction_denominator = 0.0
    if has_direction:
        for field_name, weight in direction_weights.items():
            value = updated.get(field_name)
            if pd.isna(value):
                continue
            direction_numerator += float(value) * float(weight)
            direction_denominator += float(weight)

    direction_score = None
    if direction_denominator > 0:
        direction_score = _clip_score((direction_numerator / direction_denominator) * 100.0)

    lookback_avg_range_pct = _safe_float(updated.get("lookback_avg_range_pct"), None)
    mfe_points = _safe_float(updated.get("mfe_points"), None)
    spot_at_signal = _safe_float(updated.get("spot_at_signal"), None)
    magnitude_score = None
    if has_direction and mfe_points is not None and spot_at_signal not in (None, 0):
        favorable_move_pct = abs(mfe_points) / spot_at_signal * 100.0
        baseline_range_pct = lookback_avg_range_pct if lookback_avg_range_pct not in (None, 0) else 1.0
        # Scale the best realized move by a recent "normal" session range so
        # magnitude is comparable across quiet and volatile regimes.
        magnitude_vs_range = favorable_move_pct / max(baseline_range_pct, 0.1)

        weak = thresholds["magnitude_vs_range_weak"]
        good = thresholds["magnitude_vs_range_good"]
        strong = thresholds["magnitude_vs_range_strong"]

        if magnitude_vs_range <= weak:
            magnitude_score = _clip_score((magnitude_vs_range / max(weak, 1e-6)) * 35.0)
        elif magnitude_vs_range <= good:
            span = max(good - weak, 1e-6)
            magnitude_score = _clip_score(35.0 + ((magnitude_vs_range - weak) / span) * 30.0)
        elif magnitude_vs_range <= strong:
            span = max(strong - good, 1e-6)
            magnitude_score = _clip_score(65.0 + ((magnitude_vs_range - good) / span) * 25.0)
        else:
            magnitude_score = 100.0

    timing_numerator = 0.0
    timing_denominator = 0.0
    return_floor = thresholds["timing_positive_return_floor"]
    if has_direction:
        for field_name, weight in timing_weights.items():
            value = _safe_float(updated.get(field_name), None)
            if value is None:
                continue
            # Early favorable returns get more credit because fast follow-through
            # is more actionable than the same move arriving very late.
            horizon_score = max(0.0, min(1.0, value / max(return_floor, 1e-6)))
            timing_numerator += horizon_score * float(weight)
            timing_denominator += float(weight)

    timing_score = None
    if timing_denominator > 0:
        timing_score = _clip_score((timing_numerator / timing_denominator) * 100.0)

    tradeability_score = None
    mae_points = _safe_float(updated.get("mae_points"), None)
    if has_direction and mfe_points is not None and mae_points is not None:
        adverse_points = abs(min(mae_points, 0.0))
        if adverse_points == 0:
            tradeability_ratio = float("inf")
        else:
            # This is a simple path-efficiency proxy: how much favorable move the
            # signal earned for each unit of adverse excursion.
            tradeability_ratio = abs(mfe_points) / adverse_points

        floor = thresholds["tradeability_ratio_floor"]
        good = thresholds["tradeability_ratio_good"]
        strong = thresholds["tradeability_ratio_strong"]

        if tradeability_ratio == float("inf"):
            tradeability_score = 100.0
        elif tradeability_ratio <= floor:
            tradeability_score = _clip_score((tradeability_ratio / max(floor, 1e-6)) * 35.0)
        elif tradeability_ratio <= good:
            span = max(good - floor, 1e-6)
            tradeability_score = _clip_score(35.0 + ((tradeability_ratio - floor) / span) * 35.0)
        elif tradeability_ratio <= strong:
            span = max(strong - good, 1e-6)
            tradeability_score = _clip_score(70.0 + ((tradeability_ratio - good) / span) * 25.0)
        else:
            tradeability_score = 100.0

    updated["direction_score"] = direction_score
    updated["magnitude_score"] = magnitude_score
    updated["timing_score"] = timing_score
    updated["tradeability_score"] = tradeability_score
    updated["tradeability_tier"] = _classify_tradeability_tier(tradeability_score)

    best_horizon, best_outcome_bps, peak_to_close_decay_bps, horizon_edge_label = _resolve_best_outcome_profile(updated)
    updated["best_outcome_horizon"] = best_horizon
    updated["best_outcome_bps"] = best_outcome_bps
    updated["peak_to_close_decay_bps"] = peak_to_close_decay_bps
    updated["horizon_edge_label"] = horizon_edge_label

    exit_quality_label, exit_efficiency_score = _resolve_exit_quality(updated)
    updated["exit_quality_label"] = exit_quality_label
    updated["exit_efficiency_score"] = exit_efficiency_score

    component_scores = {
        "direction_score": direction_score,
        "magnitude_score": magnitude_score,
        "timing_score": timing_score,
        "tradeability_score": tradeability_score,
    }
    if all(score is not None for score in component_scores.values()):
        composite = sum(
            component_scores[name] * score_weights[name]
            for name in component_scores
        )
        updated["composite_signal_score"] = _clip_score(composite)
    else:
        updated["composite_signal_score"] = pd.NA

    return updated


def evaluate_signal_outcomes(row: dict, realized_spot_path: pd.DataFrame, *, as_of=None) -> dict:
    """
    Purpose:
        Evaluate realized market outcomes for a captured signal against a realized spot path.
    
    Context:
        This is the core realized-outcome step in the signal-evaluation pipeline. It lines up a captured signal with future spot observations, computes horizon returns and MFE/MAE style diagnostics, and marks whether the row is still pending or complete.
    
    Inputs:
        row (dict): Signal-evaluation row to update with realized outcomes.
        realized_spot_path (pd.DataFrame): Timestamped realized spot path for the underlying after the signal was captured.
        as_of (Any): Timestamp representing how far the evaluator is allowed to look forward.
    
    Returns:
        dict: Updated evaluation row containing realized outcome checkpoints and status fields.
    
    Notes:
        Rows are marked `PENDING`, `PARTIAL`, or `COMPLETE` depending on which checkpoints are available at `as_of`, which lets the dataset be backfilled incrementally over time.
    """
    updated = dict(row)
    if realized_spot_path is None or realized_spot_path.empty:
        updated["outcome_status"] = "PENDING"
        return updated

    path = realized_spot_path.copy()
    if "timestamp" not in path.columns or "spot" not in path.columns:
        raise ValueError("Realized spot path must include 'timestamp' and 'spot' columns")

    path["timestamp"] = path["timestamp"].map(_coerce_ts)
    path["spot"] = pd.to_numeric(path["spot"], errors="coerce")
    path = path.dropna(subset=["timestamp", "spot"])
    path = path[path["spot"] > 0].sort_values("timestamp").reset_index(drop=True)
    if path.empty:
        updated["outcome_status"] = "PENDING"
        return updated

    signal_ts = _coerce_ts(updated["signal_timestamp"])
    as_of_ts = _coerce_ts(as_of) if as_of is not None else path["timestamp"].max()
    path = path[path["timestamp"] <= as_of_ts].copy()
    if path.empty:
        updated["outcome_status"] = "PENDING"
        updated["observed_minutes"] = 0.0
        updated["outcome_last_updated_at"] = as_of_ts.isoformat()
        updated["updated_at"] = as_of_ts.isoformat()
        return updated

    observed_minutes = max((as_of_ts - signal_ts).total_seconds() / 60.0, 0.0)
    updated["observed_minutes"] = round(observed_minutes, 2)
    updated["outcome_last_updated_at"] = as_of_ts.isoformat()

    entry_spot = _safe_float(updated.get("spot_at_signal"), None)
    if entry_spot in (None, 0):
        first_spot = _nearest_spot_at_or_after(path, signal_ts)
        if first_spot in (None, 0):
            updated["outcome_status"] = "PENDING"
            return updated
        entry_spot = first_spot
        updated["spot_at_signal"] = round(entry_spot, 4)

    direction_mult = _signal_direction_multiplier(updated.get("direction"))
    has_direction = direction_mult != 0

    completed_checkpoints = 0
    for horizon in SIGNAL_EVALUATION_HORIZON_MINUTES:
        target_ts = signal_ts + pd.Timedelta(minutes=horizon)
        horizon_spot = _nearest_spot_at_or_after(path, target_ts)
        if horizon_spot is None:
            continue

        updated[f"spot_{horizon}m"] = round(horizon_spot, 4)
        updated[f"realized_return_{horizon}m"] = _raw_return(entry_spot, horizon_spot)
        if has_direction:
            signed_return_bps = ((horizon_spot - entry_spot) / max(entry_spot, 1e-9)) * 10000.0 * direction_mult
            updated[f"signed_return_{horizon}m_bps"] = round(float(signed_return_bps), 2)
            updated[f"correct_{horizon}m"] = int(signed_return_bps > 0)
        completed_checkpoints += 1

    if has_direction:
        for window_minutes in [60, 120]:
            stats = _window_stats(
                path=path,
                entry_ts=signal_ts,
                end_ts=signal_ts + pd.Timedelta(minutes=window_minutes),
                direction_mult=direction_mult,
                entry_spot=entry_spot,
            )
            if stats is None:
                continue
            updated[f"mfe_{window_minutes}m_bps"] = stats["mfe_bps"]
            updated[f"mae_{window_minutes}m_bps"] = stats["mae_bps"]
            updated[f"realized_range_{window_minutes}m_bps"] = stats["range_bps"]

        full_window = path[path["timestamp"] >= signal_ts].copy()
        if not full_window.empty:
            directional_moves_points = (full_window["spot"].astype(float) - entry_spot) * direction_mult
            updated["mfe_points"] = round(float(directional_moves_points.max()), 4)
            updated["mae_points"] = round(float(directional_moves_points.min()), 4)

    close_spot = _session_close_spot(path, signal_ts)
    if close_spot is not None:
        updated["spot_close_same_day"] = round(close_spot, 4)
        updated["spot_session_close"] = round(close_spot, 4)
        if has_direction:
            signed_close_return = ((close_spot - entry_spot) / max(entry_spot, 1e-9)) * 10000.0 * direction_mult
            updated["signed_return_session_close_bps"] = round(float(signed_close_return), 2)
            updated["correct_session_close"] = int(signed_close_return > 0)
        completed_checkpoints += 1

    next_open_spot = _next_day_open_spot(path, signal_ts)
    if next_open_spot is not None:
        updated["spot_next_open"] = round(next_open_spot, 4)
        completed_checkpoints += 1

    next_close_spot = _next_day_close_spot(path, signal_ts)
    if next_close_spot is not None:
        updated["spot_next_close"] = round(next_close_spot, 4)
        completed_checkpoints += 1

    if has_direction:
        correctness_fields = [updated.get(f"correct_{horizon}m") for horizon in SIGNAL_EVALUATION_HORIZON_MINUTES]
        correctness_fields.append(updated.get("correct_session_close"))
        correctness_values = []
        for value in correctness_fields:
            if pd.isna(value):
                continue
            if value in (0, 1):
                correctness_values.append(float(value))
        if correctness_values:
            updated["directional_consistency_score"] = round(sum(correctness_values) / len(correctness_values), 4)

    updated = compute_signal_evaluation_scores(updated)

    total_checkpoints = len(SIGNAL_EVALUATION_HORIZON_MINUTES) + 3
    if completed_checkpoints == 0:
        updated["outcome_status"] = "PENDING"
    elif completed_checkpoints < total_checkpoints:
        updated["outcome_status"] = "PARTIAL"
    else:
        updated["outcome_status"] = "COMPLETE"

    updated["updated_at"] = as_of_ts.isoformat()
    return updated


def save_signal_evaluation(
    result: dict,
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    realized_spot_path: pd.DataFrame | None = None,
    as_of=None,
    notes: str | None = None,
    return_frame: bool = True,
) -> pd.DataFrame | None:
    """
    Purpose:
        Build and persist a signal-evaluation row for one engine result payload.
    
    Context:
        Used by live runtime, replay tools, and research scripts to capture a common evaluation-row shape. It can optionally enrich the row immediately when realized spot data is already available.
    
    Inputs:
        result (dict): Engine result payload containing the captured trade or no-trade state.
        dataset_path (str | Path): Destination dataset path.
        realized_spot_path (pd.DataFrame | None): Optional realized spot path used to compute outcomes immediately.
        as_of (Any): Timestamp used when resolving capture time and outcome lookahead.
        notes (str | None): Optional notes to persist with the evaluation row.
        return_frame (bool): Whether the persistence helper should return the updated dataset frame.
    
    Returns:
        pd.DataFrame | None: Updated dataset frame when requested, otherwise `None`.
    
    Notes:
        This keeps the on-write schema identical across live trading, replay analysis, and research backfills.
    """
    spot_summary = result.get("spot_summary") if isinstance(result, dict) else None
    spot_summary = spot_summary if isinstance(spot_summary, dict) else {}
    trade = result.get("trade") if isinstance(result, dict) else None
    trade = trade if isinstance(trade, dict) else {}
    capture_default = spot_summary.get("timestamp") or trade.get("valuation_time")
    row = build_signal_evaluation_row(
        result,
        notes=notes,
        captured_at=resolve_research_as_of(as_of, default=capture_default),
    )
    if realized_spot_path is not None and not realized_spot_path.empty:
        evaluation_default = (
            realized_spot_path["timestamp"].max()
            if "timestamp" in realized_spot_path.columns
            else capture_default
        )
        row = evaluate_signal_outcomes(
            row,
            realized_spot_path,
            as_of=resolve_research_as_of(as_of, default=evaluation_default),
        )
    return upsert_signal_rows([row], path=dataset_path, return_frame=return_frame)


def update_signal_dataset_outcomes(
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    as_of=None,
    fetch_spot_path_fn=None,
    fetch_spot_history_fn=fetch_realized_spot_history,
) -> pd.DataFrame:
    """
    Purpose:
        Backfill realized outcomes for incomplete rows in the signal-evaluation dataset.
    
    Context:
        Research and governance workflows often save rows before all horizons have elapsed. This helper revisits the dataset later, fetches the necessary realized spot paths, and updates any row that has become partially or fully observable.
    
    Inputs:
        dataset_path (str | Path): Signal-evaluation dataset path to update.
        as_of (Any): Timestamp that limits how far forward the backfill is allowed to look.
        fetch_spot_path_fn (Any): Optional callback that returns a realized spot path for one row.
        fetch_spot_history_fn (Any): History-fetch callback used to build realized spot paths when a row-level callback is not supplied.
    
    Returns:
        pd.DataFrame: Updated signal-evaluation dataset after incomplete rows have been refreshed.
    
    Notes:
        This helper supports delayed promotion and reporting workflows by letting the dataset mature as more realized market history becomes available.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        return upsert_signal_rows([], path=dataset_path)

    resolved_as_of = resolve_research_as_of(as_of)
    frame = load_signals_dataset(dataset_path)
    if frame.empty:
        return upsert_signal_rows([], path=dataset_path)

    records = frame.to_dict(orient="records")
    path_cache = (
        build_realized_spot_path_cache(
            [row for row in records if row.get("outcome_status") != "COMPLETE"],
            as_of=resolved_as_of,
            fetch_history_fn=fetch_spot_history_fn,
        )
        if fetch_spot_path_fn is None
        else {}
    )

    updated_rows = []
    for row_dict in records:
        if row_dict.get("outcome_status") == "COMPLETE":
            updated_rows.append(row_dict)
            continue

        if fetch_spot_path_fn is not None:
            realized_path = fetch_spot_path_fn(
                row_dict.get("symbol"),
                row_dict.get("signal_timestamp"),
                as_of=resolved_as_of,
            )
        else:
            realized_path = path_cache.get(
                (str(row_dict.get("symbol")), str(row_dict.get("signal_timestamp"))),
                pd.DataFrame(columns=["timestamp", "spot"]),
            )
        updated_rows.append(evaluate_signal_outcomes(row_dict, realized_path, as_of=resolved_as_of))

    return upsert_signal_rows(updated_rows, path=dataset_path)
