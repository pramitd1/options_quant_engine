"""
Module: signal_engine.py

Purpose:
    Assemble the final trade decision from normalized market data, analytics features, macro context, and risk controls.

Role in the System:
    Part of the signal engine layer that assembles analytics, strategy logic, and overlays into trade decisions.

Key Outputs:
    A fully explained trade or no-trade payload, including diagnostics, overlay scores, strike selection, and sizing fields.

Downstream Usage:
    Consumed by the live runtime loop, replay tooling, shadow-mode comparisons, and signal-evaluation logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import (
    BACKTEST_MIN_TRADE_STRENGTH,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
    STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT,
)
from config.event_window_policy import get_event_window_policy_config
from config.signal_policy import get_activation_score_policy_config, get_trade_runtime_thresholds
from engine.pre_market_engine import apply_pre_market_adjustments_to_signal
from engine.runtime_metadata import attach_trade_views
from engine.trading_support import (
    _clip,
    _collect_market_state,
    _compute_data_quality,
    _compute_probability_state,
    _compute_signal_state,
    _safe_float,
    _to_python_number,
    classify_execution_regime,
    classify_signal_quality,
    classify_signal_regime,
    derive_dealer_pressure_trade_modifiers,
    derive_gamma_vol_trade_modifiers,
    derive_global_risk_trade_modifiers,
    derive_option_efficiency_trade_modifiers,
    normalize_option_chain,
)
from analytics.signal_confidence import compute_signal_confidence
from analytics.probability_calibration import calibrate_probability
from macro.engine_adjustments import compute_macro_news_adjustments
from risk import (
    build_dealer_hedging_pressure_state,
    build_gamma_vol_acceleration_state,
    build_option_efficiency_state,
)
from risk.global_risk_layer import evaluate_global_risk_layer
from risk.option_efficiency_layer import score_option_efficiency_candidate
from strategy.budget_optimizer import optimize_lots
from strategy.exit_model import calculate_exit, compute_exit_timing
from strategy.strike_selector import select_best_strike
from engine.decision_journal import append_decision as _journal_append_decision
from utils.regime_normalization import canonical_gamma_regime
from utils.consistency_checks import collect_trade_consistency_findings, select_trade_escalation_findings
from strategy.score_calibration import (
    initialize_calibrator,
    apply_score_calibration,
    get_calibrator_runtime_metadata,
    create_calibration_segment_key,
    normalize_calibration_context,
)
from strategy.time_decay_model import initialize_time_decay, apply_time_decay
from strategy.path_aware_filtering import PathAwareFilter, PathPatternLibrary
from strategy.regime_conditional_thresholds import initialize_regime_thresholds, compute_regime_thresholds


_LOG = logging.getLogger(__name__)
_TIMESTAMP_ERRORS = (TypeError, ValueError, OverflowError)
_SIGNAL_STATE_ERRORS = (AttributeError, TypeError, ValueError, OverflowError)


def _as_upper(value):
    return str(value or "").upper().strip()


def _is_directional_flow(flow_label):
    return _as_upper(flow_label) in {"BULLISH_FLOW", "BEARISH_FLOW"}


def _is_convexity_active(convexity_state):
    return _as_upper(convexity_state) in {
        "UPSIDE_SQUEEZE_RISK",
        "DOWNSIDE_AIRPOCKET_RISK",
        "TWO_SIDED_VOLATILITY_RISK",
    }


def _is_dealer_structure_active(dealer_flow_state):
    return _as_upper(dealer_flow_state) in {
        "UPSIDE_HEDGING_ACCELERATION",
        "DOWNSIDE_HEDGING_ACCELERATION",
        "PINNING_DOMINANT",
        "TWO_SIDED_INSTABILITY",
    }


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item in (None, "", []):
            continue
        normalized = str(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


_DECAY_SIGNAL_STATE = {}
_PATH_SIGNAL_STATE = {}
_PATH_FILTER = None
_TIME_DECAY_MODEL_CONFIG_KEY = None
_REGIME_THRESHOLDS_CONFIG_KEY = None
_OUTCOME_HISTORY_CACHE = {
    "path": None,
    "mtime": None,
    "frame": None,
}


def _coerce_timestamp(value):
    if value is None:
        return None
    try:
        import pandas as _pd
        return _pd.Timestamp(value)
    except _TIMESTAMP_ERRORS as exc:
        _LOG.debug("signal_engine: unable to coerce valuation_time=%r into timestamp: %s", value, exc)
        return None


def _normalize_string_set(value, *, default: set[str] | None = None) -> set[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned.upper()} if cleaned else set(default or set())
    if isinstance(value, (list, tuple, set)):
        normalized = {str(item).strip().upper() for item in value if str(item).strip()}
        return normalized or set(default or set())
    return set(default or set())


def _evaluate_provider_health_override_eligibility(
    *,
    runtime_thresholds,
    provider_health_blocking_reasons,
    provider_health_summary,
    data_quality_status,
    confirmation_status,
    adjusted_trade_strength,
    min_trade_strength,
    runtime_composite_score,
    min_composite_score,
    option_chain_validation,
    provider_health,
    ranked_strikes,
    days_to_expiry,
    blocked,
    option_efficiency_score=None,
    premium_efficiency_score=None,
):
    enable_override = bool(int(_safe_float(runtime_thresholds.get("enable_provider_health_degraded_override"), 0.0)))
    if not enable_override:
        return False, {"reason": "override_disabled"}

    details = {
        "eligible": False,
        "fail_reasons": [],
    }

    if blocked and not bool(int(_safe_float(runtime_thresholds.get("provider_health_override_allow_block_status"), 0.0))):
        details["fail_reasons"].append("block_status_not_allowed")

    allowed_summary_statuses = _normalize_string_set(
        runtime_thresholds.get("provider_health_override_allowed_summary_statuses"),
        default={"CAUTION", "WEAK"},
    )
    provider_health_summary_upper = _as_upper(provider_health_summary)
    if provider_health_summary_upper not in allowed_summary_statuses:
        details["fail_reasons"].append("provider_summary_not_allowlisted")

    allowed_data_quality_statuses = _normalize_string_set(
        runtime_thresholds.get("provider_health_override_allowed_data_quality_statuses"),
        default={"GOOD", "STRONG"},
    )
    data_quality_status_upper = _as_upper(data_quality_status)
    if data_quality_status_upper not in allowed_data_quality_statuses:
        details["fail_reasons"].append("data_quality_not_allowlisted")

    dte_max = _safe_float(runtime_thresholds.get("provider_health_override_dte_max"), 1.0)
    dte_value = _safe_float(days_to_expiry, None)
    if dte_value is not None and dte_value > dte_max:
        details["fail_reasons"].append(f"dte_above_max:{dte_value}")

    require_strong_confirmation = bool(int(_safe_float(runtime_thresholds.get("provider_health_override_require_strong_confirmation"), 1.0)))
    if require_strong_confirmation and _as_upper(confirmation_status) not in {"STRONG_CONFIRMATION", "CONFIRMED"}:
        details["fail_reasons"].append("confirmation_not_strong")

    strength_buffer = int(_safe_float(runtime_thresholds.get("provider_health_override_min_strength_buffer"), 12.0))
    composite_buffer = int(_safe_float(runtime_thresholds.get("provider_health_override_min_composite_buffer"), 8.0))

    option_efficiency_value = _safe_float(option_efficiency_score, None)
    premium_efficiency_value = _safe_float(premium_efficiency_score, None)
    min_option_efficiency = _safe_float(runtime_thresholds.get("provider_health_override_min_option_efficiency_score"), 80.0)
    min_premium_efficiency = _safe_float(runtime_thresholds.get("provider_health_override_min_premium_efficiency_score"), 70.0)
    if option_efficiency_value is not None and option_efficiency_value < min_option_efficiency:
        details["fail_reasons"].append("option_efficiency_below_floor")
    if premium_efficiency_value is not None and premium_efficiency_value < min_premium_efficiency:
        details["fail_reasons"].append("premium_efficiency_below_floor")

    if (
        not blocked
        and data_quality_status_upper in {"GOOD", "STRONG"}
        and option_efficiency_value is not None
        and premium_efficiency_value is not None
        and option_efficiency_value >= min_option_efficiency
        and premium_efficiency_value >= min_premium_efficiency
    ):
        relaxed_strength_buffer = int(
            _safe_float(runtime_thresholds.get("provider_health_override_high_efficiency_strength_buffer"), 0.0)
        )
        strength_buffer = min(strength_buffer, relaxed_strength_buffer)

    if adjusted_trade_strength < (min_trade_strength + strength_buffer):
        details["fail_reasons"].append("trade_strength_buffer_not_met")
    if runtime_composite_score < (min_composite_score + composite_buffer):
        details["fail_reasons"].append("runtime_composite_buffer_not_met")

    effective_priced_ratio = _safe_float(
        option_chain_validation.get("effective_priced_ratio"),
        _safe_float(option_chain_validation.get("priced_ratio"), 0.0),
    )
    min_effective_priced_ratio = _safe_float(runtime_thresholds.get("provider_health_override_min_effective_priced_ratio"), 0.45)
    if effective_priced_ratio < min_effective_priced_ratio:
        details["fail_reasons"].append("effective_priced_ratio_below_floor")

    core_one_sided_ratio = _safe_float(provider_health.get("core_one_sided_quote_ratio"), None)
    if core_one_sided_ratio is None:
        row_count = max(int(_safe_float(option_chain_validation.get("row_count"), 0.0)), 1)
        core_one_sided_ratio = _safe_float(option_chain_validation.get("one_sided_quote_rows"), 0.0) / row_count
    max_one_sided_ratio = _safe_float(runtime_thresholds.get("provider_health_override_one_sided_quote_ratio_max"), 1.0)
    if core_one_sided_ratio > max_one_sided_ratio:
        details["fail_reasons"].append("one_sided_quote_ratio_above_cap")

    ranked_candidates = ranked_strikes or []
    if ranked_candidates:
        proxy_count = sum(
            1
            for candidate in ranked_candidates
            if bool(candidate.get("iv_is_proxy")) or bool(candidate.get("delta_is_proxy"))
        )
        proxy_ratio = proxy_count / max(len(ranked_candidates), 1)
    else:
        proxy_ratio = 1.0
    max_proxy_ratio = _safe_float(runtime_thresholds.get("provider_health_override_max_proxy_ratio"), 0.90)
    if proxy_ratio > max_proxy_ratio:
        details["fail_reasons"].append("proxy_ratio_above_cap")

    allowed_reasons = _normalize_string_set(
        runtime_thresholds.get("provider_health_override_allowed_block_reasons", ["core_iv_weak"]),
        default={"CORE_IV_WEAK"},
    )
    if blocked:
        blocking_reasons_upper = {
            str(reason).strip().upper()
            for reason in provider_health_blocking_reasons
            if str(reason).strip()
        }
        if not blocking_reasons_upper:
            details["fail_reasons"].append("missing_block_reasons")
        elif not blocking_reasons_upper.issubset(allowed_reasons):
            details["fail_reasons"].append("block_reasons_not_allowlisted")

    details["proxy_ratio"] = round(proxy_ratio, 4)
    details["effective_priced_ratio"] = round(effective_priced_ratio, 4)
    details["one_sided_quote_ratio"] = round(core_one_sided_ratio, 4)
    details["dte"] = dte_value
    details["provider_health_summary"] = provider_health_summary_upper
    details["data_quality_status"] = data_quality_status_upper
    details["option_efficiency_score"] = round(option_efficiency_value, 4) if option_efficiency_value is not None else None
    details["premium_efficiency_score"] = round(premium_efficiency_value, 4) if premium_efficiency_value is not None else None
    details["eligible"] = not details["fail_reasons"]
    return details["eligible"], details


def _ranked_strike_proxy_ratio(ranked_strikes) -> float:
    ranked_candidates = ranked_strikes or []
    if not ranked_candidates:
        return 1.0
    proxy_count = sum(
        1
        for candidate in ranked_candidates
        if bool(candidate.get("iv_is_proxy")) or bool(candidate.get("delta_is_proxy"))
    )
    return float(proxy_count) / float(max(len(ranked_candidates), 1))


def _build_signal_probability_overlay(
    *,
    direction,
    adjusted_trade_strength,
    call_probability,
    put_probability,
    market_state,
    runtime_thresholds,
):
    """Blend direction probabilities with calibrated, context-aware deflation.

    The latest signal-evaluation reports showed persistent overconfidence,
    especially in fragile risk-off sessions. The live path now shrinks
    aggressive probabilities back toward neutral when provider health, data
    quality, or macro context are weak.
    """
    runtime_thresholds = runtime_thresholds if isinstance(runtime_thresholds, dict) else {}
    market_state = market_state if isinstance(market_state, dict) else {}
    raw_score = float(_clip(_safe_float(adjusted_trade_strength, 50.0), 0.0, 100.0))
    calibrator_path = runtime_thresholds.get("signal_probability_calibrator_path")

    calibrated_probability = None
    try:
        calibrated_probability = calibrate_probability(
            raw_score,
            calibrator_path=calibrator_path,
        )
    except Exception as exc:
        _LOG.debug("signal_engine: signal probability calibration unavailable: %s", exc)

    directional_probability = None
    if direction == "CALL":
        directional_probability = _safe_float(call_probability, None)
    elif direction == "PUT":
        directional_probability = _safe_float(put_probability, None)
    else:
        call_prob = _safe_float(call_probability, None)
        put_prob = _safe_float(put_probability, None)
        directional_probability = max(v for v in (call_prob, put_prob) if v is not None) if any(
            v is not None for v in (call_prob, put_prob)
        ) else None

    if directional_probability is not None:
        directional_probability = float(_clip(directional_probability, 0.0, 1.0))
    if calibrated_probability is not None:
        calibrated_probability = float(_clip(calibrated_probability, 0.0, 1.0))

    iv_hv_regime = _as_upper(market_state.get("iv_hv_regime"))
    provider_health_summary = _as_upper(market_state.get("provider_health_summary"))
    data_quality_status = _as_upper(market_state.get("data_quality_status"))
    global_risk_state = _as_upper(market_state.get("global_risk_state"))
    option_efficiency_status = _as_upper(market_state.get("option_efficiency_status"))
    reversal_stage = _as_upper(market_state.get("reversal_stage"))
    fast_reversal_alert_level = _as_upper(market_state.get("fast_reversal_alert_level"))
    live_calibration_gate = market_state.get("live_calibration_gate") if isinstance(market_state.get("live_calibration_gate"), dict) else {}
    live_directional_gate = market_state.get("live_directional_gate") if isinstance(market_state.get("live_directional_gate"), dict) else {}
    ranked_strike_proxy_ratio = _clip(_safe_float(market_state.get("ranked_strike_proxy_ratio"), 0.0), 0.0, 1.0)
    shrink_reasons: list[str] = []

    iv_hv_probability_adjustment = 0.0
    if iv_hv_regime == "IV_CHEAP":
        iv_hv_probability_adjustment = _safe_float(runtime_thresholds.get("iv_hv_probability_bonus"), 0.03)
    elif iv_hv_regime == "IV_RICH":
        iv_hv_probability_adjustment = -abs(_safe_float(runtime_thresholds.get("iv_hv_probability_penalty"), 0.03))

    signal_success_probability = None
    if directional_probability is None and calibrated_probability is None:
        signal_success_probability = None
    elif directional_probability is None:
        signal_success_probability = calibrated_probability
    elif calibrated_probability is None:
        signal_success_probability = directional_probability
    else:
        signal_success_probability = (
            0.50 * directional_probability
            + 0.50 * calibrated_probability
            + float(iv_hv_probability_adjustment)
        )

    shrink_strength = _clip(_safe_float(runtime_thresholds.get("signal_probability_neutral_shrink"), 0.12), 0.0, 0.45)
    fragile_context = (
        provider_health_summary in {"WEAK", "CAUTION"}
        or data_quality_status in {"WEAK", "CAUTION"}
    )
    if fragile_context:
        shrink_strength = min(
            0.55,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_fragile_shrink_add"), 0.12),
        )
        shrink_reasons.append("fragile_context")
    if global_risk_state in {"RISK_OFF", "VOL_SHOCK", "EVENT_LOCKDOWN"}:
        shrink_strength = min(
            0.60,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_risk_off_shrink_add"), 0.06),
        )
        shrink_reasons.append("risk_off_context")
    if reversal_stage == "EARLY_REVERSAL_CANDIDATE":
        shrink_strength = min(
            0.68,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_early_reversal_shrink_add"), 0.06),
        )
        shrink_reasons.append("early_reversal_candidate")
    elif reversal_stage == "REVERSAL_UNRESOLVED":
        shrink_strength = min(
            0.70,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_unresolved_reversal_shrink_add"), 0.08),
        )
        shrink_reasons.append("reversal_unresolved")
    if fast_reversal_alert_level == "HIGH":
        shrink_strength = min(0.72, shrink_strength + 0.03)
        shrink_reasons.append("fast_reversal_warning")

    live_calibration_verdict = _as_upper(live_calibration_gate.get("verdict"))
    if live_calibration_verdict == "CAUTION":
        shrink_strength = min(
            0.65,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_live_calibration_caution_shrink_add"), 0.05),
        )
        shrink_reasons.append("live_calibration_caution")
    elif live_calibration_verdict == "BLOCK":
        shrink_strength = min(
            0.70,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_live_calibration_block_shrink_add"), 0.10),
        )
        shrink_reasons.append("live_calibration_block")

    max_ece = _safe_float(runtime_thresholds.get("signal_probability_live_calibration_max_ece"), 0.18)
    ece_value = _safe_float(live_calibration_gate.get("ece"), None)
    if ece_value is not None and max_ece is not None and ece_value > max_ece:
        shrink_strength = min(0.72, shrink_strength + min(0.10, max(0.0, ece_value - max_ece) * 0.75))
        shrink_reasons.append("live_calibration_ece_drift")

    max_top_overconfidence = _safe_float(
        runtime_thresholds.get("signal_probability_live_calibration_max_top_decile_overconfidence"),
        0.20,
    )
    overconfidence_value = _safe_float(live_calibration_gate.get("top_decile_overconfidence"), None)
    if (
        overconfidence_value is not None
        and max_top_overconfidence is not None
        and overconfidence_value > max_top_overconfidence
    ):
        shrink_strength = min(
            0.72,
            shrink_strength + min(0.08, max(0.0, overconfidence_value - max_top_overconfidence) * 1.0),
        )
        shrink_reasons.append("top_decile_overconfidence")

    live_directional_verdict = _as_upper(live_directional_gate.get("verdict"))
    if live_directional_verdict == "CAUTION":
        shrink_strength = min(
            0.68,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_live_directional_caution_shrink_add"), 0.05),
        )
        shrink_reasons.append("live_directional_caution")
    elif live_directional_verdict == "BLOCK":
        shrink_strength = min(
            0.72,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_live_directional_block_shrink_add"), 0.10),
        )
        shrink_reasons.append("live_directional_block")

    if option_efficiency_status.startswith("UNAVAILABLE"):
        shrink_strength = min(
            0.68,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_option_efficiency_unavailable_shrink_add"), 0.05),
        )
        shrink_reasons.append("option_efficiency_unavailable")

    proxy_ratio_high = _clip(_safe_float(runtime_thresholds.get("signal_probability_proxy_ratio_high"), 0.80), 0.0, 1.0)
    proxy_ratio_caution = _clip(_safe_float(runtime_thresholds.get("signal_probability_proxy_ratio_caution"), 0.50), 0.0, proxy_ratio_high)
    if ranked_strike_proxy_ratio >= proxy_ratio_high:
        shrink_strength = min(
            0.72,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_proxy_ratio_high_shrink_add"), 0.08),
        )
        shrink_reasons.append("proxy_heavy_structure")
    elif ranked_strike_proxy_ratio >= proxy_ratio_caution:
        shrink_strength = min(
            0.68,
            shrink_strength + _safe_float(runtime_thresholds.get("signal_probability_proxy_ratio_caution_shrink_add"), 0.04),
        )
        shrink_reasons.append("proxy_moderate_structure")

    probability_gap = None
    if directional_probability is not None and calibrated_probability is not None:
        probability_gap = abs(directional_probability - calibrated_probability)
        if probability_gap > 0.10:
            shrink_strength = min(0.65, shrink_strength + max(0.0, probability_gap - 0.10) * 0.50)

    if signal_success_probability is not None:
        neutral_anchor = 0.50
        signal_success_probability = ((1.0 - shrink_strength) * signal_success_probability) + (shrink_strength * neutral_anchor)
        if fragile_context:
            fragile_ceiling = _clip(_safe_float(runtime_thresholds.get("signal_probability_fragile_ceiling"), 0.74), 0.5, 0.95)
            signal_success_probability = min(signal_success_probability, fragile_ceiling)
        if reversal_stage == "EARLY_REVERSAL_CANDIDATE":
            early_reversal_ceiling = _clip(_safe_float(runtime_thresholds.get("signal_probability_early_reversal_ceiling"), 0.68), 0.5, 0.95)
            signal_success_probability = min(signal_success_probability, early_reversal_ceiling)
        signal_success_probability = round(float(_clip(signal_success_probability, 0.01, 0.99)), 4)

    return {
        "directional_signal_probability": round(directional_probability, 4) if directional_probability is not None else None,
        "signal_calibrated_probability": round(calibrated_probability, 4) if calibrated_probability is not None else None,
        "signal_success_probability": signal_success_probability,
        "iv_hv_probability_adjustment": round(float(iv_hv_probability_adjustment), 4),
        "probability_neutral_shrink": round(float(shrink_strength), 4),
        "probability_gap": round(float(probability_gap), 4) if probability_gap is not None else None,
        "probability_shrink_reasons": _dedupe_keep_order(shrink_reasons),
    }


def _load_recent_outcome_history_frame():
    try:
        import pandas as _pd
        from config.settings import BASE_DIR as _BASE_DIR
    except Exception:
        return None, "import_failed"

    base_dir = Path(_BASE_DIR)
    candidate_paths = [
        base_dir / "research" / "signal_evaluation" / "signals_dataset_cumul.csv",
        base_dir / "research" / "signal_evaluation" / "signals_dataset.csv",
    ]
    selected_path = next((path for path in candidate_paths if path.exists()), None)
    if selected_path is None:
        return None, "dataset_missing"

    try:
        mtime = selected_path.stat().st_mtime
    except OSError:
        mtime = None

    if (
        _OUTCOME_HISTORY_CACHE.get("path") == str(selected_path)
        and _OUTCOME_HISTORY_CACHE.get("mtime") == mtime
        and _OUTCOME_HISTORY_CACHE.get("frame") is not None
    ):
        return _OUTCOME_HISTORY_CACHE.get("frame"), "cache"

    try:
        frame = _pd.read_csv(selected_path, low_memory=False)
    except Exception as exc:
        _LOG.debug("signal_engine: unable to load outcome history frame: %s", exc)
        return None, "load_failed"

    if len(frame) > 15000:
        frame = frame.tail(15000).copy()

    _OUTCOME_HISTORY_CACHE["path"] = str(selected_path)
    _OUTCOME_HISTORY_CACHE["mtime"] = mtime
    _OUTCOME_HISTORY_CACHE["frame"] = frame
    return frame, "loaded"


def _evaluate_historical_outcome_guard(*, payload, history_frame, runtime_thresholds):
    import pandas as _pd

    if not bool(int(_safe_float(runtime_thresholds.get("enable_historical_outcome_guard"), 1.0))):
        return {"enabled": False, "verdict": "DISABLED", "reason": "guard_disabled"}

    if history_frame is None or getattr(history_frame, "empty", True):
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "history_unavailable", "sample_size": 0}

    symbol = _as_upper(payload.get("symbol"))
    direction = _as_upper(payload.get("direction"))
    if direction not in {"CALL", "PUT"}:
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "direction_unavailable", "sample_size": 0}

    frame = history_frame.copy()
    for key in ["symbol", "direction", "gamma_regime", "macro_regime", "spot_vs_flip", "signal_regime"]:
        if key in frame.columns:
            frame[key] = frame[key].astype(str).str.upper().str.strip()

    subset = frame
    if "symbol" in subset.columns and symbol:
        subset = subset[subset["symbol"] == symbol]
    if "direction" in subset.columns:
        subset = subset[subset["direction"] == direction]

    progressive_filters = [
        ["gamma_regime", "macro_regime", "spot_vs_flip", "signal_regime"],
        ["gamma_regime", "macro_regime", "spot_vs_flip"],
        ["gamma_regime", "macro_regime"],
        ["gamma_regime"],
        [],
    ]

    min_samples = max(4, int(_safe_float(runtime_thresholds.get("historical_outcome_guard_min_samples"), 12.0)))
    matched_on = ["symbol", "direction"]
    chosen = subset
    for extra_keys in progressive_filters:
        candidate = subset
        used_keys = ["symbol", "direction"]
        for key in extra_keys:
            value = _as_upper(payload.get(key))
            if not value or key not in candidate.columns:
                continue
            candidate = candidate[candidate[key] == value]
            used_keys.append(key)
        if len(candidate) >= min_samples:
            chosen = candidate
            matched_on = used_keys
            break
        if len(candidate) > 0 and len(candidate) >= len(chosen):
            chosen = candidate
            matched_on = used_keys

    if chosen.empty:
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "no_matching_history", "sample_size": 0}

    close_bps = _pd.to_numeric(chosen["signed_return_session_close_bps"] if "signed_return_session_close_bps" in chosen.columns else _pd.Series(dtype="float64"), errors="coerce")
    ret60_bps = _pd.to_numeric(chosen["signed_return_60m_bps"] if "signed_return_60m_bps" in chosen.columns else _pd.Series(dtype="float64"), errors="coerce")
    tradeability = _pd.to_numeric(chosen["tradeability_score"] if "tradeability_score" in chosen.columns else _pd.Series(dtype="float64"), errors="coerce")
    correct_60m = _pd.to_numeric(chosen["correct_60m"] if "correct_60m" in chosen.columns else _pd.Series(dtype="float64"), errors="coerce")
    correct_close = _pd.to_numeric(chosen["correct_session_close"] if "correct_session_close" in chosen.columns else _pd.Series(dtype="float64"), errors="coerce")

    avg_60m = round(float(ret60_bps.dropna().mean()), 2) if ret60_bps.notna().any() else None
    avg_close = round(float(close_bps.dropna().mean()), 2) if close_bps.notna().any() else None
    avg_tradeability = round(float(tradeability.dropna().mean()), 2) if tradeability.notna().any() else None
    hit_rate_60m = round(float(correct_60m.dropna().mean()), 4) if correct_60m.notna().any() else None
    hit_rate_close = round(float(correct_close.dropna().mean()), 4) if correct_close.notna().any() else None

    if "horizon_edge_label" in chosen.columns and chosen["horizon_edge_label"].notna().any():
        early_decay_share = float(chosen["horizon_edge_label"].astype(str).str.upper().eq("EARLY_ALPHA_DECAY").mean())
    else:
        early_decay_mask = ret60_bps.gt(0) & close_bps.lt(0)
        early_decay_share = float(early_decay_mask.mean()) if len(early_decay_mask) else 0.0

    if "exit_quality_label" in chosen.columns and chosen["exit_quality_label"].notna().any():
        stopout_share = float(
            chosen["exit_quality_label"].astype(str).str.upper().isin({"STOPPED_OUT", "AMBIGUOUS", "MISSED_EXIT"}).mean()
        )
    else:
        stopout_share = 0.0

    if "best_outcome_horizon" in chosen.columns and chosen["best_outcome_horizon"].notna().any():
        best_horizon = str(chosen["best_outcome_horizon"].dropna().astype(str).mode().iloc[0])
    else:
        horizon_means = {}
        for label, field_name in {
            "5m": "signed_return_5m_bps",
            "15m": "signed_return_15m_bps",
            "30m": "signed_return_30m_bps",
            "60m": "signed_return_60m_bps",
            "120m": "signed_return_120m_bps",
            "session_close": "signed_return_session_close_bps",
        }.items():
            if field_name in chosen.columns:
                values = _pd.to_numeric(chosen[field_name], errors="coerce")
                if values.notna().any():
                    horizon_means[label] = float(values.mean())
        best_horizon = max(horizon_means, key=horizon_means.get) if horizon_means else None

    early_decay_threshold = _safe_float(runtime_thresholds.get("historical_outcome_guard_early_decay_share_threshold"), 0.55)
    stopout_threshold = _safe_float(runtime_thresholds.get("historical_outcome_guard_stopout_share_threshold"), 0.35)
    min_tradeability = _safe_float(runtime_thresholds.get("historical_outcome_guard_min_tradeability_score"), 55.0)
    min_close_bps = _safe_float(runtime_thresholds.get("historical_outcome_guard_min_session_close_bps"), -5.0)

    exit_bias = "TAKE_PROFIT_EARLY" if (best_horizon in {"5m", "15m", "30m"} or early_decay_share >= early_decay_threshold) else "HOLD_TREND"

    if len(chosen) >= min_samples and (
        stopout_share >= stopout_threshold
        or ((avg_tradeability is not None and avg_tradeability < min_tradeability) and (avg_close is not None and avg_close < min_close_bps))
    ):
        verdict = "BLOCK"
        reason = "Historical outcome guard: similar setups have weak realized tradeability"
    elif len(chosen) >= max(6, min_samples // 2) and (
        early_decay_share >= early_decay_threshold
        or (avg_60m is not None and avg_close is not None and avg_60m > 0 and avg_close < 0)
    ):
        verdict = "CAUTION"
        reason = "Historical outcome guard: similar setups tend to decay after the early move"
    else:
        verdict = "PASS"
        reason = "Historical outcome profile is acceptable"

    return {
        "enabled": True,
        "verdict": verdict,
        "reason": reason,
        "sample_size": int(len(chosen)),
        "matched_on": matched_on,
        "avg_60m_bps": avg_60m,
        "avg_close_bps": avg_close,
        "avg_tradeability_score": avg_tradeability,
        "hit_rate_60m": hit_rate_60m,
        "hit_rate_close": hit_rate_close,
        "early_decay_share": round(float(early_decay_share), 4),
        "stopout_share": round(float(stopout_share), 4),
        "best_horizon": best_horizon,
        "exit_bias": exit_bias,
    }


def _compute_historical_outcome_guard(*, payload, runtime_thresholds):
    history_frame, source = _load_recent_outcome_history_frame()
    details = _evaluate_historical_outcome_guard(
        payload=payload,
        history_frame=history_frame,
        runtime_thresholds=runtime_thresholds,
    )
    details["history_source"] = source
    return details


def _evaluate_regime_segment_guard(*, payload, history_frame, runtime_thresholds):
    import pandas as _pd

    if not bool(int(_safe_float(runtime_thresholds.get("enable_regime_segment_guard"), 1.0))):
        return {"enabled": False, "verdict": "DISABLED", "reason": "guard_disabled", "sample_size": 0}

    if history_frame is None or getattr(history_frame, "empty", True):
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "history_unavailable", "sample_size": 0}

    base_context = payload.get("score_calibration_segment_context") if isinstance(payload.get("score_calibration_segment_context"), dict) else {}
    if not base_context:
        base_context = {
            "direction": payload.get("direction"),
            "gamma_regime": payload.get("gamma_regime"),
            "vol_regime": payload.get("volatility_regime") or payload.get("vol_regime"),
        }
    segment_context = normalize_calibration_context(base_context)
    segment_key = create_calibration_segment_key(segment_context)
    if segment_key == "default":
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "segment_context_missing", "sample_size": 0, "segment_key": segment_key}

    frame = history_frame.copy()
    if "symbol" in frame.columns:
        frame["symbol_norm"] = frame["symbol"].astype(str).str.upper().str.strip()
    else:
        frame["symbol_norm"] = ""
    if "direction" in frame.columns:
        frame["direction_norm"] = frame["direction"].astype(str).str.upper().str.strip()
    else:
        frame["direction_norm"] = "UNKNOWN"
    if "gamma_regime" in frame.columns:
        frame["gamma_regime_norm"] = frame["gamma_regime"].astype(str).str.upper().str.strip()
    else:
        frame["gamma_regime_norm"] = "UNKNOWN"

    if "vol_regime" in frame.columns:
        vol_source = frame["vol_regime"]
    elif "volatility_regime" in frame.columns:
        vol_source = frame["volatility_regime"]
    else:
        vol_source = _pd.Series(["UNKNOWN"] * len(frame), index=frame.index)
    frame["vol_regime_norm"] = vol_source.astype(str).str.upper().str.strip()

    subset = frame
    symbol = _as_upper(payload.get("symbol"))
    if symbol:
        subset = subset[subset["symbol_norm"] == symbol]
    if "direction" in segment_context:
        subset = subset[subset["direction_norm"] == segment_context.get("direction")]
    if "gamma_regime" in segment_context:
        subset = subset[subset["gamma_regime_norm"] == segment_context.get("gamma_regime")]
    if "vol_regime" in segment_context:
        subset = subset[subset["vol_regime_norm"] == segment_context.get("vol_regime")]

    sample_size = int(len(subset))
    if sample_size == 0:
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "no_matching_segment", "sample_size": 0, "segment_key": segment_key}

    close_bps = _pd.to_numeric(subset["signed_return_session_close_bps"] if "signed_return_session_close_bps" in subset.columns else _pd.Series(dtype="float64"), errors="coerce")
    ret60_bps = _pd.to_numeric(subset["signed_return_60m_bps"] if "signed_return_60m_bps" in subset.columns else _pd.Series(dtype="float64"), errors="coerce")
    tradeability = _pd.to_numeric(subset["tradeability_score"] if "tradeability_score" in subset.columns else _pd.Series(dtype="float64"), errors="coerce")
    hit_rate_60m = _pd.to_numeric(subset["correct_60m"] if "correct_60m" in subset.columns else _pd.Series(dtype="float64"), errors="coerce")

    avg_close_bps = round(float(close_bps.dropna().mean()), 2) if close_bps.notna().any() else None
    avg_60m_bps = round(float(ret60_bps.dropna().mean()), 2) if ret60_bps.notna().any() else None
    avg_tradeability_score = round(float(tradeability.dropna().mean()), 2) if tradeability.notna().any() else None
    hit_rate_60m_value = round(float(hit_rate_60m.dropna().mean()), 4) if hit_rate_60m.notna().any() else None

    min_samples = max(4, int(_safe_float(runtime_thresholds.get("regime_segment_guard_min_samples"), 10.0)))
    min_hit_rate = _safe_float(runtime_thresholds.get("regime_segment_guard_min_hit_rate_60m"), 0.48)
    min_tradeability = _safe_float(runtime_thresholds.get("regime_segment_guard_min_tradeability_score"), 55.0)
    min_close_bps = _safe_float(runtime_thresholds.get("regime_segment_guard_min_avg_close_bps"), -10.0)

    if sample_size < min_samples:
        verdict = "UNAVAILABLE"
        reason = "segment_sample_too_small"
    elif (
        hit_rate_60m_value is not None and hit_rate_60m_value < min_hit_rate
        and ((avg_tradeability_score is not None and avg_tradeability_score < min_tradeability) or (avg_close_bps is not None and avg_close_bps < min_close_bps))
    ):
        verdict = "BLOCK"
        reason = "Regime segment underperforms on hit rate and realized edge"
    elif (
        (avg_tradeability_score is not None and avg_tradeability_score < min_tradeability)
        or (avg_close_bps is not None and avg_close_bps < 0)
        or (hit_rate_60m_value is not None and hit_rate_60m_value < min_hit_rate)
    ):
        verdict = "CAUTION"
        reason = "Regime segment is fragile; require tighter promotion and faster exits"
    else:
        verdict = "PASS"
        reason = "Regime segment is within acceptable performance bounds"

    return {
        "enabled": True,
        "verdict": verdict,
        "reason": reason,
        "sample_size": sample_size,
        "segment_key": segment_key,
        "segment_context": segment_context,
        "hit_rate_60m": hit_rate_60m_value,
        "avg_60m_bps": avg_60m_bps,
        "avg_close_bps": avg_close_bps,
        "avg_tradeability_score": avg_tradeability_score,
    }


def _compute_regime_segment_guard(*, payload, runtime_thresholds):
    history_frame, source = _load_recent_outcome_history_frame()
    details = _evaluate_regime_segment_guard(
        payload=payload,
        history_frame=history_frame,
        runtime_thresholds=runtime_thresholds,
    )
    details["history_source"] = source
    return details


def _evaluate_session_risk_governor(*, payload, history_frame, runtime_thresholds):
    import pandas as _pd

    if not bool(int(_safe_float(runtime_thresholds.get("enable_session_risk_governor"), 1.0))):
        return {"enabled": False, "verdict": "DISABLED", "reason": "guard_disabled", "recent_signal_count": 0}

    if history_frame is None or getattr(history_frame, "empty", True):
        return {"enabled": True, "verdict": "UNAVAILABLE", "reason": "history_unavailable", "recent_signal_count": 0}

    symbol = _as_upper((payload or {}).get("symbol"))
    direction = _as_upper((payload or {}).get("direction"))
    valuation_time = _coerce_timestamp((payload or {}).get("valuation_time") or (payload or {}).get("as_of"))

    frame = history_frame.copy()
    if "symbol" in frame.columns:
        frame["symbol_norm"] = frame["symbol"].astype(str).str.upper().str.strip()
    else:
        frame["symbol_norm"] = ""
    if "direction" in frame.columns:
        frame["direction_norm"] = frame["direction"].astype(str).str.upper().str.strip()
    else:
        frame["direction_norm"] = "UNKNOWN"

    timestamp_col = None
    for candidate in ["timestamp", "as_of", "valuation_time", "signal_timestamp"]:
        if candidate in frame.columns:
            timestamp_col = candidate
            frame[candidate] = _pd.to_datetime(frame[candidate], errors="coerce")
            break

    subset = frame
    if symbol and "symbol_norm" in subset.columns:
        symbol_subset = subset[subset["symbol_norm"] == symbol]
        if not symbol_subset.empty:
            subset = symbol_subset

    if timestamp_col is not None and valuation_time is not None:
        day_subset = subset[subset[timestamp_col].dt.date == valuation_time.date()]
        if not day_subset.empty:
            subset = day_subset

    lookback = max(3, int(_safe_float(runtime_thresholds.get("session_risk_lookback_signals"), 6.0)))
    recent = subset.tail(lookback).copy()
    recent_signal_count = int(len(recent))
    min_samples = max(2, int(_safe_float(runtime_thresholds.get("session_risk_min_samples"), 4.0)))
    if recent_signal_count < min_samples:
        return {
            "enabled": True,
            "verdict": "UNAVAILABLE",
            "reason": "insufficient_recent_session_samples",
            "recent_signal_count": recent_signal_count,
        }

    close_bps = _pd.to_numeric(
        recent["signed_return_session_close_bps"] if "signed_return_session_close_bps" in recent.columns else _pd.Series(dtype="float64"),
        errors="coerce",
    )
    avg_close_bps = round(float(close_bps.dropna().mean()), 2) if close_bps.notna().any() else None
    loss_share = float(close_bps.lt(0).mean()) if len(close_bps) else 0.0

    if "exit_quality_label" in recent.columns:
        exit_labels = recent["exit_quality_label"].astype(str).str.upper().fillna("")
        stopout_mask = exit_labels.isin({"STOPPED_OUT", "MISSED_EXIT", "AMBIGUOUS"})
    else:
        stopout_mask = close_bps.lt(0)
    stopout_share = float(stopout_mask.mean()) if len(stopout_mask) else 0.0

    stopout_streak = 0
    max_stopout_streak = 0
    for is_stop in stopout_mask.astype(bool).tolist():
        if is_stop:
            stopout_streak += 1
            max_stopout_streak = max(max_stopout_streak, stopout_streak)
        else:
            stopout_streak = 0

    same_direction_count = 0
    if direction in {"CALL", "PUT"} and "direction_norm" in recent.columns:
        same_direction_count = int(recent[recent["direction_norm"] == direction].shape[0])

    drawdown_component = 0.0
    if avg_close_bps is not None and avg_close_bps < 0:
        drawdown_component = min(abs(float(avg_close_bps)) / 25.0, 1.0)
    budget_consumed = min(100.0, (loss_share * 40.0) + (stopout_share * 35.0) + (drawdown_component * 25.0))
    budget_remaining_pct = round(max(0.0, 100.0 - budget_consumed), 1)

    block_streak = max(1, int(_safe_float(runtime_thresholds.get("session_risk_max_stopout_streak"), 2.0)))
    max_loss_share = _clip(_safe_float(runtime_thresholds.get("session_risk_max_loss_share"), 0.60), 0.0, 1.0)
    min_avg_close_bps = _safe_float(runtime_thresholds.get("session_risk_min_avg_close_bps"), -5.0)
    caution_size_cap = _clip(_safe_float(runtime_thresholds.get("session_risk_caution_size_cap"), 0.60), 0.0, 1.0)
    block_size_cap = _clip(_safe_float(runtime_thresholds.get("session_risk_block_size_cap"), 0.35), 0.0, 1.0)
    cooldown_minutes = max(5, int(_safe_float(runtime_thresholds.get("session_risk_cooldown_minutes"), 30.0)))

    if max_stopout_streak >= block_streak and (avg_close_bps is None or avg_close_bps <= min_avg_close_bps):
        verdict = "BLOCK"
        reason = "Session risk governor: recent stop-out streak and drawdown require cooldown"
        size_cap = block_size_cap
        cooldown_active = True
    elif loss_share >= max_loss_share or stopout_share >= max_loss_share or budget_remaining_pct < 45.0:
        verdict = "CAUTION"
        reason = "Session risk governor: recent realized outcomes require reduced risk"
        size_cap = caution_size_cap
        cooldown_active = False
    else:
        verdict = "PASS"
        reason = "Session risk budget is within acceptable bounds"
        size_cap = 1.0
        cooldown_active = False

    return {
        "enabled": True,
        "verdict": verdict,
        "reason": reason,
        "recent_signal_count": recent_signal_count,
        "same_direction_count": same_direction_count,
        "avg_close_bps": avg_close_bps,
        "loss_share": round(float(loss_share), 4),
        "stopout_share": round(float(stopout_share), 4),
        "stopout_streak": int(max_stopout_streak),
        "budget_remaining_pct": budget_remaining_pct,
        "size_cap": round(float(size_cap), 4),
        "cooldown_minutes": cooldown_minutes,
        "cooldown_active": bool(cooldown_active),
    }


def _compute_session_risk_governor(*, payload, runtime_thresholds):
    history_frame, source = _load_recent_outcome_history_frame()
    details = _evaluate_session_risk_governor(
        payload=payload,
        history_frame=history_frame,
        runtime_thresholds=runtime_thresholds,
    )
    details["history_source"] = source
    return details


def _evaluate_trade_slot_governor(*, payload, history_frame, runtime_thresholds):
    import pandas as _pd

    payload = payload if isinstance(payload, dict) else {}
    operator_control_state = payload.get("operator_control_state") if isinstance(payload.get("operator_control_state"), dict) else {}

    if not bool(int(_safe_float(runtime_thresholds.get("enable_trade_slot_governor"), 1.0))):
        return {
            "enabled": False,
            "verdict": "DISABLED",
            "reason": "guard_disabled",
            "active_signal_count": 0,
            "same_direction_count": 0,
            "operator_override_active": False,
        }

    max_total_signals = max(1, int(_safe_float(runtime_thresholds.get("trade_slot_max_total_signals"), 3.0)))
    max_same_direction_signals = max(1, int(_safe_float(runtime_thresholds.get("trade_slot_max_same_direction_signals"), 2.0)))
    caution_size_cap = _clip(_safe_float(runtime_thresholds.get("trade_slot_caution_size_cap"), 0.55), 0.0, 1.0)
    override_size_cap = _clip(_safe_float(runtime_thresholds.get("trade_slot_override_size_cap"), 0.30), 0.0, 1.0)
    hold_cap_minutes = max(5, int(_safe_float(runtime_thresholds.get("trade_slot_hold_cap_minutes"), 20.0)))
    override_hold_cap_minutes = max(5, int(_safe_float(runtime_thresholds.get("trade_slot_override_hold_cap_minutes"), 15.0)))

    override_controls_enabled = bool(int(_safe_float(runtime_thresholds.get("enable_operator_override_controls"), 1.0)))
    override_mode = _as_upper(
        operator_control_state.get("slot_override")
        or operator_control_state.get("operator_override")
        or operator_control_state.get("action")
    )
    override_reason = str(
        operator_control_state.get("override_reason")
        or operator_control_state.get("note")
        or ""
    ).strip() or None

    if override_controls_enabled and (
        bool(operator_control_state.get("force_watchlist"))
        or override_mode == "FORCE_WATCHLIST"
    ):
        return {
            "enabled": True,
            "verdict": "BLOCK",
            "reason": "Trade slot governor: operator desk control forced the setup to WATCHLIST",
            "active_signal_count": 0,
            "same_direction_count": 0,
            "max_total_signals": max_total_signals,
            "max_same_direction_signals": max_same_direction_signals,
            "size_cap": 0.0,
            "hold_cap_minutes": hold_cap_minutes,
            "intraday_only": True,
            "operator_override_active": True,
            "override_mode": "FORCE_WATCHLIST",
            "override_reason": override_reason,
        }

    if history_frame is None or getattr(history_frame, "empty", True):
        return {
            "enabled": True,
            "verdict": "UNAVAILABLE",
            "reason": "history_unavailable",
            "active_signal_count": 0,
            "same_direction_count": 0,
            "max_total_signals": max_total_signals,
            "max_same_direction_signals": max_same_direction_signals,
            "size_cap": 1.0,
            "hold_cap_minutes": hold_cap_minutes,
            "intraday_only": False,
            "operator_override_active": False,
        }

    symbol = _as_upper(payload.get("symbol"))
    direction = _as_upper(payload.get("direction"))
    valuation_time = _coerce_timestamp(payload.get("valuation_time") or payload.get("as_of"))

    frame = history_frame.copy()
    if "symbol" in frame.columns:
        frame["symbol_norm"] = frame["symbol"].astype(str).str.upper().str.strip()
    else:
        frame["symbol_norm"] = ""
    if "direction" in frame.columns:
        frame["direction_norm"] = frame["direction"].astype(str).str.upper().str.strip()
    else:
        frame["direction_norm"] = "UNKNOWN"

    timestamp_col = None
    for candidate in ["timestamp", "as_of", "valuation_time", "signal_timestamp"]:
        if candidate in frame.columns:
            timestamp_col = candidate
            frame[candidate] = _pd.to_datetime(frame[candidate], errors="coerce")
            break

    subset = frame
    if symbol:
        symbol_subset = subset[subset["symbol_norm"] == symbol]
        if not symbol_subset.empty:
            subset = symbol_subset

    if timestamp_col is not None and valuation_time is not None:
        day_subset = subset[subset[timestamp_col].dt.date == valuation_time.date()]
        if not day_subset.empty:
            subset = day_subset
            subset = subset.sort_values(timestamp_col)

    if "trade_status" in subset.columns:
        active_statuses = {"TRADE", "EXECUTED", "FILLED", "OPEN", "DEGRADED_PROVIDER_TRADE"}
        status_subset = subset[subset["trade_status"].astype(str).str.upper().str.strip().isin(active_statuses)]
        if not status_subset.empty:
            subset = status_subset

    lookback = max(3, int(_safe_float(runtime_thresholds.get("trade_slot_lookback_signals"), 6.0)))
    recent = subset.tail(lookback).copy()
    active_signal_count = int(len(recent))
    same_direction_count = 0
    if direction in {"CALL", "PUT"} and "direction_norm" in recent.columns:
        same_direction_count = int(recent[recent["direction_norm"] == direction].shape[0])

    min_samples = max(1, int(_safe_float(runtime_thresholds.get("trade_slot_min_samples"), 3.0)))
    if active_signal_count < min_samples:
        return {
            "enabled": True,
            "verdict": "PASS",
            "reason": "Trade slot capacity is available",
            "active_signal_count": active_signal_count,
            "same_direction_count": same_direction_count,
            "max_total_signals": max_total_signals,
            "max_same_direction_signals": max_same_direction_signals,
            "size_cap": 1.0,
            "hold_cap_minutes": hold_cap_minutes,
            "intraday_only": False,
            "operator_override_active": False,
        }

    utilization = max(
        float(active_signal_count / max(max_total_signals, 1)),
        float(same_direction_count / max(max_same_direction_signals, 1)),
    )

    if active_signal_count > max_total_signals or same_direction_count > max_same_direction_signals:
        verdict = "BLOCK"
        reason = "Trade slot governor: symbol book already has too many same-way ideas"
        size_cap = 0.0
        intraday_only = True
    elif active_signal_count == max_total_signals or same_direction_count == max_same_direction_signals:
        verdict = "CAUTION"
        reason = "Trade slot governor: symbol book is near its governed capacity"
        size_cap = caution_size_cap
        intraday_only = True
    else:
        verdict = "PASS"
        reason = "Trade slot capacity is available"
        size_cap = 1.0
        intraday_only = False

    operator_override_active = False
    if verdict == "BLOCK" and override_controls_enabled and override_mode in {"ALLOW", "FORCE_ALLOW", "OVERRIDE"}:
        verdict = "CAUTION"
        reason = "Trade slot governor: operator override permits reduced-size entry despite a full symbol book"
        size_cap = override_size_cap
        hold_cap_minutes = min(hold_cap_minutes, override_hold_cap_minutes)
        intraday_only = True
        operator_override_active = True

    return {
        "enabled": True,
        "verdict": verdict,
        "reason": reason,
        "active_signal_count": active_signal_count,
        "same_direction_count": same_direction_count,
        "max_total_signals": max_total_signals,
        "max_same_direction_signals": max_same_direction_signals,
        "utilization_ratio": round(utilization, 4),
        "size_cap": round(float(size_cap), 4),
        "hold_cap_minutes": hold_cap_minutes,
        "intraday_only": bool(intraday_only),
        "operator_override_active": bool(operator_override_active),
        "override_mode": override_mode if operator_override_active else None,
        "override_reason": override_reason if operator_override_active else None,
    }


def _compute_trade_slot_governor(*, payload, runtime_thresholds):
    history_frame, source = _load_recent_outcome_history_frame()
    details = _evaluate_trade_slot_governor(
        payload=payload,
        history_frame=history_frame,
        runtime_thresholds=runtime_thresholds,
    )
    details["history_source"] = source
    return details


def _evaluate_trade_promotion_governor(*, payload, runtime_thresholds):
    payload = payload if isinstance(payload, dict) else {}
    if not bool(int(_safe_float(runtime_thresholds.get("enable_trade_promotion_governor"), 1.0))):
        return {
            "enabled": False,
            "verdict": "DISABLED",
            "reason": "guard_disabled",
            "promotion_state": "DISABLED",
            "replay_validation_required": False,
            "size_cap": 1.0,
        }

    trade_strength = _safe_float(payload.get("trade_strength"), 0.0)
    runtime_composite_score = _safe_float(payload.get("runtime_composite_score"), 0.0)
    success_probability = _safe_float(
        payload.get("signal_success_probability"),
        _safe_float(payload.get("hybrid_move_probability"), 0.0),
    )
    confirmation_status = _as_upper(payload.get("confirmation_status"))
    data_quality_status = _as_upper(payload.get("data_quality_status"))
    provider_health_summary = _as_upper(payload.get("provider_health_summary"))
    global_risk_state = _as_upper(payload.get("global_risk_state"))
    option_efficiency_status = _as_upper(payload.get("option_efficiency_status"))
    reversal_stage = _as_upper(payload.get("reversal_stage"))
    fast_reversal_alert_level = _as_upper(payload.get("fast_reversal_alert_level"))
    ranked_strike_proxy_ratio = _clip(_safe_float(payload.get("ranked_strike_proxy_ratio"), 0.0), 0.0, 1.0)
    live_calibration_gate = payload.get("live_calibration_gate") if isinstance(payload.get("live_calibration_gate"), dict) else {}
    live_directional_gate = payload.get("live_directional_gate") if isinstance(payload.get("live_directional_gate"), dict) else {}

    min_trade_strength = int(_safe_float(runtime_thresholds.get("min_trade_strength"), 62.0))
    min_composite_score = int(_safe_float(runtime_thresholds.get("min_composite_score"), 58.0))
    min_probability = _clip(_safe_float(runtime_thresholds.get("trade_promotion_min_probability"), 0.60), 0.0, 1.0)
    caution_size_cap = _clip(_safe_float(runtime_thresholds.get("trade_promotion_caution_size_cap"), 0.50), 0.0, 1.0)
    hold_cap_minutes = max(5, int(_safe_float(runtime_thresholds.get("trade_promotion_hold_cap_minutes"), 20.0)))
    require_confirmed = bool(int(_safe_float(runtime_thresholds.get("trade_promotion_require_confirmed_status"), 1.0)))

    fragile_context = (
        provider_health_summary in {"WEAK", "CAUTION"}
        and global_risk_state in {"RISK_OFF", "VOL_SHOCK", "EVENT_LOCKDOWN"}
    )
    if fragile_context:
        min_probability = max(
            min_probability,
            _clip(_safe_float(runtime_thresholds.get("trade_promotion_fragile_context_min_probability"), 0.68), 0.0, 1.0),
        )
        caution_size_cap = min(
            caution_size_cap,
            _clip(_safe_float(runtime_thresholds.get("trade_promotion_fragile_context_size_cap"), 0.35), 0.0, 1.0),
        )
        hold_cap_minutes = min(
            hold_cap_minutes,
            max(5, int(_safe_float(runtime_thresholds.get("trade_promotion_fragile_context_hold_cap_minutes"), 15.0))),
        )

    replay_required_reasons = []
    if trade_strength < min_trade_strength + 4:
        replay_required_reasons.append("trade_strength_near_gate")
    if runtime_composite_score < min_composite_score + 5:
        replay_required_reasons.append("runtime_composite_near_gate")
    if success_probability < min_probability:
        replay_required_reasons.append("success_probability_below_promotion_floor")
    if require_confirmed and confirmation_status not in {"CONFIRMED", "STRONG_CONFIRMATION"}:
        replay_required_reasons.append("confirmation_not_strong_enough")
    if bool(int(_safe_float(runtime_thresholds.get("trade_promotion_block_early_reversal"), 1.0))) and reversal_stage in {"EARLY_REVERSAL_CANDIDATE", "REVERSAL_UNRESOLVED"}:
        replay_required_reasons.append("early_reversal_candidate_requires_wait")
    if (
        bool(int(_safe_float(runtime_thresholds.get("trade_promotion_early_reversal_requires_strong_confirmation"), 1.0)))
        and fast_reversal_alert_level == "HIGH"
        and reversal_stage != "CONFIRMED_REVERSAL"
    ):
        replay_required_reasons.append("fast_reversal_warning_requires_wait_for_confirmation")
    if fragile_context and confirmation_status != "STRONG_CONFIRMATION":
        replay_required_reasons.append("risk_off_weak_provider_requires_strong_confirmation")
    if data_quality_status in {"CAUTION", "WEAK"}:
        replay_required_reasons.append("data_quality_not_clean")
    if fragile_context and success_probability < min_probability:
        replay_required_reasons.append("risk_off_weak_provider_probability_below_floor")
    if fragile_context and trade_strength < min_trade_strength + 6:
        replay_required_reasons.append("risk_off_weak_provider_trade_strength_buffer_not_met")
    if fragile_context and runtime_composite_score < min_composite_score + 6:
        replay_required_reasons.append("risk_off_weak_provider_composite_buffer_not_met")
    if _as_upper(live_calibration_gate.get("verdict")) == "BLOCK":
        replay_required_reasons.append("live_calibration_blocked")
    if _as_upper(live_directional_gate.get("verdict")) == "BLOCK":
        replay_required_reasons.append("live_directional_blocked")
    if option_efficiency_status.startswith("UNAVAILABLE"):
        replay_required_reasons.append("option_efficiency_unavailable")
    if ranked_strike_proxy_ratio > _clip(_safe_float(runtime_thresholds.get("trade_promotion_proxy_ratio_max"), 0.50), 0.0, 1.0):
        replay_required_reasons.append("proxy_ratio_above_promotion_cap")

    if replay_required_reasons:
        return {
            "enabled": True,
            "verdict": "BLOCK",
            "reason": "Trade promotion governor: replay validation is required before live promotion",
            "promotion_state": "REPLAY_REQUIRED",
            "replay_validation_required": True,
            "reasons": replay_required_reasons,
            "size_cap": caution_size_cap,
            "hold_cap_minutes": hold_cap_minutes,
        }

    return {
        "enabled": True,
        "verdict": "PASS",
        "reason": "Promotion evidence is sufficient for live TRADE routing",
        "promotion_state": "PROMOTE",
        "replay_validation_required": False,
        "reasons": [],
        "size_cap": 1.0,
        "hold_cap_minutes": None,
    }


def _compute_portfolio_concentration_context(*, payload, runtime_thresholds):
    import pandas as _pd

    context = {
        "enabled": bool(int(_safe_float(runtime_thresholds.get("enable_portfolio_concentration_guard"), 1.0))),
        "history_source": None,
        "reason": None,
        "symbol": _as_upper((payload or {}).get("symbol")),
        "direction": _as_upper((payload or {}).get("direction")),
        "gamma_regime": _as_upper((payload or {}).get("gamma_regime")),
        "vol_regime": _as_upper((payload or {}).get("volatility_regime") or (payload or {}).get("vol_regime")),
        "macro_regime": _as_upper((payload or {}).get("macro_regime")),
        "provider_health_summary": _as_upper((payload or {}).get("provider_health_summary")),
        "data_quality_status": _as_upper((payload or {}).get("data_quality_status")),
        "recent_signal_count": 0,
        "same_direction_count": 0,
        "same_direction_share": 0.0,
        "same_direction_avg_close_bps": None,
        "same_direction_avg_tradeability_score": None,
    }
    if not context["enabled"]:
        context["reason"] = "guard_disabled"
        return context
    if context["direction"] not in {"CALL", "PUT"}:
        context["reason"] = "direction_unavailable"
        return context

    history_frame, source = _load_recent_outcome_history_frame()
    context["history_source"] = source
    if history_frame is None or getattr(history_frame, "empty", True):
        context["reason"] = "history_unavailable"
        return context

    frame = history_frame.copy()
    if "direction" not in frame.columns:
        context["reason"] = "direction_history_unavailable"
        return context

    if "symbol" in frame.columns:
        frame["symbol_norm"] = frame["symbol"].astype(str).str.upper().str.strip()
    else:
        frame["symbol_norm"] = ""
    frame["direction_norm"] = frame["direction"].astype(str).str.upper().str.strip()
    frame = frame[frame["direction_norm"].isin(["CALL", "PUT"])]

    if context["symbol"]:
        symbol_slice = frame[frame["symbol_norm"] == context["symbol"]]
        if not symbol_slice.empty:
            frame = symbol_slice

    valuation_time = _coerce_timestamp((payload or {}).get("valuation_time") or (payload or {}).get("as_of"))
    timestamp_col = None
    for candidate in ["timestamp", "as_of", "valuation_time", "signal_timestamp"]:
        if candidate in frame.columns:
            timestamp_col = candidate
            frame[candidate] = _pd.to_datetime(frame[candidate], errors="coerce")
            break

    if timestamp_col is not None:
        if valuation_time is not None:
            day_subset = frame[frame[timestamp_col].dt.date == valuation_time.date()]
            if not day_subset.empty:
                frame = day_subset
        else:
            valid_times = frame[timestamp_col].dropna()
            if not valid_times.empty:
                latest_date = valid_times.dt.date.max()
                if latest_date is not None:
                    latest_subset = frame[frame[timestamp_col].dt.date == latest_date]
                    if not latest_subset.empty:
                        frame = latest_subset

    if "trade_status" in frame.columns:
        active_statuses = {"TRADE", "EXECUTED", "FILLED", "OPEN", "DEGRADED_PROVIDER_TRADE"}
        frame["trade_status_norm"] = frame["trade_status"].astype(str).str.upper().str.strip()
        frame = frame[frame["trade_status_norm"].isin(active_statuses)]
        if frame.empty:
            context["reason"] = "no_recent_executed_signals"
            return context

    lookback = max(3, int(_safe_float(runtime_thresholds.get("portfolio_concentration_lookback_signals"), 6.0)))
    recent = frame.tail(lookback).copy()
    context["recent_signal_count"] = int(len(recent))
    if recent.empty:
        context["reason"] = "no_recent_signals"
        return context

    same_direction = recent[recent["direction_norm"] == context["direction"]]
    same_count = int(len(same_direction))
    context["same_direction_count"] = same_count
    context["same_direction_share"] = round(float(same_count / len(recent)), 4) if len(recent) else 0.0

    if "signed_return_session_close_bps" in same_direction.columns:
        close_bps = _pd.to_numeric(same_direction["signed_return_session_close_bps"], errors="coerce")
        if close_bps.notna().any():
            context["same_direction_avg_close_bps"] = round(float(close_bps.dropna().mean()), 2)

    if "tradeability_score" in same_direction.columns:
        tradeability = _pd.to_numeric(same_direction["tradeability_score"], errors="coerce")
        if tradeability.notna().any():
            context["same_direction_avg_tradeability_score"] = round(float(tradeability.dropna().mean()), 2)

    context["reason"] = "ok"
    return context


def _apply_intraday_exit_bias(
    *,
    recommended_hold_minutes,
    max_hold_minutes,
    runtime_thresholds,
    expansion_mode,
    expansion_direction,
    direction,
    reversal_stage,
    provider_health_summary,
    data_quality_status,
    global_risk_state,
    gamma_regime,
):
    """Apply report-driven intraday exit caps for fragile live contexts."""
    runtime_thresholds = runtime_thresholds if isinstance(runtime_thresholds, dict) else {}
    recommended = max(5, int(_safe_float(recommended_hold_minutes, 30.0)))
    hold_max = max(recommended, int(_safe_float(max_hold_minutes, recommended)))
    reasons: list[str] = []

    preferred_cap = max(10, int(_safe_float(runtime_thresholds.get("preferred_intraday_exit_cap_minutes"), 75.0)))
    if not (expansion_mode and expansion_direction == direction):
        if recommended > preferred_cap or hold_max > preferred_cap:
            recommended = min(recommended, preferred_cap)
            hold_max = min(hold_max, preferred_cap)
            reasons.append("intraday_exit_bias_preferred_window")

    provider_upper = _as_upper(provider_health_summary)
    data_upper = _as_upper(data_quality_status)
    risk_upper = _as_upper(global_risk_state)
    gamma_upper = canonical_gamma_regime(gamma_regime)

    if risk_upper in {"RISK_OFF", "VOL_SHOCK", "EVENT_LOCKDOWN"} and gamma_upper == "POSITIVE_GAMMA":
        risk_off_cap = max(10, int(_safe_float(runtime_thresholds.get("positive_gamma_risk_off_hold_cap_minutes"), 60.0)))
        if recommended > risk_off_cap or hold_max > risk_off_cap:
            recommended = min(recommended, risk_off_cap)
            hold_max = min(hold_max, risk_off_cap)
            reasons.append("risk_off_positive_gamma_exit_bias")

    fragile_context = provider_upper in {"WEAK", "CAUTION"} and data_upper in {"WEAK", "CAUTION"}
    if fragile_context and risk_upper in {"RISK_OFF", "VOL_SHOCK", "EVENT_LOCKDOWN"}:
        fragile_cap = max(10, int(_safe_float(runtime_thresholds.get("fragile_risk_off_hold_cap_minutes"), 45.0)))
        if recommended > fragile_cap or hold_max > fragile_cap:
            recommended = min(recommended, fragile_cap)
            hold_max = min(hold_max, fragile_cap)
            reasons.append("fragile_risk_off_exit_bias")

    if _as_upper(reversal_stage) == "EARLY_REVERSAL_CANDIDATE" and "reversal_stage_early_hold_cap" not in reasons:
        reasons.append("reversal_stage_early_hold_cap")

    return recommended, hold_max, reasons


def _derive_advisory_size_recommendation(
    payload,
    *,
    confidence_score,
    global_risk_size_cap,
    at_flip_size_cap,
    macro_size_multiplier,
    freshness_size_cap,
    reversal_stage,
    expansion_mode,
    expansion_direction,
    direction,
    runtime_thresholds,
    regime_thresholds,
    gamma_regime,
):
    """Compute sizing as a pure recommendation without mutating the signal payload."""
    payload = payload if isinstance(payload, dict) else {}
    runtime_thresholds = runtime_thresholds if isinstance(runtime_thresholds, dict) else {}
    base_lots = max(int(_safe_float(payload.get("number_of_lots"), 0.0)), 0)

    confidence_value = _safe_float(confidence_score, 50.0)
    if confidence_value < 30.0:
        confidence_size_mult = 0.25
    elif confidence_value < 45.0:
        confidence_size_mult = 0.50
    elif confidence_value < 55.0:
        confidence_size_mult = 0.75
    elif confidence_value < 70.0:
        confidence_size_mult = 1.00
    else:
        confidence_size_mult = 1.25

    risk_size_cap = min(
        _safe_float(global_risk_size_cap, 1.0),
        _safe_float(at_flip_size_cap, 1.0),
        _safe_float(macro_size_multiplier, 1.0),
        _safe_float(freshness_size_cap, 1.0),
    )
    risk_size_cap *= confidence_size_mult

    if reversal_stage == "EARLY_REVERSAL_CANDIDATE":
        risk_size_cap *= _safe_float(runtime_thresholds.get("reversal_stage_early_size_mult"), 0.60)
    elif reversal_stage == "CONFIRMED_REVERSAL":
        risk_size_cap *= _safe_float(runtime_thresholds.get("reversal_stage_confirmed_size_mult"), 1.00)

    if expansion_mode and expansion_direction == direction:
        risk_size_cap *= _safe_float(runtime_thresholds.get("expansion_mode_size_mult"), 1.10)

    if bool(int(_safe_float(runtime_thresholds.get("enable_regime_conditional_thresholds"), 1.0))):
        risk_size_cap *= _safe_float((regime_thresholds or {}).get("position_size_multiplier"), 1.0)
    else:
        gamma_regime_upper = canonical_gamma_regime(gamma_regime)
        if gamma_regime_upper == "POSITIVE_GAMMA":
            risk_size_cap *= _safe_float(runtime_thresholds.get("positive_gamma_size_multiplier"), 0.85)
        elif gamma_regime_upper == "NEGATIVE_GAMMA":
            risk_size_cap *= _safe_float(runtime_thresholds.get("negative_gamma_size_multiplier"), 1.15)

    heat_score = _safe_float(
        payload.get("portfolio_book_heat_score"),
        _safe_float((payload.get("portfolio_concentration_guard") or {}).get("heat_score"), 0.0),
    )
    heat_label = _as_upper(
        payload.get("portfolio_book_heat_label")
        or (payload.get("portfolio_concentration_guard") or {}).get("heat_label")
        or "COOL"
    )
    trade_strength = _safe_float(payload.get("trade_strength"), 50.0)
    success_probability = _safe_float(
        payload.get("signal_success_probability"),
        _safe_float(payload.get("hybrid_move_probability"), 0.50),
    )
    success_probability = float(_clip(success_probability, 0.0, 1.0))

    entry_price = _safe_float(payload.get("entry_price"), _safe_float(payload.get("selected_option_last_price"), None))
    spot_value = _safe_float(payload.get("spot"), None)
    premium_pct_of_spot = _safe_float(payload.get("option_premium_pct_of_spot"), None)
    if premium_pct_of_spot is None and entry_price not in (None, 0.0) and spot_value not in (None, 0.0):
        premium_pct_of_spot = round((entry_price / spot_value) * 100.0, 4)
    premium_efficiency_score = _safe_float(payload.get("premium_efficiency_score"), 50.0)

    priority_score = (
        0.40 * trade_strength
        + 0.30 * confidence_value
        + 0.20 * (success_probability * 100.0)
        + 0.10 * (100.0 - heat_score)
    )
    if heat_label == "COOL":
        priority_score += _safe_float(runtime_thresholds.get("portfolio_heat_cool_priority_bonus"), 4.0)
    elif heat_label == "HOT":
        priority_score -= _safe_float(runtime_thresholds.get("portfolio_heat_hot_priority_penalty"), 8.0)
    elif heat_label == "CRITICAL":
        priority_score -= _safe_float(runtime_thresholds.get("portfolio_heat_critical_priority_penalty"), 16.0)

    premium_priority_adjustment = 0.0
    premium_size_cap = 1.0
    low_premium_threshold = _safe_float(runtime_thresholds.get("premium_preference_low_pct_of_spot"), 0.50)
    high_premium_threshold = _safe_float(runtime_thresholds.get("premium_preference_high_pct_of_spot"), 1.00)
    if premium_pct_of_spot is not None:
        if premium_pct_of_spot <= low_premium_threshold:
            premium_priority_adjustment += _safe_float(runtime_thresholds.get("premium_preference_bonus"), 6.0)
        elif premium_pct_of_spot > high_premium_threshold:
            premium_priority_adjustment -= _safe_float(runtime_thresholds.get("premium_expense_priority_penalty"), 8.0)
            premium_size_cap = min(
                premium_size_cap,
                _clip(_safe_float(runtime_thresholds.get("premium_expense_size_cap"), 0.65), 0.0, 1.0),
            )

    premium_efficiency_low = _safe_float(runtime_thresholds.get("premium_efficiency_low_threshold"), 55.0)
    premium_efficiency_high = _safe_float(runtime_thresholds.get("premium_efficiency_high_threshold"), 75.0)
    if premium_efficiency_score >= premium_efficiency_high:
        premium_priority_adjustment += _safe_float(runtime_thresholds.get("premium_efficiency_priority_bonus"), 3.0)
    elif premium_efficiency_score < premium_efficiency_low:
        premium_priority_adjustment -= _safe_float(runtime_thresholds.get("premium_efficiency_priority_penalty"), 6.0)
        premium_size_cap = min(premium_size_cap, 0.75)

    priority_score += premium_priority_adjustment
    priority_score = float(_clip(priority_score, 0.0, 100.0))

    high_threshold = _safe_float(runtime_thresholds.get("portfolio_priority_high_threshold"), 75.0)
    medium_threshold = _safe_float(runtime_thresholds.get("portfolio_priority_medium_threshold"), 60.0)
    low_threshold = _safe_float(runtime_thresholds.get("portfolio_priority_low_threshold"), 45.0)

    if bool(int(_safe_float(runtime_thresholds.get("enable_portfolio_allocation_ladder"), 1.0))) is False:
        priority_bucket = "STANDARD_PRIORITY"
        allocation_tier = "STANDARD"
        capital_fraction_max = 0.10
        allocation_multiplier = 1.0
    elif priority_score >= high_threshold:
        priority_bucket = "HIGH_PRIORITY"
        allocation_tier = "CORE"
        capital_fraction_max = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_core_fraction_max"), 0.25), 0.0, 1.0)
        allocation_multiplier = 1.0
    elif priority_score >= medium_threshold:
        priority_bucket = "MEDIUM_PRIORITY"
        allocation_tier = "TACTICAL"
        capital_fraction_max = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_tactical_fraction_max"), 0.15), 0.0, 1.0)
        allocation_multiplier = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_medium_mult"), 0.75), 0.0, 1.0)
    elif priority_score >= low_threshold:
        priority_bucket = "LOW_PRIORITY"
        allocation_tier = "SMALL"
        capital_fraction_max = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_small_fraction_max"), 0.08), 0.0, 1.0)
        allocation_multiplier = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_low_mult"), 0.50), 0.0, 1.0)
    else:
        priority_bucket = "DEFER"
        allocation_tier = "PROBE"
        capital_fraction_max = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_probe_fraction_max"), 0.04), 0.0, 1.0)
        allocation_multiplier = _clip(_safe_float(runtime_thresholds.get("portfolio_allocation_probe_mult"), 0.25), 0.0, 1.0)

    risk_size_cap *= allocation_multiplier
    risk_size_cap = min(risk_size_cap, premium_size_cap)
    if heat_label == "HOT":
        risk_size_cap = min(
            risk_size_cap,
            _clip(_safe_float(runtime_thresholds.get("portfolio_heat_hot_allocation_cap"), 0.55), 0.0, 1.0),
        )
    elif heat_label == "CRITICAL":
        risk_size_cap = min(
            risk_size_cap,
            _clip(_safe_float(runtime_thresholds.get("portfolio_heat_critical_allocation_cap"), 0.35), 0.0, 1.0),
        )
    risk_size_cap = float(_clip(risk_size_cap, 0.0, 1.0))
    advisory_lots = max(0, int(base_lots * risk_size_cap))
    if base_lots > 0 and advisory_lots == 0 and risk_size_cap > 0:
        advisory_lots = 1

    return {
        "advisory_only": True,
        "confidence_size_multiplier": round(float(confidence_size_mult), 4),
        "advisory_position_size_multiplier": round(float(risk_size_cap), 4),
        "effective_size_cap": round(float(risk_size_cap), 2),
        "advisory_lots": advisory_lots,
        "macro_size_applied": advisory_lots > 0 and advisory_lots < base_lots,
        "portfolio_priority_score": round(float(priority_score), 2),
        "portfolio_priority_bucket": priority_bucket,
        "portfolio_allocation_tier": allocation_tier,
        "portfolio_capital_fraction_max": round(float(capital_fraction_max), 4),
        "portfolio_heat_score": round(float(heat_score), 2),
        "portfolio_heat_label": heat_label,
        "premium_load_pct_of_spot": round(float(premium_pct_of_spot), 4) if premium_pct_of_spot is not None else None,
        "premium_priority_adjustment": round(float(premium_priority_adjustment), 2),
        "premium_size_cap": round(float(premium_size_cap), 4),
    }


def _apply_bearish_bias_threshold_adjustments(
    *,
    runtime_thresholds,
    direction,
    gamma_regime,
    vol_regime,
    base_min_trade_strength,
    base_min_composite_score,
):
    enabled = bool(int(_safe_float(runtime_thresholds.get("enable_bearish_bias_guard"), 1.0)))
    direction_upper = _as_upper(direction)
    gamma_upper = canonical_gamma_regime(gamma_regime)
    vol_upper = _canonical_vol_regime(vol_regime)

    details = {
        "enabled": enabled,
        "applied": False,
        "context": {
            "direction": direction_upper,
            "gamma_regime": gamma_upper,
            "vol_regime": vol_upper,
        },
    }
    min_trade_strength = int(base_min_trade_strength)
    min_composite_score = int(base_min_composite_score)

    if not enabled:
        details["reason"] = "disabled"
        return min_trade_strength, min_composite_score, details

    context_match = (
        direction_upper == "PUT"
        and gamma_upper == "NEGATIVE_GAMMA"
        and vol_upper == "VOL_EXPANSION"
    )
    if not context_match:
        details["reason"] = "context_not_matched"
        return min_trade_strength, min_composite_score, details

    composite_add = int(_safe_float(runtime_thresholds.get("bearish_bias_guard_composite_add"), 3.0))
    strength_add = int(_safe_float(runtime_thresholds.get("bearish_bias_guard_strength_add"), 2.0))
    size_cap = _clip(_safe_float(runtime_thresholds.get("bearish_bias_guard_size_cap"), 0.70), 0.0, 1.0)

    min_trade_strength = int(_clip(min_trade_strength + strength_add, 0, 100))
    min_composite_score = int(_clip(min_composite_score + composite_add, 0, 100))
    details.update(
        {
            "applied": True,
            "composite_add": composite_add,
            "strength_add": strength_add,
            "size_cap": round(float(size_cap), 4),
        }
    )
    return min_trade_strength, min_composite_score, details


def _evaluate_weak_data_circuit_breaker(
    *,
    runtime_thresholds,
    data_quality_status,
    provider_health_summary,
    confirmation_status,
    adjusted_trade_strength,
    runtime_composite_score,
    ranked_strikes,
    direction,
    gamma_regime,
    vol_regime,
    option_efficiency_status=None,
    live_calibration_gate=None,
    live_directional_gate=None,
    global_risk_state=None,
):
    enabled = bool(int(_safe_float(runtime_thresholds.get("enable_weak_data_circuit_breaker"), 1.0)))
    live_calibration_gate = live_calibration_gate if isinstance(live_calibration_gate, dict) else {}
    live_directional_gate = live_directional_gate if isinstance(live_directional_gate, dict) else {}
    global_risk_state_upper = _as_upper(
        global_risk_state.get("global_risk_state") if isinstance(global_risk_state, dict) else global_risk_state
    )
    details = {
        "enabled": enabled,
        "triggered": False,
        "trigger_reasons": [],
        "scope_override_reasons": [],
        "data_quality_status": _as_upper(data_quality_status),
        "provider_health_summary": _as_upper(provider_health_summary),
        "confirmation_status": _as_upper(confirmation_status),
        "direction": _as_upper(direction),
        "gamma_regime": canonical_gamma_regime(gamma_regime),
        "vol_regime": _canonical_vol_regime(vol_regime),
        "option_efficiency_status": _as_upper(option_efficiency_status),
        "live_calibration_verdict": _as_upper(live_calibration_gate.get("verdict")),
        "live_directional_verdict": _as_upper(live_directional_gate.get("verdict")),
        "global_risk_state": global_risk_state_upper,
    }
    if not enabled:
        details["reason"] = "disabled"
        return False, details

    watch_statuses = _normalize_string_set(
        runtime_thresholds.get("weak_data_circuit_breaker_data_quality_statuses"),
        default={"WEAK", "CAUTION"},
    )

    min_strength = int(_safe_float(runtime_thresholds.get("weak_data_circuit_breaker_min_trade_strength"), 74.0))
    min_runtime_score = int(_safe_float(runtime_thresholds.get("weak_data_circuit_breaker_min_runtime_composite_score"), 70.0))
    max_proxy_ratio = _clip(_safe_float(runtime_thresholds.get("weak_data_circuit_breaker_max_proxy_ratio"), 0.35), 0.0, 1.0)
    min_trigger_count = max(int(_safe_float(runtime_thresholds.get("weak_data_circuit_breaker_min_trigger_count"), 2.0)), 1)

    provider_statuses = _normalize_string_set(
        runtime_thresholds.get("weak_data_circuit_breaker_provider_statuses"),
        default={"WEAK", "CAUTION"},
    )
    require_strong_confirmation = bool(
        int(_safe_float(runtime_thresholds.get("weak_data_circuit_breaker_require_strong_confirmation"), 1.0))
    )
    proxy_ratio = _ranked_strike_proxy_ratio(ranked_strikes)
    details["proxy_ratio"] = round(float(proxy_ratio), 4)
    details["runtime_composite_score"] = int(runtime_composite_score)
    details["trade_strength"] = int(adjusted_trade_strength)

    hidden_fragility_min_count = max(
        int(_safe_float(runtime_thresholds.get("weak_data_circuit_breaker_hidden_fragility_min_trigger_count"), 3.0)),
        1,
    )
    if details["data_quality_status"] not in watch_statuses:
        if details["option_efficiency_status"].startswith("UNAVAILABLE"):
            details["scope_override_reasons"].append("option_efficiency_unavailable")
        if proxy_ratio > max_proxy_ratio:
            details["scope_override_reasons"].append("proxy_ratio_above_cap")
        if details["live_calibration_verdict"] == "BLOCK":
            details["scope_override_reasons"].append("live_calibration_block")
        if details["live_directional_verdict"] == "BLOCK":
            details["scope_override_reasons"].append("live_directional_block")
        if details["global_risk_state"] in {"RISK_OFF", "VOL_SHOCK", "EVENT_LOCKDOWN"}:
            details["scope_override_reasons"].append("risk_off_context")
        if len(details["scope_override_reasons"]) < hidden_fragility_min_count:
            details["reason"] = "data_quality_not_in_breaker_scope"
            return False, details
        details["reason"] = "hidden_fragility_override_scope"
        details["trigger_reasons"].extend(details["scope_override_reasons"])

    if details["provider_health_summary"] in provider_statuses:
        details["trigger_reasons"].append("provider_health_fragile")
    if require_strong_confirmation and details["confirmation_status"] not in {"STRONG_CONFIRMATION", "CONFIRMED"}:
        details["trigger_reasons"].append("confirmation_not_strong")
    if int(adjusted_trade_strength) < min_strength:
        details["trigger_reasons"].append("trade_strength_below_floor")
    if int(runtime_composite_score) < min_runtime_score:
        details["trigger_reasons"].append("runtime_composite_below_floor")
    if proxy_ratio > max_proxy_ratio and "proxy_ratio_above_cap" not in details["trigger_reasons"]:
        details["trigger_reasons"].append("proxy_ratio_above_cap")
    if details["option_efficiency_status"].startswith("UNAVAILABLE") and "option_efficiency_unavailable" not in details["trigger_reasons"]:
        details["trigger_reasons"].append("option_efficiency_unavailable")
    if details["live_calibration_verdict"] == "BLOCK" and "live_calibration_block" not in details["trigger_reasons"]:
        details["trigger_reasons"].append("live_calibration_block")
    if details["live_directional_verdict"] == "BLOCK" and "live_directional_block" not in details["trigger_reasons"]:
        details["trigger_reasons"].append("live_directional_block")
    if details["global_risk_state"] in {"RISK_OFF", "VOL_SHOCK", "EVENT_LOCKDOWN"} and "risk_off_context" not in details["trigger_reasons"]:
        details["trigger_reasons"].append("risk_off_context")
    if (
        details["direction"] == "PUT"
        and details["gamma_regime"] == "NEGATIVE_GAMMA"
        and details["vol_regime"] == "VOL_EXPANSION"
    ):
        details["trigger_reasons"].append("put_toxic_regime_context")

    details["trigger_count"] = int(len(details["trigger_reasons"]))
    details["min_trigger_count"] = int(min_trigger_count)
    details["triggered"] = bool(details["trigger_count"] >= min_trigger_count)
    return details["triggered"], details


def _state_bucket_key(symbol, selected_expiry, ts):
    """Namespace state by trading date to reduce cross-session contamination."""
    date_bucket = "NO_DATE"
    try:
        if ts is not None:
            date_bucket = str(ts.date())
    except _SIGNAL_STATE_ERRORS:
        date_bucket = "NO_DATE"
    return f"{symbol}:{selected_expiry or 'NO_EXPIRY'}:{date_bucket}"


def _prune_state_dict(state_dict: dict, now_ts, max_age_minutes: float = 24 * 60) -> None:
    """Evict entries from a state dict whose last_ts is older than max_age_minutes.

    Uses a single-pass dict comprehension rather than a two-pass collect-then-
    delete loop, halving the number of iterations over the dict.
    """
    def _age(entry) -> float:
        last_ts = entry.get("last_ts")
        try:
            return (now_ts - last_ts).total_seconds() / 60.0
        except _SIGNAL_STATE_ERRORS:
            return 0.0

    stale = {k for k, v in state_dict.items() if _age(v) > max_age_minutes}
    for k in stale:
        state_dict.pop(k, None)


def _compute_signal_elapsed_minutes(*, symbol, selected_expiry, valuation_time, direction):
    """Track signal age (minutes) by symbol+expiry and direction for time-decay."""
    if _as_upper(direction) not in {"CALL", "PUT"}:
        return 0.0

    ts = _coerce_timestamp(valuation_time)
    if ts is None:
        return 0.0

    key = _state_bucket_key(symbol, selected_expiry, ts)
    state = _DECAY_SIGNAL_STATE.get(key) or {}
    prev_direction = _as_upper(state.get("direction"))
    start_ts = state.get("start_ts")

    # Reset age when direction flips or we see this key for the first time.
    if prev_direction != _as_upper(direction) or start_ts is None:
        start_ts = ts

    elapsed_minutes = 0.0
    try:
        elapsed_minutes = max(0.0, float((ts - start_ts).total_seconds() / 60.0))
    except _SIGNAL_STATE_ERRORS as exc:
        _LOG.warning(
            "signal_engine: resetting elapsed-time state for %s after invalid timestamp arithmetic (valuation_time=%r, start_ts=%r): %s",
            key,
            valuation_time,
            start_ts,
            exc,
        )
        elapsed_minutes = 0.0
        start_ts = ts

    _DECAY_SIGNAL_STATE[key] = {
        "direction": _as_upper(direction),
        "start_ts": start_ts,
        "last_ts": ts,
    }

    # Prune stale entries opportunistically to keep in-memory state bounded.
    if len(_DECAY_SIGNAL_STATE) > 512:
        _prune_state_dict(_DECAY_SIGNAL_STATE, ts)

    return elapsed_minutes


def _compute_path_observation_bps(*, symbol, selected_expiry, valuation_time, spot, direction):
    """Track cumulative MFE/MAE (bps) from signal entry.

    MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) are
    cumulative extremes measured from the entry spot recorded on the first
    observed tick for this signal bucket.  A single-tick snapshot delta is
    **not** sufficient — the PathAwareFilter needs to know the furthest the
    market has travelled in each direction since the signal was raised.

    State stored per bucket key:
        entry_spot   – spot at the first tick (anchor for bps calculation)
        high_water   – running session high
        low_water    – running session low
        last_spot    – previous tick spot (kept for compatibility/debugging)
        last_ts      – previous tick timestamp
    """
    if _as_upper(direction) not in {"CALL", "PUT"}:
        return None, None

    ts = _coerce_timestamp(valuation_time)
    spot_now = _safe_float(spot, None)
    if ts is None or spot_now is None or spot_now <= 0:
        return None, None

    key = _state_bucket_key(symbol, selected_expiry, ts)
    state = _PATH_SIGNAL_STATE.get(key) or {}

    entry_spot = _safe_float(state.get("entry_spot"), None)
    high_water = _safe_float(state.get("high_water"), None)
    low_water = _safe_float(state.get("low_water"), None)
    last_ts = state.get("last_ts")

    # On the first tick there is no history yet — initialise anchors.
    first_tick = entry_spot is None or high_water is None or low_water is None or last_ts is None
    if first_tick:
        _PATH_SIGNAL_STATE[key] = {
            "entry_spot": spot_now,
            "high_water": spot_now,
            "low_water": spot_now,
            "last_spot": spot_now,
            "last_ts": ts,
        }
        if len(_PATH_SIGNAL_STATE) > 512:
            _prune_state_dict(_PATH_SIGNAL_STATE, ts)
        return None, None

    # Update running extremes.
    high_water = max(high_water, spot_now)
    low_water = min(low_water, spot_now)

    _PATH_SIGNAL_STATE[key] = {
        "entry_spot": entry_spot,
        "high_water": high_water,
        "low_water": low_water,
        "last_spot": spot_now,
        "last_ts": ts,
    }

    if len(_PATH_SIGNAL_STATE) > 512:
        _prune_state_dict(_PATH_SIGNAL_STATE, ts)

    # Compute MFE/MAE in basis points relative to entry.
    if _as_upper(direction) == "CALL":
        # For a long-call position: up move is favorable.
        mfe_bps = ((high_water - entry_spot) / entry_spot) * 10000.0
        mae_bps = ((low_water - entry_spot) / entry_spot) * 10000.0
    else:
        # For a long-put position: down move is favorable.
        mfe_bps = -((low_water - entry_spot) / entry_spot) * 10000.0
        mae_bps = -((high_water - entry_spot) / entry_spot) * 10000.0

    return float(mfe_bps), float(mae_bps)


def _get_path_filter():
    global _PATH_FILTER
    if _PATH_FILTER is None:
        _PATH_FILTER = PathAwareFilter(pattern_library=PathPatternLibrary())
    return _PATH_FILTER


def _canonical_vol_regime(value):
    txt = _as_upper(value)
    if txt in {"VOL_EXPANSION", "HIGH_VOL", "SHOCK_VOL", "VOLATILE"}:
        return "VOL_EXPANSION"
    if txt in {"VOL_CONTRACTION", "LOW_VOL", "COMPRESSED_VOL"}:
        return "VOL_CONTRACTION"
    return "NORMAL_VOL"


def _ensure_time_decay_model_config(runtime_thresholds):
    global _TIME_DECAY_MODEL_CONFIG_KEY
    cfg_key = (
        _safe_float(runtime_thresholds.get("time_decay_positive_gamma_half_life_m"), 90.0),
        _safe_float(runtime_thresholds.get("time_decay_negative_gamma_half_life_m"), 45.0),
        _safe_float(runtime_thresholds.get("time_decay_neutral_gamma_half_life_m"), 70.0),
        _safe_float(runtime_thresholds.get("time_decay_lambda"), 1.5),
    )
    if _TIME_DECAY_MODEL_CONFIG_KEY == cfg_key:
        return

    initialize_time_decay(
        positive_gamma_half_life_m=cfg_key[0],
        negative_gamma_half_life_m=cfg_key[1],
        neutral_gamma_half_life_m=cfg_key[2],
        steepness=max(0.1, _safe_float(cfg_key[3], 1.5)),
    )
    _TIME_DECAY_MODEL_CONFIG_KEY = cfg_key


def _ensure_regime_thresholds_config(runtime_thresholds, base_min_trade_strength, base_min_composite_score):
    global _REGIME_THRESHOLDS_CONFIG_KEY
    cfg_key = (
        int(_safe_float(base_min_composite_score, 55.0)),
        int(_safe_float(base_min_trade_strength, 62.0)),
        int(_safe_float(runtime_thresholds.get("max_intraday_hold_minutes"), 90.0)),
        1.0,
        int(_safe_float(runtime_thresholds.get("regime_positive_gamma_composite_delta"), -3.0)),
        int(_safe_float(runtime_thresholds.get("regime_positive_gamma_strength_delta"), -2.0)),
        int(_safe_float(runtime_thresholds.get("regime_positive_gamma_holding_delta_m"), 60.0)),
        _safe_float(runtime_thresholds.get("regime_positive_gamma_position_size_mult"), 1.2),
        int(_safe_float(runtime_thresholds.get("regime_negative_gamma_composite_delta"), 5.0)),
        int(_safe_float(runtime_thresholds.get("regime_negative_gamma_strength_delta"), 3.0)),
        int(_safe_float(runtime_thresholds.get("regime_negative_gamma_holding_delta_m"), -60.0)),
        _safe_float(runtime_thresholds.get("regime_negative_gamma_position_size_mult"), 0.7),
        int(_safe_float(runtime_thresholds.get("regime_neutral_gamma_composite_delta"), 0.0)),
        int(_safe_float(runtime_thresholds.get("regime_neutral_gamma_strength_delta"), 0.0)),
        0,
        _safe_float(runtime_thresholds.get("regime_neutral_gamma_position_size_mult"), 1.0),
    )
    if _REGIME_THRESHOLDS_CONFIG_KEY == cfg_key:
        return

    initialize_regime_thresholds(
        base_composite=cfg_key[0],
        base_strength=cfg_key[1],
        base_max_holding_m=cfg_key[2],
        base_position_size=cfg_key[3],
        positive_gamma_composite_delta=cfg_key[4],
        positive_gamma_strength_delta=cfg_key[5],
        positive_gamma_holding_delta_m=cfg_key[6],
        positive_gamma_position_size_mult=cfg_key[7],
        negative_gamma_composite_delta=cfg_key[8],
        negative_gamma_strength_delta=cfg_key[9],
        negative_gamma_holding_delta_m=cfg_key[10],
        negative_gamma_position_size_mult=cfg_key[11],
        neutral_gamma_composite_delta=cfg_key[12],
        neutral_gamma_strength_delta=cfg_key[13],
        neutral_gamma_holding_delta_m=cfg_key[14],
        neutral_gamma_position_size_mult=cfg_key[15],
    )
    _REGIME_THRESHOLDS_CONFIG_KEY = cfg_key


def _collect_neutralization_states(payload):
    option_efficiency_features = payload.get("option_efficiency_features")
    option_efficiency_diagnostics = payload.get("option_efficiency_diagnostics")
    option_efficiency_reasons = payload.get("option_efficiency_reasons")
    macro_adjustment_reasons = payload.get("macro_adjustment_reasons")
    global_risk_features = payload.get("global_risk_features")
    global_risk_diagnostics = payload.get("global_risk_diagnostics")

    option_efficiency_features = option_efficiency_features if isinstance(option_efficiency_features, dict) else {}
    option_efficiency_diagnostics = option_efficiency_diagnostics if isinstance(option_efficiency_diagnostics, dict) else {}
    option_efficiency_reasons = option_efficiency_reasons if isinstance(option_efficiency_reasons, list) else []
    macro_adjustment_reasons = macro_adjustment_reasons if isinstance(macro_adjustment_reasons, list) else []
    global_risk_features = global_risk_features if isinstance(global_risk_features, dict) else {}
    global_risk_diagnostics = global_risk_diagnostics if isinstance(global_risk_diagnostics, dict) else {}

    option_efficiency_status = "AVAILABLE"
    option_efficiency_reason = "features_available"
    option_efficiency_warnings = option_efficiency_diagnostics.get("warnings")
    option_efficiency_warnings = option_efficiency_warnings if isinstance(option_efficiency_warnings, list) else []
    option_efficiency_quality = _as_upper(option_efficiency_features.get("expected_move_quality"))
    option_efficiency_expected_move_points = option_efficiency_features.get("expected_move_points")
    option_efficiency_features_missing = not bool(option_efficiency_features)

    option_efficiency_unavailable = (
        option_efficiency_features_missing
        or option_efficiency_features.get("neutral_fallback")
        or "option_efficiency_neutral_fallback" in option_efficiency_reasons
        or option_efficiency_quality == "UNAVAILABLE"
        or any(_as_upper(item) == "EXPECTED_MOVE_UNAVAILABLE" for item in option_efficiency_warnings)
        or (
            option_efficiency_quality in {"DIRECT", "FALLBACK"}
            and option_efficiency_expected_move_points is None
        )
    )

    if option_efficiency_unavailable:
        option_efficiency_status = "UNAVAILABLE_NEUTRALIZED"
        if option_efficiency_features_missing:
            option_efficiency_reason = "option_efficiency_features_missing"
        elif option_efficiency_warnings:
            option_efficiency_reason = str(option_efficiency_warnings[0])
        elif option_efficiency_quality == "UNAVAILABLE":
            option_efficiency_reason = "expected_move_unavailable"
        elif option_efficiency_expected_move_points is None:
            option_efficiency_reason = "expected_move_not_computable"
        else:
            option_efficiency_reason = "option_efficiency_neutral_fallback"

    global_risk_status = "ACTIVE"
    global_risk_reason = "global_risk_features_available"
    if global_risk_features.get("market_features_neutralized"):
        global_risk_status = "LOW_CONFIDENCE_NEUTRALIZED"
        global_risk_reason = "market_features_neutralized"
    elif global_risk_diagnostics.get("fallback"):
        global_risk_status = "LOW_CONFIDENCE_NEUTRALIZED"
        global_risk_reason = "fallback_global_risk_state"

    macro_news_status = "ACTIVE"
    macro_news_reason = "headline_adjustments_available"
    if "macro_news_neutral_fallback" in macro_adjustment_reasons:
        macro_news_status = "STALE_NEUTRALIZED"
        macro_news_reason = "macro_news_neutral_fallback"
    elif _as_upper(payload.get("macro_regime")) == "MACRO_NEUTRAL":
        macro_news_status = "NEUTRAL"
        macro_news_reason = "macro_regime_neutral"

    return {
        "option_efficiency_status": option_efficiency_status,
        "option_efficiency_reason": option_efficiency_reason,
        "global_risk_status": global_risk_status,
        "global_risk_reason": global_risk_reason,
        "macro_news_status": macro_news_status,
        "macro_news_reason": macro_news_reason,
    }


def _normalize_gamma_vol_score(raw_score, normalization_scale, winsor_lower=12, winsor_upper=88):
    """Normalize gamma-vol score with winsorization to reduce outlier dominance."""
    scale = max(_safe_float(normalization_scale, 100.0), 1.0)
    raw = _safe_float(raw_score, 0.0)
    scaled = _clip((raw / scale) * 100.0, 0.0, 100.0)

    lower = _clip(_safe_float(winsor_lower, 12.0), 0.0, 95.0)
    upper = _clip(_safe_float(winsor_upper, 88.0), lower + 1.0, 100.0)
    winsorized = _clip(scaled, lower, upper)
    normalized = ((winsorized - lower) / max(upper - lower, 1.0)) * 100.0
    return int(_clip(round(normalized), 0, 100))


def _compute_runtime_composite_score(
    *,
    trade_strength,
    hybrid_move_probability,
    move_probability_score_cap,
    confirmation_status,
    data_quality_status,
    gamma_vol_acceleration_score_normalized,
    weight_trade_strength=0.50,
    weight_move_probability=0.20,
    weight_confirmation=0.15,
    weight_data_quality=0.10,
    weight_gamma_stability=0.05,
):
    confirmation_map = {
        "STRONG_CONFIRMATION": 100,
        "CONFIRMED": 85,
        "MIXED": 55,
        "CONFLICT": 25,
        "NO_DIRECTION": 10,
    }
    data_quality_map = {
        "STRONG": 100,
        "GOOD": 85,
        "CAUTION": 60,
        "WEAK": 35,
    }

    trade_strength_score = _clip(_safe_float(trade_strength, 0.0), 0, 100)
    move_probability_score = _clip(_safe_float(hybrid_move_probability, 0.0) * 100.0, 0, 100)
    move_probability_score = _clip(
        move_probability_score,
        0,
        _safe_float(move_probability_score_cap, 75.0),
    )
    confirmation_score = confirmation_map.get(_as_upper(confirmation_status), 45)
    data_quality_score = data_quality_map.get(_as_upper(data_quality_status), 50)
    gamma_stability_score = 100.0 - _clip(_safe_float(gamma_vol_acceleration_score_normalized, 0.0), 0, 100)

    w_trade = max(0.0, _safe_float(weight_trade_strength, 0.50))
    w_prob = max(0.0, _safe_float(weight_move_probability, 0.20))
    w_conf = max(0.0, _safe_float(weight_confirmation, 0.15))
    w_data = max(0.0, _safe_float(weight_data_quality, 0.10))
    w_gamma = max(0.0, _safe_float(weight_gamma_stability, 0.05))
    w_sum = w_trade + w_prob + w_conf + w_data + w_gamma
    if w_sum <= 0:
        w_trade, w_prob, w_conf, w_data, w_gamma = 0.50, 0.20, 0.15, 0.10, 0.05
        w_sum = 1.0

    w_trade /= w_sum
    w_prob /= w_sum
    w_conf /= w_sum
    w_data /= w_sum
    w_gamma /= w_sum

    composite = (
        w_trade * trade_strength_score
        + w_prob * move_probability_score
        + w_conf * confirmation_score
        + w_data * data_quality_score
        + w_gamma * gamma_stability_score
    )
    return int(_clip(round(composite), 0, 100))


def _compute_feature_reliability_overlay(option_chain_validation):
    """Summarize feature reliability weights for downstream diagnostics and weighting."""
    weights = (
        option_chain_validation.get("feature_reliability_weights")
        if isinstance(option_chain_validation, dict)
        and isinstance(option_chain_validation.get("feature_reliability_weights"), dict)
        else {}
    )
    if not weights:
        return {
            "status": "UNSPECIFIED",
            "aggregate_score": 100.0,
            "trade_strength_penalty": 0,
            "runtime_composite_penalty": 0,
            "reasons": [],
            "weights": {},
        }

    normalized = {
        "flow": _clip(_safe_float(weights.get("flow"), 1.0), 0.0, 1.0),
        "vol_surface": _clip(_safe_float(weights.get("vol_surface"), 1.0), 0.0, 1.0),
        "greeks": _clip(_safe_float(weights.get("greeks"), 1.0), 0.0, 1.0),
        "liquidity": _clip(_safe_float(weights.get("liquidity"), 1.0), 0.0, 1.0),
        "macro": _clip(_safe_float(weights.get("macro"), 1.0), 0.0, 1.0),
    }
    aggregate_score = round(
        100.0
        * (
            0.30 * normalized["flow"]
            + 0.25 * normalized["vol_surface"]
            + 0.20 * normalized["greeks"]
            + 0.15 * normalized["liquidity"]
            + 0.10 * normalized["macro"]
        ),
        2,
    )
    reasons = []
    for key, label in (
        ("flow", "flow_low_reliability"),
        ("vol_surface", "vol_surface_low_reliability"),
        ("greeks", "greeks_low_reliability"),
        ("liquidity", "liquidity_low_reliability"),
    ):
        if normalized[key] < 0.60:
            reasons.append(label)
    if normalized["macro"] < 0.70:
        reasons.append("macro_low_reliability")

    status = "ROBUST"
    if aggregate_score < 70.0:
        status = "FRAGILE"
    elif aggregate_score < 85.0:
        status = "CAUTION"

    return {
        "status": status,
        "aggregate_score": aggregate_score,
        "trade_strength_penalty": 0,
        "runtime_composite_penalty": 0,
        "reasons": reasons,
        "weights": {key: round(value, 4) for key, value in normalized.items()},
    }


def _blend_feature_reliability(weights, **components):
    total_weight = 0.0
    blended = 0.0
    normalized = weights if isinstance(weights, dict) else {}
    for key, component_weight in components.items():
        weight = max(0.0, _safe_float(component_weight, 0.0))
        if weight <= 0.0:
            continue
        blended += _clip(_safe_float(normalized.get(key), 1.0), 0.0, 1.0) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 1.0
    return round(_clip(blended / total_weight, 0.0, 1.0), 4)


def _scale_adjustment_by_reliability(score, reliability_weight):
    score_value = int(_safe_float(score, 0.0))
    weight_value = _clip(_safe_float(reliability_weight, 1.0), 0.0, 1.0)
    return int(round(score_value * weight_value))


def _scale_candidate_adjustment_by_reliability(score, reliability_weight):
    """Scale strike-candidate adjustments: dampen rewards, amplify penalties when reliability is weak."""
    score_value = int(_safe_float(score, 0.0))
    weight_value = _clip(_safe_float(reliability_weight, 1.0), 0.0, 1.0)
    if score_value >= 0:
        return int(round(score_value * weight_value))

    penalty_multiplier = 1.0 + ((1.0 - weight_value) * 0.75)
    scaled = int(round(score_value * penalty_multiplier))
    if score_value < 0 and scaled == 0:
        return -1
    return scaled


def _resolve_regime_thresholds(*, runtime_thresholds, base_min_trade_strength, base_min_composite_score, market_state):
    # Use new RegimeAdaptiveThresholds if enabled, otherwise fall back to legacy logic
    use_new_regime_thresholds = bool(int(_safe_float(runtime_thresholds.get("enable_regime_conditional_thresholds"), 1.0)))
    
    if use_new_regime_thresholds:
        _ensure_regime_thresholds_config(runtime_thresholds, base_min_trade_strength, base_min_composite_score)
        gamma_regime = canonical_gamma_regime(market_state.get("gamma_regime"))
        vol_regime = _canonical_vol_regime(market_state.get("vol_regime"))
        
        new_thresholds = compute_regime_thresholds(
            gamma_regime=gamma_regime,
            volatility_regime=vol_regime,
            spot_vs_flip=market_state.get("spot_vs_flip")
        )
        
        return {
            "effective_min_trade_strength": int(_clip(new_thresholds["effective_trade_strength"], 0, 100)),
            "effective_min_composite_score": int(_clip(new_thresholds["effective_composite_score"], 0, 100)),
            "effective_max_holding_m": int(_clip(new_thresholds.get("effective_max_holding_m", 90), 30, 480)),
            "position_size_multiplier": _safe_float(new_thresholds.get("position_size_multiplier"), 1.0),
            "adjustments": new_thresholds.get("rationale", []),
            "toxic_context": gamma_regime == "NEGATIVE_GAMMA",
        }
    
    # Legacy logic (fallback if disabled)
    adjustments = []
    effective_trade_strength = int(base_min_trade_strength)
    effective_composite = int(base_min_composite_score)

    spot_vs_flip = _as_upper(market_state.get("spot_vs_flip"))
    gamma_regime = canonical_gamma_regime(market_state.get("gamma_regime"))
    dealer_position = _as_upper(market_state.get("dealer_pos"))
    toxic_gamma = gamma_regime == "NEGATIVE_GAMMA"
    dealer_short_gamma = ("SHORT" in dealer_position) and ("GAMMA" in dealer_position)

    if spot_vs_flip == "AT_FLIP":
        add_strength = int(_safe_float(runtime_thresholds.get("regime_strength_add_at_flip"), 4.0))
        add_composite = int(_safe_float(runtime_thresholds.get("regime_composite_add_at_flip"), 3.0))
        effective_trade_strength += add_strength
        effective_composite += add_composite
        adjustments.append("at_flip_threshold_tightening")

    if toxic_gamma or dealer_short_gamma:
        add_strength = int(_safe_float(runtime_thresholds.get("regime_strength_add_toxic"), 8.0))
        add_composite = int(_safe_float(runtime_thresholds.get("regime_composite_add_toxic"), 6.0))
        effective_trade_strength += add_strength
        effective_composite += add_composite
        adjustments.append("toxic_regime_threshold_tightening")

    if gamma_regime == "POSITIVE_GAMMA":
        add_strength = int(_safe_float(runtime_thresholds.get("regime_strength_add_positive_gamma"), 5.0))
        add_composite = int(_safe_float(runtime_thresholds.get("regime_composite_add_positive_gamma"), 3.0))
        effective_trade_strength += add_strength
        effective_composite += add_composite
        adjustments.append("positive_gamma_threshold_tightening")

    if gamma_regime == "NEGATIVE_GAMMA":
        relief_strength = int(_safe_float(runtime_thresholds.get("regime_strength_relief_negative_gamma"), 2.0))
        relief_composite = int(_safe_float(runtime_thresholds.get("regime_composite_relief_negative_gamma"), 1.0))
        effective_trade_strength -= relief_strength
        effective_composite -= relief_composite
        adjustments.append("negative_gamma_threshold_relief")

    return {
        "effective_min_trade_strength": int(_clip(effective_trade_strength, 0, 100)),
        "effective_min_composite_score": int(_clip(effective_composite, 0, 100)),
        "effective_max_holding_m": int(_safe_float(runtime_thresholds.get("max_intraday_hold_minutes"), 90.0)),
        "position_size_multiplier": 1.0,
        "adjustments": adjustments,
        "toxic_context": bool(toxic_gamma or dealer_short_gamma),
    }


def _compute_structural_imbalance_audit(*, market_state, direction):
    dealer_metrics = market_state.get("dealer_metrics") if isinstance(market_state, dict) else {}
    dealer_metrics = dealer_metrics if isinstance(dealer_metrics, dict) else {}
    call_oi_change = _safe_float(dealer_metrics.get("call_oi_change"), 0.0)
    put_oi_change = _safe_float(dealer_metrics.get("put_oi_change"), 0.0)
    directional_imbalance = call_oi_change - put_oi_change

    direction = _as_upper(direction)
    alignment = "NEUTRAL"
    severity = "LOW"
    if direction == "CALL":
        alignment = "ALIGNED" if directional_imbalance >= 0 else "CONFLICT"
    elif direction == "PUT":
        alignment = "ALIGNED" if directional_imbalance <= 0 else "CONFLICT"

    abs_imbalance = abs(directional_imbalance)
    if abs_imbalance >= 200000:
        severity = "HIGH"
    elif abs_imbalance >= 80000:
        severity = "MEDIUM"

    return {
        "call_put_imbalance_score": round(directional_imbalance, 2),
        "call_put_imbalance_abs": round(abs_imbalance, 2),
        "call_put_alignment": alignment,
        "call_put_imbalance_severity": severity,
    }


def _nearest_trigger_walls(*, spot, support_wall, resistance_wall, liquidity_levels):
    """Choose nearest actionable support/resistance levels around spot."""
    spot_value = _safe_float(spot, None)
    if spot_value is None:
        return support_wall, resistance_wall

    candidates = []
    for level in [support_wall, resistance_wall]:
        val = _safe_float(level, None)
        if val is not None:
            candidates.append(val)

    if isinstance(liquidity_levels, list):
        for level in liquidity_levels:
            val = _safe_float(level, None)
            if val is not None:
                candidates.append(val)

    if not candidates:
        return support_wall, resistance_wall

    support_candidates = [lvl for lvl in candidates if lvl <= spot_value]
    resistance_candidates = [lvl for lvl in candidates if lvl >= spot_value]

    nearest_support = max(support_candidates) if support_candidates else support_wall
    nearest_resistance = min(resistance_candidates) if resistance_candidates else resistance_wall
    return nearest_support, nearest_resistance


def _build_decision_explainability(payload, *, trade_status, min_trade_strength):
    direction = payload.get("direction")
    flow_signal = _as_upper(payload.get("final_flow_signal") or payload.get("flow_signal"))
    smart_money_flow = _as_upper(payload.get("smart_money_flow"))
    confirmation_status = _as_upper(payload.get("confirmation_status"))
    signal_quality = _as_upper(payload.get("signal_quality"))
    directional_convexity_state = _as_upper(payload.get("directional_convexity_state"))
    dealer_flow_state = _as_upper(payload.get("dealer_flow_state"))
    dealer_hedging_bias = _as_upper(payload.get("dealer_hedging_bias"))
    global_risk_action = _as_upper(payload.get("global_risk_action"))
    data_quality_status = _as_upper(payload.get("data_quality_status"))
    trade_strength = int(_safe_float(payload.get("trade_strength"), 0.0))
    hybrid_move_probability = _safe_float(payload.get("hybrid_move_probability"), 0.0)
    spot = _safe_float(payload.get("spot"), 0.0)
    support_wall = payload.get("support_wall")
    resistance_wall = payload.get("resistance_wall")
    gamma_flip = payload.get("gamma_flip")
    live_calibration_gate = payload.get("live_calibration_gate") if isinstance(payload.get("live_calibration_gate"), dict) else {}
    live_directional_gate = payload.get("live_directional_gate") if isinstance(payload.get("live_directional_gate"), dict) else {}
    historical_outcome_guard = payload.get("historical_outcome_guard") if isinstance(payload.get("historical_outcome_guard"), dict) else {}
    session_risk_governor = payload.get("session_risk_governor") if isinstance(payload.get("session_risk_governor"), dict) else {}
    trade_slot_governor = payload.get("trade_slot_governor") if isinstance(payload.get("trade_slot_governor"), dict) else {}
    trade_promotion_governor = payload.get("trade_promotion_governor") if isinstance(payload.get("trade_promotion_governor"), dict) else {}
    portfolio_concentration_guard = payload.get("portfolio_concentration_guard") if isinstance(payload.get("portfolio_concentration_guard"), dict) else {}

    activation_score = 0
    acfg = get_activation_score_policy_config()
    policy_fallback_active = acfg is None
    fallback_acfg = {
        "dead_inactive_threshold": 25,
        "confirmation_score_strong": 90,
        "confirmation_score_mixed": 55,
        "confirmation_score_conflict": 25,
        "confirmation_score_no_direction": 10,
        "data_ready_strong": 90,
        "data_ready_good": 75,
        "data_ready_caution": 50,
        "data_ready_weak": 30,
        "maturity_weight_trade_strength": 0.50,
        "maturity_weight_confirmation": 0.30,
        "maturity_weight_data_ready": 0.20,
        "high_confidence_data_ready_floor": 75,
        "high_confidence_confirmation_floor": 70,
        "medium_confidence_data_ready_floor": 55,
    }
    
    # Guard: config may be None if resolution fails; use fallback
    if acfg is None:
        import logging
        logging.getLogger(__name__).error("Activation score policy config unavailable; using fallback")
        activation_score = 0
    else:
        if _is_directional_flow(flow_signal):
            activation_score += acfg.flow_bonus
        if _is_directional_flow(smart_money_flow):
            activation_score += acfg.smart_money_bonus
        if _is_convexity_active(directional_convexity_state):
            activation_score += acfg.convexity_bonus
        if _is_dealer_structure_active(dealer_flow_state):
            activation_score += acfg.dealer_structure_bonus
        if trade_strength >= max(12, int(min_trade_strength * acfg.trade_strength_min_ratio)):
            activation_score += acfg.trade_strength_bonus
        if hybrid_move_probability >= acfg.move_probability_floor:
            activation_score += acfg.move_probability_bonus
        activation_score = int(_clip(activation_score, 0, acfg.activation_cap))

    confirmation_score = 0
    if confirmation_status in {"STRONG_CONFIRMATION", "CONFIRMED"}:
        confirmation_score = acfg.confirmation_score_strong if acfg is not None else fallback_acfg["confirmation_score_strong"]
    elif confirmation_status == "MIXED":
        confirmation_score = acfg.confirmation_score_mixed if acfg is not None else fallback_acfg["confirmation_score_mixed"]
    elif confirmation_status == "CONFLICT":
        confirmation_score = acfg.confirmation_score_conflict if acfg is not None else fallback_acfg["confirmation_score_conflict"]
    elif confirmation_status == "NO_DIRECTION":
        confirmation_score = acfg.confirmation_score_no_direction if acfg is not None else fallback_acfg["confirmation_score_no_direction"]

    data_ready_score = acfg.data_ready_strong if acfg is not None else fallback_acfg["data_ready_strong"]
    if data_quality_status == "GOOD":
        data_ready_score = acfg.data_ready_good if acfg is not None else fallback_acfg["data_ready_good"]
    elif data_quality_status == "CAUTION":
        data_ready_score = acfg.data_ready_caution if acfg is not None else fallback_acfg["data_ready_caution"]
    elif data_quality_status == "WEAK":
        data_ready_score = acfg.data_ready_weak if acfg is not None else fallback_acfg["data_ready_weak"]

    maturity_score = int(
        _clip(
            ((acfg.maturity_weight_trade_strength if acfg is not None else fallback_acfg["maturity_weight_trade_strength"]) * trade_strength)
            + ((acfg.maturity_weight_confirmation if acfg is not None else fallback_acfg["maturity_weight_confirmation"]) * confirmation_score)
            + ((acfg.maturity_weight_data_ready if acfg is not None else fallback_acfg["maturity_weight_data_ready"]) * data_ready_score),
            0,
            100,
        )
    )

    explainability_confidence = "LOW"
    if data_ready_score >= (acfg.high_confidence_data_ready_floor if acfg is not None else fallback_acfg["high_confidence_data_ready_floor"]) and confirmation_score >= (acfg.high_confidence_confirmation_floor if acfg is not None else fallback_acfg["high_confidence_confirmation_floor"]):
        explainability_confidence = "HIGH"
    elif data_ready_score >= (acfg.medium_confidence_data_ready_floor if acfg is not None else fallback_acfg["medium_confidence_data_ready_floor"]):
        explainability_confidence = "MEDIUM"

    missing_requirements = []
    missing_confirmations = []
    blocked_by = []
    promotion_requirements = []
    setup_upgrade_conditions = []
    reason_details = []

    incoming_no_trade_reason_code = _as_upper(payload.get("no_trade_reason_code")) or None
    incoming_no_trade_reason = str(payload.get("no_trade_reason") or "").strip() or None

    no_trade_reason_code = None
    no_trade_reason = None
    watchlist_flag = False
    watchlist_reason = None
    setup_state = "NONE"
    setup_quality = "NONE"
    directional_resolution_needed = False
    likely_next_trigger = None

    if trade_status == "TRADE":
        decision_classification = "TRADE_READY"
        setup_state = "NONE"
        setup_quality = "READY"
    else:
        setup_quality = signal_quality or "VERY_WEAK"
        if trade_status == "DATA_INVALID":
            decision_classification = "DATA_BLOCKED"
            setup_state = "DATA_BLOCKED"
            blocked_by.append("data_quality")
            no_trade_reason_code = "DATA_QUALITY_INSUFFICIENT"
            no_trade_reason = "Trade blocked due to invalid or stale market data"
        elif trade_status == "NO_TRADE" or global_risk_action == "BLOCK":
            decision_classification = "RISK_BLOCKED"
            setup_state = "RISK_BLOCKED"
            blocked_by.append("global_risk")
            if bool(payload.get("event_lockdown_flag")):
                no_trade_reason_code = "EVENT_LOCKDOWN_BLOCK"
                no_trade_reason = "Trade blocked due to event lockdown window"
                blocked_by.append("event_lockdown")
            else:
                no_trade_reason_code = "GLOBAL_RISK_BLOCK"
                no_trade_reason = "Trade blocked by global risk overlay"
            reason_details.extend(
                payload.get("global_risk_state_reasons")
                or payload.get("global_risk_reasons")
                or []
            )
        elif trade_status == "BUDGET_FAIL":
            decision_classification = "RISK_BLOCKED"
            setup_state = "RISK_BLOCKED"
            blocked_by.append("budget")
            no_trade_reason_code = "BUDGET_CONSTRAINT_BLOCK"
            no_trade_reason = "Signal passed but budget constraint blocked execution"
        elif direction is None:
            directional_resolution_needed = True
            missing_requirements.append("missing_directional_consensus")
            missing_confirmations.append("direction")

            dead_inactive_threshold = (
                acfg.dead_inactive_threshold
                if acfg is not None
                else fallback_acfg["dead_inactive_threshold"]
            )
            if activation_score < dead_inactive_threshold:
                decision_classification = "DEAD_INACTIVE"
                setup_state = "NONE"
                no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
                no_trade_reason = "Market activity is below watchlist threshold"
                watchlist_flag = False
                watchlist_reason = None
            elif directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK":
                decision_classification = "DIRECTIONALLY_AMBIGUOUS"
                setup_state = "DIRECTION_PENDING"
                no_trade_reason_code = "TWO_SIDED_VOLATILITY_WITHOUT_EDGE"
                no_trade_reason = "Two-sided convexity risk without directional edge"
                watchlist_flag = True
                watchlist_reason = "Convexity active but directional asymmetry is unresolved"
                setup_upgrade_conditions.append("move away from gamma flip with aligned flow + dealer bias")
            elif flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"} or dealer_flow_state in {
                "UPSIDE_HEDGING_ACCELERATION",
                "DOWNSIDE_HEDGING_ACCELERATION",
                "PINNING_DOMINANT",
            }:
                decision_classification = "WATCHLIST_SETUP"
                setup_state = "DIRECTION_PENDING"
                no_trade_reason_code = "DIRECTIONAL_CONVICTION_INSUFFICIENT"
                no_trade_reason = "Directional signals are present but conviction threshold is not met"
                watchlist_flag = True
                watchlist_reason = "Setup has structure but direction is not confirmed"
            else:
                decision_classification = "DEAD_INACTIVE"
                setup_state = "NONE"
                no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
                no_trade_reason = "Market is currently inactive with no directional edge"

            if confirmation_status in {"NO_DIRECTION", "CONFLICT"}:
                missing_requirements.append("confirmation_filter_not_met")
                missing_confirmations.append("confirmation")
            elif confirmation_status in {"CONFIRMED", "STRONG_CONFIRMATION"}:
                missing_requirements.append("direction_confirmation_conflict")
                reason_details.append("secondary_blocker: confirmation reports directionality while engine direction is unresolved")

            if flow_signal == "NEUTRAL_FLOW" and smart_money_flow == "NEUTRAL_FLOW":
                missing_requirements.append("missing_flow_confirmation")

            if dealer_hedging_bias in {"PINNING", "DOWNSIDE_PINNING", "UPSIDE_PINNING"} or dealer_flow_state == "PINNING_DOMINANT":
                missing_requirements.append("pinning_structure_dampens_signal")
                promotion_requirements.append("dealer hedging bias shifts from pinning to acceleration")

            if trade_strength < min_trade_strength:
                missing_requirements.append("insufficient_trade_strength")
                promotion_requirements.append(f"trade_strength >= {int(min_trade_strength)}")
        elif trade_status == "WATCHLIST":
            watchlist_flag = True
            provider_health_summary = _as_upper(payload.get("provider_health_summary"))
            provider_health_payload = payload.get("provider_health") if isinstance(payload.get("provider_health"), dict) else {}
            provider_health_blocking_status = _as_upper(provider_health_payload.get("trade_blocking_status"))
            global_risk_overlay_reasons = {
                _as_upper(reason)
                for reason in (payload.get("global_risk_overlay_reasons") or [])
                if reason is not None
            }
            provider_health_blocked = (
                provider_health_blocking_status == "BLOCK"
                or (not provider_health_blocking_status and provider_health_summary in {"CAUTION", "WEAK"})
                or bool(incoming_no_trade_reason_code and incoming_no_trade_reason_code.startswith("PROVIDER_HEALTH_"))
                or "PROVIDER_HEALTH_CAUTION" in global_risk_overlay_reasons
                or "PROVIDER_HEALTH_WEAK" in global_risk_overlay_reasons
            )
            watchlist_message = str(payload.get("message") or "").strip()

            historical_guard_verdict = _as_upper(historical_outcome_guard.get("verdict"))
            session_risk_verdict = _as_upper(session_risk_governor.get("verdict"))
            trade_slot_verdict = _as_upper(trade_slot_governor.get("verdict"))
            trade_promotion_verdict = _as_upper(trade_promotion_governor.get("verdict"))
            portfolio_guard_verdict = _as_upper(portfolio_concentration_guard.get("verdict"))
            regime_segment_guard = payload.get("regime_segment_guard") if isinstance(payload.get("regime_segment_guard"), dict) else {}
            regime_guard_verdict = _as_upper(regime_segment_guard.get("verdict"))
            if trade_promotion_verdict == "BLOCK" or incoming_no_trade_reason_code == "TRADE_PROMOTION_GOVERNOR":
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Trade promotion governor requires replay validation"
                blocked_by.append("trade_promotion_governor")
                no_trade_reason_code = "TRADE_PROMOTION_GOVERNOR"
                no_trade_reason = trade_promotion_governor.get("reason") or watchlist_message or "Trade promotion governor downgraded this setup"
            elif trade_slot_verdict == "BLOCK" or incoming_no_trade_reason_code == "TRADE_SLOT_GOVERNOR":
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Trade slot governor downgraded the setup"
                blocked_by.append("trade_slot_governor")
                no_trade_reason_code = "TRADE_SLOT_GOVERNOR"
                no_trade_reason = trade_slot_governor.get("reason") or watchlist_message or "Trade slot governor downgraded this setup"
            elif session_risk_verdict == "BLOCK" or incoming_no_trade_reason_code == "SESSION_RISK_GOVERNOR":
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Session risk governor downgraded the setup"
                blocked_by.append("session_risk_governor")
                no_trade_reason_code = "SESSION_RISK_GOVERNOR"
                no_trade_reason = session_risk_governor.get("reason") or watchlist_message or "Session risk governor downgraded this setup"
            elif portfolio_guard_verdict == "WATCHLIST" or incoming_no_trade_reason_code == "PORTFOLIO_CONCENTRATION_GUARD":
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Portfolio concentration guard downgraded the setup"
                blocked_by.append("portfolio_concentration_guard")
                no_trade_reason_code = "PORTFOLIO_CONCENTRATION_GUARD"
                no_trade_reason = portfolio_concentration_guard.get("reason") or watchlist_message or "Portfolio concentration guard downgraded this setup"
            elif regime_guard_verdict == "BLOCK" or incoming_no_trade_reason_code == "REGIME_SEGMENT_GUARD":
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Regime segment guard downgraded the setup"
                blocked_by.append("regime_segment_guard")
                no_trade_reason_code = "REGIME_SEGMENT_GUARD"
                no_trade_reason = regime_segment_guard.get("reason") or watchlist_message or "Regime segment guard downgraded this setup"
            elif historical_guard_verdict == "BLOCK" or incoming_no_trade_reason_code == "HISTORICAL_OUTCOME_GUARD":
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Historical outcome guard downgraded the setup"
                blocked_by.append("historical_outcome_guard")
                no_trade_reason_code = "HISTORICAL_OUTCOME_GUARD"
                no_trade_reason = historical_outcome_guard.get("reason") or watchlist_message or "Historical outcome guard downgraded this setup"
            elif provider_health_blocked:
                provider_blocker = provider_health_summary
                if provider_blocker not in {"CAUTION", "WEAK"}:
                    provider_blocker = "CAUTION" if "PROVIDER_HEALTH_CAUTION" in global_risk_overlay_reasons else "WEAK"
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Provider health gates prevent trade execution"
                blocked_by.append("provider_health")
                no_trade_reason_code = f"PROVIDER_HEALTH_{provider_blocker}_BLOCK"
                no_trade_reason = f"Provider health {provider_blocker} blocks trade execution"
                if watchlist_message and watchlist_message != no_trade_reason:
                    reason_details.append(f"secondary_blocker: {watchlist_message}")
            elif data_quality_status in {"CAUTION", "WEAK"}:
                decision_classification = "WATCHLIST_CONFIRMATION_PENDING"
                setup_state = "CONFIRMATION_PENDING"
                watchlist_reason = "Data quality and confirmation filters require more clarity"
                blocked_by.append("data_quality")
                no_trade_reason_code = "DATA_QUALITY_CAUTION"
                no_trade_reason = watchlist_message or "Trade downgraded to watchlist due to cautionary data quality"
            elif global_risk_action in {"WATCHLIST", "REDUCE"}:
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Signal is structurally valid but downgraded by risk overlays"
                blocked_by.append("risk_overlay")
                no_trade_reason_code = "RISK_OVERLAY_DOWNGRADE"
                no_trade_reason = watchlist_message or "Trade downgraded to watchlist due to active risk overlay"
            else:
                decision_classification = "WATCHLIST_SETUP"
                setup_state = "CONFIRMATION_PENDING"
                watchlist_reason = "Directional thesis exists but confirmations are incomplete"

            if confirmation_status in {"CONFLICT", "NO_DIRECTION"}:
                missing_requirements.append("confirmation_filter_not_met")
                missing_confirmations.append("confirmation")

            if trade_strength < min_trade_strength:
                missing_requirements.append("insufficient_trade_strength")
                promotion_requirements.append(f"trade_strength >= {int(min_trade_strength)}")
                if not no_trade_reason_code:
                    no_trade_reason_code = "TRADE_STRENGTH_BELOW_THRESHOLD"
                    no_trade_reason = (
                        f"Setup is on watchlist: trade_strength {int(trade_strength)} "
                        f"below threshold {int(min_trade_strength)}"
                    )
                else:
                    reason_details.append(
                        f"secondary_blocker: trade_strength {int(trade_strength)} below threshold {int(min_trade_strength)}"
                    )
                low_strength_watchlist = (
                    no_trade_reason_code in {None, "TRADE_STRENGTH_BELOW_THRESHOLD"}
                    and (
                        "LOW STRENGTH" in watchlist_message.upper()
                        or "INSUFFICIENT_TRADE_STRENGTH" in global_risk_overlay_reasons
                    )
                )
                if low_strength_watchlist:
                    decision_classification = "WATCHLIST_SETUP"
                    setup_state = "CONFIRMATION_PENDING"
                    watchlist_reason = (
                        f"Trade strength {int(trade_strength)} is below execution threshold "
                        f"{int(min_trade_strength)}"
                    )
                    blocked_by = [item for item in blocked_by if item != "risk_overlay"]
                    blocked_by.append("trade_strength")

            if flow_signal == "NEUTRAL_FLOW":
                missing_requirements.append("missing_flow_confirmation")

            no_trade_reason_code = no_trade_reason_code or "FLOW_NOT_CONFIRMED"
            no_trade_reason = no_trade_reason or "Setup is on watchlist pending stronger confirmation"
        else:
            decision_classification = "WATCHLIST_SETUP" if signal_quality in {"WEAK", "MEDIUM"} else "DEAD_INACTIVE"
            setup_state = "CONFIRMATION_PENDING" if decision_classification == "WATCHLIST_SETUP" else "NONE"
            if decision_classification == "WATCHLIST_SETUP":
                watchlist_flag = True
                watchlist_reason = "Setup exists but does not yet meet execution thresholds"
            no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
            no_trade_reason = "Signal did not reach execution threshold"

    if flow_signal == "NEUTRAL_FLOW" and smart_money_flow == "NEUTRAL_FLOW":
        promotion_requirements.append("flow turns directional and aligns with smart-money flow")
        setup_upgrade_conditions.append("directional flow confirmation on both flow lenses")
    elif flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"} and smart_money_flow not in {"BULLISH_FLOW", "BEARISH_FLOW"}:
        missing_requirements.append("missing_flow_confirmation")
        promotion_requirements.append("smart-money flow confirms directional flow")

    if directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK":
        setup_upgrade_conditions.append("resolve two-sided convexity into one-sided acceleration")

    if dealer_flow_state == "PINNING_DOMINANT":
        setup_upgrade_conditions.append("pinning pressure eases and hedging acceleration emerges")

    nearest_support_wall, nearest_resistance_wall = _nearest_trigger_walls(
        spot=spot,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        liquidity_levels=payload.get("liquidity_levels"),
    )
    if flow_signal == "BEARISH_FLOW" and nearest_support_wall is not None:
        likely_next_trigger = f"break below support wall {nearest_support_wall}"
        setup_upgrade_conditions.append(f"decisive move below support wall {nearest_support_wall}")
    elif flow_signal == "BULLISH_FLOW" and nearest_resistance_wall is not None:
        likely_next_trigger = f"break above resistance wall {nearest_resistance_wall}"
        setup_upgrade_conditions.append(f"decisive move above resistance wall {nearest_resistance_wall}")

    if likely_next_trigger is None and gamma_flip is not None:
        likely_next_trigger = f"clean move away from gamma flip {gamma_flip} with confirmation"

    if payload.get("expected_move_points") is None:
        missing_requirements.append("option_efficiency_unavailable")
        promotion_requirements.append("option efficiency features become available and supportive")

    if hybrid_move_probability > 0:
        prob_value = hybrid_move_probability
        if prob_value < 0.55:
            missing_requirements.append("move_probability_not_high_enough")
            promotion_requirements.append("hybrid move probability rises above conviction floor")

    if not no_trade_reason_code and trade_status != "TRADE":
        no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
        no_trade_reason = "Setup has not met the minimum execution bar"

    # Surface runtime quality gates explicitly in explainability so operators
    # can distinguish model-health blocks from pure market-structure blocks.
    if _as_upper(live_calibration_gate.get("verdict")) == "BLOCK":
        blocked_by.append("live_calibration_gate")
        reason_details.append("secondary_blocker: live calibration gate blocked (miscalibrated recent outcomes)")
    if _as_upper(live_directional_gate.get("verdict")) == "BLOCK":
        blocked_by.append("live_directional_gate")
        reason_details.append("secondary_blocker: live directional gate blocked (stickiness/imbalance/flip-lag)")

    trade_promotion_verdict = _as_upper(trade_promotion_governor.get("verdict"))
    if trade_promotion_verdict == "BLOCK":
        blocked_by.append("trade_promotion_governor")
        promotion_state = trade_promotion_governor.get("promotion_state")
        reason_details.append(
            f"secondary_blocker: trade promotion governor blocked — replay validation required before live promotion [{promotion_state}]"
        )

    trade_slot_verdict = _as_upper(trade_slot_governor.get("verdict"))
    if trade_slot_verdict == "BLOCK":
        blocked_by.append("trade_slot_governor")
        slot_count = trade_slot_governor.get("active_signal_count")
        same_direction_count = trade_slot_governor.get("same_direction_count")
        slot_bits = []
        if slot_count is not None:
            slot_bits.append(f"{slot_count} active ideas")
        if same_direction_count is not None:
            slot_bits.append(f"{same_direction_count} same-way")
        slot_suffix = f" ({', '.join(slot_bits)})" if slot_bits else ""
        reason_details.append(
            "secondary_blocker: trade slot governor blocked — the symbol book is already full"
            f"{slot_suffix}"
        )
    elif trade_slot_verdict == "CAUTION":
        slot_count = trade_slot_governor.get("active_signal_count")
        override_active = bool(trade_slot_governor.get("operator_override_active"))
        slot_suffix = f" ({slot_count} active ideas)" if slot_count is not None else ""
        if override_active:
            reason_details.append(f"note: trade slot governor allowed a reduced-size operator override{slot_suffix}")
        else:
            reason_details.append(f"note: trade slot governor is tightening size because the symbol book is near capacity{slot_suffix}")

    session_risk_verdict = _as_upper(session_risk_governor.get("verdict"))
    if session_risk_verdict == "BLOCK":
        blocked_by.append("session_risk_governor")
        cooldown_active = bool(session_risk_governor.get("cooldown_active"))
        stopout_streak = session_risk_governor.get("stopout_streak")
        budget_remaining_pct = session_risk_governor.get("budget_remaining_pct")
        session_bits = []
        if stopout_streak is not None:
            session_bits.append(f"stopout streak {stopout_streak}")
        if budget_remaining_pct is not None:
            session_bits.append(f"budget {budget_remaining_pct}% remaining")
        if cooldown_active:
            session_bits.append("cooldown active")
        session_suffix = f" ({', '.join(session_bits)})" if session_bits else ""
        reason_details.append(
            "secondary_blocker: session risk governor blocked — recent realized losses require cooling-off"
            f"{session_suffix}"
        )
    elif session_risk_verdict == "CAUTION":
        budget_remaining_pct = session_risk_governor.get("budget_remaining_pct")
        budget_suffix = f" ({budget_remaining_pct}% remaining)" if budget_remaining_pct is not None else ""
        reason_details.append(f"note: session risk governor is reducing risk after recent losses{budget_suffix}")

    regime_segment_guard = payload.get("regime_segment_guard") if isinstance(payload.get("regime_segment_guard"), dict) else {}
    regime_guard_verdict = _as_upper(regime_segment_guard.get("verdict"))
    if regime_guard_verdict == "BLOCK":
        blocked_by.append("regime_segment_guard")
        segment_reason = str(regime_segment_guard.get("reason") or "active regime segment is underperforming")
        segment_samples = regime_segment_guard.get("sample_size")
        segment_suffix = f" ({segment_samples} matched samples)" if segment_samples is not None else ""
        reason_details.append(f"secondary_blocker: regime segment guard blocked — {segment_reason}{segment_suffix}")
    elif regime_guard_verdict == "CAUTION":
        segment_key = regime_segment_guard.get("segment_key")
        segment_suffix = f" [{segment_key}]" if segment_key not in (None, "", "nan") else ""
        reason_details.append(f"note: regime segment guard recommends tighter promotion and faster exits{segment_suffix}")

    portfolio_guard_verdict = _as_upper(portfolio_concentration_guard.get("verdict"))
    if portfolio_guard_verdict == "WATCHLIST":
        blocked_by.append("portfolio_concentration_guard")
        same_count = portfolio_concentration_guard.get("same_direction_count")
        recent_total = portfolio_concentration_guard.get("recent_signal_count")
        same_share = _safe_float(portfolio_concentration_guard.get("same_direction_share"), None)
        heat_score = _safe_float(portfolio_concentration_guard.get("heat_score"), None)
        heat_label = str(portfolio_concentration_guard.get("heat_label") or "").upper().strip()
        profile_bits = []
        if same_count is not None and recent_total is not None:
            profile_bits.append(f"{same_count}/{recent_total} same-way")
        if same_share is not None:
            profile_bits.append(f"share {same_share:.0%}")
        if heat_label:
            heat_text = heat_label.lower()
            if heat_score is not None:
                heat_text = f"{heat_text} heat {int(round(heat_score))}/100"
            profile_bits.append(heat_text)
        profile_suffix = f" ({', '.join(profile_bits)})" if profile_bits else ""
        reason_details.append(
            "secondary_blocker: portfolio concentration guard blocked — concentrated same-way options book"
            f"{profile_suffix}"
        )
    elif portfolio_guard_verdict == "REDUCE":
        same_share = _safe_float(portfolio_concentration_guard.get("same_direction_share"), None)
        heat_score = _safe_float(portfolio_concentration_guard.get("heat_score"), None)
        heat_label = str(portfolio_concentration_guard.get("heat_label") or "").upper().strip()
        detail_bits = []
        if same_share is not None:
            detail_bits.append(f"same-way share {same_share:.0%}")
        if heat_label:
            heat_text = heat_label.lower()
            if heat_score is not None:
                heat_text = f"{heat_text} heat {int(round(heat_score))}/100"
            detail_bits.append(heat_text)
        share_suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
        reason_details.append(f"note: portfolio concentration guard reduced size due to crowding{share_suffix}")

    historical_guard_verdict = _as_upper(historical_outcome_guard.get("verdict"))
    if historical_guard_verdict == "BLOCK":
        blocked_by.append("historical_outcome_guard")
        guard_reason = str(historical_outcome_guard.get("reason") or "historical regime outcomes are weak")
        samples = historical_outcome_guard.get("sample_size")
        sample_suffix = f" ({samples} matched samples)" if samples is not None else ""
        reason_details.append(f"secondary_blocker: historical outcome guard blocked — {guard_reason}{sample_suffix}")
    elif historical_guard_verdict == "CAUTION":
        exit_bias = str(historical_outcome_guard.get("exit_bias") or "TAKE_PROFIT_EARLY").lower().replace("_", " ")
        best_horizon = historical_outcome_guard.get("best_horizon")
        horizon_suffix = f" near {best_horizon}" if best_horizon not in (None, "", "nan") else ""
        reason_details.append(f"note: historical outcome profile suggests {exit_bias}{horizon_suffix}")

    dealer_liquidity_map = payload.get("dealer_liquidity_map") if isinstance(payload.get("dealer_liquidity_map"), dict) else {}
    band_reason = _as_upper(dealer_liquidity_map.get("band_reason"))
    if direction == "CALL" and band_reason == "MOVE_TO_SUPPORT":
        missing_requirements.append("dealer_liquidity_band_conflicts_with_call")
        reason_details.append("secondary_blocker: dealer liquidity map expects move to support")
        setup_upgrade_conditions.append("resolve dealer band conflict with a move toward resistance")
    elif direction == "PUT" and band_reason == "MOVE_TO_RESISTANCE":
        missing_requirements.append("dealer_liquidity_band_conflicts_with_put")
        reason_details.append("secondary_blocker: dealer liquidity map expects move to resistance")
        setup_upgrade_conditions.append("resolve dealer band conflict with a move toward support")

    # Preserve upstream reason code/reason when they were already set by earlier layers.
    if incoming_no_trade_reason_code:
        no_trade_reason_code = incoming_no_trade_reason_code
    if incoming_no_trade_reason:
        no_trade_reason = incoming_no_trade_reason

    neutralization = _collect_neutralization_states(payload)

    explainability = {
        "decision_classification": decision_classification,
        "setup_state": setup_state,
        "setup_quality": setup_quality,
        "setup_activation_score": activation_score,
        "setup_maturity_score": maturity_score,
        "explainability_confidence": explainability_confidence,
        "watchlist_flag": bool(watchlist_flag),
        "watchlist_reason": watchlist_reason,
        "no_trade_reason_code": no_trade_reason_code,
        "no_trade_reason": no_trade_reason,
        "no_trade_reason_details": _dedupe_keep_order(reason_details),
        "blocked_by": _dedupe_keep_order(blocked_by),
        "missing_confirmations": _dedupe_keep_order(missing_confirmations),
        "missing_signal_requirements": _dedupe_keep_order(missing_requirements),
        "signal_promotion_requirements": _dedupe_keep_order(promotion_requirements),
        "setup_upgrade_conditions": _dedupe_keep_order(setup_upgrade_conditions),
        "setup_upgrade_path": _dedupe_keep_order(setup_upgrade_conditions + promotion_requirements),
        "likely_next_trigger": likely_next_trigger,
        "watchlist_trigger_levels": {
            "spot": spot,
            "support_wall": support_wall,
            "resistance_wall": resistance_wall,
            "gamma_flip": gamma_flip,
        },
        "directional_resolution_needed": bool(directional_resolution_needed),
        "activation_policy_fallback_active": bool(policy_fallback_active),
        "activation_policy_fallback_reason": "activation_score_policy_config_unavailable" if policy_fallback_active else None,
        **neutralization,
    }
    return explainability


def _safe_weekday(valuation_time):
    """Return weekday int (0=Mon) from valuation_time, coercing strings via pd.Timestamp."""
    if valuation_time is None:
        import datetime as _dt
        return _dt.datetime.now().weekday()
    if hasattr(valuation_time, "weekday"):
        return valuation_time.weekday()
    try:
        import pandas as _pd
        return _pd.Timestamp(valuation_time).weekday()
    except Exception:
        import datetime as _dt
        return _dt.datetime.now().weekday()


def _estimate_days_to_expiry(option_chain_validation, valuation_time):
    """Estimate calendar days to expiry from chain validation and valuation time."""
    import datetime as _dt

    selected = (
        option_chain_validation.get("selected_expiry")
        if isinstance(option_chain_validation, dict)
        else None
    )
    if selected is None or valuation_time is None:
        return None
    try:
        import pandas as _pd
        expiry_ts = _pd.Timestamp(selected)
        val_ts = _pd.Timestamp(valuation_time)

        # Date-only expiries parse as midnight and can incorrectly look expired
        # intraday; normalize them to exchange close for same-day contracts.
        selected_str = str(selected)
        if len(selected_str) <= 10:
            expiry_ts = _pd.Timestamp(f"{expiry_ts.date()} 15:30:00", tz="Asia/Kolkata")
        if expiry_ts.tzinfo is None:
            expiry_ts = expiry_ts.tz_localize("Asia/Kolkata")
        if val_ts.tzinfo is None:
            val_ts = val_ts.tz_localize("Asia/Kolkata")

        delta = (expiry_ts - val_ts).total_seconds() / 86400.0
        return max(delta, 0.0)
    except Exception:
        return None


def generate_trade(
    symbol,
    spot,
    option_chain,
    previous_chain=None,
    previous_direction=None,
    reversal_age=None,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None,
    spot_validation=None,
    option_chain_validation=None,
    apply_budget_constraint=False,
    requested_lots=NUMBER_OF_LOTS,
    lot_size=LOT_SIZE,
    max_capital=MAX_CAPITAL_PER_TRADE,
    backtest_mode=False,
    macro_event_state=None,
    macro_news_state=None,
    global_risk_state=None,
    pre_market_context=None,
    live_calibration_gate=None,
    live_directional_gate=None,
    holding_profile="AUTO",
    valuation_time=None,
    target_profit_percent=TARGET_PROFIT_PERCENT,
    stop_loss_percent=STOP_LOSS_PERCENT,
    operator_control_state=None,
):
    """
    Purpose:
        Assemble the final trade or no-trade payload for one market snapshot.
    
    Context:
        This is the engine's top-level orchestration entry point. It sits after data normalization and analytics extraction, then layers probability estimates, macro context, risk overlays, strike selection, and position sizing into a single payload that live runtime, replay tools, and research logging can all consume.
    
    Inputs:
        symbol (Any): Underlying symbol or index identifier.
        spot (Any): Current underlying spot price.
        option_chain (Any): Current option-chain snapshot.
        previous_chain (Any): Previous option-chain snapshot used for change-sensitive features such as flow and open-interest shifts.
        day_high (Any): Session high used for intraday range context.
        day_low (Any): Session low used for intraday range context.
        day_open (Any): Session open used for intraday context and early-session fallback logic.
        prev_close (Any): Previous session close used as a reference anchor.
        lookback_avg_range_pct (Any): Historical average range percentage used to normalize today's move.
        spot_validation (Any): Validation summary for the spot snapshot.
        option_chain_validation (Any): Validation summary for the option-chain snapshot.
        apply_budget_constraint (Any): Whether capital-budget rules should be enforced during trade construction.
        requested_lots (Any): Requested lot count before any optimizer or budget cap adjusts size.
        lot_size (Any): Contract lot size used when translating premium into capital required.
        max_capital (Any): Maximum capital budget allowed for the trade.
        backtest_mode (Any): Whether the snapshot is being evaluated in a backtest or replay context.
        macro_event_state (Any): Scheduled-event state produced by the macro layer.
        macro_news_state (Any): Headline-driven macro state produced by the news layer.
        global_risk_state (Any): Precomputed cross-asset risk state, when already available.
        holding_profile (Any): Holding intent used by overnight-sensitive overlays.
        valuation_time (Any): Timestamp used when normalizing expiries and Greeks.
        target_profit_percent (Any): Target-profit percentage passed into the exit model.
        stop_loss_percent (Any): Stop-loss percentage passed into the exit model.
    
    Returns:
        dict | None: Final trade or no-trade payload. Returns `None` only when the option chain is unusable at the very first gate.
    
    Notes:
        The returned payload doubles as the live engine contract and the structured record captured by evaluation and tuning workflows, so the function keeps diagnostics and decision-state fields explicit.
    """
    if option_chain is None or option_chain.empty:
        return None

    selected_expiry = option_chain_validation.get("selected_expiry") if isinstance(option_chain_validation, dict) else None

    days_to_expiry = _estimate_days_to_expiry(option_chain_validation, valuation_time)

    # Build global context fields early so market-state fan-out can consume
    # volatility fallbacks (e.g., India VIX when provider IV is unavailable).
    _grs = global_risk_state if isinstance(global_risk_state, dict) else {}
    _grf = _grs.get("global_risk_features", {}) if isinstance(_grs.get("global_risk_features"), dict) else {}

    # Normalize provider-specific column names and enrich missing Greeks once so
    # every downstream model works off a consistent option-chain schema.
    df = normalize_option_chain(option_chain, spot=spot, valuation_time=valuation_time)
    prev_df = (
        normalize_option_chain(previous_chain, spot=spot, valuation_time=valuation_time)
        if previous_chain is not None else None
    )
    try:
        market_state = _collect_market_state(
            df,
            spot,
            symbol=symbol,
            prev_df=prev_df,
            days_to_expiry=days_to_expiry,
            india_vix_level=_grf.get("india_vix_level"),
            fallback_iv=option_chain_validation.get("fallback_iv") if isinstance(option_chain_validation, dict) else None,
        )
    except TypeError:
        market_state = _collect_market_state(
            df,
            spot,
            symbol=symbol,
            prev_df=prev_df,
            days_to_expiry=days_to_expiry,
        )

    # Build global context for v2 ML model features (available before probability).
    _mes = macro_event_state if isinstance(macro_event_state, dict) else {}
    _mns = macro_news_state if isinstance(macro_news_state, dict) else {}
    _global_ctx = {
        "india_vix_level": _grf.get("india_vix_level"),
        "india_vix_change_24h": _grf.get("india_vix_change_24h"),
        "oil_shock_score": _grf.get("oil_shock_score"),
        "commodity_risk_score": _grf.get("commodity_risk_score"),
        "volatility_shock_score": _grf.get("volatility_shock_score"),
        "macro_event_risk_score": _mes.get("macro_event_risk_score", 0.0),
        "macro_regime": _mns.get("macro_regime", _mes.get("macro_regime", "MACRO_NEUTRAL")),
        "days_to_expiry": days_to_expiry,
        "weekday": _safe_weekday(valuation_time),
    }
    
    # Inject volatility shock score into market_state for regime-aware direction weighting
    # This allows the direction decision to be sensitive to elevated volatility environments
    market_state["volatility_shock_score"] = _grf.get("volatility_shock_score", 0.0)

    probability_state = _compute_probability_state(
        df,
        spot=spot,
        symbol=symbol,
        market_state=market_state,
        day_high=day_high,
        day_low=day_low,
        day_open=day_open,
        prev_close=prev_close,
        lookback_avg_range_pct=lookback_avg_range_pct,
        global_context=_global_ctx,
    )
    intraday_range_pct = probability_state["components"]["intraday_range_pct"]

    # Keep the analytics subset explicit because these values feed both
    # confidence checks and the final audit payload consumed by research tools.
    analytics_state = {
        "flip": market_state["flip"],
        "dealer_pos": market_state["dealer_pos"],
        "vol_regime": market_state["vol_regime"],
        "gamma_regime": market_state["gamma_regime"],
        "vanna_regime": market_state["greek_exposures"].get("vanna_regime"),
        "charm_regime": market_state["greek_exposures"].get("charm_regime"),
        "final_flow_signal": market_state["final_flow_signal"],
        "hedging_bias": market_state["hedging_bias"],
        "atm_iv": market_state["atm_iv"],
        "support_wall": market_state["support_wall"],
        "resistance_wall": market_state["resistance_wall"],
        "market_gamma_summary": market_state["market_gamma_summary"],
        "provider_health": option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else None,
    }

    data_quality = _compute_data_quality(
        spot_validation=spot_validation,
        option_chain_validation=option_chain_validation,
        analytics_state=analytics_state,
        probability_state=probability_state,
    )
    feature_reliability_overlay = _compute_feature_reliability_overlay(option_chain_validation)
    feature_reliability_weights = feature_reliability_overlay.get("weights") if isinstance(feature_reliability_overlay.get("weights"), dict) else {}
    flow_reliability_weight = _blend_feature_reliability(feature_reliability_weights, flow=0.75, liquidity=0.15, greeks=0.10)
    surface_reliability_weight = _blend_feature_reliability(feature_reliability_weights, vol_surface=0.75, greeks=0.15, liquidity=0.10)
    dealer_reliability_weight = _blend_feature_reliability(feature_reliability_weights, flow=0.35, liquidity=0.35, greeks=0.30)
    option_efficiency_reliability_weight = _blend_feature_reliability(
        feature_reliability_weights,
        liquidity=0.45,
        greeks=0.30,
        vol_surface=0.25,
    )

    macro_event_state = macro_event_state if isinstance(macro_event_state, dict) else {}
    macro_event_risk_score = int(_safe_float(macro_event_state.get("macro_event_risk_score"), 0))
    event_window_status = macro_event_state.get("event_window_status", "NO_EVENT_DATA")
    event_lockdown_flag = bool(macro_event_state.get("event_lockdown_flag", False))
    minutes_to_next_event = macro_event_state.get("minutes_to_next_event")
    next_event_name = macro_event_state.get("next_event_name")
    active_event_name = macro_event_state.get("active_event_name")
    signal_state = _compute_signal_state(
        spot=spot,
        symbol=symbol,
        previous_direction=previous_direction,
        reversal_age=reversal_age,
        day_open=day_open,
        prev_close=prev_close,
        intraday_range_pct=intraday_range_pct,
        backtest_mode=backtest_mode,
        market_state=market_state,
        probability_state=probability_state,
        option_chain_validation=option_chain_validation,
        macro_news_state=macro_news_state,
        global_risk_state=global_risk_state,
    )
    direction = signal_state["direction"]
    direction_source = signal_state["direction_source"]
    expansion_mode = bool(signal_state.get("expansion_mode", False))
    expansion_direction = signal_state.get("expansion_direction")
    breakout_evidence = _safe_float(signal_state.get("breakout_evidence"), 0.0)
    reversal_context = bool(signal_state.get("reversal_context", False))
    reversal_stage = signal_state.get("reversal_stage", "NONE")
    breakout_vote_count = int(_safe_float(signal_state.get("breakout_vote_count"), 0.0))
    bull_probability = _safe_float(signal_state.get("bull_probability"), 0.5)
    bear_probability = _safe_float(signal_state.get("bear_probability"), 0.5)
    trade_strength = signal_state["trade_strength"]
    scoring_breakdown = signal_state["scoring_breakdown"]
    confirmation = signal_state["confirmation"]

    # Path-aware filter uses consecutive snapshot spot deltas as a micro-path proxy.
    path_filtering_enabled = bool(int(_safe_float(get_trade_runtime_thresholds().get("enable_path_aware_filtering"), 1.0)))
    path_check = {
        "path_status": "DISABLED",
        "score_penalty": 0,
        "entry_veto": False,
        "mfe_observed_bps": None,
        "mae_observed_bps": None,
        "mae_zscore": None,
        "reasons": [],
    }
    if path_filtering_enabled and direction in {"CALL", "PUT"}:
        _mfe_bps, _mae_bps = _compute_path_observation_bps(
            symbol=symbol,
            selected_expiry=option_chain_validation.get("selected_expiry") if isinstance(option_chain_validation, dict) else None,
            valuation_time=valuation_time,
            spot=spot,
            direction=direction,
        )
        path_filter = _get_path_filter()
        path_check = path_filter.check_path_geometry(
            gamma_regime=market_state.get("gamma_regime"),
            direction=direction,
            mfe_observed_bps=_mfe_bps,
            mae_observed_bps=_mae_bps,
            window=f"{int(_safe_float(get_trade_runtime_thresholds().get('path_filtering_entry_confirmation_window_m'), 5.0))}m",
            mae_zscore_threshold=_safe_float(get_trade_runtime_thresholds().get("path_filtering_mae_zscore_threshold"), 1.5),
            hostile_path_score_penalty=-abs(int(_safe_float(get_trade_runtime_thresholds().get("path_filtering_hostile_score_penalty"), 15.0))),
            allow_veto=bool(int(_safe_float(get_trade_runtime_thresholds().get("path_filtering_delay_entry_on_hostile"), 1.0))),
        )

        if _safe_float(path_check.get("score_penalty"), 0.0) != 0:
            confirmation["score_adjustment"] += int(_safe_float(path_check.get("score_penalty"), 0.0))
            confirmation["reasons"].append("path_aware_filter_adjustment")
            confirmation_breakdown = confirmation.get("breakdown") if isinstance(confirmation.get("breakdown"), dict) else {}
            confirmation_breakdown["path_aware_filter_score"] = int(_safe_float(path_check.get("score_penalty"), 0.0))
            confirmation["breakdown"] = confirmation_breakdown

    macro_news_adjustments = compute_macro_news_adjustments(
        direction=direction,
        macro_news_state=macro_news_state,
    )
    macro_news_state = macro_news_state if isinstance(macro_news_state, dict) else {}
    macro_news_stale = bool(macro_news_state.get("neutral_fallback", False)) and any(
        "stale" in str(item).lower() for item in (macro_news_state.get("warnings") or [])
    )
    global_risk_features_snapshot = global_risk_state.get("global_risk_features", {}) if isinstance(global_risk_state, dict) else {}
    global_macro_data_stale = bool(global_risk_features_snapshot.get("market_data_stale", False)) or bool(
        global_risk_features_snapshot.get("market_features_neutralized", False)
    )
    freshness_overlay_reasons = []
    freshness_score_penalty = 0
    freshness_size_cap = 1.0
    if macro_news_stale:
        freshness_score_penalty += int(_safe_float(get_trade_runtime_thresholds().get("headline_staleness_score_penalty"), 4.0))
        freshness_size_cap = min(
            freshness_size_cap,
            _safe_float(get_trade_runtime_thresholds().get("headline_staleness_size_cap"), 0.75),
        )
        freshness_overlay_reasons.append("headline_data_stale")
    if global_macro_data_stale:
        freshness_score_penalty += int(_safe_float(get_trade_runtime_thresholds().get("global_macro_staleness_score_penalty"), 5.0))
        freshness_size_cap = min(
            freshness_size_cap,
            _safe_float(get_trade_runtime_thresholds().get("global_macro_staleness_size_cap"), 0.70),
        )
        freshness_overlay_reasons.append("global_macro_data_stale")
    event_overlay_probability_multiplier = _safe_float(
        macro_news_adjustments.get("event_overlay_probability_multiplier"),
        1.0,
    )
    if probability_state.get("hybrid_move_probability") is not None:
        probability_state["hybrid_move_probability"] = round(
            _clip(
                _safe_float(probability_state.get("hybrid_move_probability"), 0.0)
                * event_overlay_probability_multiplier,
                0.0,
                1.0,
            ),
            4,
        )
    if bool(macro_news_adjustments.get("event_overlay_suppress_signal", False)):
        direction = None
        confirmation["reasons"].append("event_overlay_signal_suppressed")

    event_cfg = get_event_window_policy_config()

    # Scheduled events and headline overlays are scored separately so operators
    # can see whether a downgrade came from the calendar, the news tape, or both.
    macro_event_score_adjustment = 0
    if event_window_status == "PRE_EVENT_WATCH":
        macro_event_score_adjustment = (
            event_cfg.pre_event_watch_penalty_high
            if macro_event_risk_score >= event_cfg.watch_risk_threshold
            else event_cfg.pre_event_watch_penalty_normal
        )
    elif event_window_status == "POST_EVENT_COOLDOWN":
        macro_event_score_adjustment = (
            event_cfg.post_event_cooldown_penalty_high
            if macro_event_risk_score >= event_cfg.watch_risk_threshold
            else event_cfg.post_event_cooldown_penalty_normal
        )
    elif event_window_status in {"PRE_EVENT_LOCKDOWN", "LIVE_EVENT"}:
        macro_event_score_adjustment = event_cfg.lockdown_penalty

    chain_confirmation_score = int(_safe_float(confirmation.get("score_adjustment"), 0.0))
    scaled_chain_confirmation_score = _scale_adjustment_by_reliability(chain_confirmation_score, flow_reliability_weight)
    confirmation["score_adjustment"] = scaled_chain_confirmation_score + int(
        _safe_float(macro_news_adjustments["macro_confirmation_adjustment"], 0.0)
    )

    event_overlay_score = _safe_float(
        macro_news_adjustments.get("event_overlay_score_adjustment"),
        0.0,
    )
    macro_news_total_score = _safe_float(macro_news_adjustments.get("macro_adjustment_score"), 0.0)
    macro_news_core_score = macro_news_total_score - event_overlay_score
    scoring_breakdown["chain_confirmation_score_raw"] = chain_confirmation_score
    scoring_breakdown["chain_confirmation_reliability_weight"] = flow_reliability_weight
    scoring_breakdown["chain_confirmation_reliability_delta"] = scaled_chain_confirmation_score - chain_confirmation_score
    scoring_breakdown["confirmation_filter_score"] = confirmation["score_adjustment"]
    scoring_breakdown["macro_event_score"] = macro_event_score_adjustment
    scoring_breakdown["macro_news_score"] = macro_news_core_score
    scoring_breakdown["event_overlay_score"] = event_overlay_score
    global_risk_trade_modifiers = derive_global_risk_trade_modifiers(global_risk_state)
    global_risk_adjustment_score = global_risk_trade_modifiers["effective_adjustment_score"]
    scoring_breakdown["global_risk_base_adjustment_score"] = global_risk_trade_modifiers["base_adjustment_score"]
    scoring_breakdown["global_risk_feature_adjustment_score"] = global_risk_trade_modifiers["feature_adjustment_score"]
    scoring_breakdown["global_risk_adjustment_score"] = global_risk_adjustment_score

    # Each overlay returns both diagnostics and a score contribution so the
    # engine can explain not just the decision, but why the decision changed.
    macro_news_shock_norm = _clip(_safe_float(macro_news_adjustments.get("volatility_shock_score"), 0.0) / 100.0, 0.0, 1.0)
    global_risk_shock_norm = _clip(
        _safe_float(
            global_risk_state.get("global_risk_features", {}).get("volatility_shock_score")
            if isinstance(global_risk_state, dict)
            else 0.0,
            0.0,
        ),
        0.0,
        1.0,
    )
    blended_volatility_shock_score = max(macro_news_shock_norm, global_risk_shock_norm)

    gamma_vol_state = build_gamma_vol_acceleration_state(
        gamma_regime=market_state["gamma_regime"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
        dealer_hedging_bias=market_state["hedging_bias"],
        liquidity_vacuum_state=market_state["vacuum_state"],
        intraday_range_pct=intraday_range_pct,
        volatility_compression_score=(
            global_risk_state.get("global_risk_features", {}).get("volatility_compression_score")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        volatility_shock_score=blended_volatility_shock_score,
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        volatility_explosion_probability=(
            global_risk_state.get("global_risk_features", {}).get("volatility_explosion_probability")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        holding_profile=holding_profile,
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        gamma_flip_drift=market_state.get("gamma_flip_drift"),
    )
    gamma_vol_trade_modifiers = derive_gamma_vol_trade_modifiers(gamma_vol_state, direction=direction)
    gamma_vol_raw_adjustment_score = int(_safe_float(gamma_vol_trade_modifiers["effective_adjustment_score"], 0.0))
    gamma_vol_adjustment_score = _scale_adjustment_by_reliability(gamma_vol_raw_adjustment_score, surface_reliability_weight)
    scoring_breakdown["gamma_vol_base_adjustment_score"] = gamma_vol_trade_modifiers["base_adjustment_score"]
    scoring_breakdown["gamma_vol_alignment_adjustment_score"] = gamma_vol_trade_modifiers["alignment_adjustment_score"]
    scoring_breakdown["gamma_vol_reliability_weight"] = surface_reliability_weight
    scoring_breakdown["gamma_vol_reliability_delta"] = gamma_vol_adjustment_score - gamma_vol_raw_adjustment_score
    scoring_breakdown["gamma_vol_adjustment_score"] = gamma_vol_adjustment_score
    dealer_pressure_state = build_dealer_hedging_pressure_state(
        spot=spot,
        gamma_regime=market_state["gamma_regime"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
        dealer_position=market_state["dealer_pos"],
        dealer_hedging_bias=market_state["hedging_bias"],
        dealer_hedging_flow=market_state["hedging_flow"],
        market_gamma=market_state["market_gamma_summary"],
        gamma_clusters=market_state["gamma_clusters"],
        liquidity_levels=market_state["liquidity_levels"],
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        liquidity_vacuum_state=market_state["vacuum_state"],
        intraday_gamma_state=market_state["intraday_gamma_state"],
        intraday_range_pct=intraday_range_pct,
        flow_signal=market_state["flow_signal_value"],
        smart_money_flow=market_state["smart_money_signal_value"],
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        volatility_explosion_probability=(
            global_risk_state.get("global_risk_features", {}).get("volatility_explosion_probability")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        gamma_vol_acceleration_score=gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        holding_profile=holding_profile,
        max_pain_dist=market_state.get("max_pain_dist"),
        max_pain_zone=market_state.get("max_pain_zone"),
        days_to_expiry=market_state.get("days_to_expiry"),
    )
    dealer_pressure_trade_modifiers = derive_dealer_pressure_trade_modifiers(dealer_pressure_state, direction=direction)
    dealer_pressure_raw_adjustment_score = int(_safe_float(dealer_pressure_trade_modifiers["effective_adjustment_score"], 0.0))
    dealer_pressure_adjustment_score = _scale_adjustment_by_reliability(
        dealer_pressure_raw_adjustment_score,
        dealer_reliability_weight,
    )
    scoring_breakdown["dealer_pressure_base_adjustment_score"] = dealer_pressure_trade_modifiers["base_adjustment_score"]
    scoring_breakdown["dealer_pressure_alignment_adjustment_score"] = dealer_pressure_trade_modifiers["alignment_adjustment_score"]
    scoring_breakdown["dealer_pressure_reliability_weight"] = dealer_reliability_weight
    scoring_breakdown["dealer_pressure_reliability_delta"] = dealer_pressure_adjustment_score - dealer_pressure_raw_adjustment_score
    scoring_breakdown["dealer_pressure_adjustment_score"] = dealer_pressure_adjustment_score
    global_risk_features = global_risk_state.get("global_risk_features", {}) if isinstance(global_risk_state, dict) else {}
    india_vix_level = global_risk_features.get("india_vix_level")
    india_vix_change_24h = global_risk_features.get("india_vix_change_24h")
    option_efficiency_state = {}
    option_efficiency_trade_modifiers = derive_option_efficiency_trade_modifiers(option_efficiency_state)
    option_efficiency_raw_adjustment_score = int(_safe_float(option_efficiency_trade_modifiers["option_efficiency_adjustment_score"], 0.0))
    option_efficiency_adjustment_score = _scale_adjustment_by_reliability(
        option_efficiency_raw_adjustment_score,
        option_efficiency_reliability_weight,
    )
    scoring_breakdown["option_efficiency_adjustment_score"] = option_efficiency_adjustment_score
    scoring_breakdown["option_efficiency_reliability_weight"] = option_efficiency_reliability_weight
    scoring_breakdown["option_efficiency_reliability_delta"] = option_efficiency_adjustment_score - option_efficiency_raw_adjustment_score

    # Trade strength is accumulated in layers: base directional edge, then
    # confirmation/macro adjustments, then stateful risk overlays.
    adjusted_trade_strength = int(
        _clip(
            trade_strength
            + confirmation["score_adjustment"]
            + macro_event_score_adjustment
            + macro_news_core_score
            + event_overlay_score,
            0,
            100,
        )
    )
    if freshness_score_penalty > 0:
        adjusted_trade_strength = int(_clip(adjusted_trade_strength - freshness_score_penalty, 0, 100))
    adjusted_trade_strength = int(
        _clip(
            adjusted_trade_strength + global_risk_adjustment_score + gamma_vol_adjustment_score + dealer_pressure_adjustment_score,
            0,
            100,
        )
    )
    runtime_thresholds = get_trade_runtime_thresholds()
    min_trade_strength = (
        BACKTEST_MIN_TRADE_STRENGTH
        if backtest_mode
        else runtime_thresholds["min_trade_strength"]
    )
    min_composite_score = int(_safe_float(runtime_thresholds.get("min_composite_score"), 55.0))

    regime_thresholds = _resolve_regime_thresholds(
        runtime_thresholds=runtime_thresholds,
        base_min_trade_strength=min_trade_strength,
        base_min_composite_score=min_composite_score,
        market_state=market_state,
    )
    min_trade_strength = regime_thresholds["effective_min_trade_strength"]
    min_composite_score = regime_thresholds["effective_min_composite_score"]

    # Confidence-weighted gate: relax or tighten min_trade_strength based on
    # data quality and confirmation alignment, absent of full Platt scaling.
    if not backtest_mode:
        _dq_status = _as_upper(data_quality.get("status", ""))
        _conf_status = _as_upper(confirmation.get("status", ""))
        _high_confidence = (
            _dq_status == "GOOD"
            and _conf_status in {"STRONG_CONFIRMATION", "CONFIRMED"}
        )
        _low_confidence = (
            _dq_status == "WEAK"
            or _conf_status in {"CONFLICT", "NO_DIRECTION"}
        )
        _relief = int(_safe_float(runtime_thresholds.get("high_confidence_strength_relief"), 5.0))
        _surcharge = int(_safe_float(runtime_thresholds.get("low_confidence_strength_surcharge"), 8.0))
        if _high_confidence:
            min_trade_strength = int(_clip(min_trade_strength - _relief, 40, 100))
        elif _low_confidence:
            min_trade_strength = int(_clip(min_trade_strength + _surcharge, 0, 100))

    if reversal_stage == "CONFIRMED_REVERSAL":
        reversal_strength_relief = int(_safe_float(runtime_thresholds.get("reversal_stage_strength_relief"), 4.0))
        min_trade_strength = int(_clip(min_trade_strength - reversal_strength_relief, 40, 100))
    if expansion_mode and expansion_direction == direction:
        expansion_strength_relief = int(_safe_float(runtime_thresholds.get("expansion_mode_strength_relief"), 3.0))
        min_trade_strength = int(_clip(min_trade_strength - expansion_strength_relief, 40, 100))

    provider_health = option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else {}

    # Apply pre-market adjustments if in pre-market session
    if pre_market_context is not None:
        pre_market_adj = apply_pre_market_adjustments_to_signal(
            base_trade_strength=adjusted_trade_strength,
            pre_market_context=pre_market_context,
        )
        if not pre_market_adj.get("signal_eligible", True):
            # Pre-market signals disabled or quality gates failed
            direction = None
            confirmation["reasons"].append("pre_market_signal_eligibility_gate_failed")
            scoring_breakdown["pre_market_eligibility"] = False
        else:
            # Apply quality multiplier if in pre-market
            if pre_market_adj.get("quality_multiplier", 1.0) != 1.0:
                adjusted_trade_strength = int(
                    _clip(
                        adjusted_trade_strength * pre_market_adj.get("quality_multiplier", 1.0),
                        0,
                        100,
                    )
                )
                min_trade_strength = pre_market_adj.get("min_trade_strength_threshold", min_trade_strength)
                confirmation["reasons"].append("pre_market_quality_multiplier_applied")
                scoring_breakdown["pre_market_quality_multiplier"] = pre_market_adj.get("quality_multiplier", 1.0)
                scoring_breakdown["pre_market_eligibility"] = True
        scoring_breakdown["pre_market_context_available"] = True
    else:
        scoring_breakdown["pre_market_context_available"] = False

    provider_health = provider_health if isinstance(provider_health, dict) else {}
    provider_health_summary = _as_upper(provider_health.get("summary_status"))

    at_flip_penalty_applied = 0
    at_flip_size_cap = 1.0
    at_flip_toxic_context = False
    if _as_upper(market_state["spot_vs_flip"]) == "AT_FLIP":
        at_flip_penalty_applied = int(_safe_float(runtime_thresholds.get("at_flip_trade_strength_penalty"), 8.0))
        adjusted_trade_strength = int(_clip(adjusted_trade_strength - at_flip_penalty_applied, 0, 100))
        dealer_position_upper = _as_upper(market_state.get("dealer_pos"))
        at_flip_gamma_regime = canonical_gamma_regime(market_state.get("gamma_regime"))
        at_flip_toxic_context = (
            at_flip_gamma_regime == "POSITIVE_GAMMA"
            and ("SHORT" in dealer_position_upper)
            and ("GAMMA" in dealer_position_upper)
        )
        at_flip_size_cap = _safe_float(
            runtime_thresholds.get("at_flip_toxic_size_cap" if at_flip_toxic_context else "at_flip_size_cap"),
            0.50 if at_flip_toxic_context else 0.75,
        )

    gamma_vol_acceleration_score_normalized = _normalize_gamma_vol_score(
        gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        int(_safe_float(runtime_thresholds.get("gamma_vol_normalization_scale"), 100.0)),
        int(_safe_float(runtime_thresholds.get("gamma_vol_winsor_lower"), 0.0)),
        int(_safe_float(runtime_thresholds.get("gamma_vol_winsor_upper"), 100.0)),
    )
    iv_surface_residual_penalty_score = int(_safe_float(market_state.get("iv_surface_residual_penalty_score"), 0.0))
    if iv_surface_residual_penalty_score > 0:
        adjusted_trade_strength = int(_clip(adjusted_trade_strength - iv_surface_residual_penalty_score, 0, 100))

    iv_hv_adjustment_score = 0
    iv_hv_regime = _as_upper(market_state.get("iv_hv_regime"))
    if iv_hv_regime == "IV_CHEAP":
        iv_hv_adjustment_score = int(_safe_float(runtime_thresholds.get("iv_hv_cheap_trade_strength_bonus"), 2.0))
    elif iv_hv_regime == "IV_RICH":
        iv_hv_adjustment_score = -abs(int(_safe_float(runtime_thresholds.get("iv_hv_rich_trade_strength_penalty"), 3.0)))
    if iv_hv_adjustment_score != 0:
        adjusted_trade_strength = int(_clip(adjusted_trade_strength + iv_hv_adjustment_score, 0, 100))

    structural_imbalance_audit = _compute_structural_imbalance_audit(
        market_state=market_state,
        direction=direction,
    )

    scoring_breakdown["base_trade_strength"] = trade_strength
    scoring_breakdown["freshness_uncertainty_score"] = -freshness_score_penalty
    scoring_breakdown["at_flip_trade_strength_penalty"] = -at_flip_penalty_applied
    scoring_breakdown["iv_surface_residual_penalty"] = -iv_surface_residual_penalty_score
    scoring_breakdown["iv_hv_adjustment_score"] = iv_hv_adjustment_score
    scoring_breakdown["total_score"] = adjusted_trade_strength
    signal_regime = classify_signal_regime(
        direction=direction,
        adjusted_trade_strength=adjusted_trade_strength,
        final_flow_signal=market_state["final_flow_signal"],
        gamma_regime=market_state["gamma_regime"],
        confirmation_status=confirmation["status"],
        event_lockdown_flag=event_lockdown_flag or macro_news_adjustments["event_lockdown_flag"],
        data_quality_status=data_quality["status"],
    )

    # This payload is intentionally verbose because it serves three audiences:
    # the live trader, the risk overlays, and the offline evaluation dataset.
    base_payload = {
        "symbol": symbol,
        "spot": round(spot, 2),
        "ranked_strike_candidates": [],
        "gamma_exposure": round(market_state["gamma"], 2) if market_state["gamma"] is not None else None,
        "market_gamma": market_state["market_gamma_summary"],
        "gamma_flip": _to_python_number(market_state["flip"]),
        "spot_vs_flip": market_state["spot_vs_flip"],
        "gamma_regime": market_state["gamma_regime"],
        "gamma_clusters": market_state["gamma_clusters"],
        "delta_exposure": market_state["greek_exposures"].get("delta_exposure"),
        "gamma_exposure_greeks": market_state["greek_exposures"].get("gamma_exposure_greeks"),
        "theta_exposure": market_state["greek_exposures"].get("theta_exposure"),
        "vega_exposure": market_state["greek_exposures"].get("vega_exposure"),
        "rho_exposure": market_state["greek_exposures"].get("rho_exposure"),
        "vanna_exposure": market_state["greek_exposures"].get("vanna_exposure"),
        "charm_exposure": market_state["greek_exposures"].get("charm_exposure"),
        "vanna_regime": market_state["greek_exposures"].get("vanna_regime"),
        "charm_regime": market_state["greek_exposures"].get("charm_regime"),
        "greeks_data_warning": market_state["greek_exposures"].get("greeks_data_warning"),
        "missing_greek_columns": market_state["greek_exposures"].get("missing_greek_columns", []),
        "dealer_position": market_state["dealer_pos"],
        "dealer_inventory_basis": market_state["dealer_metrics"].get("basis"),
        "call_oi_change": market_state["dealer_metrics"].get("call_oi_change"),
        "put_oi_change": market_state["dealer_metrics"].get("put_oi_change"),
        "net_oi_change_bias": market_state["dealer_metrics"].get("net_oi_change_bias"),
        "dealer_hedging_flow": market_state["hedging_flow"],
        "dealer_hedging_bias": market_state["hedging_bias"],
        "intraday_gamma_state": market_state["intraday_gamma_state"],
        "volatility_regime": market_state["vol_regime"],
        "vol_surface_regime": market_state["surface_regime"],
        "iv_surface_residual_status": market_state.get("iv_surface_residual_status"),
        "iv_surface_residual_rmse": market_state.get("iv_surface_residual_rmse"),
        "iv_surface_residual_outlier_ratio": market_state.get("iv_surface_residual_outlier_ratio"),
        "iv_surface_term_structure_inversion_count": market_state.get("iv_surface_term_structure_inversion_count"),
        "iv_surface_residual_penalty_score": iv_surface_residual_penalty_score,
        "iv_surface_residual_penalty_reasons": market_state.get("iv_surface_residual_penalty_reasons", []),
        "atm_iv": round(float(market_state["atm_iv"]), 2) if market_state["atm_iv"] is not None else None,
        "max_pain": market_state.get("max_pain"),
        "max_pain_dist": market_state.get("max_pain_dist"),
        "max_pain_zone": market_state.get("max_pain_zone"),
        "atm_straddle_price": market_state.get("atm_straddle_price"),
        "expected_move_up": market_state.get("expected_move_up"),
        "expected_move_down": market_state.get("expected_move_down"),
        # Keep this field as straddle-derived expected-move percent for
        # consistency with atm_straddle_price in user-facing output.
        "expected_move_pct": market_state.get("expected_move_pct"),
        "volume_pcr": market_state.get("volume_pcr"),
        "volume_pcr_atm": market_state.get("volume_pcr_atm"),
        "volume_pcr_regime": market_state.get("volume_pcr_regime"),
        "iv_hv_spread": market_state.get("iv_hv_spread"),
        "iv_hv_spread_relative": market_state.get("iv_hv_spread_relative"),
        "iv_hv_regime": market_state.get("iv_hv_regime", "UNAVAILABLE"),
        "realized_hv_pct": market_state.get("realized_hv_pct"),
        "gamma_flip_drift": market_state.get("gamma_flip_drift"),
        "flow_signal": market_state["flow_signal_value"],
        "smart_money_flow": market_state["smart_money_signal_value"],
        "final_flow_signal": market_state["final_flow_signal"],
        "gamma_event": market_state["gamma_event"],
        "support_wall": market_state["support_wall"],
        "resistance_wall": market_state["resistance_wall"],
        "liquidity_levels": market_state["liquidity_levels"],
        "liquidity_voids": market_state["voids"],
        "liquidity_void_signal": market_state["void_signal"],
        "liquidity_vacuum_zones": market_state["vacuum_zones"],
        "liquidity_vacuum_state": market_state["vacuum_state"],
        "dealer_liquidity_map": market_state["dealer_liquidity_map"],
        "market_state_timings": market_state.get("market_state_timings"),
        "rule_move_probability": probability_state["rule_move_probability"],
        "ml_move_probability": probability_state["ml_move_probability"],
        "hybrid_move_probability": probability_state["hybrid_move_probability"],
        "large_move_probability": probability_state["hybrid_move_probability"],  # legacy alias — use hybrid_move_probability
        "move_probability_components": probability_state["components"],
        "spot_validation": spot_validation,
        "option_chain_validation": option_chain_validation,
        "provider_health": option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else None,
        "provider_health_score": provider_health.get("market_data_readiness_score"),
        "provider_health_tier": provider_health.get("market_data_readiness_tier"),
        "data_readiness_score": provider_health.get("market_data_readiness_score"),
        "data_confidence_tier": provider_health.get("market_data_readiness_tier"),
        "analytics_usable": bool(option_chain_validation.get("analytics_usable")) if isinstance(option_chain_validation, dict) else False,
        "execution_suggestion_usable": bool(option_chain_validation.get("execution_suggestion_usable")) if isinstance(option_chain_validation, dict) else False,
        "tradable_data": option_chain_validation.get("tradable_data") if isinstance(option_chain_validation, dict) else None,
        "feature_reliability_weights": option_chain_validation.get("feature_reliability_weights") if isinstance(option_chain_validation, dict) else None,
        "feature_reliability_status": feature_reliability_overlay.get("status"),
        "feature_reliability_score": feature_reliability_overlay.get("aggregate_score"),
        "feature_reliability_penalty_score": feature_reliability_overlay.get("trade_strength_penalty"),
        "feature_reliability_composite_penalty": feature_reliability_overlay.get("runtime_composite_penalty"),
        "feature_reliability_reasons": feature_reliability_overlay.get("reasons", []),
        "provider_health_summary": provider_health_summary,
        "data_quality_score": data_quality["score"],
        "data_quality_status": data_quality["status"],
        "data_quality_reasons": data_quality["reasons"],
        "analytics_quality": data_quality["analytics_quality"],
        "confirmation_status": confirmation["status"],
        "confirmation_veto": confirmation["veto"],
        "confirmation_reasons": confirmation["reasons"],
        "confirmation_breakdown": confirmation["breakdown"],
        "expansion_mode": expansion_mode,
        "expansion_direction": expansion_direction,
        "breakout_evidence": round(float(breakout_evidence or 0.0), 3),
        "path_aware_status": path_check.get("path_status"),
        "path_aware_score_penalty": int(_safe_float(path_check.get("score_penalty"), 0.0)),
        "path_aware_entry_veto": bool(path_check.get("entry_veto", False)),
        "path_aware_mfe_observed_bps": _to_python_number(path_check.get("mfe_observed_bps")),
        "path_aware_mae_observed_bps": _to_python_number(path_check.get("mae_observed_bps")),
        "path_aware_mae_zscore": _to_python_number(path_check.get("mae_zscore")),
        "path_aware_reasons": path_check.get("reasons", []),
        "direction_source": direction_source,
        "direction_vote_shadow": signal_state.get("direction_vote_shadow"),
        "direction_source_vote": signal_state.get("direction_source_vote"),
        "direction_vote_call_probability": _to_python_number(signal_state.get("direction_vote_call_probability")),
        "direction_vote_put_probability": _to_python_number(signal_state.get("direction_vote_put_probability")),
        "direction_head_enabled": bool(signal_state.get("direction_head_enabled", False)),
        "direction_head_direction": signal_state.get("direction_head_direction"),
        "direction_head_probability_up": _to_python_number(signal_state.get("direction_head_probability_up")),
        "direction_head_probability_up_raw": _to_python_number(signal_state.get("direction_head_probability_up_raw")),
        "direction_head_uncertainty": _to_python_number(signal_state.get("direction_head_uncertainty")),
        "direction_head_confidence": _to_python_number(signal_state.get("direction_head_confidence")),
        "direction_head_disagreement_with_vote": _to_python_number(signal_state.get("direction_head_disagreement_with_vote")),
        "direction_head_microstructure_friction_score": _to_python_number(signal_state.get("direction_head_microstructure_friction_score")),
        "direction_head_calibration_applied": bool(signal_state.get("direction_head_calibration_applied", False)),
        "direction_head_used_for_final": bool(signal_state.get("direction_head_used_for_final", False)),
        "direction_call_probability": round(float(_clip(bull_probability, 0.0, 1.0)), 4),
        "direction_put_probability": round(float(_clip(bear_probability, 0.0, 1.0)), 4),
        "trade_strength": adjusted_trade_strength,
        "signal_quality": classify_signal_quality(adjusted_trade_strength),
        "signal_regime": signal_regime,
        "scoring_breakdown": scoring_breakdown,
        "macro_event_risk_score": macro_event_risk_score,
        "event_window_status": event_window_status,
        "event_lockdown_flag": event_lockdown_flag,
        "minutes_to_next_event": minutes_to_next_event,
        "next_event_name": next_event_name,
        "active_event_name": active_event_name,
        "macro_regime": macro_news_adjustments["macro_regime"],
        "macro_sentiment_score": macro_news_adjustments["macro_sentiment_score"],
        "macro_news_volatility_shock_score": macro_news_adjustments["volatility_shock_score"],
        "news_confidence_score": macro_news_adjustments["news_confidence_score"],
        "macro_adjustment_score": macro_news_adjustments["macro_adjustment_score"],
        "macro_confirmation_adjustment": macro_news_adjustments["macro_confirmation_adjustment"],
        "macro_position_size_multiplier": macro_news_adjustments["macro_position_size_multiplier"],
        "macro_adjustment_reasons": macro_news_adjustments["macro_adjustment_reasons"],
        "headline_data_stale": macro_news_stale,
        "global_macro_data_stale": global_macro_data_stale,
        "freshness_overlay_reasons": freshness_overlay_reasons,
        "headline_data_stale": macro_news_stale,
        "global_macro_data_stale": global_macro_data_stale,
        "event_intelligence_enabled": bool((macro_news_state or {}).get("event_intelligence_enabled", False)),
        "event_bullish_score": ((macro_news_state or {}).get("event_features") or {}).get("bullish_event_score"),
        "event_bearish_score": ((macro_news_state or {}).get("event_features") or {}).get("bearish_event_score"),
        "event_vol_expansion_score": ((macro_news_state or {}).get("event_features") or {}).get("vol_expansion_score"),
        "event_vol_compression_score": ((macro_news_state or {}).get("event_features") or {}).get("vol_compression_score"),
        "event_uncertainty_score": ((macro_news_state or {}).get("event_features") or {}).get("event_uncertainty_score"),
        "event_gap_risk_score": ((macro_news_state or {}).get("event_features") or {}).get("gap_risk_score"),
        "event_catalyst_alignment_score": ((macro_news_state or {}).get("event_features") or {}).get("catalyst_alignment_score"),
        "event_contradictory_penalty": ((macro_news_state or {}).get("event_features") or {}).get("contradictory_event_penalty"),
        "event_cluster_score": ((macro_news_state or {}).get("event_features") or {}).get("recent_event_cluster_score"),
        "event_decayed_signal": ((macro_news_state or {}).get("event_features") or {}).get("decayed_event_signal"),
        "event_relevance_score": ((macro_news_state or {}).get("event_features") or {}).get("routed_event_relevance_score"),
        "event_count": ((macro_news_state or {}).get("event_features") or {}).get("event_count"),
        "event_routed_count": ((macro_news_state or {}).get("event_features") or {}).get("routed_event_count"),
        "event_explanations": (macro_news_state or {}).get("event_explanations", []),
        "event_overlay_probability_multiplier": macro_news_adjustments.get("event_overlay_probability_multiplier", 1.0),
        "event_overlay_size_multiplier": macro_news_adjustments.get("event_overlay_size_multiplier", 1.0),
        "event_overlay_score_adjustment": macro_news_adjustments.get("event_overlay_score_adjustment", 0),
        "event_overlay_suppress_signal": bool(macro_news_adjustments.get("event_overlay_suppress_signal", False)),
        "event_overlay_reasons": macro_news_adjustments.get("event_overlay_reasons", []),
        "global_risk_state": global_risk_state.get("global_risk_state") if isinstance(global_risk_state, dict) else "GLOBAL_NEUTRAL",
        "global_risk_state_score": global_risk_state.get("global_risk_score") if isinstance(global_risk_state, dict) else 0,
        "global_risk_overlay_score": None,
        "global_risk_score": global_risk_state.get("global_risk_score") if isinstance(global_risk_state, dict) else 0,
        "global_risk_state_reasons": global_risk_state.get("global_risk_reasons") if isinstance(global_risk_state, dict) else [],
        "global_risk_overlay_reasons": [],
        "overnight_gap_risk_score": global_risk_state.get("overnight_gap_risk_score") if isinstance(global_risk_state, dict) else 0,
        "volatility_expansion_risk_score": global_risk_state.get("volatility_expansion_risk_score") if isinstance(global_risk_state, dict) else 0,
        "overnight_hold_allowed": global_risk_trade_modifiers["overnight_hold_allowed"],
        "overnight_hold_reason": global_risk_trade_modifiers["overnight_hold_reason"],
        "overnight_risk_penalty": global_risk_trade_modifiers["overnight_risk_penalty"],
        "overnight_trade_block": global_risk_trade_modifiers["overnight_trade_block"],
        "global_risk_adjustment_score": global_risk_adjustment_score,
        "gamma_vol_acceleration_score": gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        "gamma_vol_acceleration_score_normalized": gamma_vol_acceleration_score_normalized,
        "squeeze_risk_state": gamma_vol_trade_modifiers["squeeze_risk_state"],
        "directional_convexity_state": gamma_vol_trade_modifiers["directional_convexity_state"],
        "upside_squeeze_risk": gamma_vol_trade_modifiers["upside_squeeze_risk"],
        "downside_airpocket_risk": gamma_vol_trade_modifiers["downside_airpocket_risk"],
        "overnight_convexity_risk": gamma_vol_trade_modifiers["overnight_convexity_risk"],
        "overnight_convexity_penalty": gamma_vol_trade_modifiers["overnight_convexity_penalty"],
        "overnight_convexity_boost": gamma_vol_trade_modifiers["overnight_convexity_boost"],
        "gamma_vol_adjustment_score": gamma_vol_adjustment_score,
        "dealer_hedging_pressure_score": dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
        "dealer_flow_state": dealer_pressure_trade_modifiers["dealer_flow_state"],
        "upside_hedging_pressure": dealer_pressure_trade_modifiers["upside_hedging_pressure"],
        "downside_hedging_pressure": dealer_pressure_trade_modifiers["downside_hedging_pressure"],
        "pinning_pressure_score": dealer_pressure_trade_modifiers["pinning_pressure_score"],
        "overnight_hedging_risk": dealer_pressure_trade_modifiers["overnight_hedging_risk"],
        "overnight_dealer_pressure_penalty": dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"],
        "overnight_dealer_pressure_boost": dealer_pressure_trade_modifiers["overnight_dealer_pressure_boost"],
        "dealer_pressure_adjustment_score": dealer_pressure_adjustment_score,
        "oil_shock_score": global_risk_trade_modifiers["oil_shock_score"],
        "market_volatility_shock_score": global_risk_trade_modifiers["volatility_shock_score"],
        "commodity_risk_score": global_risk_trade_modifiers["commodity_risk_score"],
        "usdinr_change_24h": global_risk_features_snapshot.get("usdinr_change_24h"),
        "dxy_change_24h": global_risk_features_snapshot.get("dxy_change_24h"),
        "gift_nifty_change_24h": global_risk_features_snapshot.get("gift_nifty_change_24h"),
        "currency_shock_score": global_risk_features_snapshot.get("currency_shock_score"),
        "dxy_shock_score": global_risk_features_snapshot.get("dxy_shock_score"),
        "gift_nifty_lead_score": global_risk_features_snapshot.get("gift_nifty_lead_score"),
        "macro_uncertainty_score": global_risk_features_snapshot.get("macro_uncertainty_score"),
        "risk_off_intensity": (
            global_risk_state.get("global_risk_features", {}).get("risk_off_intensity")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        "volatility_compression_score": (
            global_risk_state.get("global_risk_features", {}).get("volatility_compression_score")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        "volatility_explosion_probability": global_risk_trade_modifiers["volatility_explosion_probability"],
        "global_risk_features": global_risk_state.get("global_risk_features") if isinstance(global_risk_state, dict) else {},
        "global_risk_diagnostics": global_risk_state.get("global_risk_diagnostics") if isinstance(global_risk_state, dict) else {},
        "call_put_imbalance_score": structural_imbalance_audit["call_put_imbalance_score"],
        "call_put_imbalance_abs": structural_imbalance_audit["call_put_imbalance_abs"],
        "call_put_alignment": structural_imbalance_audit["call_put_alignment"],
        "call_put_imbalance_severity": structural_imbalance_audit["call_put_imbalance_severity"],
        "gamma_vol_features": gamma_vol_state.get("gamma_vol_features") if isinstance(gamma_vol_state, dict) else {},
        "gamma_vol_diagnostics": gamma_vol_state.get("gamma_vol_diagnostics") if isinstance(gamma_vol_state, dict) else {},
        "dealer_pressure_features": dealer_pressure_state.get("dealer_pressure_features") if isinstance(dealer_pressure_state, dict) else {},
        "dealer_pressure_diagnostics": dealer_pressure_state.get("dealer_pressure_diagnostics") if isinstance(dealer_pressure_state, dict) else {},
        "budget_constraint_applied": apply_budget_constraint,
        "lot_size": lot_size,
        "requested_lots": requested_lots,
        "max_capital_per_trade": max_capital,
        "at_flip_trade_strength_penalty": at_flip_penalty_applied,
        "at_flip_size_cap": round(at_flip_size_cap, 2),
        "at_flip_toxic_context": at_flip_toxic_context,
        "regime_toxic_context": regime_thresholds["toxic_context"],
        "regime_threshold_adjustments": regime_thresholds["adjustments"],
        "min_trade_strength_threshold": min_trade_strength,
        "min_composite_score_threshold": min_composite_score,
        "score_calibration_enabled": bool(int(_safe_float(runtime_thresholds.get("enable_score_calibration"), 1.0))),
        "score_calibration_applied": False,
        "score_calibration_backend": runtime_thresholds.get("calibration_backend", "isotonic"),
        "score_calibration_artifact_path": runtime_thresholds.get("runtime_score_calibrator_path"),
        "score_calibration_segment_key": None,
        "score_calibration_segment_context": {},
        "time_decay_enabled": bool(int(_safe_float(runtime_thresholds.get("enable_time_decay_model"), 1.0))),
        "time_decay_applied": False,
        "time_decay_fallback_used": False,
        "time_decay_elapsed_source": None,
        "runtime_composite_score": None,
        "time_decay_elapsed_minutes": None,
        "time_decay_factor": None,
        "live_calibration_gate": live_calibration_gate if isinstance(live_calibration_gate, dict) else {},
        "live_directional_gate": live_directional_gate if isinstance(live_directional_gate, dict) else {},
        "operator_control_state": operator_control_state if isinstance(operator_control_state, dict) else {},
        "backtest_mode": backtest_mode,
    }

    portfolio_concentration_context = _compute_portfolio_concentration_context(
        payload={
            **base_payload,
            "direction": direction,
            "gamma_regime": market_state.get("gamma_regime"),
            "volatility_regime": market_state.get("vol_regime"),
        },
        runtime_thresholds=runtime_thresholds,
    )

    # The global risk layer is the final pre-trade gate. It can block, downgrade
    # to watchlist, or cap size even when the analytics stack is directionally strong.
    global_risk = evaluate_global_risk_layer(
        data_quality=data_quality,
        confirmation=confirmation,
        adjusted_trade_strength=adjusted_trade_strength,
        min_trade_strength=min_trade_strength,
        event_window_status=event_window_status,
        macro_event_risk_score=macro_event_risk_score,
        event_lockdown_flag=event_lockdown_flag,
        next_event_name=next_event_name,
        active_event_name=active_event_name,
        macro_news_adjustments=macro_news_adjustments,
        global_risk_state=global_risk_state,
        holding_profile=holding_profile,
        portfolio_context=portfolio_concentration_context,
    )
    base_payload.update(
        {
            "global_risk_state": global_risk["global_risk_state"],
            "global_risk_state_score": global_risk_state.get("global_risk_score") if isinstance(global_risk_state, dict) else 0,
            "global_risk_overlay_score": global_risk["global_risk_score"],
            "global_risk_score": global_risk["global_risk_score"],
            "global_risk_state_reasons": global_risk_state.get("global_risk_reasons") if isinstance(global_risk_state, dict) else [],
            "global_risk_overlay_reasons": global_risk["global_risk_reasons"],
            "overnight_gap_risk_score": global_risk["overnight_gap_risk_score"],
            "volatility_expansion_risk_score": global_risk["volatility_expansion_risk_score"],
            "overnight_hold_allowed": (
                global_risk_trade_modifiers["overnight_hold_allowed"]
                and gamma_vol_trade_modifiers["overnight_hold_allowed"]
                and dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                and option_efficiency_trade_modifiers["overnight_hold_allowed"]
            ),
            "overnight_hold_reason": (
                global_risk_trade_modifiers["overnight_hold_reason"]
                if not global_risk_trade_modifiers["overnight_hold_allowed"]
                else (
                    gamma_vol_trade_modifiers["overnight_hold_reason"]
                    if not gamma_vol_trade_modifiers["overnight_hold_allowed"]
                    else (
                        dealer_pressure_trade_modifiers["overnight_hold_reason"]
                        if not dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                        else option_efficiency_trade_modifiers["overnight_hold_reason"]
                    )
                )
            ),
            "overnight_risk_penalty": (
                global_risk_trade_modifiers["overnight_risk_penalty"]
                + gamma_vol_trade_modifiers["overnight_convexity_penalty"]
                + dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"]
                + option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"]
            ),
            "overnight_trade_block": (
                global_risk_trade_modifiers["overnight_trade_block"]
                or not gamma_vol_trade_modifiers["overnight_hold_allowed"]
                or not dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                or not option_efficiency_trade_modifiers["overnight_hold_allowed"]
            ),
            "global_risk_adjustment_score": global_risk_adjustment_score,
            "gamma_vol_acceleration_score": gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
            "squeeze_risk_state": gamma_vol_trade_modifiers["squeeze_risk_state"],
            "directional_convexity_state": gamma_vol_trade_modifiers["directional_convexity_state"],
            "upside_squeeze_risk": gamma_vol_trade_modifiers["upside_squeeze_risk"],
            "downside_airpocket_risk": gamma_vol_trade_modifiers["downside_airpocket_risk"],
            "overnight_convexity_risk": gamma_vol_trade_modifiers["overnight_convexity_risk"],
            "overnight_convexity_penalty": gamma_vol_trade_modifiers["overnight_convexity_penalty"],
            "overnight_convexity_boost": gamma_vol_trade_modifiers["overnight_convexity_boost"],
            "gamma_vol_adjustment_score": gamma_vol_adjustment_score,
            "dealer_hedging_pressure_score": dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
            "dealer_flow_state": dealer_pressure_trade_modifiers["dealer_flow_state"],
            "upside_hedging_pressure": dealer_pressure_trade_modifiers["upside_hedging_pressure"],
            "downside_hedging_pressure": dealer_pressure_trade_modifiers["downside_hedging_pressure"],
            "pinning_pressure_score": dealer_pressure_trade_modifiers["pinning_pressure_score"],
            "overnight_hedging_risk": dealer_pressure_trade_modifiers["overnight_hedging_risk"],
            "overnight_dealer_pressure_penalty": dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"],
            "overnight_dealer_pressure_boost": dealer_pressure_trade_modifiers["overnight_dealer_pressure_boost"],
            "dealer_pressure_adjustment_score": dealer_pressure_adjustment_score,
            "expected_move_points": option_efficiency_trade_modifiers["expected_move_points"],
            "expected_move_pct_model": option_efficiency_trade_modifiers["expected_move_pct"],
            "expected_move_quality": option_efficiency_trade_modifiers["expected_move_quality"],
            "target_reachability_score": option_efficiency_trade_modifiers["target_reachability_score"],
            "premium_efficiency_score": option_efficiency_trade_modifiers["premium_efficiency_score"],
            "strike_efficiency_score": option_efficiency_trade_modifiers["strike_efficiency_score"],
            "option_efficiency_score": option_efficiency_trade_modifiers["option_efficiency_score"],
            "option_efficiency_adjustment_score": option_efficiency_adjustment_score,
            "overnight_option_efficiency_penalty": option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"],
            "strike_moneyness_bucket": option_efficiency_trade_modifiers["strike_moneyness_bucket"],
            "strike_distance_from_spot": option_efficiency_trade_modifiers["strike_distance_from_spot"],
            "payoff_efficiency_hint": option_efficiency_trade_modifiers["payoff_efficiency_hint"],
            "oil_shock_score": global_risk_trade_modifiers["oil_shock_score"],
            "market_volatility_shock_score": global_risk_trade_modifiers["volatility_shock_score"],
            "india_vix_level": india_vix_level,
            "india_vix_change_24h": india_vix_change_24h,
            "commodity_risk_score": global_risk_trade_modifiers["commodity_risk_score"],
            "risk_off_intensity": global_risk["global_risk_features"].get("risk_off_intensity", 0.0),
            "volatility_compression_score": global_risk["global_risk_features"].get("volatility_compression_score", 0.0),
            "volatility_explosion_probability": global_risk_trade_modifiers["volatility_explosion_probability"],
            "global_risk_level": global_risk["global_risk_level"],
            "global_risk_action": global_risk["global_risk_action"],
            "global_risk_size_cap": global_risk["global_risk_size_cap"],
            "global_risk_reasons": global_risk["global_risk_reasons"],
            "global_risk_features": global_risk["global_risk_features"],
            "global_risk_diagnostics": global_risk["global_risk_diagnostics"],
            "portfolio_concentration_guard": global_risk.get("portfolio_concentration_guard", portfolio_concentration_context),
            "portfolio_concentration_recent_signal_count": (global_risk.get("portfolio_concentration_guard") or {}).get("recent_signal_count"),
            "portfolio_concentration_same_direction_count": (global_risk.get("portfolio_concentration_guard") or {}).get("same_direction_count"),
            "portfolio_concentration_same_direction_share": (global_risk.get("portfolio_concentration_guard") or {}).get("same_direction_share"),
            "portfolio_book_heat_score": (global_risk.get("portfolio_concentration_guard") or {}).get("heat_score"),
            "portfolio_book_heat_label": (global_risk.get("portfolio_concentration_guard") or {}).get("heat_label"),
            "gamma_vol_reasons": gamma_vol_state.get("gamma_vol_reasons", []) if isinstance(gamma_vol_state, dict) else [],
            "gamma_vol_features": gamma_vol_state.get("gamma_vol_features", {}) if isinstance(gamma_vol_state, dict) else {},
            "gamma_vol_diagnostics": gamma_vol_state.get("gamma_vol_diagnostics", {}) if isinstance(gamma_vol_state, dict) else {},
            "dealer_pressure_reasons": dealer_pressure_state.get("dealer_pressure_reasons", []) if isinstance(dealer_pressure_state, dict) else [],
            "dealer_pressure_features": dealer_pressure_state.get("dealer_pressure_features", {}) if isinstance(dealer_pressure_state, dict) else {},
            "dealer_pressure_diagnostics": dealer_pressure_state.get("dealer_pressure_diagnostics", {}) if isinstance(dealer_pressure_state, dict) else {},
            "option_efficiency_reasons": option_efficiency_state.get("option_efficiency_reasons", []) if isinstance(option_efficiency_state, dict) else [],
            "option_efficiency_features": option_efficiency_state.get("option_efficiency_features", {}) if isinstance(option_efficiency_state, dict) else {},
            "option_efficiency_diagnostics": option_efficiency_state.get("option_efficiency_diagnostics", {}) if isinstance(option_efficiency_state, dict) else {},
        }
    )

    # Expiry-day contracts should not be marked overnight-holdable.
    expiry_day_contract = False
    try:
        if valuation_time is not None and selected_expiry is not None:
            expiry_day_contract = pd.Timestamp(selected_expiry).date() == pd.Timestamp(valuation_time).date()
    except Exception:
        expiry_day_contract = bool(_safe_float(days_to_expiry, None) == 0.0)

    base_payload["expiry_day_contract"] = expiry_day_contract
    if expiry_day_contract:
        base_payload["overnight_hold_allowed"] = False
        base_payload["overnight_trade_block"] = True
        base_payload["overnight_hold_reason"] = "expiry_day_contract_roll_required"

    def _finalize(payload, trade_status, message):
        """
        Purpose:
            Finalize the response payload with execution status and regime
            metadata.

        Context:
            Used at every exit path in `generate_trade` so blocked trades,
            watchlist outcomes, and executable trades all share the same output
            contract.

        Inputs:
            payload (Any): Base response payload that already contains shared diagnostics for the current snapshot.
            trade_status (Any): Final trade-status label such as `OK`, `NO_TRADE`, or a validation-specific code.
            message (Any): Human-readable explanation attached to the final payload.

        Returns:
            dict: Final response payload ready for runtime consumption and
            signal-evaluation logging.

        Notes:
            Centralizing this bookkeeping keeps decision branches focused on why
            the trade changed state rather than how the payload is shaped.
        """
        consistency_findings = collect_trade_consistency_findings(payload)
        consistency_issue_count = len(consistency_findings)
        escalation_state = select_trade_escalation_findings(payload, consistency_findings)
        consistency_escalation_findings = escalation_state["matched_findings"]
        payload["consistency_check_findings"] = consistency_findings
        payload["consistency_check_issue_count"] = consistency_issue_count
        payload["consistency_check_critical_issue_count"] = len(consistency_escalation_findings)
        payload["consistency_check_status"] = "WARN" if consistency_issue_count else "PASS"
        payload["consistency_check_policy"] = escalation_state["policy"]

        final_trade_status = trade_status
        final_message = message
        payload["consistency_check_escalated"] = False
        if trade_status == "TRADE" and consistency_escalation_findings:
            first_critical = consistency_escalation_findings[0]
            first_message = str(first_critical.get("message") or "consistency warning")
            final_trade_status = "WATCHLIST"
            final_message = f"Consistency check warning: {first_message}"
            payload["consistency_check_escalated"] = True
            payload["consistency_check_escalation_reason"] = first_message
            payload["no_trade_reason_code"] = "CONSISTENCY_CHECK_WARN"
            payload["no_trade_reason"] = first_message

        payload["message"] = final_message
        payload["trade_status"] = final_trade_status
        execution_size_multiplier = min(
            _safe_float(macro_news_adjustments.get("macro_position_size_multiplier"), 1.0),
            _safe_float(global_risk.get("global_risk_size_cap"), 1.0),
        )
        payload["execution_regime"] = classify_execution_regime(
            trade_status=final_trade_status,
            signal_regime=signal_regime,
            data_quality_score=data_quality["score"],
            macro_position_size_multiplier=execution_size_multiplier,
        )
        explainability = _build_decision_explainability(
            payload,
            trade_status=final_trade_status,
            min_trade_strength=int(_safe_float(payload.get("effective_min_trade_strength_threshold"), min_trade_strength)),
        )
        payload.update(explainability)
        payload["explainability"] = explainability

        # Note: confidence already computed early in the pipeline for sizing decisions.
        # Just ensure it's in the final payload (it should already be there).
        if "signal_confidence_score" not in payload:
            confidence = compute_signal_confidence(payload)
            payload["signal_confidence_score"] = confidence["confidence_score"]
            payload["signal_confidence_level"] = confidence["confidence_level"]

        # Immutable audit log — best-effort, never raises.
        if not backtest_mode:
            _journal_append_decision(payload, parameter_pack_name=None)

        return attach_trade_views(payload)

    if global_risk["risk_trade_status"] == "DATA_INVALID":
        return _finalize(base_payload, "DATA_INVALID", global_risk["risk_message"])

    if global_risk["risk_trade_status"] == "GLOBAL_RISK_BLOCKED":
        return _finalize(
            base_payload,
            "NO_TRADE",
            global_risk["risk_message"] or "Trade blocked due to elevated global risk conditions",
        )

    if global_risk_trade_modifiers["force_no_trade"] or global_risk["risk_trade_status"] == "EVENT_LOCKDOWN":
        return _finalize(base_payload, "NO_TRADE", global_risk["risk_message"] or "Trade blocked due to global event lockdown")

    if direction is None:
        return _finalize(base_payload, "NO_SIGNAL", "No trade signal")

    if bool(base_payload.get("path_aware_entry_veto")):
        return _finalize(base_payload, "WATCHLIST", "Path-aware filter vetoed entry")

    if (
        market_state["final_flow_signal"] == "NEUTRAL_FLOW"
        and probability_state["hybrid_move_probability"] is not None
        and probability_state["hybrid_move_probability"] < runtime_thresholds["neutral_flow_probability_floor"]
    ):
        return _finalize(base_payload, "NO_SIGNAL", "No trade signal: neutral flow and insufficient directional edge")

    ranked_strikes = []
    strike = None

    # Strike ranking only happens after the directional thesis survives macro
    # and risk gating. That keeps expensive contract-specific work off the
    # path for obvious no-trade scenarios.
    if direction is not None:
        def option_efficiency_candidate_hook(row, candidate_context=None):
            """
            Purpose:
                Score a strike candidate with contract-level option-efficiency
                heuristics.

            Context:
                Used by strike ranking after the engine has already chosen a
                direction. This lets contract selection account for payoff
                geometry, expected move, and overlay state without mutating the
                base strike-ranking model.

            Inputs:
                row (Any): Candidate option row under evaluation.

            Returns:
                dict: Optional score adjustment plus option-efficiency
                diagnostics for the candidate strike.

            Notes:
                The hook is nested because it depends on the fully assembled
                signal state for the current snapshot.
            """
            candidate_context = candidate_context if isinstance(candidate_context, dict) else {}
            row_payload = dict(row) if isinstance(row, dict) else row
            if isinstance(row_payload, dict) and candidate_context:
                # Support strike-selector hook contract that provides a compact
                # context payload for candidate diagnostics.
                row_payload.setdefault("strikePrice", candidate_context.get("strike"))
                row_payload.setdefault("lastPrice", candidate_context.get("last_price"))
                row_payload.setdefault("totalTradedVolume", candidate_context.get("volume"))
                row_payload.setdefault("openInterest", candidate_context.get("open_interest"))
                row_payload.setdefault("IV", candidate_context.get("iv"))

            hook_payload = score_option_efficiency_candidate(
                row_payload,
                spot=spot,
                direction=direction,
                atm_iv=market_state["atm_iv"],
                india_vix_level=india_vix_level,
                india_vix_change_24h=india_vix_change_24h,
                selected_expiry=(
                    option_chain_validation.get("selected_expiry")
                    if isinstance(option_chain_validation, dict)
                    else None
                ),
                valuation_time=valuation_time,
                hybrid_move_probability=probability_state["hybrid_move_probability"],
                gamma_regime=market_state["gamma_regime"],
                volatility_regime=market_state["vol_regime"],
                volatility_shock_score=global_risk_features.get("volatility_shock_score"),
                volatility_compression_score=global_risk_features.get("volatility_compression_score"),
                macro_event_risk_score=macro_event_risk_score,
                global_risk_state=base_payload.get("global_risk_state"),
                gamma_vol_acceleration_score=gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
                dealer_hedging_pressure_score=dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
                liquidity_vacuum_state=market_state["vacuum_state"],
                support_wall=market_state["support_wall"],
                resistance_wall=market_state["resistance_wall"],
            )
            hook_payload = hook_payload if isinstance(hook_payload, dict) else {}
            raw_adjustment = int(_safe_float(hook_payload.get("score_adjustment"), 0.0))
            strike_reliability_weight = _blend_feature_reliability(
                feature_reliability_weights,
                liquidity=0.55,
                vol_surface=0.35,
                greeks=0.10,
            )
            scaled_adjustment = _scale_candidate_adjustment_by_reliability(raw_adjustment, strike_reliability_weight)
            hook_payload["score_adjustment_raw"] = raw_adjustment
            hook_payload["score_adjustment"] = scaled_adjustment
            hook_payload["strike_reliability_weight"] = strike_reliability_weight
            hook_payload["strike_reliability_delta"] = scaled_adjustment - raw_adjustment
            return hook_payload

        strike, ranked_strikes = select_best_strike(
            option_chain=df,
            direction=direction,
            spot=spot,
            support_wall=market_state["support_wall"],
            resistance_wall=market_state["resistance_wall"],
            gamma_clusters=market_state["gamma_clusters"],
            lot_size=lot_size,
            max_capital=max_capital if apply_budget_constraint else None,
            candidate_score_hook=option_efficiency_candidate_hook,
            gamma_regime=market_state["gamma_regime"],
            spot_vs_flip=market_state["spot_vs_flip"],
            dealer_hedging_bias=market_state["hedging_bias"],
            gamma_flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
            atm_iv=market_state["atm_iv"],
            india_vix_level=india_vix_level,
            days_to_expiry=days_to_expiry,
            max_pain=market_state.get("max_pain"),
            vol_surface_regime=market_state["surface_regime"],
            volatility_shock_score=market_state.get("volatility_shock_score", 0.0),
            directional_call_probability=bull_probability,
            directional_put_probability=bear_probability,
        )

    base_payload["ranked_strike_candidates"] = ranked_strikes

    if strike is None:
        base_payload["direction"] = direction
        return _finalize(base_payload, "NO_SIGNAL", "No valid strike found")

    option_type = "CE" if direction == "CALL" else "PE"

    option_row = df[
        (df["strikePrice"] == strike) &
        (df["OPTION_TYP"] == option_type)
    ]

    if option_row.empty:
        base_payload["direction"] = direction
        return _finalize(base_payload, "NO_SIGNAL", "Selected strike/option type not available")

    entry_price = float(option_row.iloc[0]["lastPrice"])
    target, stop_loss = calculate_exit(
        entry_price,
        target_profit_percent=target_profit_percent,
        stop_loss_percent=stop_loss_percent,
    )

    # --- Time-based exit recommendation ------------------------------------
    _gr_features = global_risk_state.get("global_risk_features", {}) if isinstance(global_risk_state, dict) else {}
    _mtc_raw = _gr_features.get("minutes_to_close")
    _mtc = _safe_float(_mtc_raw, 0.0) if _mtc_raw is not None else None
    _mso = round(375.0 - _mtc, 2) if _mtc is not None else None
    exit_timing = compute_exit_timing(
        trade_strength=adjusted_trade_strength,
        gamma_regime=market_state["gamma_regime"],
        vol_regime=market_state["vol_regime"],
        minutes_since_open=_mso,
        minutes_to_close=_mtc,
    )
    hold_cap_minutes = int(_safe_float(runtime_thresholds.get("max_intraday_hold_minutes"), 90.0))
    hold_cap_minutes = min(hold_cap_minutes, int(_safe_float(regime_thresholds.get("effective_max_holding_m"), hold_cap_minutes)))
    if regime_thresholds["toxic_context"] or at_flip_toxic_context:
        hold_cap_minutes = min(
            hold_cap_minutes,
            int(_safe_float(runtime_thresholds.get("toxic_regime_hold_cap_minutes"), 60.0)),
        )
    recommended_hold_minutes = min(int(exit_timing["recommended_hold_minutes"]), hold_cap_minutes)
    max_hold_minutes = min(int(exit_timing["max_hold_minutes"]), hold_cap_minutes)
    exit_timing_reasons = list(exit_timing.get("exit_timing_reasons") or [])

    if reversal_stage == "EARLY_REVERSAL_CANDIDATE":
        early_hold_mult = _safe_float(runtime_thresholds.get("reversal_stage_early_hold_mult"), 0.60)
        recommended_hold_minutes = max(10, int(recommended_hold_minutes * early_hold_mult))
        max_hold_minutes = max(recommended_hold_minutes, int(max_hold_minutes * early_hold_mult))
        exit_timing_reasons.append("reversal_stage_early_hold_cap")
    if expansion_mode and expansion_direction == direction:
        expansion_hold_mult = _safe_float(runtime_thresholds.get("expansion_mode_hold_mult"), 0.75)
        recommended_hold_minutes = max(10, int(recommended_hold_minutes * expansion_hold_mult))
        max_hold_minutes = max(recommended_hold_minutes, int(max_hold_minutes * expansion_hold_mult))
        exit_timing_reasons.append("expansion_mode_fast_hold_cap")
    if recommended_hold_minutes < int(exit_timing["recommended_hold_minutes"]) or max_hold_minutes < int(exit_timing["max_hold_minutes"]):
        exit_timing_reasons.append(f"hard_hold_cap_applied_{hold_cap_minutes}m")

    recommended_hold_minutes, max_hold_minutes, live_exit_bias_reasons = _apply_intraday_exit_bias(
        recommended_hold_minutes=recommended_hold_minutes,
        max_hold_minutes=max_hold_minutes,
        runtime_thresholds=runtime_thresholds,
        expansion_mode=expansion_mode,
        expansion_direction=expansion_direction,
        direction=direction,
        reversal_stage=reversal_stage,
        provider_health_summary=provider_health_summary,
        data_quality_status=data_quality.get("status"),
        global_risk_state=base_payload.get("global_risk_state"),
        gamma_regime=market_state.get("gamma_regime"),
    )
    for _reason in live_exit_bias_reasons:
        if _reason not in exit_timing_reasons:
            exit_timing_reasons.append(_reason)

    option_row_dict = option_row.iloc[0].to_dict()

    ranked_candidate = None
    for candidate in ranked_strikes or []:
        if (
            _safe_float(candidate.get("strike"), None) == _safe_float(strike, None)
            and str(candidate.get("option_type") or "").upper().strip() == option_type
        ):
            ranked_candidate = candidate
            break
    resolved_delta_input = option_row_dict.get("DELTA")
    if _safe_float(resolved_delta_input, None) in (None, 0.0) and isinstance(ranked_candidate, dict):
        resolved_delta_input = ranked_candidate.get("delta")
    selected_option_iv = _safe_float(
        option_row_dict.get("impliedVolatility", option_row_dict.get("IV")),
        _safe_float((ranked_candidate or {}).get("iv"), None),
    )
    selected_option_ba_spread_ratio = _safe_float(
        option_row_dict.get("_normalized_ba_spread_ratio"),
        _safe_float((ranked_candidate or {}).get("ba_spread_ratio"), None),
    )
    selected_option_capital_per_lot = _safe_float(
        (ranked_candidate or {}).get("capital_per_lot"),
        round(entry_price * float(lot_size), 2) if lot_size is not None else None,
    )

    # Once a specific contract is chosen, recompute option-efficiency metrics
    # with contract-level Greeks, expiry, and payoff geometry.
    option_efficiency_state = build_option_efficiency_state(
        spot=spot,
        atm_iv=market_state["atm_iv"],
        india_vix_level=india_vix_level,
        india_vix_change_24h=india_vix_change_24h,
        fallback_iv=option_row_dict.get("impliedVolatility", option_row_dict.get("IV")),
        expiry_value=option_row_dict.get(
            "EXPIRY_DT",
            option_chain_validation.get("selected_expiry") if isinstance(option_chain_validation, dict) else None,
        ),
        valuation_time=valuation_time,
        time_to_expiry_years=option_row_dict.get("TTE"),
        direction=direction,
        strike=strike,
        option_type=option_type,
        entry_price=entry_price,
        target=target,
        stop_loss=stop_loss,
        trade_strength=adjusted_trade_strength,
        hybrid_move_probability=probability_state["hybrid_move_probability"],
        rule_move_probability=probability_state["rule_move_probability"],
        ml_move_probability=probability_state["ml_move_probability"],
        gamma_regime=market_state["gamma_regime"],
        volatility_regime=market_state["vol_regime"],
        volatility_shock_score=global_risk_features.get("volatility_shock_score"),
        volatility_compression_score=global_risk_features.get("volatility_compression_score"),
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=base_payload.get("global_risk_state"),
        gamma_vol_acceleration_score=gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        dealer_hedging_pressure_score=dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
        liquidity_vacuum_state=market_state["vacuum_state"],
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        delta=resolved_delta_input,
        holding_profile=holding_profile,
    )
    option_efficiency_trade_modifiers = derive_option_efficiency_trade_modifiers(option_efficiency_state)
    option_efficiency_raw_adjustment_score = int(_safe_float(option_efficiency_trade_modifiers["option_efficiency_adjustment_score"], 0.0))
    option_efficiency_adjustment_score = _scale_adjustment_by_reliability(
        option_efficiency_raw_adjustment_score,
        option_efficiency_reliability_weight,
    )
    adjusted_trade_strength = int(_clip(adjusted_trade_strength + option_efficiency_adjustment_score, 0, 100))
    scoring_breakdown["option_efficiency_adjustment_score"] = option_efficiency_adjustment_score
    scoring_breakdown["option_efficiency_reliability_weight"] = option_efficiency_reliability_weight
    scoring_breakdown["option_efficiency_reliability_delta"] = option_efficiency_adjustment_score - option_efficiency_raw_adjustment_score
    feature_reliability_penalty_score = max(
        0,
        chain_confirmation_score - scaled_chain_confirmation_score,
        gamma_vol_raw_adjustment_score - gamma_vol_adjustment_score,
        dealer_pressure_raw_adjustment_score - dealer_pressure_adjustment_score,
        option_efficiency_raw_adjustment_score - option_efficiency_adjustment_score,
    )
    feature_reliability_composite_penalty = int(_clip(round(feature_reliability_penalty_score * 0.67), 0, 8))
    scoring_breakdown["feature_reliability_penalty"] = -feature_reliability_penalty_score
    scoring_breakdown["total_score"] = adjusted_trade_strength
    base_payload["trade_strength"] = adjusted_trade_strength
    base_payload["signal_quality"] = classify_signal_quality(adjusted_trade_strength)
    base_payload["feature_reliability_penalty_score"] = feature_reliability_penalty_score
    base_payload["feature_reliability_composite_penalty"] = feature_reliability_composite_penalty
    signal_regime = classify_signal_regime(
        direction=direction,
        adjusted_trade_strength=adjusted_trade_strength,
        final_flow_signal=market_state["final_flow_signal"],
        gamma_regime=market_state["gamma_regime"],
        confirmation_status=confirmation["status"],
        event_lockdown_flag=event_lockdown_flag or macro_news_adjustments["event_lockdown_flag"],
        data_quality_status=data_quality["status"],
    )
    base_payload["signal_regime"] = signal_regime

    base_payload.update({
        "direction": direction,
        "strike": _to_python_number(strike),
        "option_type": option_type,
        "reversal_context": reversal_context,
        "reversal_stage": reversal_stage,
        "fast_reversal_active": signal_state.get("fast_reversal_active"),
        "fast_reversal_alert_level": signal_state.get("fast_reversal_alert_level"),
        "fast_reversal_warning_direction": signal_state.get("fast_reversal_warning_direction"),
        "fast_reversal_prior_direction": signal_state.get("fast_reversal_prior_direction"),
        "fast_reversal_evidence_score": signal_state.get("fast_reversal_evidence_score"),
        "fast_reversal_reasons": signal_state.get("fast_reversal_reasons"),
        "fast_reversal_promotion_bias": signal_state.get("fast_reversal_promotion_bias"),
        "breakout_vote_count": breakout_vote_count,
        "expansion_mode": expansion_mode,
        "expansion_direction": expansion_direction,
        "breakout_evidence": round(float(breakout_evidence or 0.0), 3),
        "entry_price": round(entry_price, 2),
        "selected_option_last_price": round(entry_price, 2),
        "selected_option_volume": _to_python_number(
            _safe_float(
                option_row_dict.get("totalTradedVolume", option_row_dict.get("VOLUME")),
                _safe_float((ranked_candidate or {}).get("volume"), None),
            )
        ),
        "selected_option_open_interest": _to_python_number(
            _safe_float(
                option_row_dict.get("openInterest", option_row_dict.get("OPEN_INT")),
                _safe_float((ranked_candidate or {}).get("open_interest"), None),
            )
        ),
        "selected_option_iv": round(float(selected_option_iv), 4) if selected_option_iv not in (None, 0.0) else None,
        "selected_option_iv_is_proxy": bool((ranked_candidate or {}).get("iv_is_proxy", False)),
        "selected_option_iv_proxy_source": (ranked_candidate or {}).get("iv_proxy_source"),
        "selected_option_delta": round(float(resolved_delta_input), 6) if _safe_float(resolved_delta_input, None) is not None else None,
        "selected_option_delta_is_proxy": bool((ranked_candidate or {}).get("delta_is_proxy", False)),
        "selected_option_delta_proxy_source": (ranked_candidate or {}).get("delta_proxy_source"),
        "selected_option_gamma": _to_python_number(_safe_float(option_row_dict.get("GAMMA"), None)),
        "selected_option_theta": _to_python_number(_safe_float(option_row_dict.get("THETA"), None)),
        "selected_option_vega": _to_python_number(_safe_float(option_row_dict.get("VEGA"), None)),
        "selected_option_vanna": _to_python_number(_safe_float(option_row_dict.get("VANNA"), None)),
        "selected_option_charm": _to_python_number(_safe_float(option_row_dict.get("CHARM"), None)),
        "selected_option_capital_per_lot": _to_python_number(selected_option_capital_per_lot),
        "selected_option_ba_spread_ratio": _to_python_number(selected_option_ba_spread_ratio),
        "selected_option_ba_spread_pct": round(float(selected_option_ba_spread_ratio) * 100.0, 4) if selected_option_ba_spread_ratio is not None else None,
        "selected_option_score": _to_python_number(_safe_float((ranked_candidate or {}).get("score"), None)),
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
        "recommended_hold_minutes": recommended_hold_minutes,
        "max_hold_minutes": max_hold_minutes,
        "exit_urgency": exit_timing["exit_urgency"],
        "exit_timing_reasons": exit_timing_reasons,
        "expected_move_points": option_efficiency_trade_modifiers["expected_move_points"],
        "expected_move_pct_model": option_efficiency_trade_modifiers["expected_move_pct"],
        "expected_move_quality": option_efficiency_trade_modifiers["expected_move_quality"],
        "target_reachability_score": option_efficiency_trade_modifiers["target_reachability_score"],
        "premium_efficiency_score": option_efficiency_trade_modifiers["premium_efficiency_score"],
        "strike_efficiency_score": option_efficiency_trade_modifiers["strike_efficiency_score"],
        "option_efficiency_score": option_efficiency_trade_modifiers["option_efficiency_score"],
        "option_efficiency_adjustment_score": option_efficiency_adjustment_score,
        "option_efficiency_reliability_weight": option_efficiency_reliability_weight,
        "overnight_option_efficiency_penalty": option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"],
        "strike_moneyness_bucket": option_efficiency_trade_modifiers["strike_moneyness_bucket"],
        "strike_distance_from_spot": option_efficiency_trade_modifiers["strike_distance_from_spot"],
        "payoff_efficiency_hint": option_efficiency_trade_modifiers["payoff_efficiency_hint"],
        "option_efficiency_reasons": option_efficiency_state.get("option_efficiency_reasons", []),
        "option_efficiency_features": option_efficiency_state.get("option_efficiency_features", {}),
        "option_efficiency_diagnostics": option_efficiency_state.get("option_efficiency_diagnostics", {}),
        "option_efficiency_delta_source": (
            "RANKED_CANDIDATE_FALLBACK"
            if _safe_float(option_row_dict.get("DELTA"), None) in (None, 0.0) and _safe_float(resolved_delta_input, None) not in (None, 0.0)
            else "CHAIN_DELTA"
        ),
    })
    if base_payload.get("expected_move_pct") is None:
        base_payload["expected_move_pct"] = base_payload.get("expected_move_pct_model")

    # --- Overnight hold consolidation -----------------------------------------
    _overnight_layers = [
        ("global_risk", global_risk_trade_modifiers),
        ("gamma_vol", gamma_vol_trade_modifiers),
        ("dealer_pressure", dealer_pressure_trade_modifiers),
        ("option_efficiency", option_efficiency_trade_modifiers),
    ]
    _overnight_allowed = True
    _overnight_reason = option_efficiency_trade_modifiers["overnight_hold_reason"]
    for _layer_name, _layer_mods in _overnight_layers:
        if not _layer_mods["overnight_hold_allowed"]:
            _overnight_allowed = False
            _overnight_reason = _layer_mods["overnight_hold_reason"]
            break

    _penalty_keys = {
        "global_risk": "overnight_risk_penalty",
        "gamma_vol": "overnight_convexity_penalty",
        "dealer_pressure": "overnight_dealer_pressure_penalty",
        "option_efficiency": "overnight_option_efficiency_penalty",
    }
    _overnight_penalty = sum(
        int(_safe_float(_layer_mods.get(_penalty_keys[_layer_name]), 0.0))
        for _layer_name, _layer_mods in _overnight_layers
    )

    base_payload.update(
        {
            "overnight_hold_allowed": _overnight_allowed,
            "overnight_hold_reason": _overnight_reason,
            "overnight_risk_penalty": _overnight_penalty,
            "overnight_trade_block": not _overnight_allowed,
        }
    )

    # Budget controls are applied after the signal is fully validated so they
    # affect only advisory sizing, not the informational content of the signal itself.
    budget_signal_only = False
    if apply_budget_constraint:
        budget_info = optimize_lots(
            entry_price=entry_price,
            lot_size=lot_size,
            max_capital=max_capital,
            requested_lots=requested_lots,
        )

        base_payload.update(budget_info)

        if not budget_info["budget_ok"]:
            budget_signal_only = True
            base_payload["budget_signal_only"] = True
            base_payload["number_of_lots"] = 0
            base_payload["optimized_lots"] = 0
        else:
            base_payload["number_of_lots"] = budget_info.get("optimized_lots", requested_lots)
            base_payload["optimized_lots"] = base_payload["number_of_lots"]
            base_payload["budget_signal_only"] = False
    else:
        base_payload["number_of_lots"] = requested_lots
        base_payload["optimized_lots"] = requested_lots
        base_payload["budget_signal_only"] = False

    base_payload["capital_per_lot"] = round(entry_price * lot_size, 2)
    base_payload["capital_required"] = round(entry_price * lot_size * base_payload["number_of_lots"], 2)

    # Compute signal confidence for diagnostics and advisory sizing only.
    pre_finalize_payload = dict(base_payload)
    signal_confidence = compute_signal_confidence(pre_finalize_payload)
    base_payload["signal_confidence_score"] = signal_confidence["confidence_score"]
    base_payload["signal_confidence_level"] = signal_confidence["confidence_level"]

    signal_probability_overlay = _build_signal_probability_overlay(
        direction=direction,
        adjusted_trade_strength=adjusted_trade_strength,
        call_probability=bull_probability,
        put_probability=bear_probability,
        market_state={
            **(market_state if isinstance(market_state, dict) else {}),
            "provider_health_summary": provider_health_summary,
            "data_quality_status": data_quality.get("status"),
            "global_risk_state": base_payload.get("global_risk_state"),
            "option_efficiency_status": base_payload.get("option_efficiency_status"),
            "ranked_strike_proxy_ratio": _ranked_strike_proxy_ratio(ranked_strikes),
            "reversal_stage": reversal_stage,
            "fast_reversal_alert_level": signal_state.get("fast_reversal_alert_level"),
            "live_calibration_gate": live_calibration_gate if isinstance(live_calibration_gate, dict) else {},
            "live_directional_gate": live_directional_gate if isinstance(live_directional_gate, dict) else {},
        },
        runtime_thresholds=runtime_thresholds,
    )
    base_payload.update(signal_probability_overlay)

    advisory_sizing = _derive_advisory_size_recommendation(
        base_payload,
        confidence_score=_safe_float(signal_confidence.get("confidence_score"), 50.0),
        global_risk_size_cap=_safe_float(global_risk["global_risk_size_cap"], 1.0),
        at_flip_size_cap=_safe_float(at_flip_size_cap, 1.0),
        macro_size_multiplier=_safe_float(macro_news_adjustments.get("macro_position_size_multiplier"), 1.0),
        freshness_size_cap=_safe_float(freshness_size_cap, 1.0),
        reversal_stage=reversal_stage,
        expansion_mode=expansion_mode,
        expansion_direction=expansion_direction,
        direction=direction,
        runtime_thresholds=runtime_thresholds,
        regime_thresholds=regime_thresholds,
        gamma_regime=market_state.get("gamma_regime"),
    )
    base_payload["effective_size_cap"] = advisory_sizing["effective_size_cap"]
    base_payload["macro_suggested_lots"] = advisory_sizing["advisory_lots"]
    base_payload["macro_size_applied"] = advisory_sizing["macro_size_applied"]
    base_payload["advisory_only"] = advisory_sizing["advisory_only"]
    base_payload["advisory_lots"] = advisory_sizing["advisory_lots"]
    base_payload["confidence_size_multiplier"] = advisory_sizing["confidence_size_multiplier"]
    base_payload["advisory_position_size_multiplier"] = advisory_sizing["advisory_position_size_multiplier"]
    base_payload["advisory_capital_required"] = round(entry_price * lot_size * advisory_sizing["advisory_lots"], 2)
    base_payload["portfolio_priority_score"] = advisory_sizing.get("portfolio_priority_score")
    base_payload["portfolio_priority_bucket"] = advisory_sizing.get("portfolio_priority_bucket")
    base_payload["portfolio_allocation_tier"] = advisory_sizing.get("portfolio_allocation_tier")
    base_payload["portfolio_capital_fraction_max"] = advisory_sizing.get("portfolio_capital_fraction_max")
    base_payload["portfolio_book_heat_score"] = advisory_sizing.get("portfolio_heat_score", base_payload.get("portfolio_book_heat_score"))
    base_payload["portfolio_book_heat_label"] = advisory_sizing.get("portfolio_heat_label", base_payload.get("portfolio_book_heat_label"))
    base_payload["premium_load_pct_of_spot"] = advisory_sizing.get("premium_load_pct_of_spot")
    base_payload["premium_priority_adjustment"] = advisory_sizing.get("premium_priority_adjustment")
    base_payload["premium_size_cap"] = advisory_sizing.get("premium_size_cap")

    if global_risk["risk_trade_status"] == "WATCHLIST":
        return _finalize(base_payload, "WATCHLIST", global_risk["risk_message"])

    composite_probability_input = probability_state["hybrid_move_probability"]
    overlay_probability = _safe_float(base_payload.get("signal_success_probability"), None)
    if overlay_probability is not None:
        if composite_probability_input is None:
            composite_probability_input = overlay_probability
        else:
            composite_probability_input = round(
                (float(composite_probability_input) + float(overlay_probability)) / 2.0,
                4,
            )
    base_payload["runtime_probability_input"] = composite_probability_input

    runtime_composite_score = _compute_runtime_composite_score(
        trade_strength=adjusted_trade_strength,
        hybrid_move_probability=composite_probability_input,
        move_probability_score_cap=runtime_thresholds.get("move_probability_score_cap"),
        confirmation_status=confirmation["status"],
        data_quality_status=data_quality["status"],
        gamma_vol_acceleration_score_normalized=gamma_vol_acceleration_score_normalized,
        weight_trade_strength=runtime_thresholds.get("composite_weight_trade_strength", 0.50),
        weight_move_probability=runtime_thresholds.get("composite_weight_move_probability", 0.20),
        weight_confirmation=runtime_thresholds.get("composite_weight_confirmation", 0.15),
        weight_data_quality=runtime_thresholds.get("composite_weight_data_quality", 0.10),
        weight_gamma_stability=runtime_thresholds.get("composite_weight_gamma_stability", 0.05),
    )
    if feature_reliability_composite_penalty > 0:
        runtime_composite_score = int(_clip(runtime_composite_score - feature_reliability_composite_penalty, 0, 100))
    
    # Apply score calibration if enabled
    enable_calibration = bool(int(_safe_float(runtime_thresholds.get("enable_score_calibration"), 1.0)))
    calibration_backend = runtime_thresholds.get("calibration_backend", "isotonic")
    calibrator_path = runtime_thresholds.get("runtime_score_calibrator_path")
    calibration_context = {
        "direction": direction,
        "gamma_regime": canonical_gamma_regime(market_state.get("gamma_regime")),
        "vol_regime": _canonical_vol_regime(market_state.get("vol_regime")),
    }
    base_payload["score_calibration_enabled"] = enable_calibration
    base_payload["score_calibration_backend"] = calibration_backend if enable_calibration else None
    base_payload["score_calibration_artifact_path"] = calibrator_path if enable_calibration else None
    if enable_calibration:
        runtime_composite_score = apply_score_calibration(
            raw_composite_score=runtime_composite_score,
            calibration_backend=calibration_backend,
            calibrator_path=calibrator_path,
            calibration_context=calibration_context,
        )
        calibration_metadata = get_calibrator_runtime_metadata(
            calibrator_path,
            calibration_context=calibration_context,
        )
        base_payload["score_calibration_applied"] = bool(calibration_metadata.get("calibrator_loaded"))
        loaded_artifact_path = calibration_metadata.get("loaded_artifact_path")
        if loaded_artifact_path:
            base_payload["score_calibration_artifact_path"] = loaded_artifact_path
        base_payload["score_calibration_segment_key"] = calibration_metadata.get("selected_segment_key")
        base_payload["score_calibration_segment_context"] = calibration_metadata.get("selected_segment_context") or {}

    regime_segment_guard = _compute_regime_segment_guard(
        payload={**base_payload, "direction": direction, "vol_regime": calibration_context.get("vol_regime")},
        runtime_thresholds=runtime_thresholds,
    )
    base_payload["regime_segment_guard"] = regime_segment_guard
    base_payload["regime_segment_key"] = regime_segment_guard.get("segment_key")
    base_payload["regime_segment_samples"] = regime_segment_guard.get("sample_size")
    base_payload["regime_segment_hit_rate_60m"] = regime_segment_guard.get("hit_rate_60m")
    base_payload["regime_segment_avg_60m_bps"] = regime_segment_guard.get("avg_60m_bps")
    base_payload["regime_segment_avg_close_bps"] = regime_segment_guard.get("avg_close_bps")
    base_payload["regime_segment_avg_tradeability_score"] = regime_segment_guard.get("avg_tradeability_score")

    effective_min_trade_strength, effective_min_composite_score, bearish_bias_guard = _apply_bearish_bias_threshold_adjustments(
        runtime_thresholds=runtime_thresholds,
        direction=direction,
        gamma_regime=market_state.get("gamma_regime"),
        vol_regime=market_state.get("vol_regime"),
        base_min_trade_strength=min_trade_strength,
        base_min_composite_score=min_composite_score,
    )
    regime_segment_verdict = _as_upper(regime_segment_guard.get("verdict"))
    if regime_segment_verdict == "CAUTION":
        effective_min_trade_strength = int(_clip(
            effective_min_trade_strength + int(_safe_float(runtime_thresholds.get("regime_segment_guard_caution_strength_add"), 3.0)),
            0,
            100,
        ))
        effective_min_composite_score = int(_clip(
            effective_min_composite_score + int(_safe_float(runtime_thresholds.get("regime_segment_guard_caution_composite_add"), 3.0)),
            0,
            100,
        ))
        hold_cap = max(int(_safe_float(runtime_thresholds.get("regime_segment_guard_hold_cap_minutes"), 35.0)), 5)
        base_payload["recommended_hold_minutes"] = min(int(_safe_float(base_payload.get("recommended_hold_minutes"), hold_cap)), hold_cap)
        base_payload["max_hold_minutes"] = min(int(_safe_float(base_payload.get("max_hold_minutes"), hold_cap)), hold_cap)
        current_cap = _clip(_safe_float(base_payload.get("effective_size_cap"), 1.0), 0.0, 1.0)
        segment_cap = _clip(_safe_float(runtime_thresholds.get("regime_segment_guard_size_cap"), 0.75), 0.0, 1.0)
        base_payload["effective_size_cap"] = round(min(current_cap, segment_cap), 2)
    base_payload["bearish_bias_guard"] = bearish_bias_guard
    base_payload["effective_min_trade_strength_threshold"] = int(effective_min_trade_strength)
    base_payload["effective_min_composite_score_threshold"] = int(effective_min_composite_score)
    if bearish_bias_guard.get("applied"):
        guard_size_cap = _clip(_safe_float(bearish_bias_guard.get("size_cap"), 1.0), 0.0, 1.0)
        current_cap = _clip(_safe_float(base_payload.get("effective_size_cap"), 1.0), 0.0, 1.0)
        guarded_cap = min(current_cap, guard_size_cap)
        base_payload["effective_size_cap"] = round(guarded_cap, 2)
        current_lots = max(int(_safe_float(base_payload.get("macro_suggested_lots"), 0.0)), 0)
        guarded_lots = max(int(current_lots * guarded_cap), 0)
        if current_lots > 0 and guarded_lots == 0 and guarded_cap > 0:
            guarded_lots = 1
        if guarded_lots > 0 or current_lots == 0:
            base_payload["macro_suggested_lots"] = guarded_lots if current_lots == 0 else min(current_lots, guarded_lots)
            base_payload["advisory_lots"] = base_payload["macro_suggested_lots"]
            base_payload["advisory_position_size_multiplier"] = round(guarded_cap, 4)
            if "entry_price" in base_payload:
                base_payload["advisory_capital_required"] = round(entry_price * lot_size * base_payload["advisory_lots"], 2)
    
    # Apply time-decay model if enabled
    enable_decay = bool(int(_safe_float(runtime_thresholds.get("enable_time_decay_model"), 1.0)))
    base_payload["time_decay_enabled"] = enable_decay
    decay_elapsed_source = None
    decay_minutes_elapsed = _safe_float(runtime_thresholds.get("time_decay_elapsed_minutes"), None)
    if decay_minutes_elapsed is not None:
        decay_elapsed_source = "configured_minutes"
    if decay_minutes_elapsed is None:
        decay_minutes_elapsed = _compute_signal_elapsed_minutes(
            symbol=symbol,
            selected_expiry=selected_expiry,
            valuation_time=valuation_time,
            direction=direction,
        )
        decay_elapsed_source = "signal_tracking"
    if decay_minutes_elapsed in (None, 0.0) and reversal_age is not None:
        per_snapshot_m = _safe_float(runtime_thresholds.get("time_decay_minutes_per_snapshot"), 5.0)
        decay_minutes_elapsed = max(0.0, _safe_float(reversal_age, 0.0)) * max(0.0, _safe_float(per_snapshot_m, 5.0))
        decay_elapsed_source = "reversal_age_fallback"
        base_payload["time_decay_fallback_used"] = True

    decay_minutes_elapsed = max(0.0, _safe_float(decay_minutes_elapsed, 0.0))
    base_payload["time_decay_elapsed_source"] = decay_elapsed_source
    base_payload["time_decay_elapsed_minutes"] = round(decay_minutes_elapsed, 2)

    if enable_decay and decay_minutes_elapsed > 0:
        _ensure_time_decay_model_config(runtime_thresholds)
        gamma_regime = canonical_gamma_regime(market_state.get("gamma_regime"))
        vol_regime = _canonical_vol_regime(market_state.get("vol_regime"))
        decay_factor = apply_time_decay(
            minutes_elapsed=decay_minutes_elapsed,
            gamma_regime=gamma_regime,
            lambda_param=runtime_thresholds.get("time_decay_lambda", 1.5),
            volatility_regime=vol_regime,
        )
        base_payload["time_decay_factor"] = round(_safe_float(decay_factor, 1.0), 6)
        base_payload["time_decay_applied"] = True
        runtime_composite_score = int(runtime_composite_score * decay_factor)
    elif enable_decay:
        base_payload["time_decay_factor"] = 1.0
    
    base_payload["runtime_composite_score"] = runtime_composite_score

    if regime_segment_verdict == "BLOCK":
        base_payload["no_trade_reason_code"] = "REGIME_SEGMENT_GUARD"
        base_payload["no_trade_reason"] = regime_segment_guard.get("reason") or "Regime segment guard downgraded the setup"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Regime segment guard routed TRADE to WATCHLIST",
        )

    if adjusted_trade_strength < effective_min_trade_strength:
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Trade strength {adjusted_trade_strength} below threshold {effective_min_trade_strength}",
        )

    if runtime_composite_score < effective_min_composite_score:
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Runtime composite score {runtime_composite_score} below threshold {effective_min_composite_score}",
        )

    if bool(base_payload.get("analytics_usable")) and not bool(base_payload.get("execution_suggestion_usable")):
        tradable_data = base_payload.get("tradable_data") if isinstance(base_payload.get("tradable_data"), dict) else {}
        tradable_reasons = tradable_data.get("reasons") if isinstance(tradable_data.get("reasons"), list) else []
        reason_suffix = f" ({', '.join(str(r) for r in tradable_reasons)})" if tradable_reasons else ""
        base_payload["no_trade_reason_code"] = "EXECUTION_DATA_UNUSABLE"
        base_payload["no_trade_reason"] = "Execution suggestion blocked: tradable data quality is below execution threshold"
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Execution suggestion blocked by tradable-data gate{reason_suffix}",
        )

    weak_data_shadow_triggered, weak_data_shadow = _evaluate_weak_data_circuit_breaker(
        runtime_thresholds=runtime_thresholds,
        data_quality_status=data_quality.get("status"),
        provider_health_summary=provider_health_summary,
        confirmation_status=confirmation.get("status"),
        adjusted_trade_strength=adjusted_trade_strength,
        runtime_composite_score=runtime_composite_score,
        ranked_strikes=ranked_strikes,
        direction=direction,
        gamma_regime=market_state.get("gamma_regime"),
        vol_regime=market_state.get("vol_regime"),
        option_efficiency_status=base_payload.get("option_efficiency_status"),
        live_calibration_gate=live_calibration_gate,
        live_directional_gate=live_directional_gate,
        global_risk_state=base_payload.get("global_risk_state"),
    )
    base_payload["weak_data_circuit_breaker_shadow"] = weak_data_shadow

    provider_health_blocking_status = _as_upper(provider_health.get("trade_blocking_status"))
    provider_health_blocking_reasons = provider_health.get("trade_blocking_reasons") if isinstance(provider_health.get("trade_blocking_reasons"), list) else []

    def _evaluate_provider_health_override(*, blocked: bool):
        return _evaluate_provider_health_override_eligibility(
            runtime_thresholds=runtime_thresholds,
            provider_health_blocking_reasons=provider_health_blocking_reasons,
            provider_health_summary=provider_health_summary,
            data_quality_status=data_quality.get("status"),
            confirmation_status=confirmation.get("status"),
            adjusted_trade_strength=adjusted_trade_strength,
            min_trade_strength=min_trade_strength,
            runtime_composite_score=runtime_composite_score,
            min_composite_score=min_composite_score,
            option_chain_validation=option_chain_validation or {},
            provider_health=provider_health or {},
            ranked_strikes=ranked_strikes,
            days_to_expiry=days_to_expiry,
            blocked=blocked,
            option_efficiency_score=base_payload.get("option_efficiency_score"),
            premium_efficiency_score=base_payload.get("premium_efficiency_score"),
        )

    def _apply_provider_health_override(*, reason_label: str):
        override_size_cap = _clip(_safe_float(runtime_thresholds.get("provider_health_override_size_cap"), 0.35), 0.0, 1.0)
        current_size_cap = _clip(_safe_float(base_payload.get("effective_size_cap"), 1.0), 0.0, 1.0)
        new_size_cap = min(current_size_cap, override_size_cap)
        base_payload["effective_size_cap"] = round(new_size_cap, 2)

        original_lots = max(int(_safe_float(base_payload.get("macro_suggested_lots"), requested_lots)), 0)
        constrained_lots = max(int(original_lots * new_size_cap), 0)
        if original_lots > 0 and constrained_lots == 0 and new_size_cap > 0:
            constrained_lots = 1
        base_payload["macro_suggested_lots"] = constrained_lots
        base_payload["advisory_lots"] = constrained_lots
        base_payload["advisory_position_size_multiplier"] = round(new_size_cap, 4)

        if "entry_price" in base_payload:
            base_payload["advisory_capital_required"] = round(entry_price * lot_size * constrained_lots, 2)

        override_hold_cap = max(int(_safe_float(runtime_thresholds.get("provider_health_override_hold_cap_minutes"), 35.0)), 5)
        base_payload["recommended_hold_minutes"] = min(int(_safe_float(base_payload.get("recommended_hold_minutes"), override_hold_cap)), override_hold_cap)
        base_payload["max_hold_minutes"] = min(int(_safe_float(base_payload.get("max_hold_minutes"), override_hold_cap)), override_hold_cap)
        base_payload["overnight_hold_allowed"] = False
        base_payload["overnight_trade_block"] = True
        base_payload["overnight_hold_reason"] = "provider_health_degraded_override_no_overnight"

        base_payload["provider_health_override_active"] = True
        base_payload["provider_health_override_mode"] = "DEGRADED_PROVIDER_TRADE"
        base_payload["provider_health_override_reason"] = reason_label
        base_payload["provider_health_override_constraints"] = [
            f"size_cap:{round(new_size_cap, 2)}",
            f"max_hold_minutes:{override_hold_cap}",
            "no_overnight",
        ]

        return _finalize(
            base_payload,
            "TRADE",
            "Tradable signal generated in degraded provider-health override mode",
        )

    if provider_health_blocking_status == "BLOCK":
        override_allowed, override_details = _evaluate_provider_health_override(blocked=True)
        if override_allowed:
            base_payload["provider_health_override_diagnostics"] = override_details
            return _apply_provider_health_override(reason_label="provider_health_block_override")
        reason_suffix = f" ({', '.join(str(r) for r in provider_health_blocking_reasons)})" if provider_health_blocking_reasons else ""
        base_payload["provider_health_override_diagnostics"] = override_details
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Provider health BLOCK routes TRADE to WATCHLIST{reason_suffix}",
        )

    caution_blocks_trade = bool(int(_safe_float(runtime_thresholds.get("provider_health_caution_blocks_trade"), 1.0)))
    if caution_blocks_trade and not provider_health_blocking_status and provider_health_summary in {"CAUTION", "WEAK"}:
        override_allowed, override_details = _evaluate_provider_health_override(blocked=False)
        if override_allowed:
            base_payload["provider_health_override_diagnostics"] = override_details
            return _apply_provider_health_override(reason_label="provider_health_caution_override")
        base_payload["provider_health_override_diagnostics"] = override_details
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Provider health {provider_health_summary} blocks TRADE and routes to WATCHLIST",
        )

    weak_data_triggered, weak_data_breaker = weak_data_shadow_triggered, weak_data_shadow
    base_payload["weak_data_circuit_breaker"] = weak_data_breaker
    if weak_data_triggered:
        base_payload["no_trade_reason_code"] = "WEAK_DATA_CIRCUIT_BREAKER"
        base_payload["no_trade_reason"] = "Weak-data circuit breaker routed trade to WATCHLIST"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Weak-data circuit breaker routed TRADE to WATCHLIST",
        )

    historical_outcome_guard = _compute_historical_outcome_guard(
        payload=base_payload,
        runtime_thresholds=runtime_thresholds,
    )
    base_payload["historical_outcome_guard"] = historical_outcome_guard
    base_payload["historical_outcome_samples"] = historical_outcome_guard.get("sample_size")
    base_payload["historical_best_horizon"] = historical_outcome_guard.get("best_horizon")
    base_payload["historical_exit_bias"] = historical_outcome_guard.get("exit_bias")
    base_payload["historical_avg_60m_bps"] = historical_outcome_guard.get("avg_60m_bps")
    base_payload["historical_avg_close_bps"] = historical_outcome_guard.get("avg_close_bps")
    base_payload["historical_avg_tradeability_score"] = historical_outcome_guard.get("avg_tradeability_score")

    outcome_guard_verdict = _as_upper(historical_outcome_guard.get("verdict"))
    if outcome_guard_verdict == "CAUTION":
        hold_cap = max(int(_safe_float(runtime_thresholds.get("historical_outcome_guard_hold_cap_minutes"), 30.0)), 5)
        size_cap = _clip(_safe_float(runtime_thresholds.get("historical_outcome_guard_size_cap"), 0.70), 0.0, 1.0)
        base_payload["recommended_hold_minutes"] = min(int(_safe_float(base_payload.get("recommended_hold_minutes"), hold_cap)), hold_cap)
        base_payload["max_hold_minutes"] = min(int(_safe_float(base_payload.get("max_hold_minutes"), hold_cap)), hold_cap)
        base_payload["overnight_hold_allowed"] = False
        base_payload["historical_outcome_guard_active"] = True
        base_payload["historical_outcome_guard_mode"] = "EARLY_EXIT_BIAS"
        current_size_cap = _clip(_safe_float(base_payload.get("effective_size_cap"), 1.0), 0.0, 1.0)
        new_size_cap = min(current_size_cap, size_cap)
        base_payload["effective_size_cap"] = round(new_size_cap, 2)
        existing_lots = max(int(_safe_float(base_payload.get("macro_suggested_lots"), 0.0)), 0)
        resized_lots = max(int(existing_lots * new_size_cap), 0)
        if existing_lots > 0 and resized_lots == 0 and new_size_cap > 0:
            resized_lots = 1
        if existing_lots > 0:
            base_payload["macro_suggested_lots"] = min(existing_lots, resized_lots)
            base_payload["advisory_lots"] = base_payload["macro_suggested_lots"]
            base_payload["advisory_position_size_multiplier"] = round(new_size_cap, 4)
    elif outcome_guard_verdict == "BLOCK":
        base_payload["historical_outcome_guard_active"] = True
        base_payload["no_trade_reason_code"] = "HISTORICAL_OUTCOME_GUARD"
        base_payload["no_trade_reason"] = historical_outcome_guard.get("reason") or "Historical outcome guard downgraded the setup"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Historical outcome guard routed TRADE to WATCHLIST",
        )

    session_risk_governor = _compute_session_risk_governor(
        payload={**base_payload, "valuation_time": valuation_time, "direction": direction},
        runtime_thresholds=runtime_thresholds,
    )
    base_payload["session_risk_governor"] = session_risk_governor
    base_payload["session_risk_recent_signal_count"] = session_risk_governor.get("recent_signal_count")
    base_payload["session_risk_stopout_streak"] = session_risk_governor.get("stopout_streak")
    base_payload["session_risk_budget_remaining_pct"] = session_risk_governor.get("budget_remaining_pct")
    base_payload["session_risk_cooldown_active"] = session_risk_governor.get("cooldown_active")

    session_risk_verdict = _as_upper(session_risk_governor.get("verdict"))
    if session_risk_verdict == "CAUTION":
        hold_cap = max(int(_safe_float(runtime_thresholds.get("session_risk_hold_cap_minutes"), 25.0)), 5)
        size_cap = _clip(_safe_float(session_risk_governor.get("size_cap"), _safe_float(runtime_thresholds.get("session_risk_caution_size_cap"), 0.60)), 0.0, 1.0)
        base_payload["recommended_hold_minutes"] = min(int(_safe_float(base_payload.get("recommended_hold_minutes"), hold_cap)), hold_cap)
        base_payload["max_hold_minutes"] = min(int(_safe_float(base_payload.get("max_hold_minutes"), hold_cap)), hold_cap)
        current_size_cap = _clip(_safe_float(base_payload.get("effective_size_cap"), 1.0), 0.0, 1.0)
        new_size_cap = min(current_size_cap, size_cap)
        base_payload["effective_size_cap"] = round(new_size_cap, 2)
        existing_lots = max(int(_safe_float(base_payload.get("macro_suggested_lots"), 0.0)), 0)
        resized_lots = max(int(existing_lots * new_size_cap), 0)
        if existing_lots > 0 and resized_lots == 0 and new_size_cap > 0:
            resized_lots = 1
        if existing_lots > 0:
            base_payload["macro_suggested_lots"] = min(existing_lots, resized_lots)
            base_payload["advisory_lots"] = base_payload["macro_suggested_lots"]
            base_payload["advisory_position_size_multiplier"] = round(new_size_cap, 4)
    elif session_risk_verdict == "BLOCK":
        base_payload["no_trade_reason_code"] = "SESSION_RISK_GOVERNOR"
        base_payload["no_trade_reason"] = session_risk_governor.get("reason") or "Session risk governor downgraded the setup"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Session risk governor routed TRADE to WATCHLIST",
        )

    trade_slot_governor = _compute_trade_slot_governor(
        payload={
            **base_payload,
            "valuation_time": valuation_time,
            "direction": direction,
            "operator_control_state": operator_control_state if isinstance(operator_control_state, dict) else {},
        },
        runtime_thresholds=runtime_thresholds,
    )
    base_payload["trade_slot_governor"] = trade_slot_governor
    base_payload["trade_slot_active_signal_count"] = trade_slot_governor.get("active_signal_count")
    base_payload["trade_slot_same_direction_count"] = trade_slot_governor.get("same_direction_count")
    base_payload["operator_override_active"] = trade_slot_governor.get("operator_override_active")
    base_payload["operator_override_reason"] = trade_slot_governor.get("override_reason")
    base_payload["operator_override_mode"] = trade_slot_governor.get("override_mode")

    trade_slot_verdict = _as_upper(trade_slot_governor.get("verdict"))
    if trade_slot_verdict == "CAUTION":
        hold_cap = max(int(_safe_float(trade_slot_governor.get("hold_cap_minutes"), _safe_float(runtime_thresholds.get("trade_slot_hold_cap_minutes"), 20.0))), 5)
        size_cap = _clip(_safe_float(trade_slot_governor.get("size_cap"), _safe_float(runtime_thresholds.get("trade_slot_caution_size_cap"), 0.55)), 0.0, 1.0)
        base_payload["recommended_hold_minutes"] = min(int(_safe_float(base_payload.get("recommended_hold_minutes"), hold_cap)), hold_cap)
        base_payload["max_hold_minutes"] = min(int(_safe_float(base_payload.get("max_hold_minutes"), hold_cap)), hold_cap)
        current_size_cap = _clip(_safe_float(base_payload.get("effective_size_cap"), 1.0), 0.0, 1.0)
        new_size_cap = min(current_size_cap, size_cap)
        base_payload["effective_size_cap"] = round(new_size_cap, 2)
        existing_lots = max(int(_safe_float(base_payload.get("macro_suggested_lots"), 0.0)), 0)
        resized_lots = max(int(existing_lots * new_size_cap), 0)
        if existing_lots > 0 and resized_lots == 0 and new_size_cap > 0:
            resized_lots = 1
        if existing_lots > 0:
            base_payload["macro_suggested_lots"] = min(existing_lots, resized_lots)
            base_payload["advisory_lots"] = base_payload["macro_suggested_lots"]
            base_payload["advisory_position_size_multiplier"] = round(new_size_cap, 4)
        if bool(trade_slot_governor.get("intraday_only")):
            base_payload["overnight_hold_allowed"] = False
            base_payload["overnight_trade_block"] = True
            base_payload["overnight_hold_reason"] = "trade_slot_governor_requires_intraday_only"
    elif trade_slot_verdict == "BLOCK":
        base_payload["no_trade_reason_code"] = "TRADE_SLOT_GOVERNOR"
        base_payload["no_trade_reason"] = trade_slot_governor.get("reason") or "Trade slot governor downgraded the setup"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Trade slot governor routed TRADE to WATCHLIST",
        )

    trade_promotion_governor = _evaluate_trade_promotion_governor(
        payload=base_payload,
        runtime_thresholds=runtime_thresholds,
    )
    base_payload["trade_promotion_governor"] = trade_promotion_governor
    base_payload["replay_validation_required"] = trade_promotion_governor.get("replay_validation_required")
    base_payload["promotion_state"] = trade_promotion_governor.get("promotion_state")

    trade_promotion_verdict = _as_upper(trade_promotion_governor.get("verdict"))
    if trade_promotion_verdict == "BLOCK":
        base_payload["no_trade_reason_code"] = "TRADE_PROMOTION_GOVERNOR"
        base_payload["no_trade_reason"] = trade_promotion_governor.get("reason") or "Trade promotion governor downgraded the setup"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Trade promotion governor routed TRADE to WATCHLIST",
        )

    if budget_signal_only:
        base_payload["no_trade_reason_code"] = "BUDGET_SIGNAL_ONLY"
        base_payload["no_trade_reason"] = "Signal remains valid, but execution sizing is withheld by the budget constraint"
        return _finalize(
            base_payload,
            "WATCHLIST",
            "Signal generated; execution sizing withheld due to budget constraint",
        )

    if apply_budget_constraint:
        base_payload["message"] = "Tradable signal generated with advisory sizing separation"
    else:
        base_payload["message"] = "Tradable signal generated"

    return _finalize(
        base_payload,
        "TRADE",
        "Tradable signal generated with advisory sizing separation" if apply_budget_constraint else "Tradable signal generated",
    )
