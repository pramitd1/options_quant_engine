"""
Signal evaluation and calibration research dataset builder.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.signal_evaluation_scoring import (
    SIGNAL_EVALUATION_DIRECTION_WEIGHTS,
    SIGNAL_EVALUATION_SCORE_WEIGHTS,
    SIGNAL_EVALUATION_THRESHOLDS,
    SIGNAL_EVALUATION_TIMING_WEIGHTS,
)
from config.settings import BASE_DIR
from data.spot_downloader import normalize_underlying_symbol
from research.signal_evaluation.dataset import SIGNAL_DATASET_PATH, upsert_signal_rows


IST_TIMEZONE = "Asia/Kolkata"
EVALUATION_WINDOW_MINUTES = 120
HORIZON_MINUTES = [5, 15, 30, 60]


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _coerce_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(IST_TIMEZONE)
    return ts.tz_convert(IST_TIMEZONE)


def _signal_direction_multiplier(direction: str | None) -> int:
    return 1 if str(direction or "").upper() == "CALL" else -1


def _bucket_trade_strength(value) -> str | None:
    value = _safe_float(value, None)
    if value is None:
        return None
    if value >= 80:
        return "80_100"
    if value >= 65:
        return "65_79"
    if value >= 50:
        return "50_64"
    if value >= 35:
        return "35_49"
    return "0_34"


def _bucket_probability(value) -> str | None:
    value = _safe_float(value, None)
    if value is None:
        return None
    if value >= 0.80:
        return "0.80_1.00"
    if value >= 0.65:
        return "0.65_0.79"
    if value >= 0.50:
        return "0.50_0.64"
    if value >= 0.35:
        return "0.35_0.49"
    return "0.00_0.34"


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


def build_signal_evaluation_row(result: dict, *, notes: str | None = None) -> dict:
    if not result or not result.get("trade"):
        raise ValueError("Result payload must include a trade object")

    trade = result["trade"] or {}
    spot_summary = result.get("spot_summary", {}) or {}
    signal_timestamp = spot_summary.get("timestamp") or trade.get("valuation_time")
    if signal_timestamp is None:
        raise ValueError("Signal row requires a stable signal timestamp")

    signal_id = build_signal_id(
        signal_timestamp=signal_timestamp,
        source=result.get("source"),
        mode=result.get("mode"),
        symbol=result.get("symbol"),
        selected_expiry=trade.get("selected_expiry"),
        direction=trade.get("direction"),
        strike=trade.get("strike"),
        option_type=trade.get("option_type"),
    )

    provider_health = trade.get("provider_health") or result.get("option_chain_validation", {}).get("provider_health", {}) or {}
    regime_fingerprint, regime_fingerprint_id = build_regime_fingerprint(trade, provider_health)
    saved_paths = result.get("saved_paths") or {}
    now_ts = pd.Timestamp.now(tz=IST_TIMEZONE).isoformat()

    row = {
        "signal_id": signal_id,
        "signal_timestamp": _coerce_ts(signal_timestamp).isoformat(),
        "source": str(result.get("source") or "").upper().strip(),
        "mode": str(result.get("mode") or "").upper().strip(),
        "symbol": normalize_underlying_symbol(result.get("symbol")),
        "ticker": spot_summary.get("ticker") or result.get("spot_snapshot", {}).get("ticker"),
        "selected_expiry": trade.get("selected_expiry"),
        "direction": trade.get("direction"),
        "option_type": trade.get("option_type"),
        "strike": trade.get("strike"),
        "entry_price": trade.get("entry_price"),
        "target": trade.get("target"),
        "stop_loss": trade.get("stop_loss"),
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
        "dealer_position": trade.get("dealer_position"),
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
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
        "created_at": now_ts,
        "updated_at": now_ts,
        "outcome_last_updated_at": pd.NA,
        "outcome_status": "PENDING",
        "observed_minutes": 0.0,
        "evaluation_window_minutes": EVALUATION_WINDOW_MINUTES,
        "directional_consistency_score": pd.NA,
        "signal_calibration_bucket": _bucket_trade_strength(trade.get("trade_strength")),
        "probability_calibration_bucket": _bucket_probability(trade.get("hybrid_move_probability")),
        "notes": notes,
    }

    for horizon in HORIZON_MINUTES:
        row[f"spot_{horizon}m"] = pd.NA
        row[f"signed_return_{horizon}m_bps"] = pd.NA
        row[f"correct_{horizon}m"] = pd.NA

    row["spot_session_close"] = pd.NA
    row["signed_return_session_close_bps"] = pd.NA
    row["correct_session_close"] = pd.NA
    row["mfe_60m_bps"] = pd.NA
    row["mae_60m_bps"] = pd.NA
    row["mfe_120m_bps"] = pd.NA
    row["mae_120m_bps"] = pd.NA
    row["realized_range_60m_bps"] = pd.NA
    row["realized_range_120m_bps"] = pd.NA
    return row


def _nearest_spot_at_or_after(path: pd.DataFrame, target_ts: pd.Timestamp):
    candidates = path[path["timestamp"] >= target_ts]
    if candidates.empty:
        return None
    return _safe_float(candidates.iloc[0]["spot"], None)


def _window_stats(path: pd.DataFrame, entry_ts: pd.Timestamp, end_ts: pd.Timestamp, direction_mult: int, entry_spot: float):
    window = path[(path["timestamp"] >= entry_ts) & (path["timestamp"] <= end_ts)].copy()
    if window.empty:
        return None

    signed_moves_bps = ((window["spot"].astype(float) - entry_spot) / max(entry_spot, 1e-9)) * 10000.0 * direction_mult
    raw_moves_bps = ((window["spot"].astype(float) - entry_spot) / max(entry_spot, 1e-9)) * 10000.0

    return {
        "mfe_bps": round(float(signed_moves_bps.max()), 2),
        "mae_bps": round(float(signed_moves_bps.min()), 2),
        "range_bps": round(float(raw_moves_bps.max() - raw_moves_bps.min()), 2),
    }


def _session_close_spot(path: pd.DataFrame, signal_ts: pd.Timestamp):
    same_day = path[path["timestamp"].dt.date == signal_ts.date()]
    if same_day.empty:
        return None
    return _safe_float(same_day.iloc[-1]["spot"], None)


def _next_day_rows(path: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.DataFrame:
    future_days = path[path["timestamp"].dt.date > signal_ts.date()].copy()
    if future_days.empty:
        return future_days
    first_next_day = future_days["timestamp"].dt.date.min()
    return future_days[future_days["timestamp"].dt.date == first_next_day].copy()


def _next_day_open_spot(path: pd.DataFrame, signal_ts: pd.Timestamp):
    next_day = _next_day_rows(path, signal_ts)
    if next_day.empty:
        return None
    return _safe_float(next_day.iloc[0]["spot"], None)


def _next_day_close_spot(path: pd.DataFrame, signal_ts: pd.Timestamp):
    next_day = _next_day_rows(path, signal_ts)
    if next_day.empty:
        return None
    return _safe_float(next_day.iloc[-1]["spot"], None)


def _raw_return(entry_spot: float, horizon_spot: float):
    if entry_spot in (None, 0) or horizon_spot is None:
        return None
    return round(float((horizon_spot - entry_spot) / entry_spot), 6)


def _clip_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 2)


def compute_signal_evaluation_scores(row: dict) -> dict:
    updated = dict(row)

    direction_numerator = 0.0
    direction_denominator = 0.0
    for field_name, weight in SIGNAL_EVALUATION_DIRECTION_WEIGHTS.items():
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
    if mfe_points is not None and spot_at_signal not in (None, 0):
        favorable_move_pct = abs(mfe_points) / spot_at_signal * 100.0
        baseline_range_pct = lookback_avg_range_pct if lookback_avg_range_pct not in (None, 0) else 1.0
        magnitude_vs_range = favorable_move_pct / max(baseline_range_pct, 0.1)

        weak = SIGNAL_EVALUATION_THRESHOLDS["magnitude_vs_range_weak"]
        good = SIGNAL_EVALUATION_THRESHOLDS["magnitude_vs_range_good"]
        strong = SIGNAL_EVALUATION_THRESHOLDS["magnitude_vs_range_strong"]

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
    return_floor = SIGNAL_EVALUATION_THRESHOLDS["timing_positive_return_floor"]
    for field_name, weight in SIGNAL_EVALUATION_TIMING_WEIGHTS.items():
        value = _safe_float(updated.get(field_name), None)
        if value is None:
            continue
        horizon_score = max(0.0, min(1.0, value / max(return_floor, 1e-6)))
        timing_numerator += horizon_score * float(weight)
        timing_denominator += float(weight)

    timing_score = None
    if timing_denominator > 0:
        timing_score = _clip_score((timing_numerator / timing_denominator) * 100.0)

    tradeability_score = None
    mae_points = _safe_float(updated.get("mae_points"), None)
    if mfe_points is not None and mae_points is not None:
        adverse_points = abs(min(mae_points, 0.0))
        if adverse_points == 0:
            tradeability_ratio = float("inf")
        else:
            tradeability_ratio = abs(mfe_points) / adverse_points

        floor = SIGNAL_EVALUATION_THRESHOLDS["tradeability_ratio_floor"]
        good = SIGNAL_EVALUATION_THRESHOLDS["tradeability_ratio_good"]
        strong = SIGNAL_EVALUATION_THRESHOLDS["tradeability_ratio_strong"]

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

    component_scores = {
        "direction_score": direction_score,
        "magnitude_score": magnitude_score,
        "timing_score": timing_score,
        "tradeability_score": tradeability_score,
    }
    if all(score is not None for score in component_scores.values()):
        composite = sum(
            component_scores[name] * SIGNAL_EVALUATION_SCORE_WEIGHTS[name]
            for name in component_scores
        )
        updated["composite_signal_score"] = _clip_score(composite)
    else:
        updated["composite_signal_score"] = pd.NA

    return updated


def evaluate_signal_outcomes(row: dict, realized_spot_path: pd.DataFrame, *, as_of=None) -> dict:
    updated = dict(row)
    if realized_spot_path is None or realized_spot_path.empty:
        updated["outcome_status"] = "PENDING"
        return updated

    path = realized_spot_path.copy()
    if "timestamp" not in path.columns or "spot" not in path.columns:
        raise ValueError("Realized spot path must include 'timestamp' and 'spot' columns")

    path["timestamp"] = path["timestamp"].map(_coerce_ts)
    path["spot"] = pd.to_numeric(path["spot"], errors="coerce")
    path = path.dropna(subset=["timestamp", "spot"]).sort_values("timestamp").reset_index(drop=True)
    if path.empty:
        updated["outcome_status"] = "PENDING"
        return updated

    signal_ts = _coerce_ts(updated["signal_timestamp"])
    as_of_ts = _coerce_ts(as_of) if as_of is not None else path["timestamp"].max()
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

    completed_checkpoints = 0
    for horizon in HORIZON_MINUTES:
        target_ts = signal_ts + pd.Timedelta(minutes=horizon)
        horizon_spot = _nearest_spot_at_or_after(path, target_ts)
        if horizon_spot is None:
            continue

        signed_return_bps = ((horizon_spot - entry_spot) / max(entry_spot, 1e-9)) * 10000.0 * direction_mult
        updated[f"spot_{horizon}m"] = round(horizon_spot, 4)
        updated[f"realized_return_{horizon}m"] = _raw_return(entry_spot, horizon_spot)
        updated[f"signed_return_{horizon}m_bps"] = round(float(signed_return_bps), 2)
        updated[f"correct_{horizon}m"] = int(signed_return_bps > 0)
        completed_checkpoints += 1

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
        signed_close_return = ((close_spot - entry_spot) / max(entry_spot, 1e-9)) * 10000.0 * direction_mult
        updated["spot_close_same_day"] = round(close_spot, 4)
        updated["spot_session_close"] = round(close_spot, 4)
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

    correctness_fields = [updated.get(f"correct_{horizon}m") for horizon in HORIZON_MINUTES]
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

    total_checkpoints = len(HORIZON_MINUTES) + 3
    if completed_checkpoints == 0:
        updated["outcome_status"] = "PENDING"
    elif completed_checkpoints < total_checkpoints:
        updated["outcome_status"] = "PARTIAL"
    else:
        updated["outcome_status"] = "COMPLETE"

    updated["updated_at"] = pd.Timestamp.now(tz=IST_TIMEZONE).isoformat()
    return updated


def _normalize_symbol_to_yfinance(symbol: str) -> str:
    normalized = normalize_underlying_symbol(symbol)
    ticker_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^NSEFIN",
    }
    if normalized in ticker_map:
        return ticker_map[normalized]
    if normalized.startswith("^") or "." in normalized:
        return normalized
    return f"{normalized}.NS"


def fetch_realized_spot_path(symbol: str, signal_timestamp, *, as_of=None, interval: str = "5m") -> pd.DataFrame:
    signal_ts = _coerce_ts(signal_timestamp)
    end_ts = _coerce_ts(as_of) if as_of is not None else pd.Timestamp.now(tz=IST_TIMEZONE)
    fetch_end_ts = max(end_ts, signal_ts + pd.Timedelta(days=2))

    ticker = yf.Ticker(_normalize_symbol_to_yfinance(symbol))
    frame = ticker.history(
        start=(signal_ts - pd.Timedelta(days=1)).tz_convert("UTC").to_pydatetime(),
        end=(fetch_end_ts + pd.Timedelta(days=1)).tz_convert("UTC").to_pydatetime(),
        interval=interval,
        auto_adjust=False,
    )

    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "spot"])

    history = frame.reset_index()
    ts_col = "Datetime" if "Datetime" in history.columns else "Date"
    history["timestamp"] = history[ts_col].map(_coerce_ts)
    history["spot"] = pd.to_numeric(history.get("Close"), errors="coerce")
    history = history.dropna(subset=["timestamp", "spot"])
    history = history.loc[(history["timestamp"] >= signal_ts) & (history["timestamp"] <= fetch_end_ts)].reset_index(drop=True)
    return history[["timestamp", "spot"]]


def save_signal_evaluation(
    result: dict,
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    realized_spot_path: pd.DataFrame | None = None,
    as_of=None,
    notes: str | None = None,
) -> pd.DataFrame:
    row = build_signal_evaluation_row(result, notes=notes)
    if realized_spot_path is not None and not realized_spot_path.empty:
        row = evaluate_signal_outcomes(row, realized_spot_path, as_of=as_of)
    return upsert_signal_rows([row], path=dataset_path)


def update_signal_dataset_outcomes(
    *,
    dataset_path: str | Path = SIGNAL_DATASET_PATH,
    as_of=None,
    fetch_spot_path_fn=fetch_realized_spot_path,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        return upsert_signal_rows([], path=dataset_path)

    frame = pd.read_csv(dataset_path)
    if frame.empty:
        return upsert_signal_rows([], path=dataset_path)

    updated_rows = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        if row_dict.get("outcome_status") == "COMPLETE":
            updated_rows.append(row_dict)
            continue

        realized_path = fetch_spot_path_fn(
            row_dict.get("symbol"),
            row_dict.get("signal_timestamp"),
            as_of=as_of,
        )
        updated_rows.append(evaluate_signal_outcomes(row_dict, realized_path, as_of=as_of))

    return upsert_signal_rows(updated_rows, path=dataset_path)
