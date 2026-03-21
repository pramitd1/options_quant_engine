"""
Holistic Backtest Runner — Signal-Centric
============================================
Evaluates the signal engine against the full 2012-2026 historical NSE option
chain database.  The approach mirrors the live signal evaluation pipeline:

  - For each trading day, resolve upcoming expiries (nearest N).
  - For each expiry, run the signal engine → capture the full signal payload.
  - Evaluate every signal against the realized spot path (subsequent days through
    expiry) using the same scoring framework as the live evaluator.
  - Aggregate signal quality metrics across the entire date range.

This runner is **signal-centric, not trade-centric**: it does not simulate
positions, open/close trades, or compute portfolio PnL.  Every day × expiry
produces an independent signal evaluation record, scored on direction accuracy,
magnitude, timing, and tradeability against what actually happened.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from app.engine_runner import run_preloaded_engine_snapshot
from config.settings import (
    BACKTEST_ENABLE_BUDGET,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
    STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT,
)
from data.expiry_resolver import (
    filter_option_chain_by_expiry,
    ordered_expiries,
)
from data.historical_snapshot import (
    SPOT_FILE,
    get_available_dates,
    replay_historical_snapshot,
)
from research.signal_evaluation.evaluator import (
    build_signal_evaluation_row,
    compute_signal_evaluation_scores,
    evaluate_signal_outcomes,
)
from utils.numerics import safe_float as _safe_float

log = logging.getLogger(__name__)

# EOD horizon day offsets used for outcome evaluation
EOD_HORIZON_DAYS = (1, 2, 3, 5)

# ---------------------------------------------------------------
# Spot-path builder (daily resolution from historical data)
# ---------------------------------------------------------------

_spot_df_cache: pd.DataFrame | None = None


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no external dependency)."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _estimate_abs_option_delta(
    *,
    direction_mult: int,
    entry_spot: float,
    strike: float,
    signal_date: date,
    expiry_date: date | None,
    atm_iv_scaled: float | None,
) -> tuple[float, str]:
    """Estimate absolute option delta for premium->spot conversion.

    Preference order:
      1) Black-Scholes style estimate using ATM IV and time-to-expiry.
      2) Smooth moneyness fallback when inputs are incomplete.
    """
    # Fallback first so every path has a stable estimate.
    moneyness = (entry_spot - strike) / entry_spot if entry_spot else 0.0
    if direction_mult == 1:
        fallback = 1.0 / (1.0 + math.exp(-8.0 * moneyness))
    else:
        fallback = 1.0 / (1.0 + math.exp(8.0 * moneyness))
    fallback = float(np.clip(fallback, 0.10, 0.90))

    if direction_mult == 0 or entry_spot <= 0 or strike <= 0 or expiry_date is None:
        return fallback, "moneyness_fallback"

    days_to_expiry = max((expiry_date - signal_date).days, 1)
    t_years = max(days_to_expiry / 365.0, 1.0 / 365.0)
    sigma = _safe_float(atm_iv_scaled, None)
    if sigma is None:
        sigma = 0.20
    sigma = float(np.clip(sigma, 0.05, 2.50))

    try:
        vol_term = sigma * math.sqrt(t_years)
        if vol_term <= 0:
            return fallback, "moneyness_fallback"
        d1 = (
            math.log(entry_spot / strike)
            + 0.5 * sigma * sigma * t_years
        ) / vol_term
        call_delta = _norm_cdf(d1)
        put_delta = call_delta - 1.0
        abs_delta = call_delta if direction_mult == 1 else abs(put_delta)
        abs_delta = float(np.clip(abs_delta, 0.05, 0.95))
        return abs_delta, "bs_atm_iv"
    except Exception:
        return fallback, "moneyness_fallback"


def _load_spot_daily() -> pd.DataFrame:
    """Load and cache the daily spot OHLCV parquet."""
    global _spot_df_cache
    if _spot_df_cache is not None:
        return _spot_df_cache
    if not SPOT_FILE.exists():
        return pd.DataFrame()
    _spot_df_cache = pd.read_parquet(SPOT_FILE)
    _spot_df_cache["date"] = pd.to_datetime(_spot_df_cache["date"])
    _spot_df_cache = _spot_df_cache.sort_values("date").reset_index(drop=True)
    return _spot_df_cache


def build_realized_spot_path(
    signal_date: date,
    expiry_date: date | None = None,
    max_days: int = 10,
) -> pd.DataFrame:
    """Build a daily-resolution realized spot path for outcome evaluation.

    Returns DataFrame with columns [timestamp, spot] compatible with
    ``evaluate_signal_outcomes``.  Timestamps are set to 15:30 IST (market
    close) so the evaluator's minute-based horizon snapping resolves cleanly.
    """
    spot = _load_spot_daily()
    if spot.empty:
        return pd.DataFrame(columns=["timestamp", "spot"])

    start = pd.to_datetime(signal_date)
    if expiry_date is not None:
        end = pd.to_datetime(expiry_date) + timedelta(days=2)
    else:
        end = start + timedelta(days=max_days)

    mask = (spot["date"] >= start) & (spot["date"] <= end)
    window = spot.loc[mask].copy()

    if window.empty:
        return pd.DataFrame(columns=["timestamp", "spot"])

    # Build intraday-style timestamps at market close (15:30 IST)
    rows = []
    for _, r in window.iterrows():
        d = r["date"]
        # Market open proxy
        rows.append({
            "timestamp": pd.Timestamp(d.year, d.month, d.day, 9, 15,
                                      tz="Asia/Kolkata"),
            "spot": float(r["open"]),
        })
        # Intraday high / low as mid-session proxies
        rows.append({
            "timestamp": pd.Timestamp(d.year, d.month, d.day, 11, 30,
                                      tz="Asia/Kolkata"),
            "spot": float(r["high"]),
        })
        rows.append({
            "timestamp": pd.Timestamp(d.year, d.month, d.day, 13, 0,
                                      tz="Asia/Kolkata"),
            "spot": float(r["low"]),
        })
        # Market close
        rows.append({
            "timestamp": pd.Timestamp(d.year, d.month, d.day, 15, 30,
                                      tz="Asia/Kolkata"),
            "spot": float(r["close"]),
        })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------
# EOD outcome evaluation (extends the live evaluator)
# ---------------------------------------------------------------

def evaluate_eod_outcomes(
    row: dict,
    realized_spot_path: pd.DataFrame,
    available_dates: list[date],
) -> dict:
    """Evaluate a signal against the daily-resolution realized spot path.

    This first runs the standard ``evaluate_signal_outcomes`` (which fills the
    5m/15m/30m/60m/120m horizon fields via nearest-timestamp snapping), then
    adds EOD-specific outcome fields:

    - ``spot_Nd`` / ``return_Nd_bps`` / ``correct_Nd`` for N in (1, 2, 3, 5)
    - ``spot_at_expiry`` / ``return_at_expiry_bps`` / ``correct_at_expiry``
    - ``eod_mfe_bps`` / ``eod_mae_bps`` (daily-resolution MFE/MAE to expiry)
    - ``target_hit`` / ``stop_loss_hit`` flags
    """
    # 1. Run the standard evaluator for baseline fields
    updated = evaluate_signal_outcomes(row, realized_spot_path)

    entry_spot = updated.get("spot_at_signal")
    if not entry_spot or entry_spot == 0:
        return updated

    direction = updated.get("direction")
    direction_mult = 1 if direction == "CALL" else (-1 if direction == "PUT" else 0)
    signal_date = pd.to_datetime(updated["signal_timestamp"]).date()

    # 2. EOD horizon fields (T+1, T+2, T+3, T+5 trading days)
    spot_daily = _load_spot_daily()
    if spot_daily.empty:
        return updated

    future_dates = [d for d in available_dates if d > signal_date]

    for offset in EOD_HORIZON_DAYS:
        if offset - 1 >= len(future_dates):
            continue
        target_date = future_dates[offset - 1]
        target_row = spot_daily[spot_daily["date"].dt.date == target_date]
        if target_row.empty:
            continue
        horizon_close = float(target_row.iloc[0]["close"])
        updated[f"spot_{offset}d"] = round(horizon_close, 4)
        if direction_mult != 0:
            ret_bps = ((horizon_close - entry_spot) / entry_spot) * 10000.0 * direction_mult
            updated[f"return_{offset}d_bps"] = round(ret_bps, 2)
            updated[f"correct_{offset}d"] = int(ret_bps > 0)

    # 3. Spot at expiry
    expiry_str = updated.get("selected_expiry")
    if expiry_str:
        try:
            expiry_date = pd.to_datetime(expiry_str).date()
        except Exception:
            expiry_date = None
        if expiry_date:
            exp_row = spot_daily[spot_daily["date"].dt.date == expiry_date]
            if exp_row.empty:
                # Use nearest prior date
                prior = spot_daily[spot_daily["date"].dt.date <= expiry_date]
                if not prior.empty:
                    exp_row = prior.tail(1)
            if not exp_row.empty:
                expiry_close = float(exp_row.iloc[0]["close"])
                updated["spot_at_expiry"] = round(expiry_close, 4)
                if direction_mult != 0:
                    ret_bps = ((expiry_close - entry_spot) / entry_spot) * 10000.0 * direction_mult
                    updated["return_at_expiry_bps"] = round(ret_bps, 2)
                    updated["correct_at_expiry"] = int(ret_bps > 0)

    # 4. Daily MFE / MAE from signal date to expiry (or +10 days)
    if direction_mult != 0 and expiry_str:
        try:
            exp_d = pd.to_datetime(expiry_str).date()
        except Exception:
            exp_d = signal_date + timedelta(days=10)
        window = spot_daily[
            (spot_daily["date"].dt.date > signal_date) &
            (spot_daily["date"].dt.date <= exp_d)
        ]
        if not window.empty:
            highs = window["high"].astype(float)
            lows = window["low"].astype(float)
            if direction_mult == 1:  # CALL
                mfe_bps = ((highs.max() - entry_spot) / entry_spot) * 10000.0
                mae_bps = ((lows.min() - entry_spot) / entry_spot) * 10000.0
            else:  # PUT
                mfe_bps = ((entry_spot - lows.min()) / entry_spot) * 10000.0
                mae_bps = ((entry_spot - highs.max()) / entry_spot) * 10000.0
            updated["eod_mfe_bps"] = round(mfe_bps, 2)
            updated["eod_mae_bps"] = round(mae_bps, 2)

    # 5. Target / stop-loss hit check (delta-adjusted)
    # Target/SL are in option premium space; convert to spot moves using an
    # estimated absolute option delta so we can compare against spot high/low.
    target = updated.get("target")
    stop_loss = updated.get("stop_loss")
    if target and stop_loss and expiry_str:
        try:
            exp_d = pd.to_datetime(expiry_str).date()
        except Exception:
            exp_d = signal_date + timedelta(days=10)
        window = spot_daily[
            (spot_daily["date"].dt.date > signal_date) &
            (spot_daily["date"].dt.date <= exp_d)
        ]
        entry_price = updated.get("entry_price", 0)
        if entry_price and not window.empty:
            strike = updated.get("strike", entry_spot)
            try:
                strike = float(strike)
            except (TypeError, ValueError):
                strike = entry_spot
            est_delta, est_delta_source = _estimate_abs_option_delta(
                direction_mult=direction_mult,
                entry_spot=float(entry_spot),
                strike=float(strike),
                signal_date=signal_date,
                expiry_date=exp_d,
                atm_iv_scaled=_safe_float(updated.get("atm_iv_scaled"), None),
            )
            updated["target_sl_delta_used"] = round(est_delta, 4)
            updated["target_sl_delta_source"] = est_delta_source

            # Convert option premium targets to spot point moves
            target_premium_move = float(target) - entry_price
            sl_premium_move = entry_price - float(stop_loss)
            target_spot_move = abs(target_premium_move / est_delta) if est_delta > 0 else float("inf")
            sl_spot_move = abs(sl_premium_move / est_delta) if est_delta > 0 else float("inf")

            for _, wr in window.iterrows():
                spot_high = float(wr["high"])
                spot_low = float(wr["low"])
                if direction_mult == 1:
                    favorable = spot_high - entry_spot
                    adverse = entry_spot - spot_low
                else:
                    favorable = entry_spot - spot_low
                    adverse = spot_high - entry_spot
                if favorable >= target_spot_move:
                    updated["target_hit"] = True
                    updated["target_hit_date"] = str(wr["date"].date() if hasattr(wr["date"], "date") else wr["date"])
                    break
                if adverse >= sl_spot_move:
                    updated["stop_loss_hit"] = True
                    updated["stop_loss_hit_date"] = str(wr["date"].date() if hasattr(wr["date"], "date") else wr["date"])
                    break

        if "target_hit" not in updated:
            updated["target_hit"] = False
        if "stop_loss_hit" not in updated:
            updated["stop_loss_hit"] = False

    return updated


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def run_holistic_backtest(
    symbol: str = "NIFTY",
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    max_expiries: int = 3,
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
    compute_iv: bool = True,
    include_global_market: bool = True,
    include_macro_events: bool = True,
    min_quality_score: float = 40.0,
    evaluate_outcomes: bool = True,
    progress_callback=None,
    prediction_method: str | None = None,
) -> dict:
    """Run a signal-centric historical backtest.

    For each trading day in the date range, resolves up to ``max_expiries``
    upcoming expiries, runs the signal engine for each, captures every signal
    in the 136-column evaluation schema, then evaluates each signal against
    the realized daily spot path.

    Parameters
    ----------
    symbol : str
        Underlying (default NIFTY).
    start_date, end_date : date | str | None
        Date range.  ``None`` ⇒ earliest / latest available.
    max_expiries : int
        How many upcoming expiries to evaluate per day (default 3).
    target_profit_percent, stop_loss_percent : float
        TP / SL thresholds passed to the signal engine.
    compute_iv : bool
        Compute implied vol via Newton-Raphson.
    include_global_market : bool
        Wire real historical global market context.
    include_macro_events : bool
        Wire real historical macro event schedule.
    min_quality_score : float
        Skip days with chain quality below this threshold.
    evaluate_outcomes : bool
        Whether to evaluate signals against realized spot path.
    progress_callback : callable | None
        ``fn(current_idx, total, trade_date)`` for progress reporting.
    prediction_method : str | None
        Override the global ``PREDICTION_METHOD`` for this run.
        E.g. ``"pure_ml"``, ``"pure_rule"``, ``"research_dual_model"``.
        ``None`` (default) uses the global setting.

    Returns
    -------
    dict with keys: ok, signals, daily_summary, metrics, metadata.
    """
    t0 = time.time()

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()

    available = get_available_dates(symbol)
    if not available:
        return _empty_result(symbol, "No trading dates available in database")

    if start_date:
        available = [d for d in available if d >= start_date]
    if end_date:
        available = [d for d in available if d <= end_date]

    if not available:
        return _empty_result(symbol, "No dates in range after filtering")

    total_days = len(available)
    log.info(
        "Holistic backtest (signal-centric): %s from %s to %s (%d days, max %d expiries/day)",
        symbol, available[0], available[-1], total_days, max_expiries,
    )

    all_signals: list[dict] = []
    daily_summary: list[dict] = []
    skipped_days = 0
    previous_chain: pd.DataFrame | None = None

    # -- Activate predictor override for this backtest run --
    _pred_factory = None
    _saved_predictor = None
    _saved_override = None
    if prediction_method:
        from engine.predictors import factory as _pf
        _pred_factory = _pf
        _saved_predictor = _pf._ACTIVE_PREDICTOR
        _saved_override = _pf._METHOD_OVERRIDE
        registry = _pf._ensure_registry()
        cls = registry.get(prediction_method)
        if cls is None:
            return _empty_result(
                symbol,
                f"Unknown prediction_method: {prediction_method!r}. "
                f"Available: {', '.join(sorted(registry))}",
            )
        _pf._ACTIVE_PREDICTOR = cls()
        _pf._METHOD_OVERRIDE = prediction_method
        log.info("Backtest predictor override → %s", prediction_method)

    try:

        for idx, trade_date in enumerate(available):
            if progress_callback:
                progress_callback(idx, total_days, trade_date)

            # 1. Build snapshot
            snap = replay_historical_snapshot(
                trade_date,
                symbol,
                compute_iv=compute_iv,
                include_global_market=include_global_market,
                include_macro_events=include_macro_events,
            )

            if not snap["ok"]:
                skipped_days += 1
                daily_summary.append(_day_stat(trade_date, "NO_DATA", 0))
                continue

            if snap["quality_score"] < min_quality_score:
                skipped_days += 1
                daily_summary.append(_day_stat(trade_date, "LOW_QUALITY", snap["quality_score"]))
                continue

            option_chain = snap["option_chain"]
            spot_snapshot = snap["spot_snapshot"]

            # 2. Resolve upcoming expiries (nearest N)
            expiries = ordered_expiries(option_chain)
            if not expiries:
                skipped_days += 1
                daily_summary.append(_day_stat(trade_date, "NO_EXPIRY", snap["quality_score"]))
                continue

            target_expiries = expiries[:max_expiries]
            day_signals = []

            # 3. Run engine for each expiry
            for expiry in target_expiries:
                expiry_chain = filter_option_chain_by_expiry(option_chain, expiry)
                if expiry_chain is None or expiry_chain.empty:
                    continue

                signal_result = run_preloaded_engine_snapshot(
                    symbol=symbol,
                    mode="BACKTEST",
                    source="HISTORICAL_HOLISTIC",
                    spot_snapshot=spot_snapshot,
                    option_chain=expiry_chain,
                    previous_chain=previous_chain,
                    apply_budget_constraint=BACKTEST_ENABLE_BUDGET,
                    requested_lots=NUMBER_OF_LOTS,
                    lot_size=LOT_SIZE,
                    max_capital=MAX_CAPITAL_PER_TRADE,
                    capture_signal_evaluation=False,
                    enable_shadow_logging=False,
                    global_market_snapshot=snap["global_market_snapshot"],
                    macro_event_state=snap["macro_event_state"],
                    target_profit_percent=target_profit_percent,
                    stop_loss_percent=stop_loss_percent,
                )

                if not signal_result.get("ok", False):
                    continue

                # 4. Capture signal evaluation row
                trade = signal_result.get("trade")
                if trade is None:
                    continue

                try:
                    sig_row = build_signal_evaluation_row(
                        signal_result,
                        notes=f"holistic_backtest|expiry={expiry}",
                        captured_at=f"{trade_date}T15:30:00+05:30",
                    )
                except (ValueError, KeyError):
                    continue

                # 5. Evaluate against realized path
                if evaluate_outcomes:
                    expiry_date = None
                    try:
                        expiry_date = pd.to_datetime(expiry).date()
                    except Exception:
                        pass
                    spot_path = build_realized_spot_path(trade_date, expiry_date)
                    if not spot_path.empty:
                        sig_row = evaluate_eod_outcomes(
                            sig_row, spot_path, available,
                        )

                day_signals.append(sig_row)

            all_signals.extend(day_signals)

            daily_summary.append({
                "date": str(trade_date),
                "status": "EVALUATED",
                "quality": snap["quality_score"],
                "expiries_evaluated": len(target_expiries),
                "signals_generated": len(day_signals),
                "trade_signals": sum(
                    1 for s in day_signals if s.get("trade_status") == "TRADE"
                ),
            })

            previous_chain = option_chain.copy()

        # Aggregate metrics
        metrics = _compute_signal_metrics(all_signals)
        elapsed = round(time.time() - t0, 2)

        return {
            "ok": True,
            "symbol": symbol,
            "date_range": {
                "start": str(available[0]) if available else None,
                "end": str(available[-1]) if available else None,
            },
            "total_days": total_days,
            "skipped_days": skipped_days,
            "evaluated_days": total_days - skipped_days,
            "total_signals": len(all_signals),
            "signals": all_signals,
            "daily_summary": daily_summary,
            "elapsed_seconds": elapsed,
            "data_source": "NSE_BHAV_COPY_PARQUET",
            "global_market_enabled": include_global_market,
            "macro_events_enabled": include_macro_events,
            "parameters": {
                "max_expiries": max_expiries,
                "target_profit_percent": target_profit_percent,
                "stop_loss_percent": stop_loss_percent,
                "min_quality_score": min_quality_score,
                "compute_iv": compute_iv,
                "evaluate_outcomes": evaluate_outcomes,
                "prediction_method": prediction_method,
            },
            "metrics": metrics,
        }

    finally:
        if _pred_factory is not None:
            _pred_factory._ACTIVE_PREDICTOR = _saved_predictor
            _pred_factory._METHOD_OVERRIDE = _saved_override
            log.info("Backtest predictor restored")


# ---------------------------------------------------------------
# Signal-centric aggregation
# ---------------------------------------------------------------

def _compute_signal_metrics(signals: list[dict]) -> dict:
    """Aggregate quality metrics across all signal evaluation records."""
    if not signals:
        return {
            "total_signals": 0,
            "trade_signals": 0,
            "no_trade_signals": 0,
        }

    trade_signals = [s for s in signals if s.get("trade_status") == "TRADE"]
    no_trade_signals = [s for s in signals if s.get("trade_status") != "TRADE"]

    def _avg(vals):
        valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return round(sum(valid) / len(valid), 4) if valid else None

    def _safe_get(s, key):
        v = s.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Directional accuracy for TRADE signals
    correct_keys = [f"correct_{d}d" for d in EOD_HORIZON_DAYS] + ["correct_at_expiry"]
    accuracy = {}
    for key in correct_keys:
        vals = [_safe_get(s, key) for s in trade_signals]
        accuracy[key] = _avg(vals)

    # Score distributions for TRADE signals
    score_keys = ["direction_score", "magnitude_score", "timing_score",
                  "tradeability_score", "composite_signal_score"]
    scores = {}
    for key in score_keys:
        vals = [_safe_get(s, key) for s in trade_signals]
        scores[key] = _avg(vals)

    # Target / stop-loss hit rates
    target_hits = sum(1 for s in trade_signals if s.get("target_hit") is True)
    sl_hits = sum(1 for s in trade_signals if s.get("stop_loss_hit") is True)

    # Trade strength distribution
    strength_vals = [_safe_get(s, "trade_strength") for s in trade_signals]
    strength_vals = [v for v in strength_vals if v is not None]

    # Direction distribution
    directions = {}
    for s in trade_signals:
        d = s.get("direction", "NONE")
        directions[d] = directions.get(d, 0) + 1

    # Regime distribution
    regimes = {}
    for s in trade_signals:
        r = s.get("signal_regime", "UNKNOWN")
        regimes[r] = regimes.get(r, 0) + 1

    return {
        "total_signals": len(signals),
        "trade_signals": len(trade_signals),
        "no_trade_signals": len(no_trade_signals),
        "trade_rate": round(len(trade_signals) / len(signals), 4) if signals else 0,
        "directional_accuracy": accuracy,
        "avg_scores": scores,
        "target_hit_rate": round(target_hits / len(trade_signals), 4) if trade_signals else 0,
        "stop_loss_hit_rate": round(sl_hits / len(trade_signals), 4) if trade_signals else 0,
        "avg_trade_strength": _avg(strength_vals),
        "direction_distribution": directions,
        "regime_distribution": regimes,
        "avg_eod_mfe_bps": _avg([_safe_get(s, "eod_mfe_bps") for s in trade_signals]),
        "avg_eod_mae_bps": _avg([_safe_get(s, "eod_mae_bps") for s in trade_signals]),
    }


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _day_stat(trade_date, status, quality) -> dict:
    return {
        "date": str(trade_date),
        "status": status,
        "quality": quality,
        "expiries_evaluated": 0,
        "signals_generated": 0,
        "trade_signals": 0,
    }


def _empty_result(symbol: str, message: str) -> dict:
    return {
        "ok": False,
        "symbol": symbol,
        "total_days": 0,
        "total_signals": 0,
        "signals": [],
        "daily_summary": [],
        "message": message,
        "metrics": _compute_signal_metrics([]),
    }
