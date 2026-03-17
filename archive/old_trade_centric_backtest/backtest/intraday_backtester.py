"""
Module: intraday_backtester.py

Purpose:
    Implement intraday backtester logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""

from app.engine_runner import run_preloaded_engine_snapshot
from config.settings import (
    BACKTEST_SIGNAL_PERSISTENCE,
    BACKTEST_MAX_HOLD_BARS,
    BACKTEST_ENABLE_BUDGET,
    BACKTEST_STARTING_CAPITAL,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
    GLOBAL_MARKET_DATA_ENABLED,
    TARGET_PROFIT_PERCENT,
    STOP_LOSS_PERCENT,
)
from data.historical_option_chain import load_option_chain
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry
from backtest.pnl_engine import calculate_trade_pnl
from backtest.performance_metrics import compute_performance_metrics


def _finalize_open_trade(open_trade, exit_snapshot, exit_ts):
    """
    Purpose:
        Process finalize open trade for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        open_trade (Any): Input associated with open trade.
        exit_snapshot (Any): Input associated with exit snapshot.
        exit_ts (Any): Timestamp associated with exit.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    pnl_result = calculate_trade_pnl(open_trade["trade"], exit_snapshot)
    trade = open_trade["trade"]

    return {
        "entry_timestamp": open_trade["entry_timestamp"],
        "exit_timestamp": exit_ts,
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "strike": trade.get("strike"),
        "entry_price": trade.get("entry_price"),
        "target": trade.get("target"),
        "stop_loss": trade.get("stop_loss"),
        "trade_strength": trade.get("trade_strength"),
        "signal_quality": trade.get("signal_quality"),
        "lot_size": trade.get("lot_size"),
        "number_of_lots": trade.get("optimized_lots", trade.get("number_of_lots", 1)),
        "gross_pnl": pnl_result.get("gross_pnl", 0),
        "pnl": pnl_result.get("pnl", 0),
        "charges": pnl_result.get("charges", 0),
        "exit_reason": pnl_result.get("exit_reason"),
        "exit_price": pnl_result.get("exit_price"),
        "bars_held": open_trade["bars_held"]
    }


def _build_backtest_spot_snapshot(symbol, ts, snapshot_chain, previous_chain=None):
    """
    Purpose:
        Synthesize the spot snapshot contract expected by the shared runtime
        orchestration path.

    Context:
        Internal backtest helper that adapts historical option-chain bars into
        the same spot-summary shape used by the live engine runner.

    Inputs:
        symbol (Any): Underlying symbol or index identifier.
        ts (Any): Timestamp associated with the current historical bar.
        snapshot_chain (Any): Current option-chain snapshot.
        previous_chain (Any): Previous snapshot used to approximate
            `prev_close` when available.

    Returns:
        dict: Spot snapshot shaped for `run_preloaded_engine_snapshot`.

    Notes:
        Daily synthetic backtest bars do not carry true intraday OHLC data, so
        the current spot is reused as a neutral proxy for open, high, and low.
        This preserves runner parity while making the approximation explicit.
    """

    spot = (
        float(snapshot_chain["spot"].iloc[0])
        if snapshot_chain is not None and not snapshot_chain.empty and "spot" in snapshot_chain.columns
        else float(snapshot_chain["strikePrice"].median())
    )
    previous_spot = None
    if previous_chain is not None and not previous_chain.empty:
        if "spot" in previous_chain.columns:
            previous_spot = float(previous_chain["spot"].iloc[0])
        else:
            previous_spot = float(previous_chain["strikePrice"].median())

    timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return {
        "symbol": str(symbol or "").upper().strip(),
        "spot": spot,
        "day_open": spot,
        "day_high": spot,
        "day_low": spot,
        "prev_close": previous_spot if previous_spot is not None else spot,
        "timestamp": timestamp,
        "lookback_avg_range_pct": None,
    }


def _neutral_backtest_snapshot(symbol, ts):
    """Return a neutral global-market snapshot when cross-asset data is unavailable."""
    timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return {
        "symbol": str(symbol or "").upper().strip(),
        "provider": "BACKTEST_NEUTRAL",
        "as_of": timestamp,
        "data_available": False,
        "neutral_fallback": True,
        "issues": [],
        "warnings": ["historical_global_market_snapshot_unavailable"],
        "stale": False,
        "lookback_days": None,
        "market_inputs": {},
    }


def _backtest_global_market_snapshot(symbol, ts, *, _cache=None):
    """
    Build a global-market snapshot for backtest evaluation.

    Attempts to fetch real cross-asset data via yfinance (when enabled),
    caching per date to avoid redundant API calls.  Falls back to a
    neutral snapshot when market data is unavailable.
    """
    date_key = ts.date() if hasattr(ts, "date") else str(ts)[:10]
    cache_key = (str(symbol or "").upper().strip(), date_key)

    if _cache is not None and cache_key in _cache:
        return _cache[cache_key]

    if GLOBAL_MARKET_DATA_ENABLED:
        try:
            from data.global_market_snapshot import build_global_market_snapshot

            snapshot = build_global_market_snapshot(symbol, as_of=ts)
        except Exception:
            snapshot = _neutral_backtest_snapshot(symbol, ts)
    else:
        snapshot = _neutral_backtest_snapshot(symbol, ts)

    if _cache is not None:
        _cache[cache_key] = snapshot

    return snapshot


def run_intraday_backtest(
    symbol: str,
    years: int = 1,
    signal_persistence: int = BACKTEST_SIGNAL_PERSISTENCE,
    max_hold_bars: int = BACKTEST_MAX_HOLD_BARS,
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
    data_source=None,
):
    """
    Purpose:
        Process run intraday backtest for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        years (int): Input associated with years.
        signal_persistence (int): Input associated with signal persistence.
        max_hold_bars (int): Input associated with max hold bars.
        target_profit_percent (float): Input associated with target profit percent.
        stop_loss_percent (float): Input associated with stop loss percent.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    historical_df = load_option_chain(symbol=symbol, years=years, data_source=data_source)

    if historical_df is None or historical_df.empty:
        return {
            "symbol": symbol,
            "years": years,
            "total_trades": 0,
            "message": "No historical data available",
            "trades": []
        }

    historical_df = historical_df.sort_values("timestamp").reset_index(drop=True)
    grouped = list(historical_df.groupby("timestamp"))
    total_snapshots = len(grouped)

    trade_log = []
    previous_chain = None
    open_trade = None
    global_market_cache = {}

    last_direction = None
    direction_count = 0

    for idx, (ts, snapshot) in enumerate(grouped):
        snapshot_chain = snapshot.copy()
        if snapshot_chain.empty:
            continue

        signal_expiry = resolve_selected_expiry(snapshot_chain)
        signal_chain = filter_option_chain_by_expiry(snapshot_chain, signal_expiry)
        if signal_chain is None or signal_chain.empty:
            continue

        spot_snapshot = _build_backtest_spot_snapshot(
            symbol,
            ts,
            signal_chain,
            previous_chain=previous_chain,
        )

        if open_trade is not None:
            open_trade["bars_held"] += 1
            exit_chain = snapshot_chain.copy()
            selected_expiry = open_trade["trade"].get("selected_expiry")
            if selected_expiry:
                filtered_exit_chain = filter_option_chain_by_expiry(exit_chain, selected_expiry)
                if filtered_exit_chain is not None and not filtered_exit_chain.empty:
                    exit_chain = filtered_exit_chain

            should_exit = False
            pnl_preview = calculate_trade_pnl(open_trade["trade"], exit_chain)
            exit_reason = pnl_preview.get("exit_reason")

            if exit_reason in ["TARGET_HIT", "STOP_LOSS_HIT"]:
                should_exit = True
            elif open_trade["bars_held"] >= max_hold_bars:
                should_exit = True
            elif idx == total_snapshots - 1:
                should_exit = True

            if should_exit:
                trade_log.append(_finalize_open_trade(open_trade, exit_chain, ts))
                open_trade = None

        if open_trade is not None:
            previous_chain = signal_chain.copy()
            continue

        signal_result = run_preloaded_engine_snapshot(
            symbol=symbol,
            mode="BACKTEST",
            source="HISTORICAL",
            spot_snapshot=spot_snapshot,
            option_chain=signal_chain,
            previous_chain=previous_chain,
            apply_budget_constraint=BACKTEST_ENABLE_BUDGET,
            requested_lots=NUMBER_OF_LOTS,
            lot_size=LOT_SIZE,
            max_capital=MAX_CAPITAL_PER_TRADE,
            capture_signal_evaluation=False,
            enable_shadow_logging=False,
            global_market_snapshot=_backtest_global_market_snapshot(symbol, ts, _cache=global_market_cache),
            target_profit_percent=target_profit_percent,
            stop_loss_percent=stop_loss_percent,
        )
        if not signal_result.get("ok", False):
            raise ValueError(signal_result.get("error") or "Backtest snapshot evaluation failed")

        trade = signal_result.get("execution_trade") or signal_result.get("trade")

        if trade is None:
            previous_chain = signal_chain.copy()
            continue

        direction = trade.get("direction")

        if direction == last_direction and direction is not None:
            direction_count += 1
        else:
            direction_count = 1 if direction is not None else 0
            last_direction = direction

        if (
            trade.get("trade_status") == "TRADE"
            and direction is not None
            and direction_count >= signal_persistence
        ):
            open_trade = {
                "entry_timestamp": ts,
                "trade": trade,
                "bars_held": 0
            }

        previous_chain = signal_chain.copy()

    metrics = compute_performance_metrics(
        trade_log,
        starting_capital=BACKTEST_STARTING_CAPITAL
    )

    result = {
        "symbol": symbol,
        "years": years,
        "bar_interval": "1d_default_synthetic",
        "total_snapshots": total_snapshots,
        "total_rows": int(len(historical_df)),
        "total_trades": len(trade_log),
        "trades": trade_log,
        **metrics
    }

    if len(trade_log) == 0:
        result["message"] = "Historical data loaded, but strategy generated no tradable signals"

    return result


def intraday_backtester(symbol: str, years: int = 1):
    """
    Purpose:
        Process intraday backtester for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        years (int): Input associated with years.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return run_intraday_backtest(symbol, years)
