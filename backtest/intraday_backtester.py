"""
Historical bar-based backtester.

The default historical builder creates one synthetic snapshot per trading
day, so persistence and holding periods are measured in bars unless a
finer-grained dataset is supplied.
"""

from config.settings import (
    BACKTEST_SIGNAL_PERSISTENCE,
    BACKTEST_MAX_HOLD_BARS,
    BACKTEST_ENABLE_BUDGET,
    BACKTEST_STARTING_CAPITAL,
    TARGET_PROFIT_PERCENT,
    STOP_LOSS_PERCENT,
)
from data.historical_option_chain import load_option_chain
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry
from engine.signal_engine import generate_trade
from backtest.pnl_engine import calculate_trade_pnl
from backtest.performance_metrics import compute_performance_metrics


def _finalize_open_trade(open_trade, exit_snapshot, exit_ts):
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


def run_intraday_backtest(
    symbol: str,
    years: int = 1,
    signal_persistence: int = BACKTEST_SIGNAL_PERSISTENCE,
    max_hold_bars: int = BACKTEST_MAX_HOLD_BARS,
    target_profit_percent: float = TARGET_PROFIT_PERCENT,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
):
    historical_df = load_option_chain(symbol=symbol, years=years)

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

        spot = float(signal_chain["spot"].iloc[0]) if "spot" in signal_chain.columns else float(signal_chain["strikePrice"].median())

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

        trade = generate_trade(
            symbol=symbol,
            spot=spot,
            option_chain=signal_chain,
            previous_chain=previous_chain,
            apply_budget_constraint=BACKTEST_ENABLE_BUDGET,
            backtest_mode=True,
            target_profit_percent=target_profit_percent,
            stop_loss_percent=stop_loss_percent,
        )

        if trade is None:
            previous_chain = signal_chain.copy()
            continue

        trade["selected_expiry"] = signal_expiry

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
    return run_intraday_backtest(symbol, years)
