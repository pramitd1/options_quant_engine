from config.settings import (
    BACKTEST_ENTRY_SLIPPAGE_BPS,
    BACKTEST_EXIT_SLIPPAGE_BPS,
    BACKTEST_SPREAD_BPS,
    BACKTEST_COMMISSION_PER_ORDER,
)


def _bps_adjust(price: float, bps: float, side: str) -> float:
    mult = bps / 10000.0
    if side == "BUY":
        return price * (1 + mult)
    if side == "SELL":
        return price * (1 - mult)
    return price


def _find_option_row(option_chain, strike, option_type):
    rows = option_chain[
        (option_chain["strikePrice"] == strike) &
        (option_chain["OPTION_TYP"] == option_type)
    ]
    return None if rows.empty else rows.iloc[0]


def calculate_trade_pnl(trade: dict, exit_snapshot):
    if not trade:
        return {
            "pnl": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "exit_price": None,
            "exit_reason": "NO_TRADE",
            "charges": 0.0
        }

    strike = trade.get("strike")
    option_type = trade.get("option_type")
    raw_entry_price = float(trade.get("entry_price", 0))
    target = float(trade.get("target", 0))
    stop_loss = float(trade.get("stop_loss", 0))
    lot_size = int(trade.get("lot_size", 1))
    number_of_lots = int(trade.get("optimized_lots", trade.get("number_of_lots", 1)))

    row = _find_option_row(exit_snapshot, strike, option_type)
    if row is None:
        return {
            "pnl": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "exit_price": None,
            "exit_reason": "OPTION_NOT_FOUND",
            "charges": 0.0
        }

    raw_current_price = float(row.get("lastPrice", 0))

    entry_price = _bps_adjust(
        raw_entry_price,
        BACKTEST_ENTRY_SLIPPAGE_BPS + BACKTEST_SPREAD_BPS / 2,
        "BUY"
    )
    tradable_exit_price = _bps_adjust(
        raw_current_price,
        BACKTEST_EXIT_SLIPPAGE_BPS + BACKTEST_SPREAD_BPS / 2,
        "SELL"
    )

    if tradable_exit_price >= target:
        exit_price = target
        exit_reason = "TARGET_HIT"
    elif tradable_exit_price <= stop_loss:
        exit_price = stop_loss
        exit_reason = "STOP_LOSS_HIT"
    else:
        exit_price = tradable_exit_price
        exit_reason = "TIME_EXIT"

    gross_pnl = (exit_price - entry_price) * lot_size * number_of_lots
    charges = 2 * BACKTEST_COMMISSION_PER_ORDER
    net_pnl = gross_pnl - charges

    return {
        "pnl": round(net_pnl, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "charges": round(charges, 2)
    }


def pnl_engine(trade: dict, exit_snapshot):
    return calculate_trade_pnl(trade, exit_snapshot)