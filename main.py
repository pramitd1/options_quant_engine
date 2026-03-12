import time

from config.settings import (
    DEFAULT_SYMBOL,
    REFRESH_INTERVAL,
    LOT_SIZE,
    NUMBER_OF_LOTS,
    MAX_CAPITAL_PER_TRADE
)

from data.spot_downloader import get_spot_price
from data.data_source_router import DataSourceRouter

from engine.trading_engine import generate_trade

from analytics.gamma_exposure import calculate_gamma_exposure
from analytics.gamma_flip import gamma_flip_level
from analytics.dealer_inventory import dealer_inventory_position
from analytics.volatility_regime import detect_volatility_regime
from analytics.dealer_gamma_path import simulate_gamma_path, detect_gamma_squeeze
from analytics.options_flow_imbalance import flow_signal
from analytics.liquidity_heatmap import strongest_liquidity_levels
from analytics.liquidity_void import detect_liquidity_voids, liquidity_void_signal
from analytics.smart_money_flow import smart_money_signal
from analytics.liquidity_vacuum import detect_liquidity_vacuum, vacuum_direction
from analytics.market_gamma_map import (
    calculate_market_gamma,
    market_gamma_regime,
    largest_gamma_strikes
)
from analytics.gamma_walls import classify_walls
from analytics.dealer_hedging_flow import dealer_hedging_flow
from analytics.dealer_hedging_simulator import (
    simulate_dealer_hedging,
    hedging_bias
)
from analytics.volatility_surface import atm_vol, vol_regime
from analytics.intraday_gamma_shift import gamma_shift_signal

from visualization.dealer_dashboard import print_dealer_dashboard


TRADER_VIEW_KEYS = [
    "symbol",
    "spot",
    "direction",
    "direction_source",
    "strike",
    "option_type",
    "entry_price",
    "target",
    "stop_loss",
    "trade_strength",
    "signal_quality",
    "trade_status",
    "budget_constraint_applied",
    "lot_size",
    "requested_lots",
    "number_of_lots",
    "optimized_lots",
    "capital_per_lot",
    "capital_required",
    "max_affordable_lots",
    "budget_ok",
    "hybrid_move_probability",
    "large_move_probability",
    "ml_move_probability",
]


def choose_data_source():
    print("\nChoose data source:")
    print("1. Zerodha")
    print("2. NSE")

    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        return "ZERODHA"

    if choice == "2":
        return "NSE"

    print("Invalid choice. Defaulting to NSE.")
    return "NSE"


def choose_budget_mode():
    """
    Ask whether budget should be applied in decision making.
    """
    print("\nApply budget constraint in trade decision?")
    print("1. Yes")
    print("2. No")

    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        return True

    if choice == "2":
        return False

    print("Invalid choice. Defaulting to No.")
    return False


def get_budget_inputs(apply_budget_constraint: bool):
    """
    Ask for lot/capital inputs only if budget mode is enabled.
    """
    if not apply_budget_constraint:
        return LOT_SIZE, NUMBER_OF_LOTS, MAX_CAPITAL_PER_TRADE

    lot_size_input = input(f"Enter lot size [{LOT_SIZE}]: ").strip()
    lots_input = input(f"Enter number of lots [{NUMBER_OF_LOTS}]: ").strip()
    capital_input = input(
        f"Enter max capital per trade [{MAX_CAPITAL_PER_TRADE}]: "
    ).strip()

    lot_size = int(lot_size_input) if lot_size_input else LOT_SIZE
    requested_lots = int(lots_input) if lots_input else NUMBER_OF_LOTS
    max_capital = float(capital_input) if capital_input else MAX_CAPITAL_PER_TRADE

    return lot_size, requested_lots, max_capital


def safe_call(func, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


def print_trader_view(trade):
    print("\nTRADER VIEW")
    print("---------------------------")

    for key in TRADER_VIEW_KEYS:
        if key in trade:
            print(f"{key}: {trade.get(key)}")


def build_non_overlapping_trade_output(trade):
    """
    Remove fields already shown in Trader View from the later raw output block.
    """
    filtered = {}

    for key, value in trade.items():
        if key not in TRADER_VIEW_KEYS:
            filtered[key] = value

    return filtered


def main():
    symbol = input(
        "Enter symbol (NIFTY / BANKNIFTY / FINNIFTY / STOCK): "
    ).strip().upper() or DEFAULT_SYMBOL

    source = choose_data_source()
    apply_budget_constraint = choose_budget_mode()
    lot_size, requested_lots, max_capital = get_budget_inputs(apply_budget_constraint)

    print("\nRunning Quant Engine for:", symbol)
    print("Data Source:", source)
    print("Budget Constraint Applied:", apply_budget_constraint)

    data_router = DataSourceRouter(source)
    previous_chain = None

    try:
        while True:
            try:
                spot = get_spot_price(symbol)
                print("\nSpot Price:", spot)

                option_chain = data_router.get_option_chain(symbol)

                if option_chain is None or option_chain.empty:
                    print("Option chain unavailable")
                    time.sleep(REFRESH_INTERVAL)
                    continue

                print("Option chain rows:", len(option_chain))

                gamma = safe_call(
                    calculate_gamma_exposure,
                    option_chain,
                    spot,
                    default=None
                )

                flip = safe_call(
                    gamma_flip_level,
                    option_chain,
                    default=None
                )

                dealer_pos = safe_call(
                    dealer_inventory_position,
                    option_chain,
                    default=None
                )

                vol_regime_value = safe_call(
                    detect_volatility_regime,
                    option_chain,
                    default=None
                )

                gamma_path = safe_call(
                    simulate_gamma_path,
                    option_chain,
                    spot,
                    default=([], [])
                )

                if isinstance(gamma_path, tuple) and len(gamma_path) == 2:
                    prices, gamma_curve = gamma_path
                else:
                    prices, gamma_curve = [], []

                gamma_event = safe_call(
                    detect_gamma_squeeze,
                    prices,
                    gamma_curve,
                    default=None
                )

                flow = safe_call(
                    flow_signal,
                    option_chain,
                    default=None
                )

                smart_flow = safe_call(
                    smart_money_signal,
                    option_chain,
                    default=None
                )

                liquidity_levels = safe_call(
                    strongest_liquidity_levels,
                    option_chain,
                    default=[]
                )

                voids = safe_call(
                    detect_liquidity_voids,
                    option_chain,
                    default=[]
                )

                void_sig = safe_call(
                    liquidity_void_signal,
                    spot,
                    voids,
                    default=None
                )

                vacuum_zones = safe_call(
                    detect_liquidity_vacuum,
                    option_chain,
                    default=[]
                )

                vacuum_state = safe_call(
                    vacuum_direction,
                    spot,
                    vacuum_zones,
                    default=None
                )

                market_gamma = safe_call(
                    calculate_market_gamma,
                    option_chain,
                    default=None
                )

                gamma_regime = safe_call(
                    market_gamma_regime,
                    market_gamma,
                    default=None
                )

                gamma_clusters = safe_call(
                    largest_gamma_strikes,
                    market_gamma,
                    default=None
                )

                walls = safe_call(
                    classify_walls,
                    option_chain,
                    default={}
                ) or {}

                support_wall = walls.get("support_wall") if isinstance(walls, dict) else None
                resistance_wall = walls.get("resistance_wall") if isinstance(walls, dict) else None

                hedging_flow = safe_call(
                    dealer_hedging_flow,
                    option_chain,
                    default=None
                )

                hedging_sim = safe_call(
                    simulate_dealer_hedging,
                    option_chain,
                    default={}
                )

                hedging_bias_value = safe_call(
                    hedging_bias,
                    hedging_sim,
                    default=None
                )

                atm_iv_value = safe_call(
                    atm_vol,
                    option_chain,
                    spot,
                    default=None
                )

                vol_surface_regime = None
                if atm_iv_value is not None:
                    vol_surface_regime = safe_call(
                        vol_regime,
                        atm_iv_value,
                        default=None
                    )

                intraday_gamma_state = None
                if previous_chain is not None:
                    intraday_gamma_state = safe_call(
                        gamma_shift_signal,
                        previous_chain,
                        option_chain,
                        spot,
                        default=None
                    )

                if flip is None:
                    spot_vs_flip = "UNKNOWN"
                elif abs(spot - flip) <= 25:
                    spot_vs_flip = "AT_FLIP"
                elif spot > flip:
                    spot_vs_flip = "ABOVE_FLIP"
                else:
                    spot_vs_flip = "BELOW_FLIP"

                dashboard_summary = {
                    "spot": round(spot, 2),
                    "gamma_exposure": round(gamma, 2) if gamma is not None else None,
                    "market_gamma": market_gamma,
                    "gamma_flip": flip,
                    "spot_vs_flip": spot_vs_flip,
                    "gamma_regime": gamma_regime,
                    "gamma_clusters": gamma_clusters,
                    "dealer_position": dealer_pos,
                    "dealer_hedging_flow": hedging_flow,
                    "dealer_hedging_bias": hedging_bias_value,
                    "intraday_gamma_state": intraday_gamma_state,
                    "volatility_regime": vol_regime_value,
                    "vol_surface_regime": vol_surface_regime,
                    "atm_iv": atm_iv_value,
                    "flow_signal": flow,
                    "smart_money_flow": smart_flow,
                    "final_flow_signal": None,
                    "gamma_event": gamma_event,
                    "support_wall": support_wall,
                    "resistance_wall": resistance_wall,
                    "liquidity_levels": liquidity_levels,
                    "liquidity_voids": voids,
                    "liquidity_void_signal": void_sig,
                    "liquidity_vacuum_zones": vacuum_zones,
                    "liquidity_vacuum_state": vacuum_state
                }

                trade = generate_trade(
                    symbol=symbol,
                    spot=spot,
                    option_chain=option_chain,
                    previous_chain=previous_chain,
                    apply_budget_constraint=apply_budget_constraint,
                    requested_lots=requested_lots,
                    lot_size=lot_size,
                    max_capital=max_capital
                )

                if trade:
                    print_trader_view(trade)

                    dashboard_for_print = dict(dashboard_summary)

                    for key in [
                        "market_gamma",
                        "gamma_regime",
                        "gamma_clusters",
                        "dealer_hedging_flow",
                        "dealer_hedging_bias",
                        "intraday_gamma_state",
                        "vol_surface_regime",
                        "atm_iv",
                        "final_flow_signal",
                        "support_wall",
                        "resistance_wall",
                        "liquidity_vacuum_zones",
                        "liquidity_vacuum_state",
                        "dealer_liquidity_map",
                        "large_move_probability",
                        "ml_move_probability",
                        "hybrid_move_probability",
                        "scoring_breakdown",
                    ]:
                        if key in trade:
                            dashboard_for_print[key] = trade.get(key)

                    print_dealer_dashboard(dashboard_for_print)

                    print("\nQUANT TRADE SIGNAL")
                    print("---------------------------")
                    non_overlapping_output = build_non_overlapping_trade_output(trade)
                    for key, value in non_overlapping_output.items():
                        print(f"{key}: {value}")
                else:
                    print("\nNo trade signal")
                    print_dealer_dashboard(dashboard_summary)

                previous_chain = option_chain.copy()

            except Exception as e:
                print("\nEngine error:", e)

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\nEngine stopped by user")

    finally:
        data_router.close()


if __name__ == "__main__":
    main()