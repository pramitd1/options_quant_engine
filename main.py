import os
import time
import warnings
from getpass import getpass

import pandas as pd
from urllib3.exceptions import NotOpenSSLWarning


warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

from config.settings import (
    DEFAULT_SYMBOL,
    DEFAULT_DATA_SOURCE,
    REFRESH_INTERVAL,
    NSE_REFRESH_INTERVAL,
    ICICI_REFRESH_INTERVAL,
    LOT_SIZE,
    NUMBER_OF_LOTS,
    MAX_CAPITAL_PER_TRADE,
    DATA_SOURCE_OPTIONS,
)

from data.spot_downloader import get_spot_snapshot, save_spot_snapshot
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
    "selected_expiry",
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
    "data_quality_score",
    "data_quality_status",
]


def _refresh_interval_for_source(source: str) -> int:
    source = source.upper().strip()
    if source == "NSE":
        return NSE_REFRESH_INTERVAL
    if source == "ICICI":
        return ICICI_REFRESH_INTERVAL
    return REFRESH_INTERVAL


def choose_data_source():
    print("\nChoose data source:")
    for idx, source in enumerate(DATA_SOURCE_OPTIONS, start=1):
        print(f"{idx}. {source}")

    choice = input(f"Enter choice (1-{len(DATA_SOURCE_OPTIONS)}) [{DEFAULT_DATA_SOURCE}]: ").strip()

    if not choice:
        return DEFAULT_DATA_SOURCE

    try:
        index = int(choice) - 1
        if 0 <= index < len(DATA_SOURCE_OPTIONS):
            return DATA_SOURCE_OPTIONS[index]
    except Exception:
        pass

    print(f"Invalid choice. Defaulting to {DEFAULT_DATA_SOURCE}.")
    return DEFAULT_DATA_SOURCE


def choose_budget_mode():
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


def _prompt_runtime_secret(env_name: str, label: str, secret: bool = False):
    existing = os.getenv(env_name, "").strip()
    prompt = f"{label} [{ 'saved' if existing else 'required' }]: "

    value = getpass(prompt) if secret else input(prompt).strip()
    if value:
        os.environ[env_name] = value
        return

    if existing:
        return

    print(f"{env_name} is still not set.")


def prompt_provider_credentials(source: str):
    source = source.upper().strip()

    if source == "ZERODHA":
        print("\nEnter Zerodha credentials. Leave blank to use values already loaded from .env or shell.")
        _prompt_runtime_secret("ZERODHA_API_KEY", "Zerodha API key")
        _prompt_runtime_secret("ZERODHA_API_SECRET", "Zerodha API secret", secret=True)
        _prompt_runtime_secret("ZERODHA_ACCESS_TOKEN", "Zerodha access token", secret=True)

    if source == "ICICI":
        print("\nEnter ICICI Breeze credentials. Leave blank to use values already loaded from .env or shell.")
        _prompt_runtime_secret("ICICI_BREEZE_API_KEY", "ICICI Breeze API key")
        _prompt_runtime_secret("ICICI_BREEZE_SECRET_KEY", "ICICI Breeze secret key", secret=True)
        _prompt_runtime_secret("ICICI_BREEZE_SESSION_TOKEN", "ICICI Breeze session token", secret=True)


def get_budget_inputs(apply_budget_constraint: bool):
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
            print(f"{key:26}: {trade.get(key)}")


def build_non_overlapping_trade_output(trade):
    filtered = {}

    for key, value in trade.items():
        if key not in TRADER_VIEW_KEYS:
            filtered[key] = value

    return filtered


def validate_option_chain(option_chain):
    issues = []
    warnings = []

    if option_chain is None:
        issues.append("option_chain_none")
        return {
            "is_valid": False,
            "issues": issues,
            "warnings": warnings,
            "row_count": 0,
            "ce_rows": 0,
            "pe_rows": 0,
            "priced_rows": 0,
        }

    if option_chain.empty:
        issues.append("option_chain_empty")
        return {
            "is_valid": False,
            "issues": issues,
            "warnings": warnings,
            "row_count": 0,
            "ce_rows": 0,
            "pe_rows": 0,
            "priced_rows": 0,
        }

    required_cols = ["strikePrice", "OPTION_TYP", "lastPrice"]
    missing_cols = [col for col in required_cols if col not in option_chain.columns]
    if missing_cols:
        issues.append(f"missing_columns:{','.join(missing_cols)}")

    row_count = len(option_chain)
    ce_rows = 0
    pe_rows = 0
    priced_rows = 0
    iv_rows = 0
    selected_expiry = None

    try:
        ce_rows = int((option_chain["OPTION_TYP"] == "CE").sum())
        pe_rows = int((option_chain["OPTION_TYP"] == "PE").sum())
    except Exception:
        warnings.append("option_type_count_failed")

    try:
        priced_rows = int((pd.to_numeric(option_chain["lastPrice"], errors="coerce") > 0).sum())
    except Exception:
        warnings.append("priced_row_count_failed")

    try:
        iv_rows = int((pd.to_numeric(option_chain.get("impliedVolatility", option_chain.get("IV")), errors="coerce") > 0).sum())
    except Exception:
        warnings.append("iv_row_count_failed")

    try:
        if "EXPIRY_DT" in option_chain.columns:
            expiries = [
                str(value) for value in option_chain["EXPIRY_DT"].dropna().astype(str).unique().tolist()
                if str(value).strip()
            ]
            if len(expiries) == 1:
                selected_expiry = expiries[0]
            elif len(expiries) > 1:
                selected_expiry = f"MULTI({len(expiries)})"
    except Exception:
        warnings.append("expiry_summary_failed")

    if row_count < 20:
        issues.append(f"too_few_rows:{row_count}")

    if ce_rows == 0:
        issues.append("no_ce_rows")

    if pe_rows == 0:
        issues.append("no_pe_rows")

    if priced_rows == 0:
        issues.append("no_priced_rows")

    if priced_rows > 0 and priced_rows < max(10, int(0.2 * row_count)):
        warnings.append(f"low_priced_row_ratio:{priced_rows}/{row_count}")

    if row_count > 0 and iv_rows == 0:
        warnings.append("no_positive_iv_rows")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "row_count": row_count,
        "ce_rows": ce_rows,
        "pe_rows": pe_rows,
        "priced_rows": priced_rows,
        "iv_rows": iv_rows,
        "selected_expiry": selected_expiry,
    }


def print_validation_block(title, validation):
    print(f"\n{title}")
    print("---------------------------")
    for key, value in validation.items():
        print(f"{key:26}: {value}")


def print_key_value_block(title, values):
    print(f"\n{title}")
    print("---------------------------")
    for key, value in values.items():
        print(f"{key:26}: {value}")


def _format_output_value(value, max_items=8):
    if isinstance(value, float):
        return round(value, 2)

    if isinstance(value, list):
        if len(value) <= max_items:
            return value
        return f"{value[:max_items]} ... (+{len(value) - max_items} more)"

    if isinstance(value, dict):
        items = list(value.items())
        preview = {k: v for k, v in items[:max_items]}
        if len(items) <= max_items:
            return preview
        return f"{preview} ... (+{len(items) - max_items} more)"

    return value


def print_signal_summary(trade):
    compact = {
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "direction_source": trade.get("direction_source"),
        "selected_expiry": trade.get("selected_expiry"),
        "strike": trade.get("strike"),
        "option_type": trade.get("option_type"),
        "entry_price": trade.get("entry_price"),
        "target": trade.get("target"),
        "stop_loss": trade.get("stop_loss"),
        "trade_strength": trade.get("trade_strength"),
        "signal_quality": trade.get("signal_quality"),
        "hybrid_move_probability": trade.get("hybrid_move_probability"),
        "flow_signal": trade.get("final_flow_signal"),
        "gamma_regime": trade.get("gamma_regime"),
        "spot_vs_flip": trade.get("spot_vs_flip"),
        "dealer_position": trade.get("dealer_position"),
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
        "atm_iv": trade.get("atm_iv"),
        "vol_surface_regime": trade.get("vol_surface_regime"),
        "capital_required": trade.get("capital_required"),
        "data_quality_score": trade.get("data_quality_score"),
        "data_quality_status": trade.get("data_quality_status"),
        "message": trade.get("message"),
    }
    print_key_value_block("QUANT TRADE SIGNAL", compact)


def print_ranked_candidates_table(candidates, expiry=None):
    if not candidates:
        return

    title = "RANKED STRIKES"
    if expiry:
        title = f"RANKED STRIKES ({expiry})"

    print(f"\n{title}")
    print("---------------------------")
    print(f"{'strike':>8} {'ltp':>10} {'iv':>8} {'volume':>12} {'oi':>12} {'score':>8}")

    for row in candidates[:5]:
        print(
            f"{str(row.get('strike')):>8} "
            f"{str(_format_output_value(row.get('last_price'))):>10} "
            f"{str(_format_output_value(row.get('iv'))):>8} "
            f"{str(row.get('volume')):>12} "
            f"{str(row.get('open_interest')):>12} "
            f"{str(row.get('score')):>8}"
        )


def print_diagnostics(trade):
    diagnostic_keys = [
        "gamma_clusters",
        "liquidity_levels",
        "liquidity_voids",
        "liquidity_vacuum_zones",
        "dealer_liquidity_map",
        "move_probability_components",
        "spot_validation",
        "option_chain_validation",
        "data_quality_reasons",
        "confirmation_status",
        "confirmation_veto",
        "confirmation_reasons",
        "confirmation_breakdown",
        "scoring_breakdown",
    ]

    diagnostics = {}
    for key in diagnostic_keys:
        if key in trade:
            diagnostics[key] = _format_output_value(trade.get(key))

    if diagnostics:
        print_key_value_block("DIAGNOSTICS", diagnostics)

    if trade.get("ranked_strike_candidates"):
        print_ranked_candidates_table(
            trade.get("ranked_strike_candidates"),
            expiry=trade.get("selected_expiry"),
        )


def main():
    symbol = input(
        "Enter symbol (NIFTY / BANKNIFTY / FINNIFTY / STOCK): "
    ).strip().upper() or DEFAULT_SYMBOL

    source = choose_data_source()
    prompt_provider_credentials(source)
    apply_budget_constraint = choose_budget_mode()
    lot_size, requested_lots, max_capital = get_budget_inputs(apply_budget_constraint)
    refresh_interval = _refresh_interval_for_source(source)

    print("\nRunning Quant Engine for:", symbol)
    print("Data Source:", source)
    print("Budget Constraint Applied:", apply_budget_constraint)

    try:
        data_router = DataSourceRouter(source)
    except Exception as e:
        print(f"\nFailed to initialize data source '{source}': {e}")
        print("Check credentials, session tokens, installed packages, and expiry configuration.")
        return

    previous_chain = None
    saved_one_spot_snapshot = False

    try:
        while True:
            try:
                spot_snapshot = get_spot_snapshot(symbol)
                spot_validation = spot_snapshot.get("validation", {})

                if not saved_one_spot_snapshot:
                    try:
                        saved_path = save_spot_snapshot(spot_snapshot)
                        print(f"\nSaved one live spot snapshot to: {saved_path}")
                    except Exception as save_err:
                        print(f"\nCould not save spot snapshot: {save_err}")
                    saved_one_spot_snapshot = True

                print_validation_block("SPOT VALIDATION", spot_validation)

                if not spot_validation.get("is_valid", False):
                    print("\nSpot snapshot invalid. Skipping this cycle.")
                    time.sleep(refresh_interval)
                    continue

                spot = float(spot_snapshot["spot"])
                day_open = spot_snapshot.get("day_open")
                day_high = spot_snapshot.get("day_high")
                day_low = spot_snapshot.get("day_low")
                prev_close = spot_snapshot.get("prev_close")
                spot_timestamp = spot_snapshot.get("timestamp")
                lookback_avg_range_pct = spot_snapshot.get("lookback_avg_range_pct")

                print_key_value_block("SPOT SNAPSHOT", {
                    "spot": spot,
                    "day_open": day_open,
                    "day_high": day_high,
                    "day_low": day_low,
                    "prev_close": prev_close,
                    "timestamp": spot_timestamp,
                    "lookback_avg_range_pct": lookback_avg_range_pct,
                })

                option_chain = data_router.get_option_chain(symbol)
                option_chain_validation = validate_option_chain(option_chain)
                print_validation_block("OPTION CHAIN VALIDATION", option_chain_validation)

                if not option_chain_validation.get("is_valid", False):
                    print("\nOption chain invalid. Skipping this cycle.")
                    time.sleep(refresh_interval)
                    continue

                print(f"\n{'option_chain_rows':26}: {len(option_chain)}")

                gamma = safe_call(calculate_gamma_exposure, option_chain, spot, default=None)
                flip = safe_call(gamma_flip_level, option_chain, default=None)
                dealer_pos = safe_call(dealer_inventory_position, option_chain, default=None)
                vol_regime_value = safe_call(detect_volatility_regime, option_chain, default=None)

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

                gamma_event = safe_call(detect_gamma_squeeze, prices, gamma_curve, default=None)
                flow = safe_call(flow_signal, option_chain, default=None)
                smart_flow = safe_call(smart_money_signal, option_chain, default=None)
                liquidity_levels = safe_call(strongest_liquidity_levels, option_chain, default=[])
                voids = safe_call(detect_liquidity_voids, option_chain, default=[])
                void_sig = safe_call(liquidity_void_signal, spot, voids, default=None)
                vacuum_zones = safe_call(detect_liquidity_vacuum, option_chain, default=[])
                vacuum_state = safe_call(vacuum_direction, spot, vacuum_zones, default=None)
                market_gamma = safe_call(calculate_market_gamma, option_chain, default=None)
                gamma_regime = safe_call(market_gamma_regime, market_gamma, default=None)
                gamma_clusters = safe_call(largest_gamma_strikes, market_gamma, default=None)
                walls = safe_call(classify_walls, option_chain, default={}) or {}

                support_wall = walls.get("support_wall") if isinstance(walls, dict) else None
                resistance_wall = walls.get("resistance_wall") if isinstance(walls, dict) else None

                hedging_flow = safe_call(dealer_hedging_flow, option_chain, default=None)
                hedging_sim = safe_call(simulate_dealer_hedging, option_chain, default={})
                hedging_bias_value = safe_call(hedging_bias, hedging_sim, default=None)
                atm_iv_value = safe_call(atm_vol, option_chain, spot, default=None)

                vol_surface_regime = None
                if atm_iv_value is not None:
                    vol_surface_regime = safe_call(vol_regime, atm_iv_value, default=None)

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
                    "spot_timestamp": spot_timestamp,
                    "day_open": day_open,
                    "day_high": day_high,
                    "day_low": day_low,
                    "prev_close": prev_close,
                    "lookback_avg_range_pct": lookback_avg_range_pct,
                    "spot_validation": spot_validation,
                    "option_chain_validation": option_chain_validation,
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
                    day_high=day_high,
                    day_low=day_low,
                    day_open=day_open,
                    prev_close=prev_close,
                    lookback_avg_range_pct=lookback_avg_range_pct,
                    spot_validation=spot_validation,
                    option_chain_validation=option_chain_validation,
                    apply_budget_constraint=apply_budget_constraint,
                    requested_lots=requested_lots,
                    lot_size=lot_size,
                    max_capital=max_capital
                )

                if trade:
                    trade["selected_expiry"] = option_chain_validation.get("selected_expiry")
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
                        "rule_move_probability",
                        "move_probability_components",
                        "scoring_breakdown",
                        "spot_validation",
                        "option_chain_validation",
                        "data_quality_score",
                        "data_quality_status",
                        "data_quality_reasons",
                        "analytics_quality",
                        "confirmation_status",
                        "confirmation_veto",
                        "confirmation_reasons",
                        "confirmation_breakdown",
                        "ranked_strike_candidates",
                    ]:
                        if key in trade:
                            dashboard_for_print[key] = trade.get(key)

                    print_dealer_dashboard(dashboard_for_print)

                    print_signal_summary(trade)
                    print_diagnostics(trade)
                else:
                    print("\nNo trade signal")
                    print_dealer_dashboard(dashboard_summary)

                previous_chain = option_chain.copy()

            except Exception as e:
                print("\nEngine error:", e)

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nEngine stopped by user")

    finally:
        data_router.close()


if __name__ == "__main__":
    main()
