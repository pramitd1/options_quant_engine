import argparse
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
from data.replay_loader import (
    latest_replay_snapshot_paths,
    load_option_chain_snapshot,
    load_spot_snapshot,
    save_option_chain_snapshot,
)
from data.option_chain_validation import validate_option_chain
from data.expiry_resolver import (
    filter_option_chain_by_expiry,
    resolve_selected_expiry,
)
from macro.scheduled_event_risk import evaluate_scheduled_event_risk
from macro.macro_news_aggregator import build_macro_news_state
from news.service import build_default_headline_service

from engine.trading_engine import generate_trade
from engine.runtime_metadata import TRADER_VIEW_KEYS
from research.signal_evaluation import (
    CAPTURE_POLICY_ALL,
    SIGNAL_DATASET_PATH,
    normalize_capture_policy,
    save_signal_evaluation,
    should_capture_signal,
)

from analytics.gamma_exposure import calculate_gamma_exposure
from analytics.gamma_flip import gamma_flip_level
from analytics.dealer_inventory import dealer_inventory_metrics, dealer_inventory_position
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
from analytics.greeks_engine import enrich_chain_with_greeks, summarize_greek_exposures

from visualization.dealer_dashboard import print_dealer_dashboard


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


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--replay", action="store_true", help="Run once using saved spot and option-chain snapshots")
    parser.add_argument("--replay-spot", help="Path to a saved spot snapshot JSON file")
    parser.add_argument("--replay-chain", help="Path to a saved option-chain snapshot CSV/JSON file")
    parser.add_argument("--replay-dir", default="debug_samples", help="Directory used to auto-discover latest replay snapshots")
    parser.add_argument("--replay-source", default="REPLAY", help="Label to display as the data source during replay mode")
    parser.add_argument(
        "--signal-capture-policy",
        default=CAPTURE_POLICY_ALL,
        help="Signal capture policy: TRADE_ONLY, ACTIONABLE, or ALL_SIGNALS",
    )
    return parser.parse_args()


def choose_underlying_symbol():
    raw_symbol = input(
        "Enter symbol (NIFTY / BANKNIFTY / FINNIFTY / STOCK): "
    ).strip().upper()

    if not raw_symbol:
        return DEFAULT_SYMBOL

    if raw_symbol != "STOCK":
        return raw_symbol

    stock_symbol = input(
        "Enter stock option underlying symbol (example: RELIANCE / SBIN / TCS): "
    ).strip().upper()

    if stock_symbol:
        return stock_symbol

    print(f"No stock symbol entered. Defaulting to {DEFAULT_SYMBOL}.")
    return DEFAULT_SYMBOL


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


def print_validation_block(title, validation):
    print(f"\n{title}")
    print("---------------------------")
    preferred_order = [
        "validation_mode",
        "is_valid",
        "live_trading_valid",
        "replay_analysis_valid",
        "is_stale",
        "age_minutes",
        "issues",
        "warnings",
    ]

    printed = set()
    for key in preferred_order:
        if key in validation:
            print(f"{key:26}: {validation.get(key)}")
            printed.add(key)

    for key, value in validation.items():
        if key in printed:
            continue
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
        "macro_event_risk_score": trade.get("macro_event_risk_score"),
        "event_window_status": trade.get("event_window_status"),
        "event_lockdown_flag": trade.get("event_lockdown_flag"),
        "minutes_to_next_event": trade.get("minutes_to_next_event"),
        "next_event_name": trade.get("next_event_name"),
        "macro_regime": trade.get("macro_regime"),
        "macro_sentiment_score": trade.get("macro_sentiment_score"),
        "volatility_shock_score": trade.get("volatility_shock_score"),
        "news_confidence_score": trade.get("news_confidence_score"),
        "macro_adjustment_score": trade.get("macro_adjustment_score"),
        "macro_position_size_multiplier": trade.get("macro_position_size_multiplier"),
        "macro_suggested_lots": trade.get("macro_suggested_lots"),
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
        "provider_health",
        "data_quality_reasons",
        "macro_adjustment_reasons",
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
    args = parse_runtime_args()
    signal_capture_policy = normalize_capture_policy(args.signal_capture_policy)
    symbol = choose_underlying_symbol()
    headline_service = build_default_headline_service()

    source = args.replay_source.upper().strip() if args.replay else choose_data_source()
    if not args.replay:
        prompt_provider_credentials(source)
    apply_budget_constraint = choose_budget_mode()
    lot_size, requested_lots, max_capital = get_budget_inputs(apply_budget_constraint)
    refresh_interval = 0 if args.replay else _refresh_interval_for_source(source)

    print("\nRunning Quant Engine for:", symbol)
    print("Data Source:", source)
    print("Budget Constraint Applied:", apply_budget_constraint)

    if args.replay:
        spot_replay_path = args.replay_spot
        chain_replay_path = args.replay_chain
        if not spot_replay_path or not chain_replay_path:
            discovered_spot, discovered_chain = latest_replay_snapshot_paths(symbol, replay_dir=args.replay_dir)
            spot_replay_path = spot_replay_path or discovered_spot
            chain_replay_path = chain_replay_path or discovered_chain

        if not spot_replay_path or not chain_replay_path:
            print("\nReplay mode requires both a spot snapshot and an option-chain snapshot.")
            print("Use --replay-spot / --replay-chain, or keep snapshots in debug_samples for auto-discovery.")
            return

        print(f"Replay Spot Snapshot       : {spot_replay_path}")
        print(f"Replay Option Chain File  : {chain_replay_path}")
        data_router = None
    else:
        try:
            data_router = DataSourceRouter(source)
        except Exception as e:
            print(f"\nFailed to initialize data source '{source}': {e}")
            print("Check credentials, session tokens, installed packages, and expiry configuration.")
            return

    previous_chain = None
    saved_one_spot_snapshot = False
    saved_one_option_chain_snapshot = False
    latest_saved_spot_path = None
    latest_saved_chain_path = None

    try:
        while True:
            try:
                if args.replay:
                    spot_snapshot = load_spot_snapshot(spot_replay_path)
                else:
                    spot_snapshot = get_spot_snapshot(symbol)
                spot_validation = spot_snapshot.get("validation", {})

                if not args.replay and not saved_one_spot_snapshot:
                    try:
                        saved_path = save_spot_snapshot(spot_snapshot)
                        latest_saved_spot_path = saved_path
                        print(f"\nSaved one live spot snapshot to: {saved_path}")
                    except Exception as save_err:
                        print(f"\nCould not save spot snapshot: {save_err}")
                    saved_one_spot_snapshot = True

                print_validation_block("SPOT VALIDATION", spot_validation)

                if not spot_validation.get("is_valid", False):
                    print("\nSpot snapshot invalid. Skipping this cycle.")
                    if args.replay:
                        break
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

                macro_event_state = evaluate_scheduled_event_risk(
                    symbol=symbol,
                    as_of=spot_timestamp,
                )

                print_key_value_block("MACRO EVENT RISK", {
                    "macro_event_risk_score": macro_event_state.get("macro_event_risk_score"),
                    "event_window_status": macro_event_state.get("event_window_status"),
                    "event_lockdown_flag": macro_event_state.get("event_lockdown_flag"),
                    "minutes_to_next_event": macro_event_state.get("minutes_to_next_event"),
                    "next_event_name": macro_event_state.get("next_event_name"),
                })

                headline_state = headline_service.fetch(
                    symbol=symbol,
                    as_of=spot_timestamp,
                    replay_mode=args.replay,
                )
                macro_news_state = build_macro_news_state(
                    event_state=macro_event_state,
                    headline_state=headline_state,
                    as_of=spot_timestamp,
                ).to_dict()

                print_key_value_block("MACRO / NEWS REGIME", {
                    "macro_regime": macro_news_state.get("macro_regime"),
                    "macro_regime_reasons": macro_news_state.get("macro_regime_reasons"),
                    "macro_event_risk_score": macro_news_state.get("macro_event_risk_score"),
                    "macro_sentiment_score": macro_news_state.get("macro_sentiment_score"),
                    "volatility_shock_score": macro_news_state.get("volatility_shock_score"),
                    "event_lockdown_flag": macro_news_state.get("event_lockdown_flag"),
                    "news_confidence_score": macro_news_state.get("news_confidence_score"),
                    "headline_velocity": macro_news_state.get("headline_velocity"),
                    "headline_count": macro_news_state.get("headline_count"),
                    "classified_headline_count": macro_news_state.get("classified_headline_count"),
                    "next_event_name": macro_news_state.get("next_event_name"),
                    "neutral_fallback": macro_news_state.get("neutral_fallback"),
                })

                macro_news_details = {
                    "headline_provider": headline_state.provider_name,
                    "headline_data_available": headline_state.data_available,
                    "headline_is_stale": headline_state.is_stale,
                    "headline_warnings": headline_state.warnings,
                    "headline_issues": headline_state.issues,
                    "provider_metadata": headline_state.provider_metadata,
                    "classification_preview": macro_news_state.get("classification_preview"),
                }
                if (
                    headline_state.warnings
                    or headline_state.issues
                    or headline_state.provider_metadata
                    or macro_news_state.get("classification_preview")
                ):
                    print_key_value_block(
                        "MACRO / NEWS DETAILS",
                        {key: _format_output_value(value) for key, value in macro_news_details.items()},
                    )

                if args.replay:
                    option_chain = load_option_chain_snapshot(chain_replay_path)
                else:
                    option_chain = data_router.get_option_chain(symbol)
                resolved_expiry = resolve_selected_expiry(option_chain)
                option_chain = filter_option_chain_by_expiry(option_chain, resolved_expiry)
                option_chain_validation = validate_option_chain(option_chain)
                print_validation_block("OPTION CHAIN VALIDATION", option_chain_validation)

                if not option_chain_validation.get("is_valid", False):
                    print("\nOption chain invalid. Skipping this cycle.")
                    if args.replay:
                        break
                    time.sleep(refresh_interval)
                    continue

                print(f"\n{'option_chain_rows':26}: {len(option_chain)}")

                if not args.replay and not saved_one_option_chain_snapshot:
                    try:
                        saved_chain_path = save_option_chain_snapshot(
                            option_chain,
                            symbol=symbol,
                            source=source,
                        )
                        latest_saved_chain_path = saved_chain_path
                        print(f"Saved one live option chain snapshot to: {saved_chain_path}")
                    except Exception as save_err:
                        print(f"Could not save option chain snapshot: {save_err}")
                    saved_one_option_chain_snapshot = True

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
                    max_capital=max_capital,
                    macro_event_state=macro_event_state,
                    macro_news_state=macro_news_state,
                    valuation_time=spot_timestamp,
                )

                if trade:
                    trade["selected_expiry"] = option_chain_validation.get("selected_expiry")

                    signal_capture_payload = {
                        "ok": True,
                        "mode": "REPLAY" if args.replay else "LIVE",
                        "source": source,
                        "symbol": symbol,
                        "saved_paths": {
                            "spot": latest_saved_spot_path,
                            "chain": latest_saved_chain_path,
                        },
                        "spot_snapshot": spot_snapshot,
                        "spot_summary": {
                            "spot": spot,
                            "day_open": day_open,
                            "day_high": day_high,
                            "day_low": day_low,
                            "prev_close": prev_close,
                            "timestamp": spot_timestamp,
                            "lookback_avg_range_pct": lookback_avg_range_pct,
                            "ticker": spot_snapshot.get("ticker"),
                        },
                        "option_chain_validation": option_chain_validation,
                        "trade": trade,
                    }

                    if should_capture_signal(trade, signal_capture_policy):
                        try:
                            save_signal_evaluation(signal_capture_payload)
                            print(f"\n{'signal_capture':26}: CAPTURED -> {SIGNAL_DATASET_PATH}")
                        except Exception as capture_err:
                            print(f"\n{'signal_capture':26}: FAILED ({type(capture_err).__name__}: {capture_err})")
                    else:
                        print(f"\n{'signal_capture':26}: SKIPPED_POLICY ({signal_capture_policy})")

                    print_trader_view(trade)

                    dashboard_for_print = dict(trade)
                    dashboard_for_print.update({
                        "spot": round(spot, 2),
                        "spot_timestamp": spot_timestamp,
                        "day_open": day_open,
                        "day_high": day_high,
                        "day_low": day_low,
                        "prev_close": prev_close,
                        "lookback_avg_range_pct": lookback_avg_range_pct,
                        "spot_validation": spot_validation,
                        "option_chain_validation": option_chain_validation,
                        "macro_regime": macro_news_state.get("macro_regime"),
                        "macro_sentiment_score": macro_news_state.get("macro_sentiment_score"),
                        "volatility_shock_score": macro_news_state.get("volatility_shock_score"),
                        "news_confidence_score": macro_news_state.get("news_confidence_score"),
                        "headline_velocity": macro_news_state.get("headline_velocity"),
                    })

                    print_dealer_dashboard(dashboard_for_print)

                    print_signal_summary(trade)
                    print_diagnostics(trade)
                else:
                    print("\nNo trade signal")
                    print_key_value_block("ENGINE STATUS", {"message": "No trade payload returned"})

                previous_chain = option_chain.copy()

            except Exception as e:
                print("\nEngine error:", e)

            if args.replay:
                break

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\nEngine stopped by user")

    finally:
        if data_router is not None:
            data_router.close()


if __name__ == "__main__":
    main()
