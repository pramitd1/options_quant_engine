"""
Module: main.py

Purpose:
    Provide the interactive CLI entry point for live, replay, and data-capture runs of the options engine.

Role in the System:
    Part of the repository entry layer that wires operator choices and runtime flags into the engine runner.

Key Outputs:
    Parsed runtime configuration, operator prompts, saved snapshots, and calls into the application runner.

Downstream Usage:
    Used directly by operators and indirectly by replay, capture, and shadow-evaluation workflows.
"""
import argparse
import json
import os
import time
import warnings
from getpass import getpass
from pathlib import Path

from urllib3.exceptions import NotOpenSSLWarning


warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

from config.settings import (
    DEFAULT_SYMBOL,
    DEFAULT_DATA_SOURCE,
    OUTPUT_MODE,
    REFRESH_INTERVAL,
    NSE_REFRESH_INTERVAL,
    ICICI_REFRESH_INTERVAL,
    LOT_SIZE,
    NUMBER_OF_LOTS,
    MAX_CAPITAL_PER_TRADE,
    DATA_SOURCE_OPTIONS,
)

from app.engine_runner import run_engine_snapshot
from config.policy_resolver import (
    evaluate_regime_pack_switch,
    get_active_parameter_pack,
    get_regime_switch_policy,
    set_active_parameter_pack,
    suggest_regime_pack,
)
from notifications.telegram_alert import maybe_alert as _telegram_maybe_alert
from app.terminal_output import render_snapshot, _resolve_top_liquidity_walls
from data.data_source_router import DataSourceRouter
from data.replay_loader import save_option_chain_snapshot
from data.spot_downloader import save_spot_snapshot
from news.service import build_default_headline_service
from engine.runtime_metadata import TRADER_VIEW_KEYS
from research.signal_evaluation import (
    CAPTURE_POLICY_ALL,
    normalize_capture_policy,
)


def _refresh_interval_for_source(source: str) -> int:
    """
    Purpose:
        Resolve the polling interval implied by the selected data-source provider.
    
    Context:
        Part of the repository entry layer that translates CLI inputs into runtime behavior.
    
    Inputs:
        source (str): Market-data source label selected for the current runtime session.
    
    Returns:
        int: Refresh interval, in seconds, for the selected market-data provider.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
    """
    source = source.upper().strip()
    if source == "NSE":
        return NSE_REFRESH_INTERVAL
    if source == "ICICI":
        return ICICI_REFRESH_INTERVAL
    return REFRESH_INTERVAL


def _append_regime_switch_log(record: dict, relative_path: str) -> None:
    """Append one regime-switch decision record to a JSONL log file."""
    try:
        log_path = Path(relative_path)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Logging should never break live signal generation.
        return


def choose_data_source():
    """
    Purpose:
        Prompt the operator to choose which market-data provider should feed the current session.
    
    Context:
        Part of the interactive runtime setup path that collects operator intent and resolves the settings used by the signal engine.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        str: Operator-selected provider name, with the configured default used as a fallback.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
    """
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
    """
    Purpose:
        Parse CLI flags that control replay mode, capture policy, and runtime wiring.
    
    Context:
        Part of the repository entry layer that translates CLI inputs into runtime behavior.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments for the current runtime session.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
    """
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
    parser.add_argument(
        "--output-mode",
        default=None,
        choices=["COMPACT", "STANDARD", "FULL_DEBUG"],
        help="Terminal verbosity: COMPACT, STANDARD (default), or FULL_DEBUG",
    )
    parser.add_argument(
        "--market-levels-sort",
        default="GROUPED",
        choices=["GROUPED", "NEAREST"],
        help="Sort mode for COMPACT market level table: GROUPED (default) or NEAREST",
    )
    return parser.parse_args()


def choose_underlying_symbol():
    """
    Purpose:
        Prompt the operator to choose the index or stock symbol to evaluate.
    
    Context:
        Part of the interactive runtime setup path that collects operator intent and resolves the settings used by the signal engine.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        str: Operator-selected symbol, with the repository default used when no input is supplied.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
    """
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
    """
    Purpose:
        Prompt the operator to decide whether trade construction should honor the configured capital budget.
    
    Context:
        Part of the interactive runtime setup path that collects operator intent and resolves the settings used by the signal engine.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        bool: `True` when trade construction should enforce budget constraints.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
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


def choose_output_mode(default: str = "STANDARD") -> str:
    """Prompt the operator to select the terminal output verbosity level."""
    print("\nSelect output mode:")
    print("1. COMPACT      — minimal execution surface")
    print("2. STANDARD     — scoring + confirmation diagnostics")
    print("3. FULL_DEBUG   — every field the engine produces")

    mapping = {"1": "COMPACT", "2": "STANDARD", "3": "FULL_DEBUG"}
    choice = input(f"Enter choice (1/2/3) [default: {default}]: ").strip()

    if choice in mapping:
        return mapping[choice]
    if not choice:
        return default
    print(f"Invalid choice. Defaulting to {default}.")
    return default


def _prompt_runtime_secret(env_name: str, label: str, secret: bool = False):
    """
    Purpose:
        Read a provider credential from the environment or an interactive prompt.
    
    Context:
        Part of the interactive runtime setup path that collects operator intent and resolves the settings used by the signal engine.
    
    Inputs:
        env_name (str): Environment-variable name associated with a provider credential.
        label (str): Human-readable label shown to the operator while prompting for a credential.
        secret (bool): Whether the prompt should hide typed input such as session tokens or secrets.
    
    Returns:
        None: The helper updates the environment in place when the operator supplies a value.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
    """
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
    """
    Purpose:
        Collect any provider credentials required before the live runtime starts.
    
    Context:
        Part of the interactive runtime setup path that collects operator intent and resolves the settings used by the signal engine.
    
    Inputs:
        source (str): Market-data source label selected for the current runtime session.
    
    Returns:
        None: The function operates through prompts and environment-variable updates.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
    """
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
    """
    Purpose:
        Resolve the lot-size and capital settings that should govern trade construction.
    
    Context:
        Part of the interactive runtime setup path that collects operator intent and resolves the settings used by the signal engine.
    
    Inputs:
        apply_budget_constraint (bool): Whether the operator chose to enforce capital-budget rules during trade construction.
    
    Returns:
        tuple[int, int, float]: Lot size, requested lots, and maximum capital to use for trade construction.
    
    Notes:
        Operator prompts are kept explicit so live, replay, and capture workflows remain easy to audit.
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


def print_trader_view(trade):
    """
    Purpose:
        Render the compact trader-facing subset of the final trade payload.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        trade (Any): Final trade or no-trade payload produced by the signal engine.
    
    Returns:
        None: The function writes the trader-facing summary directly to stdout.
    
    Notes:
        Formatting is intentionally handled outside the signal engine so display concerns do not leak into trading logic.
    """
    print("\nTRADER VIEW")
    print("---------------------------")

    for key in TRADER_VIEW_KEYS:
        if key in trade:
            print(f"{key:26}: {trade.get(key)}")


def print_validation_block(title, validation):
    """
    Purpose:
        Render one validation payload in a stable operator-facing order.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        title (Any): Section title to display in the CLI output.
        validation (Any): Validation payload to display in a stable, operator-readable order.
    
    Returns:
        None: The function writes the validation block directly to stdout.
    
    Notes:
        Formatting is intentionally handled outside the signal engine so display concerns do not leak into trading logic.
    """
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
    """
    Purpose:
        Render a generic key-value section in the CLI output.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        title (Any): Section title to display in the CLI output.
        values (Any): Mapping of keys to values that should be printed as a formatted block.
    
    Returns:
        None: The function writes the formatted section directly to stdout.
    
    Notes:
        Formatting is intentionally handled outside the signal engine so display concerns do not leak into trading logic.
    """
    print(f"\n{title}")
    print("---------------------------")
    for key, value in values.items():
        print(f"{key:26}: {value}")


def _format_output_value(value, max_items=8):
    """
    Purpose:
        Convert nested runtime values into a compact CLI-friendly representation.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        value (Any): Arbitrary runtime value that may need truncation or formatting for CLI output.
        max_items (Any): Maximum number of list or dictionary items to preview before truncating the display.
    
    Returns:
        Any: Scalar or compact preview value suitable for operator-facing CLI output.
    
    Notes:
        Formatting is intentionally handled outside the signal engine so display concerns do not leak into trading logic.
    """
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


def _build_trade_triggers(trade):
    """Return a short, human-readable list of execution triggers."""
    if not isinstance(trade, dict):
        return []

    triggers = []
    direction = str(trade.get("direction") or "").upper().strip()
    flow_signal = str(trade.get("final_flow_signal") or "").upper().strip()
    spot_vs_flip = str(trade.get("spot_vs_flip") or "").upper().strip()
    dealer_bias = str(trade.get("dealer_hedging_bias") or "").upper().strip()
    confirmation = str(trade.get("confirmation_status") or "").upper().strip()
    macro_regime = str(trade.get("macro_regime") or "").upper().strip()
    move_prob = trade.get("hybrid_move_probability")

    if flow_signal == "BEARISH_FLOW" and direction == "PUT":
        triggers.append("Bearish flow aligned with PUT direction")
    elif flow_signal == "BULLISH_FLOW" and direction == "CALL":
        triggers.append("Bullish flow aligned with CALL direction")

    if spot_vs_flip == "BELOW_FLIP" and direction == "PUT":
        triggers.append("Spot trading below gamma flip supports downside setup")
    elif spot_vs_flip == "ABOVE_FLIP" and direction == "CALL":
        triggers.append("Spot trading above gamma flip supports upside setup")
    elif spot_vs_flip == "AT_FLIP":
        triggers.append("Spot holding near gamma flip keeps move sensitivity elevated")

    if dealer_bias == "DOWNSIDE_PINNING" and direction == "PUT":
        triggers.append("Dealer hedging bias favors downside pressure")
    elif dealer_bias == "UPSIDE_PINNING" and direction == "CALL":
        triggers.append("Dealer hedging bias favors upside pressure")
    elif dealer_bias in {"UPSIDE_HEDGING_ACCELERATION", "DOWNSIDE_HEDGING_ACCELERATION"}:
        triggers.append("Dealer hedging acceleration is aligned with the move")

    if confirmation == "STRONG_CONFIRMATION":
        triggers.append("Confirmation filter is fully aligned")
    elif confirmation == "CONFIRMED":
        triggers.append("Confirmation filter supports execution")

    if isinstance(move_prob, (int, float)):
        if move_prob >= 0.65:
            triggers.append(f"Move probability elevated at {move_prob:.0%}")
        elif move_prob >= 0.55:
            triggers.append(f"Move probability supportive at {move_prob:.0%}")

    if macro_regime == "RISK_OFF" and direction == "PUT":
        triggers.append("Risk-off macro regime supports PUT exposure")
    elif macro_regime == "RISK_ON" and direction == "CALL":
        triggers.append("Risk-on macro regime supports CALL exposure")

    deduped = []
    seen = set()
    for trigger in triggers:
        if trigger not in seen:
            deduped.append(trigger)
            seen.add(trigger)
    return deduped[:5]


def print_dealer_dashboard(summary: dict):
    """
    Purpose:
        Render the market-state and dealer-positioning dashboard for the current snapshot.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        summary (dict): Market-state summary assembled for the dealer-positioning dashboard.
    
    Returns:
        None: The function writes the dealer dashboard directly to stdout.
    
    Notes:
        These views mirror the same payload captured by research logging so operators and researchers can inspect a common signal contract.
    """
    print("\nDEALER POSITIONING DASHBOARD")
    print("--------------------------------------------------")
    top_support_walls, top_resistance_walls = _resolve_top_liquidity_walls(
        summary,
        top_n=3,
        formatted=True,
    )
    summary = dict(summary)
    summary["top_support_walls"] = top_support_walls
    summary["top_resistance_walls"] = top_resistance_walls

    ordered_keys = [
        ("Spot Price", "spot"),
        ("Gamma Exposure", "gamma_exposure"),
        ("Market Gamma", "market_gamma"),
        ("Delta Exposure", "delta_exposure"),
        ("Greek Gamma Exp", "gamma_exposure_greeks"),
        ("Theta Exposure", "theta_exposure"),
        ("Vega Exposure", "vega_exposure"),
        ("Rho Exposure", "rho_exposure"),
        ("Vanna Exposure", "vanna_exposure"),
        ("Charm Exposure", "charm_exposure"),
        ("Gamma Flip Level", "gamma_flip"),
        ("Spot vs Flip", "spot_vs_flip"),
        ("Gamma Regime", "gamma_regime"),
        ("Vanna Regime", "vanna_regime"),
        ("Charm Regime", "charm_regime"),
        ("Gamma Clusters", "gamma_clusters"),
        ("Dealer Inventory", "dealer_position"),
        ("Dealer Inv Basis", "dealer_inventory_basis"),
        ("Call OI Change", "call_oi_change"),
        ("Put OI Change", "put_oi_change"),
        ("Net OI Bias", "net_oi_change_bias"),
        ("Dealer Hedging Flow", "dealer_hedging_flow"),
        ("Dealer Hedging Bias", "dealer_hedging_bias"),
        ("Intraday Gamma State", "intraday_gamma_state"),
        ("Volatility Regime", "volatility_regime"),
        ("Vol Surface Regime", "vol_surface_regime"),
        ("ATM IV", "atm_iv"),
        ("Flow Signal", "flow_signal"),
        ("Smart Money Flow", "smart_money_flow"),
        ("Final Flow Signal", "final_flow_signal"),
        ("Signal Regime", "signal_regime"),
        ("Execution Regime", "execution_regime"),
        ("Macro Event Risk", "macro_event_risk_score"),
        ("Event Window", "event_window_status"),
        ("Event Lockdown", "event_lockdown_flag"),
        ("Min To Next Event", "minutes_to_next_event"),
        ("Next Event", "next_event_name"),
        ("Macro Regime", "macro_regime"),
        ("Macro Sentiment", "macro_sentiment_score"),
        ("Vol Shock Score", "macro_news_volatility_shock_score"),
        ("India VIX Level", "india_vix_level"),
        ("India VIX Change 24h", "india_vix_change_24h"),
        ("News Confidence", "news_confidence_score"),
        ("Headline Velocity", "headline_velocity"),
        ("Macro Adj Score", "macro_adjustment_score"),
        ("Macro Size Mult", "macro_position_size_multiplier"),
        ("Macro Lots Hook", "macro_suggested_lots"),
        ("Gamma Event", "gamma_event"),
        ("Top Support Walls", "top_support_walls"),
        ("Top Resistance Walls", "top_resistance_walls"),
        ("Liquidity Levels", "liquidity_levels"),
        ("Liquidity Voids", "liquidity_voids"),
        ("Liquidity Void Signal", "liquidity_void_signal"),
        ("Liquidity Vacuum Zones", "liquidity_vacuum_zones"),
        ("Liquidity Vacuum State", "liquidity_vacuum_state"),
        ("Provider Health", "provider_health"),
    ]

    for label, key in ordered_keys:
        print(f"{label:22}: {_format_output_value(summary.get(key))}")

    dealer_map = summary.get("dealer_liquidity_map")
    if dealer_map:
        print(f"{'Dealer Liquidity Map':22}: {_format_output_value(dealer_map)}")

    print(f"{'Large Move Probability':22}: {_format_output_value(summary.get('large_move_probability'))}")
    print(f"{'ML Move Probability':22}: {_format_output_value(summary.get('ml_move_probability'))}")
    print("--------------------------------------------------")

    scoring_breakdown = summary.get("scoring_breakdown")
    if scoring_breakdown:
        print("SCORING BREAKDOWN")
        print("--------------------------------------------------")
        for key, value in scoring_breakdown.items():
            print(f"{key:22}: {_format_output_value(value)}")
        print("--------------------------------------------------")


def print_signal_summary(trade):
    """
    Purpose:
        Render the compact trade summary that operators use to understand the current signal.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        trade (Any): Final trade or no-trade payload produced by the signal engine.
    
    Returns:
        None: The function writes the signal summary directly to stdout.
    
    Notes:
        These views mirror the same payload captured by research logging so operators and researchers can inspect a common signal contract.
    """
    top_support_walls, top_resistance_walls = _resolve_top_liquidity_walls(
        trade,
        top_n=3,
        formatted=True,
    )

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
        "decision_classification": trade.get("decision_classification"),
        "setup_state": trade.get("setup_state"),
        "setup_quality": trade.get("setup_quality"),
        "watchlist_flag": trade.get("watchlist_flag"),
        "watchlist_reason": trade.get("watchlist_reason"),
        "hybrid_move_probability": trade.get("hybrid_move_probability"),
        "flow_signal": trade.get("final_flow_signal"),
        "gamma_regime": trade.get("gamma_regime"),
        "spot_vs_flip": trade.get("spot_vs_flip"),
        "top_resistance_walls": top_resistance_walls,
        "top_support_walls": top_support_walls,
        "dealer_position": trade.get("dealer_position"),
        "dealer_hedging_bias": trade.get("dealer_hedging_bias"),
        "macro_event_risk_score": trade.get("macro_event_risk_score"),
        "event_window_status": trade.get("event_window_status"),
        "event_lockdown_flag": trade.get("event_lockdown_flag"),
        "minutes_to_next_event": trade.get("minutes_to_next_event"),
        "next_event_name": trade.get("next_event_name"),
        "macro_regime": trade.get("macro_regime"),
        "macro_sentiment_score": trade.get("macro_sentiment_score"),
        "macro_news_volatility_shock_score": trade.get("macro_news_volatility_shock_score"),
        "news_confidence_score": trade.get("news_confidence_score"),
        "macro_adjustment_score": trade.get("macro_adjustment_score"),
        "macro_position_size_multiplier": trade.get("macro_position_size_multiplier"),
        "macro_suggested_lots": trade.get("macro_suggested_lots"),
        "global_risk_state": trade.get("global_risk_state"),
        "global_risk_state_score": trade.get("global_risk_state_score"),
        "global_risk_overlay_score": trade.get("global_risk_overlay_score", trade.get("global_risk_score")),
        "gamma_vol_acceleration_score": trade.get("gamma_vol_acceleration_score"),
        "squeeze_risk_state": trade.get("squeeze_risk_state"),
        "directional_convexity_state": trade.get("directional_convexity_state"),
        "dealer_hedging_pressure_score": trade.get("dealer_hedging_pressure_score"),
        "dealer_flow_state": trade.get("dealer_flow_state"),
        "expected_move_points": trade.get("expected_move_points"),
        "expected_move_pct": trade.get("expected_move_pct"),
        "expected_move_quality": trade.get("expected_move_quality"),
        "target_reachability_score": trade.get("target_reachability_score"),
        "premium_efficiency_score": trade.get("premium_efficiency_score"),
        "strike_efficiency_score": trade.get("strike_efficiency_score"),
        "option_efficiency_score": trade.get("option_efficiency_score"),
        "overnight_gap_risk_score": trade.get("overnight_gap_risk_score"),
        "overnight_hold_allowed": trade.get("overnight_hold_allowed"),
        "overnight_hold_reason": trade.get("overnight_hold_reason"),
        "overnight_risk_penalty": trade.get("overnight_risk_penalty"),
        "atm_iv": trade.get("atm_iv"),
        "vol_surface_regime": trade.get("vol_surface_regime"),
        "capital_required": trade.get("capital_required"),
        "data_quality_score": trade.get("data_quality_score"),
        "data_quality_status": trade.get("data_quality_status"),
        "message": trade.get("message"),
    }
    print_key_value_block("QUANT TRADE SIGNAL", compact)

    if trade.get("direction"):
        triggers = _build_trade_triggers(trade)
        if triggers:
            print("\nTRADE TRIGGERS")
            print("---------------------------")
            for trigger in triggers:
                print(f"  • {trigger}")

    explainability = {
        "decision_classification": trade.get("decision_classification"),
        "no_trade_reason_code": trade.get("no_trade_reason_code"),
        "no_trade_reason": trade.get("no_trade_reason"),
        "missing_signal_requirements": trade.get("missing_signal_requirements"),
        "setup_upgrade_conditions": trade.get("setup_upgrade_conditions"),
        "likely_next_trigger": trade.get("likely_next_trigger"),
        "watchlist_flag": trade.get("watchlist_flag"),
        "watchlist_reason": trade.get("watchlist_reason"),
        "option_efficiency_status": trade.get("option_efficiency_status"),
        "option_efficiency_reason": trade.get("option_efficiency_reason"),
        "global_risk_status": trade.get("global_risk_status"),
        "global_risk_reason": trade.get("global_risk_reason"),
        "macro_news_status": trade.get("macro_news_status"),
        "macro_news_reason": trade.get("macro_news_reason"),
    }
    print_key_value_block("EXPLAINABILITY", explainability)


def print_ranked_candidates_table(candidates, expiry=None):
    """
    Purpose:
        Render the highest-ranked strike candidates selected by the strategy layer.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        candidates (Any): Ranked strike-candidate records returned by the strategy layer.
        expiry (Any): Selected expiry label shown alongside the ranked-candidate table, when available.
    
    Returns:
        None: The function writes the candidate table directly to stdout when candidates are available.
    
    Notes:
        Formatting is intentionally handled outside the signal engine so display concerns do not leak into trading logic.
    """
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
    """
    Purpose:
        Render the detailed diagnostic payload that supports the current trade decision.
    
    Context:
        Helper in the operator-facing CLI layer. It turns structured engine output into readable terminal sections without changing the underlying payload.
    
    Inputs:
        trade (Any): Final trade or no-trade payload produced by the signal engine.
    
    Returns:
        None: The function writes the diagnostic section directly to stdout.
    
    Notes:
        These views mirror the same payload captured by research logging so operators and researchers can inspect a common signal contract.
    """
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
        "global_risk_state_reasons",
        "global_risk_overlay_reasons",
        "global_risk_diagnostics",
        "global_risk_features",
        "gamma_vol_reasons",
        "gamma_vol_diagnostics",
        "gamma_vol_features",
        "dealer_pressure_reasons",
        "dealer_pressure_diagnostics",
        "dealer_pressure_features",
        "option_efficiency_reasons",
        "option_efficiency_diagnostics",
        "option_efficiency_features",
        "no_trade_reason_details",
        "blocked_by",
        "missing_confirmations",
        "signal_promotion_requirements",
        "setup_upgrade_path",
        "watchlist_trigger_levels",
        "directional_resolution_needed",
        "explainability",
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
    """
    Purpose:
        Run the interactive repository entry point for live snapshots or replay mode.
    
    Context:
        Top-level repository entry point that wires prompts, data routing, engine execution, and operator-facing output into one interactive workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        None: The function runs the interactive runtime loop until interrupted or a replay snapshot is processed.
    
    Notes:
        The entry point keeps live mode and replay mode on the same signal path so diagnostics remain comparable across environments.
    """
    args = parse_runtime_args()
    signal_capture_policy = normalize_capture_policy(args.signal_capture_policy)
    output_mode = args.output_mode or OUTPUT_MODE
    symbol = choose_underlying_symbol()
    headline_service = build_default_headline_service()

    source = args.replay_source.upper().strip() if args.replay else choose_data_source()
    if not args.replay:
        prompt_provider_credentials(source)
    apply_budget_constraint = choose_budget_mode()
    lot_size, requested_lots, max_capital = get_budget_inputs(apply_budget_constraint)
    output_mode = choose_output_mode(default=output_mode)
    refresh_interval = 0 if args.replay else _refresh_interval_for_source(source)

    print("\nRunning Quant Engine for:", symbol)
    print("Data Source:", source)
    print("Budget Constraint Applied:", apply_budget_constraint)
    print("Output Mode:", output_mode)

    if args.replay:
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
    replay_paths_printed = False
    _prev_trade_status: str | None = None
    _regime_switch_state = {
        "last_regime_signature": "",
        "consecutive_regime_hits": 0,
        "last_switch_ts": None,
        "active_since_ts": time.time(),
    }

    try:
        while True:
            result = run_engine_snapshot(
                symbol=symbol,
                mode="REPLAY" if args.replay else "LIVE",
                source=source,
                apply_budget_constraint=apply_budget_constraint,
                requested_lots=requested_lots,
                lot_size=lot_size,
                max_capital=max_capital,
                replay_spot=args.replay_spot,
                replay_chain=args.replay_chain,
                replay_dir=args.replay_dir,
                capture_signal_evaluation=True,
                signal_capture_policy=signal_capture_policy,
                previous_chain=previous_chain,
                holding_profile="AUTO",
                headline_service=headline_service,
                data_router=data_router,
            )

            if not result.get("ok"):
                print("\nEngine error:", result.get("error", "Unknown engine error"))
                if args.replay:
                    break
                time.sleep(refresh_interval)
                continue

            replay_paths = result.get("replay_paths") or {}
            if args.replay and replay_paths and not replay_paths_printed:
                print(f"Replay Spot Snapshot       : {replay_paths.get('spot')}")
                print(f"Replay Option Chain File  : {replay_paths.get('chain')}")
                replay_paths_printed = True

            spot_snapshot = result.get("spot_snapshot", {})
            spot_validation = result.get("spot_validation", {})
            spot_summary = result.get("spot_summary", {})
            macro_event_state = result.get("macro_event_state", {})
            headline_state = result.get("headline_state", {})
            macro_news_state = result.get("macro_news_state", {})
            global_market_snapshot = result.get("global_market_snapshot", {})
            global_risk_state = result.get("global_risk_state", {})
            option_chain_validation = result.get("option_chain_validation", {})
            option_chain_frame = result.get("option_chain_frame")
            full_trade = result.get("trade") or result.get("trade_audit")
            execution_trade = result.get("execution_trade")
            if execution_trade is None and isinstance(full_trade, dict):
                execution_trade = full_trade.get("execution_trade")
            trade_for_display = full_trade or execution_trade

            if not args.replay and not saved_one_spot_snapshot:
                try:
                    saved_path = save_spot_snapshot(spot_snapshot)
                    print(f"\nSaved one live spot snapshot to: {saved_path}")
                except Exception as save_err:
                    print(f"\nCould not save spot snapshot: {save_err}")
                saved_one_spot_snapshot = True

            if not spot_validation.get("is_valid", False):
                print("\nSpot snapshot invalid. Skipping this cycle.")
                if args.replay:
                    break
                time.sleep(refresh_interval)
                continue

            if not option_chain_validation.get("is_valid", False):
                print("\nOption chain invalid. Skipping this cycle.")
                if args.replay:
                    break
                time.sleep(refresh_interval)
                continue

            if not args.replay and not saved_one_option_chain_snapshot:
                try:
                    saved_chain_path = save_option_chain_snapshot(
                        option_chain_frame,
                        symbol=symbol,
                        source=source,
                    )
                    print(f"Saved one live option chain snapshot to: {saved_chain_path}")
                except Exception as save_err:
                    print(f"Could not save option chain snapshot: {save_err}")
                saved_one_option_chain_snapshot = True

            if trade_for_display:
                signal_capture_status = result.get("signal_capture_status", "SKIPPED")
                if signal_capture_status == "CAPTURED":
                    print(f"\n{'signal_capture':26}: CAPTURED -> {result.get('signal_dataset_path')}")
                elif signal_capture_status.startswith("FAILED:"):
                    print(
                        f"\n{'signal_capture':26}: {signal_capture_status}"
                        f" ({result.get('signal_capture_error', 'unknown error')})"
                    )
                elif signal_capture_status.startswith("SKIPPED_POLICY:"):
                    print(f"\n{'signal_capture':26}: {signal_capture_status}")

            render_snapshot(
                output_mode,
                result=result,
                spot_summary=spot_summary,
                spot_validation=spot_validation,
                option_chain_validation=option_chain_validation,
                macro_event_state=macro_event_state,
                macro_news_state=macro_news_state,
                global_risk_state=global_risk_state,
                global_market_snapshot=global_market_snapshot,
                headline_state=headline_state,
                trade=trade_for_display,
                execution_trade=execution_trade,
                market_levels_sort_mode=args.market_levels_sort,
            )

            previous_chain = option_chain_frame.copy()

            # Telegram push alert on meaningful state transitions.
            if trade_for_display and isinstance(trade_for_display, dict):
                _telegram_maybe_alert(trade_for_display, prev_status=_prev_trade_status)
                _prev_trade_status = str(trade_for_display.get("trade_status") or "")

            # Regime-conditional auto-switching: suggest a new parameter pack
            # based on the gamma + vol regime resolved in this snapshot.
            if trade_for_display and isinstance(trade_for_display, dict):
                _gamma_r = trade_for_display.get("gamma_regime")
                _vol_r = trade_for_display.get("vol_surface_regime")
                _global_risk = trade_for_display.get("global_risk_state")
                _macro_r = trade_for_display.get("macro_regime")
                _event_bucket = trade_for_display.get("event_risk_bucket")
                _overnight_bucket = (
                    "OVERNIGHT_ALLOWED"
                    if bool(trade_for_display.get("overnight_hold_allowed"))
                    else "OVERNIGHT_BLOCKED"
                )
                _regime_confidence = trade_for_display.get("regime_confidence")

                _suggested_pack = suggest_regime_pack(
                    _gamma_r,
                    _vol_r,
                    global_risk_state=_global_risk,
                    macro_regime=_macro_r,
                    event_risk_bucket=_event_bucket,
                    overnight_bucket=_overnight_bucket,
                )

                if _suggested_pack:
                    _policy = get_regime_switch_policy()
                    _active_pack = get_active_parameter_pack().get("name")
                    _signature = "|".join(
                        [
                            str(_gamma_r or ""),
                            str(_vol_r or ""),
                            str(_global_risk or ""),
                            str(_macro_r or ""),
                            str(_event_bucket or ""),
                            str(_overnight_bucket or ""),
                        ]
                    )
                    _decision = evaluate_regime_pack_switch(
                        suggested_pack=_suggested_pack,
                        current_pack=_active_pack,
                        regime_signature=_signature,
                        switch_state=_regime_switch_state,
                        required_consecutive=_policy["required_consecutive"],
                        cooldown_seconds=_policy["cooldown_seconds"],
                        min_dwell_seconds=_policy["min_dwell_seconds"],
                        regime_confidence=_regime_confidence,
                        min_regime_confidence=_policy["min_regime_confidence"],
                    )
                    _regime_switch_state = _decision.get("state", _regime_switch_state)
                    if _decision.get("apply"):
                        set_active_parameter_pack(_suggested_pack)

                    if _policy.get("log_decisions", True):
                        _append_regime_switch_log(
                            {
                                "timestamp": time.time(),
                                "gamma_regime": _gamma_r,
                                "vol_regime": _vol_r,
                                "global_risk_state": _global_risk,
                                "macro_regime": _macro_r,
                                "event_risk_bucket": _event_bucket,
                                "overnight_bucket": _overnight_bucket,
                                "regime_signature": _signature,
                                "active_pack_before": _active_pack,
                                "suggested_pack": _suggested_pack,
                                "decision_apply": bool(_decision.get("apply")),
                                "decision_reason": _decision.get("reason"),
                                "regime_confidence": _regime_confidence,
                            },
                            str(_policy.get("decision_log_path", "logs/regime_switch_decisions.jsonl")),
                        )

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
