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
import os
import time
import warnings
from getpass import getpass

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

from app.engine_runner import run_engine_snapshot
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
        ("Vol Shock Score", "volatility_shock_score"),
        ("News Confidence", "news_confidence_score"),
        ("Headline Velocity", "headline_velocity"),
        ("Macro Adj Score", "macro_adjustment_score"),
        ("Macro Size Mult", "macro_position_size_multiplier"),
        ("Macro Lots Hook", "macro_suggested_lots"),
        ("Gamma Event", "gamma_event"),
        ("Support Wall", "support_wall"),
        ("Resistance Wall", "resistance_wall"),
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
        "global_risk_state": trade.get("global_risk_state"),
        "global_risk_score": trade.get("global_risk_score"),
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
        "global_risk_reasons",
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
            trade = result.get("execution_trade") or result.get("trade")

            if not args.replay and not saved_one_spot_snapshot:
                try:
                    saved_path = save_spot_snapshot(spot_snapshot)
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

            print_key_value_block("SPOT SNAPSHOT", {
                "spot": spot_summary.get("spot"),
                "day_open": spot_summary.get("day_open"),
                "day_high": spot_summary.get("day_high"),
                "day_low": spot_summary.get("day_low"),
                "prev_close": spot_summary.get("prev_close"),
                "timestamp": spot_summary.get("timestamp"),
                "lookback_avg_range_pct": spot_summary.get("lookback_avg_range_pct"),
            })

            print_key_value_block("MACRO EVENT RISK", {
                "macro_event_risk_score": macro_event_state.get("macro_event_risk_score"),
                "event_window_status": macro_event_state.get("event_window_status"),
                "event_lockdown_flag": macro_event_state.get("event_lockdown_flag"),
                "minutes_to_next_event": macro_event_state.get("minutes_to_next_event"),
                "next_event_name": macro_event_state.get("next_event_name"),
            })

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
            print_key_value_block("GLOBAL RISK STATE", {
                "global_risk_state": global_risk_state.get("global_risk_state"),
                "global_risk_score": global_risk_state.get("global_risk_score"),
                "overnight_gap_risk_score": global_risk_state.get("overnight_gap_risk_score"),
                "volatility_expansion_risk_score": global_risk_state.get("volatility_expansion_risk_score"),
                "overnight_hold_allowed": global_risk_state.get("overnight_hold_allowed"),
                "overnight_hold_reason": global_risk_state.get("overnight_hold_reason"),
                "overnight_risk_penalty": global_risk_state.get("overnight_risk_penalty"),
                "global_risk_adjustment_score": global_risk_state.get("global_risk_adjustment_score"),
                "global_risk_reasons": global_risk_state.get("global_risk_reasons"),
            })
            print_key_value_block("GLOBAL MARKET SNAPSHOT", {
                "provider": global_market_snapshot.get("provider"),
                "data_available": global_market_snapshot.get("data_available"),
                "stale": global_market_snapshot.get("stale"),
                "oil_change_24h": global_market_snapshot.get("market_inputs", {}).get("oil_change_24h"),
                "vix_change_24h": global_market_snapshot.get("market_inputs", {}).get("vix_change_24h"),
                "sp500_change_24h": global_market_snapshot.get("market_inputs", {}).get("sp500_change_24h"),
                "us10y_change_bp": global_market_snapshot.get("market_inputs", {}).get("us10y_change_bp"),
                "usdinr_change_24h": global_market_snapshot.get("market_inputs", {}).get("usdinr_change_24h"),
                "warnings": global_market_snapshot.get("warnings"),
            })

            macro_news_details = {
                "headline_provider": headline_state.get("provider_name"),
                "headline_data_available": headline_state.get("data_available"),
                "headline_is_stale": headline_state.get("is_stale"),
                "headline_warnings": headline_state.get("warnings"),
                "headline_issues": headline_state.get("issues"),
                "provider_metadata": headline_state.get("provider_metadata"),
                "classification_preview": macro_news_state.get("classification_preview"),
            }
            if (
                headline_state.get("warnings")
                or headline_state.get("issues")
                or headline_state.get("provider_metadata")
                or macro_news_state.get("classification_preview")
            ):
                print_key_value_block(
                    "MACRO / NEWS DETAILS",
                    {key: _format_output_value(value) for key, value in macro_news_details.items()},
                )

            print_validation_block("OPTION CHAIN VALIDATION", option_chain_validation)

            if not option_chain_validation.get("is_valid", False):
                print("\nOption chain invalid. Skipping this cycle.")
                if args.replay:
                    break
                time.sleep(refresh_interval)
                continue

            print(f"\n{'option_chain_rows':26}: {result.get('option_chain_rows')}")

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

            if trade:
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

                print_trader_view(trade)

                dashboard_for_print = dict(trade)
                dashboard_for_print.update({
                    "spot": spot_summary.get("spot"),
                    "spot_timestamp": spot_summary.get("timestamp"),
                    "day_open": spot_summary.get("day_open"),
                    "day_high": spot_summary.get("day_high"),
                    "day_low": spot_summary.get("day_low"),
                    "prev_close": spot_summary.get("prev_close"),
                    "lookback_avg_range_pct": spot_summary.get("lookback_avg_range_pct"),
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

            previous_chain = option_chain_frame.copy()

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
