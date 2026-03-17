#!/usr/bin/env python3
"""
Regime Simulation Script
========================

Replays saved option-chain + spot snapshots through the full signal engine
while injecting synthetic macro, global-risk, and headline contexts that
force specific regime conditions (RISK_ON, RISK_OFF, NEGATIVE_GAMMA,
VOL_COMPRESSION, etc.).

The output is written to a separate simulation dataset
(research/signal_evaluation/simulation_signals_dataset.csv) so that the
primary signal evaluation dataset remains purely market-driven.

Usage
-----
  # Simulate all built-in regime scenarios using the latest saved snapshot
  python scripts/simulate_regime_signals.py

  # Simulate a specific regime only
  python scripts/simulate_regime_signals.py --regime RISK_ON

  # Use a specific spot and chain snapshot
  python scripts/simulate_regime_signals.py \
      --spot debug_samples/NIFTY_spot_snapshot_2026-03-17T11-35-00+05-30.json \
      --chain debug_samples/NIFTY_ICICI_option_chain_snapshot_2026-03-17T11-36-24.157999+05-30.csv

  # List available regime scenarios
  python scripts/simulate_regime_signals.py --list

  # Use a custom scenario JSON file
  python scripts/simulate_regime_signals.py --scenario-file config/my_scenarios.json

  # Dry run — show what would be simulated without writing to the dataset
  python scripts/simulate_regime_signals.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.engine_runner import run_preloaded_engine_snapshot
from backtest.macro_news_scenario_runner import _build_headline_state
from backtest.scenario_utils import load_scenarios
from config.settings import (
    BASE_DIR,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
    STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT,
)
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot
from macro.scheduled_event_risk import evaluate_scheduled_event_risk
from research.signal_evaluation import should_capture_signal
from research.signal_evaluation.evaluator import save_signal_evaluation


DEFAULT_SCENARIO_FILE = Path(BASE_DIR) / "config" / "regime_simulation_scenarios.json"
SIMULATION_DATASET_PATH = Path(BASE_DIR) / "research" / "signal_evaluation" / "simulation_signals_dataset.csv"


class _SimulationSignalCaptureSink:
    """Captures signals to a separate simulation dataset file."""

    def apply(
        self,
        *,
        result_payload,
        trade,
        capture_signal_evaluation,
        signal_capture_policy,
    ) -> None:
        if capture_signal_evaluation and should_capture_signal(trade, signal_capture_policy):
            try:
                save_signal_evaluation(
                    result_payload,
                    dataset_path=SIMULATION_DATASET_PATH,
                    as_of=(result_payload.get("spot_summary", {}) or {}).get("timestamp"),
                    return_frame=False,
                )
                result_payload["signal_capture_status"] = "CAPTURED"
            except Exception as exc:
                result_payload["signal_capture_status"] = f"FAILED:{type(exc).__name__}"
                result_payload["signal_capture_error"] = str(exc)
        elif capture_signal_evaluation and trade:
            result_payload["signal_capture_status"] = f"SKIPPED_POLICY:{signal_capture_policy}"


def _latest_snapshot_pair(symbol: str = "NIFTY", replay_dir: str = "debug_samples"):
    """Discover the most recent spot + chain snapshot pair for the given symbol."""
    base = Path(BASE_DIR) / replay_dir
    spots = sorted(base.glob(f"{symbol}_spot_snapshot_*.json"))
    # Skip empty chain files (< 100 bytes)
    chains = sorted(
        p for p in base.glob(f"{symbol}_*_option_chain_snapshot_*.csv")
        if p.stat().st_size > 100
    )
    if not spots or not chains:
        return None, None
    return str(spots[-1]), str(chains[-1])


def _load_scenarios_file(path: str | None) -> list[dict]:
    scenario_path = Path(path) if path else DEFAULT_SCENARIO_FILE
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path}")
        sys.exit(1)
    return load_scenarios(scenario_path, list_key="scenarios")


def simulate_regime(
    scenario: dict,
    spot_snapshot: dict,
    option_chain,
    *,
    dry_run: bool = False,
) -> dict | None:
    """Run one regime scenario through the full signal engine pipeline."""

    name = scenario["name"]
    symbol = scenario.get("symbol", spot_snapshot.get("symbol", "NIFTY"))
    as_of = scenario.get("as_of", spot_snapshot.get("timestamp"))

    # Stamp the scenario's as_of into a copy of the spot snapshot so
    # each scenario produces a unique signal_id (the evaluator keys on
    # spot_summary.timestamp).
    sim_spot = dict(spot_snapshot)
    if as_of:
        sim_spot["timestamp"] = as_of

    # Build macro event state from scenario or fresh evaluation
    macro_event_state = scenario.get("macro_event_state")
    if macro_event_state is None:
        macro_event_state = evaluate_scheduled_event_risk(
            symbol=symbol,
            as_of=as_of,
            events=scenario.get("events", []),
            enabled=True,
        )

    # Build headline state from scenario headlines
    headline_state = _build_headline_state(
        scenario.get("headlines", []),
        as_of,
        provider_name="REGIME_SIMULATION",
    )

    # Global market snapshot (cross-asset risk surface)
    global_market_snapshot = scenario.get("global_market_snapshot")

    if dry_run:
        print(f"  [DRY RUN] Would simulate: {name}")
        print(f"            Symbol: {symbol}")
        print(f"            As-of: {as_of}")
        print(f"            Macro regime hint: {scenario.get('macro_event_state', {}).get('macro_event_risk_score', 'default')}")
        print(f"            Global risk hint: {_describe_global_snapshot(global_market_snapshot)}")
        return None

    result = run_preloaded_engine_snapshot(
        symbol=symbol,
        mode="SIMULATION",
        source="REGIME_SIMULATION",
        spot_snapshot=sim_spot,
        option_chain=option_chain,
        apply_budget_constraint=False,
        requested_lots=NUMBER_OF_LOTS,
        lot_size=LOT_SIZE,
        max_capital=MAX_CAPITAL_PER_TRADE,
        capture_signal_evaluation=True,
        signal_capture_policy="ALL_SIGNALS",
        signal_capture_sink=_SimulationSignalCaptureSink(),
        holding_profile=scenario.get("holding_profile", "AUTO"),
        macro_event_state=macro_event_state,
        headline_state=headline_state,
        global_market_snapshot=global_market_snapshot,
        target_profit_percent=TARGET_PROFIT_PERCENT,
        stop_loss_percent=STOP_LOSS_PERCENT,
    )

    trade = result.get("trade", {})
    return {
        "scenario": name,
        "ok": result.get("ok", False),
        "trade_status": trade.get("trade_status", "UNKNOWN"),
        "direction": trade.get("direction"),
        "macro_regime": trade.get("macro_regime"),
        "global_risk_state": trade.get("global_risk_state"),
        "gamma_regime": trade.get("gamma_regime"),
        "volatility_regime": trade.get("volatility_regime"),
        "trade_strength": trade.get("trade_strength"),
        "signal_quality": trade.get("signal_quality"),
        "signal_capture_status": result.get("signal_capture_status"),
    }


def _describe_global_snapshot(snapshot: dict | None) -> str:
    if snapshot is None:
        return "none (use live)"
    inputs = snapshot.get("market_inputs", {})
    parts = []
    for key in ("oil_change_24h", "vix_change_24h", "sp500_change_24h"):
        val = inputs.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return ", ".join(parts) if parts else "custom"


def main():
    parser = argparse.ArgumentParser(
        description="Simulate signal engine behavior under synthetic regime conditions",
    )
    parser.add_argument("--regime", default=None, help="Run only the named regime scenario")
    parser.add_argument("--spot", default=None, help="Path to spot snapshot JSON")
    parser.add_argument("--chain", default=None, help="Path to option chain CSV")
    parser.add_argument("--scenario-file", default=None, help="Custom scenario JSON file")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol to simulate (default: NIFTY)")
    parser.add_argument("--list", action="store_true", help="List available regime scenarios and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be simulated without executing")
    args = parser.parse_args()

    scenarios = _load_scenarios_file(args.scenario_file)

    if args.list:
        print(f"\nAvailable regime scenarios ({len(scenarios)}):")
        print("-" * 50)
        for s in scenarios:
            desc = s.get("description", "")
            print(f"  {s['name']:<35} {desc}")
        return

    if args.regime:
        scenarios = [s for s in scenarios if s["name"] == args.regime]
        if not scenarios:
            print(f"Regime scenario not found: {args.regime}")
            sys.exit(1)

    # Resolve snapshot pair
    spot_path = args.spot
    chain_path = args.chain
    if not spot_path or not chain_path:
        auto_spot, auto_chain = _latest_snapshot_pair(args.symbol)
        spot_path = spot_path or auto_spot
        chain_path = chain_path or auto_chain
    if not spot_path or not chain_path:
        print(f"No saved snapshots found for {args.symbol} in debug_samples/")
        print("Run the engine once with --save-snapshots, or specify --spot and --chain paths.")
        sys.exit(1)

    print(f"\nRegime Simulation")
    print(f"{'=' * 60}")
    print(f"Spot snapshot  : {Path(spot_path).name}")
    print(f"Chain snapshot : {Path(chain_path).name}")
    print(f"Scenarios      : {len(scenarios)}")
    print(f"{'=' * 60}")

    spot_snapshot = load_spot_snapshot(spot_path)
    option_chain = load_option_chain_snapshot(chain_path)

    results = []
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        result = simulate_regime(
            scenario,
            spot_snapshot,
            option_chain,
            dry_run=args.dry_run,
        )
        if result is not None:
            results.append(result)
            print(f"  Status: {result['trade_status']}")
            print(f"  Macro Regime: {result['macro_regime']}")
            print(f"  Global Risk: {result['global_risk_state']}")
            print(f"  Gamma Regime: {result['gamma_regime']}")
            print(f"  Vol Regime: {result['volatility_regime']}")
            print(f"  Trade Strength: {result['trade_strength']}")
            print(f"  Signal Capture: {result['signal_capture_status']}")

    if results:
        print(f"\n{'=' * 60}")
        print(f"Simulation complete: {len(results)} scenarios evaluated")
        captured = sum(1 for r in results if r.get("signal_capture_status", "").startswith("CAPTURED"))
        print(f"Signals captured to dataset: {captured}")
        print(f"Dataset: research/signal_evaluation/simulation_signals_dataset.csv")


if __name__ == "__main__":
    main()
