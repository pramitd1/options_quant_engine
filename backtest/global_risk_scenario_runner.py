"""
Module: global_risk_scenario_runner.py

Purpose:
    Implement global risk scenario runner logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.scenario_utils import load_scenarios
from config.settings import BASE_DIR
from risk import build_global_risk_state


def _load_scenarios(path: str | None = None):
    """
    Purpose:
        Process load scenarios for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (str | None): Input associated with path.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    path = Path(path or (Path(BASE_DIR) / "config/global_risk_scenarios.json"))
    return load_scenarios(path, list_key="scenarios")


def run_scenario(scenario: dict):
    """
    Purpose:
        Process run scenario for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        scenario (dict): Input associated with scenario.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    global_risk_state = build_global_risk_state(
        macro_event_state=scenario.get("macro_event_state"),
        macro_news_state=scenario.get("macro_news_state"),
        global_market_snapshot=scenario.get("global_market_snapshot"),
        holding_profile=scenario.get("holding_profile", "AUTO"),
        as_of=scenario.get("as_of"),
    )
    features = global_risk_state.get("global_risk_features", {})

    return {
        "name": scenario["name"],
        "symbol": scenario["symbol"],
        "as_of": scenario["as_of"],
        "holding_profile": scenario.get("holding_profile", "AUTO"),
        "global_risk_state": global_risk_state,
        "features": {
            "oil_shock_score": features.get("oil_shock_score"),
            "commodity_risk_score": features.get("commodity_risk_score"),
            "volatility_shock_score": features.get("volatility_shock_score"),
            "volatility_compression_score": features.get("volatility_compression_score"),
            "volatility_explosion_probability": features.get("volatility_explosion_probability"),
            "macro_event_risk_score": features.get("macro_event_risk_score"),
        },
        "expected": scenario.get("expected", {}),
    }


def main():
    """
    Purpose:
        Run the module entry point for command-line or operational execution.

    Context:
        Function inside the `global risk scenario runner` module. The module sits in the backtest layer that replays historical scenarios and scores realized performance.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Exit status or workflow result returned by the implementation.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, help="Optional scenario name from config/global_risk_scenarios.json")
    parser.add_argument("--scenario-file", default=None, help="Optional custom scenario file path")
    args = parser.parse_args()

    scenarios = _load_scenarios(args.scenario_file)
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario.get("name") == args.scenario]
        if not scenarios:
            raise ValueError(f"Scenario not found: {args.scenario}")

    for scenario in scenarios:
        result = run_scenario(scenario)
        state = result["global_risk_state"]
        print("\nSCENARIO")
        print("---------------------------")
        print(f"name                          : {result['name']}")
        print(f"symbol                        : {result['symbol']}")
        print(f"as_of                         : {result['as_of']}")
        print(f"holding_profile               : {result['holding_profile']}")
        print(f"global_risk_state             : {state['global_risk_state']}")
        print(f"global_risk_score             : {state['global_risk_score']}")
        print(f"overnight_hold_allowed        : {state['overnight_hold_allowed']}")
        print(f"overnight_hold_reason         : {state['overnight_hold_reason']}")
        print(f"overnight_risk_penalty        : {state['overnight_risk_penalty']}")
        print(f"oil_shock_score               : {result['features']['oil_shock_score']}")
        print(f"commodity_risk_score          : {result['features']['commodity_risk_score']}")
        print(f"volatility_shock_score        : {result['features']['volatility_shock_score']}")
        print(f"volatility_compression_score  : {result['features']['volatility_compression_score']}")
        print(f"volatility_explosion_prob     : {result['features']['volatility_explosion_probability']}")
        if result["expected"]:
            print(f"expected                      : {result['expected']}")


if __name__ == "__main__":
    main()
