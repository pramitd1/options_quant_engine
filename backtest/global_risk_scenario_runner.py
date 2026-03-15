"""
Scenario runner for validating the global risk layer using local fixtures only.

Usage:
    python -m backtest.global_risk_scenario_runner
    python -m backtest.global_risk_scenario_runner --scenario volatility_compression_breakout
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
    path = Path(path or (Path(BASE_DIR) / "config/global_risk_scenarios.json"))
    return load_scenarios(path, list_key="scenarios")


def run_scenario(scenario: dict):
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
