"""
Module: dealer_hedging_pressure_scenario_runner.py

Purpose:
    Implement dealer hedging pressure scenario runner logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""

from __future__ import annotations

from pathlib import Path

from backtest.scenario_utils import find_named_scenario, load_scenarios
from risk.dealer_hedging_pressure_layer import build_dealer_hedging_pressure_state


def load_dealer_pressure_scenarios(path: str | Path = "config/dealer_hedging_pressure_scenarios.json") -> list[dict]:
    """
    Purpose:
        Process load dealer pressure scenarios for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        list[dict]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return load_scenarios(path)


def run_dealer_pressure_scenario(name: str, path: str | Path = "config/dealer_hedging_pressure_scenarios.json") -> dict:
    """
    Purpose:
        Process run dealer pressure scenario for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        name (str): Input associated with name.
        path (str | Path): Input associated with path.
    
    Returns:
        dict: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    scenario = find_named_scenario(name, path=path)
    state = build_dealer_hedging_pressure_state(**scenario.get("inputs", {}))
    return {
        "scenario": scenario.get("name"),
        "inputs": scenario.get("inputs", {}),
        "expected": scenario.get("expected", {}),
        "dealer_pressure_state": state,
    }
