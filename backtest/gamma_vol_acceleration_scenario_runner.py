"""
Module: gamma_vol_acceleration_scenario_runner.py

Purpose:
    Implement gamma vol acceleration scenario runner logic used by historical replay and backtest evaluation.

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
from risk.gamma_vol_acceleration_layer import build_gamma_vol_acceleration_state


def load_gamma_vol_scenarios(path: str | Path = "config/gamma_vol_acceleration_scenarios.json") -> list[dict]:
    """
    Purpose:
        Process load gamma vol scenarios for downstream use.
    
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


def run_gamma_vol_scenario(name: str, path: str | Path = "config/gamma_vol_acceleration_scenarios.json") -> dict:
    """
    Purpose:
        Process run gamma vol scenario for downstream use.
    
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
    state = build_gamma_vol_acceleration_state(**scenario.get("inputs", {}))
    return {
        "scenario": scenario.get("name"),
        "inputs": scenario.get("inputs", {}),
        "expected": scenario.get("expected", {}),
        "gamma_vol_state": state,
    }
