"""
Module: snapshot_context.py

Purpose:
    Define the SnapshotContext dataclass that bundles the common fields shared
    across engine evaluation, shadow comparison, and runtime sink calls.

Role in the System:
    Part of the application layer. Reduces function signatures from 20–30
    keyword arguments to a single context object, making the orchestration
    code easier to read, refactor, and extend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class SnapshotContext:
    """Bundle of validated market inputs shared across engine evaluations."""

    symbol: str
    mode: str
    source: str
    spot: float
    option_chain: pd.DataFrame
    spot_validation: dict
    option_chain_validation: dict
    macro_event_state: dict
    global_market_snapshot: dict
    holding_profile: str

    # Spot scalars
    day_open: Any = None
    day_high: Any = None
    day_low: Any = None
    prev_close: Any = None
    spot_timestamp: Any = None
    lookback_avg_range_pct: Any = None

    # Full snapshots
    spot_snapshot: dict = field(default_factory=dict)
    headline_state: Any = None

    # Optional chain context
    previous_chain: Optional[pd.DataFrame] = None

    # Sizing
    apply_budget_constraint: bool = False
    requested_lots: int = 1
    lot_size: int = 1
    max_capital: float = 50000.0

    # Mode flags
    backtest_mode: bool = False
    target_profit_percent: float = 30.0
    stop_loss_percent: float = 15.0

    # Resolved artifacts
    resolved_expiry: Any = None
    option_chain_frame: Optional[pd.DataFrame] = None
