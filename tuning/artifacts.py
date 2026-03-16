"""
Module: artifacts.py

Purpose:
    Implement artifacts utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def append_jsonl_record(record: dict[str, Any], path: str | Path) -> Path:
    """
    Purpose:
        Process append jsonl record for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        record (dict[str, Any]): Input associated with record.
        path (str | Path): Input associated with path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record or {}), sort_keys=True))
        handle.write("\n")
    return ledger_path


def load_jsonl_frame(path: str | Path) -> pd.DataFrame:
    """
    Purpose:
        Process load jsonl frame for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    ledger_path = Path(path)
    if not ledger_path.exists():
        return pd.DataFrame()

    records = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)
