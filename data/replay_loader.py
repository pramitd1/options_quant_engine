"""
Module: replay_loader.py

Purpose:
    Load replay artifacts for replay or research workflows.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.spot_downloader import validate_spot_snapshot


def load_spot_snapshot(path: str) -> dict:
    """
    Purpose:
        Process load spot snapshot for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str): Input associated with path.
    
    Returns:
        dict: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    snapshot_path = Path(path)
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    # Recompute freshness/validation at load time, but distinguish between
    # live-trading freshness and replay-analysis usability.
    snapshot["validation"] = validate_spot_snapshot(snapshot, replay_mode=True)
    return snapshot


def load_option_chain_snapshot(path: str) -> pd.DataFrame:
    """
    Purpose:
        Process load option chain snapshot for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str): Input associated with path.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    snapshot_path = Path(path)
    suffix = snapshot_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(snapshot_path)

    if suffix == ".json":
        with open(snapshot_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "rows" in payload:
            return pd.DataFrame(payload["rows"])
        return pd.DataFrame(payload)

    raise ValueError(f"Unsupported replay option-chain file type: {snapshot_path.suffix}")


def save_option_chain_snapshot(
    option_chain: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    output_dir: str = "debug_samples",
) -> str:
    """
    Purpose:
        Process save option chain snapshot for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (pd.DataFrame): Input associated with option chain.
        symbol (str): Underlying symbol or index identifier.
        source (str): Data-source label associated with the current snapshot.
        output_dir (str): Input associated with output dir.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol = str(symbol or "UNKNOWN").upper().strip()
    source = str(source or "UNKNOWN").upper().strip()
    timestamp = pd.Timestamp.now(tz="Asia/Kolkata").isoformat().replace(":", "-")
    filename = out_dir / f"{symbol}_{source}_option_chain_snapshot_{timestamp}.csv"
    option_chain.to_csv(filename, index=False)
    return str(filename)


def latest_replay_snapshot_paths(symbol: str, replay_dir: str = "debug_samples") -> tuple[str | None, str | None]:
    """
    Purpose:
        Process latest replay snapshot paths for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        replay_dir (str): Input associated with replay dir.
    
    Returns:
        tuple[str | None, str | None]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    directory = Path(replay_dir)
    if not directory.exists():
        return None, None

    symbol = str(symbol or "").upper().strip()
    spot_candidates = sorted(directory.glob(f"{symbol}_spot_snapshot_*.json"))
    chain_candidates = sorted(directory.glob(f"{symbol}_*_option_chain_snapshot_*.csv"))

    spot_path = str(spot_candidates[-1]) if spot_candidates else None
    chain_path = str(chain_candidates[-1]) if chain_candidates else None
    return spot_path, chain_path
