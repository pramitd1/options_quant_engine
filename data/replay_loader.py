"""
Replay snapshot helpers for after-hours engine validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.spot_downloader import validate_spot_snapshot


def load_spot_snapshot(path: str) -> dict:
    snapshot_path = Path(path)
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    # Recompute freshness/validation at load time, but distinguish between
    # live-trading freshness and replay-analysis usability.
    snapshot["validation"] = validate_spot_snapshot(snapshot, replay_mode=True)
    return snapshot


def load_option_chain_snapshot(path: str) -> pd.DataFrame:
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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol = str(symbol or "UNKNOWN").upper().strip()
    source = str(source or "UNKNOWN").upper().strip()
    timestamp = pd.Timestamp.now(tz="Asia/Kolkata").isoformat().replace(":", "-")
    filename = out_dir / f"{symbol}_{source}_option_chain_snapshot_{timestamp}.csv"
    option_chain.to_csv(filename, index=False)
    return str(filename)


def latest_replay_snapshot_paths(symbol: str, replay_dir: str = "debug_samples") -> tuple[str | None, str | None]:
    directory = Path(replay_dir)
    if not directory.exists():
        return None, None

    symbol = str(symbol or "").upper().strip()
    spot_candidates = sorted(directory.glob(f"{symbol}_spot_snapshot_*.json"))
    chain_candidates = sorted(directory.glob(f"{symbol}_*_option_chain_snapshot_*.csv"))

    spot_path = str(spot_candidates[-1]) if spot_candidates else None
    chain_path = str(chain_candidates[-1]) if chain_candidates else None
    return spot_path, chain_path
