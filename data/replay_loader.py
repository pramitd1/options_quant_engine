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
import re
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

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
        try:
            return pd.read_csv(snapshot_path)
        except EmptyDataError as exc:
            raise ValueError(f"Replay option-chain snapshot is empty or malformed: {snapshot_path}") from exc

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
    selection = resolve_replay_snapshot_paths(symbol, replay_dir=replay_dir)
    return selection["spot_path"], selection["chain_path"]


def _extract_chain_timestamp(path: Path) -> pd.Timestamp | None:
    """Parse timestamp token from option-chain snapshot filename."""
    match = re.search(r"_option_chain_snapshot_(.+)\.csv$", path.name)
    if not match:
        return None
    token = match.group(1)
    # Filenames store timestamp separators as hyphens, e.g.
    # 2026-03-20T12-10-00+05-30 -> 2026-03-20T12:10:00+05:30
    token = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", token)
    token = re.sub(r"\+(\d{2})-(\d{2})$", r"+\1:\2", token)
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def list_replay_chain_snapshots(
    symbol: str,
    *,
    replay_dir: str = "debug_samples",
    source_label: str | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Return valid replay chain snapshot paths and skipped-file diagnostics.

    Invalid files (empty, malformed, or with unparsable timestamp token) are
    skipped so callers can safely offer only loadable snapshot choices.
    """
    directory = Path(replay_dir)
    if not directory.exists():
        return [], []

    symbol_token = str(symbol or "").upper().strip()
    source_token = str(source_label or "").upper().strip()
    pattern = (
        f"{symbol_token}_{source_token}_option_chain_snapshot_*.csv"
        if source_token
        else f"{symbol_token}_*_option_chain_snapshot_*.csv"
    )

    candidates = sorted(directory.glob(pattern))
    valid: list[tuple[pd.Timestamp, Path]] = []
    skipped: list[dict] = []

    for path in candidates:
        timestamp = _extract_chain_timestamp(path)
        if timestamp is None:
            skipped.append({"path": str(path), "reason": "bad_timestamp_token"})
            continue

        if path.stat().st_size <= 1:
            skipped.append({"path": str(path), "reason": "empty_file"})
            continue

        try:
            probe = pd.read_csv(path, nrows=1)
            if len(probe.columns) == 0:
                skipped.append({"path": str(path), "reason": "no_columns"})
                continue
        except EmptyDataError:
            skipped.append({"path": str(path), "reason": "empty_file"})
            continue
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"unreadable:{type(exc).__name__}"})
            continue

        valid.append((timestamp, path))

    valid.sort(key=lambda item: item[0])
    return [str(path) for _, path in valid], skipped


def resolve_replay_snapshot_paths(
    symbol: str,
    *,
    replay_dir: str = "debug_samples",
    source_label: str | None = None,
) -> dict:
    """Resolve replay spot/chain paths with source-aware, valid-file selection."""
    directory = Path(replay_dir)
    if not directory.exists():
        return {
            "spot_path": None,
            "chain_path": None,
            "selection_reason": "replay_dir_missing",
            "source_label": str(source_label or "").upper().strip() or None,
            "skipped_chain_files": [],
        }

    symbol_token = str(symbol or "").upper().strip()
    spot_candidates = sorted(directory.glob(f"{symbol_token}_spot_snapshot_*.json"))
    spot_path = str(spot_candidates[-1]) if spot_candidates else None

    chain_paths, skipped_chain_files = list_replay_chain_snapshots(
        symbol,
        replay_dir=replay_dir,
        source_label=source_label,
    )
    chain_path = chain_paths[-1] if chain_paths else None

    if chain_path:
        reason = "latest_valid_for_source" if source_label else "latest_valid_any_source"
    else:
        reason = "no_valid_chain_snapshot"

    return {
        "spot_path": spot_path,
        "chain_path": chain_path,
        "selection_reason": reason,
        "source_label": str(source_label or "").upper().strip() or None,
        "skipped_chain_files": skipped_chain_files,
    }
