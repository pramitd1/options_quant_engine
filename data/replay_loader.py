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


_NEAREST_INDEX_CACHE: dict[tuple[str, str, str], dict] = {}


def _resolve_snapshot_output_dir(output_dir: str, *, snapshot_kind: str) -> Path:
    base_dir = Path(output_dir)
    if base_dir == Path("debug_samples"):
        return base_dir / snapshot_kind
    return base_dir


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


def validate_snapshot_consistency(
    spot_snapshot: dict | None,
    option_chain: pd.DataFrame | None,
) -> dict:
    """
    Validate that spot and option-chain timestamps are reasonably aligned.
    
    Returns a dict with:
        is_consistent: bool
        warnings: list[str]
    """
    warnings = []
    
    if spot_snapshot is None or option_chain is None:
        return {"is_consistent": True, "warnings": []}
    
    spot_ts = spot_snapshot.get("timestamp")
    spot_dt = None
    if spot_ts is None:
        # Missing timestamp is OK - no data to conflict
        pass
    else:
        try:
            spot_dt = pd.to_datetime(spot_ts) if isinstance(spot_ts, str) else spot_ts
        except Exception:
            warnings.append("spot_snapshot_timestamp_unparseable")
            spot_dt = None
    
    # Compare against true snapshot timestamps only. Expiry metadata is not a
    # chain-capture timestamp and must not be used for temporal alignment checks.
    chain_ts_candidates = [
        option_chain.get("TIMESTAMP"),
        option_chain.get("timestamp"),
        option_chain.get("snapshot_timestamp"),
        option_chain.get("as_of"),
        option_chain.get("trade_timestamp"),
    ]
    chain_ts = next((ts for ts in chain_ts_candidates if ts is not None and not ts.empty), None)
    
    if chain_ts is not None and spot_dt is not None:
        try:
            if isinstance(chain_ts, pd.Series):
                first_chain_dt = pd.to_datetime(chain_ts.iloc[0])
            else:
                first_chain_dt = pd.to_datetime(chain_ts)
            
            # Timestamps should be within 30 seconds (allow slight clock skew)
            delta = abs((spot_dt - first_chain_dt).total_seconds())
            if delta > 30:
                warnings.append(f"spot_chain_timestamp_mismatch_delta_sec_{delta}")
        except Exception:
            pass
    
    is_consistent = len(warnings) == 0
    return {"is_consistent": is_consistent, "warnings": warnings}


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
    out_dir = _resolve_snapshot_output_dir(output_dir, snapshot_kind="option_chain_snapshots")
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


def _extract_spot_timestamp(path: Path) -> pd.Timestamp | None:
    """Parse timestamp token from spot snapshot filename."""
    match = re.search(r"_spot_snapshot_(.+)\.json$", path.name)
    if not match:
        return None
    token = match.group(1)
    token = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", token)
    token = re.sub(r"\+(\d{2})-(\d{2})$", r"+\1:\2", token)
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def list_replay_spot_snapshots(
    symbol: str,
    *,
    replay_dir: str = "debug_samples",
) -> tuple[list[str], list[dict]]:
    """Return valid replay spot snapshot paths and skipped-file diagnostics."""
    directory = Path(replay_dir)
    if not directory.exists():
        return [], []

    symbol_token = str(symbol or "").upper().strip()
    candidates = sorted(directory.rglob(f"{symbol_token}_spot_snapshot_*.json"))
    valid: list[tuple[pd.Timestamp, Path]] = []
    skipped: list[dict] = []

    for path in candidates:
        timestamp = _extract_spot_timestamp(path)
        if timestamp is None:
            skipped.append({"path": str(path), "reason": "bad_timestamp_token"})
            continue
        if path.stat().st_size <= 1:
            skipped.append({"path": str(path), "reason": "empty_file"})
            continue
        valid.append((timestamp, path))

    valid.sort(key=lambda item: item[0])
    return [str(path) for _, path in valid], skipped


def resolve_nearest_replay_snapshot_paths(
    symbol: str,
    *,
    target_timestamp,
    replay_dir: str = "debug_samples",
    source_label: str | None = None,
    max_spot_delta_seconds: float = 600.0,
    max_chain_delta_seconds: float = 900.0,
) -> dict:
    """Resolve nearest valid replay spot/chain snapshots around a target timestamp."""
    target_ts = pd.to_datetime(target_timestamp, errors="coerce")
    if pd.isna(target_ts):
        return {
            "spot_path": None,
            "chain_path": None,
            "spot_delta_seconds": None,
            "chain_delta_seconds": None,
            "selection_reason": "bad_target_timestamp",
            "source_label": str(source_label or "").upper().strip() or None,
        }

    cache_key = (
        str(Path(replay_dir).resolve()),
        str(symbol or "").upper().strip(),
        str(source_label or "").upper().strip(),
    )
    indexed = _NEAREST_INDEX_CACHE.get(cache_key)
    if indexed is None:
        spot_paths, skipped_spot_files = list_replay_spot_snapshots(symbol, replay_dir=replay_dir)
        chain_paths, skipped_chain_files = list_replay_chain_snapshots(
            symbol,
            replay_dir=replay_dir,
            source_label=source_label,
        )
        spot_pairs = []
        for path_str in spot_paths:
            ts = _extract_spot_timestamp(Path(path_str))
            if ts is not None:
                spot_pairs.append((ts, path_str))
        chain_pairs = []
        for path_str in chain_paths:
            ts = _extract_chain_timestamp(Path(path_str))
            if ts is not None:
                chain_pairs.append((ts, path_str))
        indexed = {
            "spot_pairs": spot_pairs,
            "chain_pairs": chain_pairs,
            "skipped_spot_files": skipped_spot_files,
            "skipped_chain_files": skipped_chain_files,
        }
        _NEAREST_INDEX_CACHE[cache_key] = indexed
    else:
        skipped_spot_files = indexed.get("skipped_spot_files", [])
        skipped_chain_files = indexed.get("skipped_chain_files", [])

    best_spot_path = None
    best_chain_path = None
    best_spot_delta = None
    best_chain_delta = None

    for ts, path_str in indexed.get("spot_pairs", []):
        delta = abs((ts - target_ts).total_seconds())
        if best_spot_delta is None or delta < best_spot_delta:
            best_spot_delta = delta
            best_spot_path = path_str

    for ts, path_str in indexed.get("chain_pairs", []):
        delta = abs((ts - target_ts).total_seconds())
        if best_chain_delta is None or delta < best_chain_delta:
            best_chain_delta = delta
            best_chain_path = path_str

    if best_spot_delta is not None and best_spot_delta > max_spot_delta_seconds:
        best_spot_path = None
    if best_chain_delta is not None and best_chain_delta > max_chain_delta_seconds:
        best_chain_path = None

    if best_spot_path or best_chain_path:
        reason = "nearest_snapshot_by_timestamp"
    else:
        reason = "no_snapshot_within_tolerance"

    return {
        "spot_path": best_spot_path,
        "chain_path": best_chain_path,
        "spot_delta_seconds": round(best_spot_delta, 3) if best_spot_delta is not None else None,
        "chain_delta_seconds": round(best_chain_delta, 3) if best_chain_delta is not None else None,
        "selection_reason": reason,
        "source_label": str(source_label or "").upper().strip() or None,
        "skipped_spot_files": skipped_spot_files,
        "skipped_chain_files": skipped_chain_files,
    }


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

    candidates = sorted(directory.rglob(pattern))
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
    spot_candidates = sorted(directory.rglob(f"{symbol_token}_spot_snapshot_*.json"))
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
