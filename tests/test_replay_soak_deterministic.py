import re
from pathlib import Path

import pandas as pd
import pytest

from data.option_chain_validation import validate_option_chain
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot
from engine import signal_engine as se
from engine.signal_engine import generate_trade


DEBUG_SAMPLES_DIR = Path("debug_samples")


def _parse_spot_timestamp(path: Path) -> pd.Timestamp | None:
    match = re.search(r"_spot_snapshot_(.+)\.json$", path.name)
    if not match:
        return None
    token = match.group(1)
    # Filenames encode ":" as "-".
    token = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", token)
    token = re.sub(r"\+(\d{2})-(\d{2})$", r"+\1:\2", token)
    parsed = pd.to_datetime(token, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _parse_chain_timestamp(path: Path) -> pd.Timestamp | None:
    match = re.search(r"_option_chain_snapshot_(.+)\.csv$", path.name)
    if not match:
        return None
    token = match.group(1)
    token = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", token)
    token = re.sub(r"\+(\d{2})-(\d{2})$", r"+\1:\2", token)
    parsed = pd.to_datetime(token, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _build_replay_pairs(*, symbol: str = "NIFTY", source: str = "ICICI", max_pairs: int = 12) -> list[tuple[Path, Path]]:
    spot_files = sorted(DEBUG_SAMPLES_DIR.glob(f"{symbol}_spot_snapshot_*.json"))
    chain_files = sorted(DEBUG_SAMPLES_DIR.glob(f"{symbol}_{source}_option_chain_snapshot_*.csv"))

    spot_index = []
    for path in spot_files:
        ts = _parse_spot_timestamp(path)
        if ts is not None:
            spot_index.append((ts, path))

    pairs: list[tuple[Path, Path, pd.Timestamp]] = []
    for chain_path in chain_files:
        chain_ts = _parse_chain_timestamp(chain_path)
        if chain_ts is None:
            continue

        best_spot = None
        best_delta = None
        for spot_ts, spot_path in spot_index:
            delta = abs((chain_ts - spot_ts).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_spot = spot_path

        # Keep the pair only when a close-in-time spot snapshot exists.
        if best_spot is not None and best_delta is not None and best_delta <= 10 * 60:
            pairs.append((best_spot, chain_path, chain_ts))

    pairs.sort(key=lambda item: item[2])
    return [(spot_path, chain_path) for spot_path, chain_path, _ in pairs[:max_pairs]]


def _trade_signature(trade: dict | None) -> tuple:
    if not isinstance(trade, dict):
        return (None, None, None, None, None, None)
    return (
        trade.get("trade_status"),
        trade.get("direction"),
        trade.get("runtime_composite_score"),
        trade.get("trade_strength"),
        trade.get("path_aware_status"),
        trade.get("time_decay_factor"),
    )


def _run_replay_pass(pairs: list[tuple[Path, Path]]) -> list[tuple]:
    se._DECAY_SIGNAL_STATE.clear()
    se._PATH_SIGNAL_STATE.clear()

    previous_chain = None
    previous_direction = None
    signatures: list[tuple] = []

    for spot_path, chain_path in pairs:
        spot_snapshot = load_spot_snapshot(str(spot_path))
        option_chain = load_option_chain_snapshot(str(chain_path))
        spot = float(spot_snapshot.get("spot"))

        spot_validation = spot_snapshot.get("validation") or {}
        option_chain_validation = validate_option_chain(option_chain, spot=spot)

        trade = generate_trade(
            symbol="NIFTY",
            spot=spot,
            option_chain=option_chain,
            previous_chain=previous_chain,
            previous_direction=previous_direction,
            day_high=spot_snapshot.get("day_high"),
            day_low=spot_snapshot.get("day_low"),
            day_open=spot_snapshot.get("day_open"),
            prev_close=spot_snapshot.get("prev_close"),
            lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
            spot_validation=spot_validation,
            option_chain_validation=option_chain_validation,
            backtest_mode=True,
            valuation_time=spot_snapshot.get("timestamp"),
        )

        signatures.append(_trade_signature(trade))
        previous_chain = option_chain
        previous_direction = trade.get("direction") if isinstance(trade, dict) else None

    return signatures


@pytest.mark.replay_soak
def test_deterministic_replay_soak_signatures_are_stable():
    pairs = _build_replay_pairs(max_pairs=12)
    if len(pairs) < 8:
        pytest.skip("Need at least 8 replay pairs in debug_samples for soak coverage")

    first_pass = _run_replay_pass(pairs)
    second_pass = _run_replay_pass(pairs)

    assert len(first_pass) == len(pairs)
    assert first_pass == second_pass
