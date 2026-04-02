#!/usr/bin/env python3
"""Replay sanity check for time-decay activation across consecutive snapshots."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.option_chain_validation import validate_option_chain
from data.spot_downloader import validate_spot_snapshot
from engine.signal_engine import generate_trade


def _safe(v, default=None):
    return default if v is None else v


def main() -> int:
    spot_path = "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-16T12-35-00+05-30.json"
    chain_path = "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-16T12-34-24.280934+05-30.csv"

    with open(spot_path, "r", encoding="utf-8") as f:
        spot_snapshot = json.load(f)

    option_chain = pd.read_csv(chain_path)
    spot = float(spot_snapshot.get("spot"))
    spot_ts = pd.Timestamp(spot_snapshot.get("timestamp"))

    spot_validation = spot_snapshot.get("validation") or validate_spot_snapshot(
        spot_snapshot,
        replay_mode=True,
    )
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    base_kwargs = dict(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_validation,
        option_chain_validation=option_chain_validation,
        backtest_mode=False,
    )

    trade_1 = generate_trade(**base_kwargs, valuation_time=spot_ts)
    trade_2 = generate_trade(**base_kwargs, valuation_time=spot_ts + timedelta(minutes=5))

    for i, t in enumerate([trade_1, trade_2], start=1):
        if not isinstance(t, dict):
            print(f"SNAPSHOT_{i}: payload_missing")
            continue
        print(
            f"SNAPSHOT_{i}: "
            f"status={t.get('trade_status')} "
            f"direction={t.get('direction')} "
            f"gamma_regime={t.get('gamma_regime')} "
            f"path_status={t.get('path_aware_status')} "
            f"path_penalty={t.get('path_aware_score_penalty')} "
            f"elapsed_m={t.get('time_decay_elapsed_minutes')} "
            f"decay_factor={t.get('time_decay_factor')} "
            f"runtime_composite={t.get('runtime_composite_score')}"
        )

    m2 = _safe((trade_2 or {}).get("time_decay_elapsed_minutes"), 0.0)
    f2 = _safe((trade_2 or {}).get("time_decay_factor"), 1.0)

    elapsed_nonzero = float(m2) > 0.0
    factor_non_one = float(f2) < 1.0

    print(f"SANITY: elapsed_nonzero_second_snapshot={elapsed_nonzero}")
    print(f"SANITY: factor_non_one_second_snapshot={factor_non_one}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
