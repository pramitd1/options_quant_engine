"""
Module: replay_regression.py

Purpose:
    Implement replay regression logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.engine_runner import run_preloaded_engine_snapshot
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot


def _extract_ts_from_filename(path: Path, marker: str):
    """
    Purpose:
        Extract timestamp from filename from the supplied payload.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
        marker (str): Input associated with marker.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    name = path.name
    try:
        start = name.index(marker) + len(marker)
        end = name.rindex(path.suffix)
    except ValueError:
        return None
    raw = name[start:end]
    return raw


def _parse_embedded_timestamp(path: Path, marker: str):
    """
    Purpose:
        Process parse embedded timestamp for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        path (Path): Input associated with path.
        marker (str): Input associated with marker.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    raw = _extract_ts_from_filename(path, marker)
    if raw is None:
        return None
    try:
        date_part, time_part = raw.split("T", 1)
        if "+" in time_part:
            time_core, tz_part = time_part.split("+", 1)
            tz_sign = "+"
        elif "-" in time_part[1:]:
            time_core, tz_part = time_part.rsplit("-", 1)
            tz_sign = "-"
        else:
            time_core, tz_part, tz_sign = time_part, None, ""

        time_core = time_core.replace("-", ":")
        if tz_part is not None:
            tz_part = tz_part.replace("-", ":", 1)
            normalized = f"{date_part}T{time_core}{tz_sign}{tz_part}"
        else:
            normalized = f"{date_part}T{time_core}"
        return __import__("pandas").Timestamp(normalized)
    except Exception:
        return None


def _find_spot_snapshots(symbol: str, replay_dir: str):
    """
    Purpose:
        Process find spot snapshots for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        replay_dir (str): Input associated with replay dir.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    directory = Path(replay_dir)
    return sorted(directory.glob(f"{symbol.upper()}_spot_snapshot_*.json"))


def _replay_global_market_snapshot(symbol: str, spot_snapshot: dict) -> dict:
    """
    Purpose:
        Provide a neutral global-market snapshot for offline replay analysis.

    Context:
        Internal helper used when archived spot and option-chain snapshots are
        re-evaluated through the shared runtime orchestration path.

    Inputs:
        symbol (str): Underlying symbol or index identifier.
        spot_snapshot (dict): Spot snapshot associated with the replay case.

    Returns:
        dict: Neutral global-market snapshot shaped like the live payload.

    Notes:
        Replay regression is intended to measure signal behavior from archived
        inputs, so it avoids blending in present-day cross-asset market data.
    """

    return {
        "symbol": str(symbol or "").upper().strip(),
        "provider": "REPLAY_NEUTRAL",
        "as_of": spot_snapshot.get("timestamp"),
        "data_available": False,
        "neutral_fallback": True,
        "issues": [],
        "warnings": ["replay_global_market_snapshot_unavailable"],
        "stale": False,
        "lookback_days": None,
        "market_inputs": {},
    }


def _find_chain_snapshots(symbol: str, source: str, replay_dir: str):
    """
    Purpose:
        Process find chain snapshots for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        source (str): Data-source label associated with the current snapshot.
        replay_dir (str): Input associated with replay dir.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    directory = Path(replay_dir)
    source = source.upper().strip()
    return sorted(directory.glob(f"{symbol.upper()}_{source}_option_chain_snapshot_*.csv"))


def _nearest_spot_snapshot(chain_path: Path, spot_paths: list[Path]):
    """
    Purpose:
        Process nearest spot snapshot for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        chain_path (Path): Input associated with chain path.
        spot_paths (list[Path]): Input associated with spot paths.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not spot_paths:
        return None

    chain_ts = _parse_embedded_timestamp(chain_path, "_option_chain_snapshot_")
    if chain_ts is None:
        return spot_paths[-1]

    ranked = []
    for spot_path in spot_paths:
        spot_ts = _parse_embedded_timestamp(spot_path, "_spot_snapshot_")
        if spot_ts is None:
            continue
        ranked.append((abs((chain_ts - spot_ts).total_seconds()), spot_path))

    if ranked:
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    return spot_paths[-1]


def _count_bucket(trade):
    """
    Purpose:
        Process count bucket for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        trade (Any): Input associated with trade.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not trade:
        return "NO_SIGNAL"

    if trade.get("trade_status") != "TRADE":
        return "NO_SIGNAL"

    direction = str(trade.get("direction") or "").upper()
    if direction == "CALL":
        return "CALL"
    if direction == "PUT":
        return "PUT"
    return "NO_SIGNAL"


def run_regression(symbol: str, source: str, replay_dir: str, limit: int | None = None):
    """
    Purpose:
        Process run regression for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        source (str): Data-source label associated with the current snapshot.
        replay_dir (str): Input associated with replay dir.
        limit (int | None): Input associated with limit.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    spot_paths = _find_spot_snapshots(symbol, replay_dir)
    chain_paths = _find_chain_snapshots(symbol, source, replay_dir)

    if not chain_paths:
        raise ValueError(f"No replay option-chain snapshots found for {symbol}/{source} in {replay_dir}")

    if limit is not None:
        chain_paths = chain_paths[-limit:]

    results = []
    bucket_counts = Counter()
    source_counts = Counter()

    for chain_path in chain_paths:
        spot_path = _nearest_spot_snapshot(chain_path, spot_paths)
        if spot_path is None:
            continue

        spot_snapshot = load_spot_snapshot(str(spot_path))
        option_chain = load_option_chain_snapshot(str(chain_path))
        resolved_expiry = resolve_selected_expiry(option_chain)
        option_chain = filter_option_chain_by_expiry(option_chain, resolved_expiry)

        signal_result = run_preloaded_engine_snapshot(
            symbol=symbol.upper().strip(),
            mode="REPLAY",
            source=source.upper().strip(),
            spot_snapshot=spot_snapshot,
            option_chain=option_chain,
            previous_chain=None,
            apply_budget_constraint=False,
            requested_lots=1,
            lot_size=1,
            max_capital=float("inf"),
            capture_signal_evaluation=False,
            enable_shadow_logging=False,
            global_market_snapshot=_replay_global_market_snapshot(symbol, spot_snapshot),
        )
        if not signal_result.get("ok", False):
            raise ValueError(signal_result.get("error") or "Replay snapshot evaluation failed")

        trade = signal_result.get("execution_trade") or signal_result.get("trade")

        bucket = _count_bucket(trade)
        bucket_counts[bucket] += 1
        if trade and trade.get("direction_source"):
            source_counts[trade["direction_source"]] += 1

        results.append({
            "spot_snapshot": str(spot_path),
            "chain_snapshot": str(chain_path),
            "bucket": bucket,
            "direction": trade.get("direction") if trade else None,
            "trade_status": trade.get("trade_status") if trade else None,
            "direction_source": trade.get("direction_source") if trade else None,
            "trade_strength": trade.get("trade_strength") if trade else None,
            "selected_expiry": trade.get("selected_expiry") if trade else None,
        })

    return results, bucket_counts, source_counts


def main():
    """
    Purpose:
        Run the module entry point for command-line or operational execution.

    Context:
        Function inside the `replay regression` module. The module sits in the backtest layer that replays historical scenarios and scores realized performance.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Exit status or workflow result returned by the implementation.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Underlying symbol, for example NIFTY or RELIANCE")
    parser.add_argument("--source", default="ICICI", help="Snapshot source label, for example ICICI or NSE")
    parser.add_argument("--replay-dir", default="debug_samples", help="Directory containing saved replay snapshots")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of latest option-chain snapshots to evaluate")
    args = parser.parse_args()

    results, bucket_counts, source_counts = run_regression(
        symbol=args.symbol,
        source=args.source,
        replay_dir=args.replay_dir,
        limit=args.limit,
    )

    print("\nREPLAY REGRESSION")
    print("---------------------------")
    print(f"{'symbol':18}: {args.symbol.upper().strip()}")
    print(f"{'source':18}: {args.source.upper().strip()}")
    print(f"{'cases':18}: {len(results)}")
    print(f"{'call_count':18}: {bucket_counts.get('CALL', 0)}")
    print(f"{'put_count':18}: {bucket_counts.get('PUT', 0)}")
    print(f"{'no_signal_count':18}: {bucket_counts.get('NO_SIGNAL', 0)}")

    if results:
        total = max(len(results), 1)
        print(f"{'call_ratio':18}: {round(bucket_counts.get('CALL', 0) / total, 3)}")
        print(f"{'put_ratio':18}: {round(bucket_counts.get('PUT', 0) / total, 3)}")
        print(f"{'no_signal_ratio':18}: {round(bucket_counts.get('NO_SIGNAL', 0) / total, 3)}")

    if source_counts:
        print("\nDIRECTION SOURCES")
        print("---------------------------")
        for key, value in source_counts.most_common():
            print(f"{key:24}: {value}")

    if results:
        print("\nLATEST CASES")
        print("---------------------------")
        for item in results[-5:]:
            print(
                f"{item['bucket']:10} | {item['trade_status'] or 'NONE':10} | "
                f"{item['direction_source'] or 'NONE':24} | {item['chain_snapshot']}"
            )


if __name__ == "__main__":
    main()
