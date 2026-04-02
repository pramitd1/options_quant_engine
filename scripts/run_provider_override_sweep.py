import argparse
import glob
import json
import os
import re
from datetime import datetime

import config.signal_policy as sp
from app.engine_runner import run_engine_snapshot

SPOT_RE = re.compile(r"spot_snapshot_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")
CHAIN_RE = re.compile(r"option_chain_snapshot_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


def parse_ts(path, rx):
    match = rx.search(path)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")


def build_pairs(limit=25):
    spots = []
    for path in glob.glob("debug_samples/**/*spot_snapshot*.json", recursive=True):
        ts = parse_ts(path, SPOT_RE)
        if ts is not None:
            spots.append((ts, path))
    spots.sort()

    chains = []
    for path in glob.glob("debug_samples/**/*option_chain_snapshot*.csv", recursive=True):
        ts = parse_ts(path, CHAIN_RE)
        if ts is not None:
            chains.append((ts, path))
    chains.sort()
    chains = chains[-limit:]

    pairs = []
    for chain_ts, chain_path in chains:
        prior_spots = [spot_path for spot_ts, spot_path in spots if spot_ts <= chain_ts]
        if prior_spots:
            pairs.append((prior_spots[-1], chain_path))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=int, required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sp.TRADE_RUNTIME_THRESHOLDS["provider_health_override_min_composite_buffer"] = args.buffer

    rows = []
    for spot_path, chain_path in build_pairs(limit=args.limit):
        result = run_engine_snapshot(
            symbol="NIFTY",
            mode="REPLAY",
            source=f"AB_BUF_{args.buffer}",
            apply_budget_constraint=False,
            requested_lots=1,
            lot_size=65,
            max_capital=20000,
            replay_spot=spot_path,
            replay_chain=chain_path,
            replay_dir="debug_samples",
            capture_signal_evaluation=False,
            signal_capture_policy="ALL_SIGNALS",
            previous_chain=None,
            holding_profile="AUTO",
            headline_service=None,
            data_router=None,
        )
        trade = result.get("trade") or {}
        rows.append(
            {
                "spot": os.path.basename(spot_path),
                "chain": os.path.basename(chain_path),
                "trade_status": trade.get("trade_status"),
                "provider_override_active": bool(trade.get("provider_health_override_active")),
                "runtime_composite_score": trade.get("runtime_composite_score"),
                "min_composite_score_threshold": trade.get("min_composite_score_threshold"),
                "message": trade.get("message"),
            }
        )

    payload = {
        "buffer": args.buffer,
        "count": len(rows),
        "trade_count": sum(1 for r in rows if r.get("trade_status") == "TRADE"),
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"wrote={args.out}")
    print(f"buffer={args.buffer} trade_count={payload['trade_count']} count={payload['count']}")


if __name__ == "__main__":
    main()
