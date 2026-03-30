import glob
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
    for path in glob.glob("debug_samples/*spot_snapshot*.json"):
        ts = parse_ts(path, SPOT_RE)
        if ts is not None:
            spots.append((ts, path))
    spots.sort()

    chains = []
    for path in glob.glob("debug_samples/*option_chain_snapshot*.csv"):
        ts = parse_ts(path, CHAIN_RE)
        if ts is not None:
            chains.append((ts, path))
    chains.sort()

    chains = chains[-limit:]
    pairs = []
    for chain_ts, chain_path in chains:
        prior_spots = [spot_path for spot_ts, spot_path in spots if spot_ts <= chain_ts]
        if not prior_spots:
            continue
        pairs.append((prior_spots[-1], chain_path))
    return pairs


def run_with_buffer(pairs, buffer):
    sp.TRADE_RUNTIME_THRESHOLDS["provider_health_override_min_composite_buffer"] = buffer
    rows = []
    for spot_path, chain_path in pairs:
        result = run_engine_snapshot(
            symbol="NIFTY",
            mode="REPLAY",
            source=f"AB_BUF_{buffer}",
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
    return rows


def main():
    pairs = build_pairs(limit=25)
    baseline = run_with_buffer(pairs, 8)
    tuned = run_with_buffer(pairs, 4)

    baseline_trade_count = sum(1 for row in baseline if row["trade_status"] == "TRADE")
    tuned_trade_count = sum(1 for row in tuned if row["trade_status"] == "TRADE")
    additional_trades = tuned_trade_count - baseline_trade_count

    print(f"paired_snapshots={len(pairs)}")
    print(f"baseline_buffer=8 trade_count={baseline_trade_count}")
    print(f"tuned_buffer=4 trade_count={tuned_trade_count}")
    print(f"additional_trades={additional_trades}")

    changed = []
    for baseline_row, tuned_row in zip(baseline, tuned):
        if baseline_row["trade_status"] != tuned_row["trade_status"]:
            changed.append((baseline_row, tuned_row))

    print(f"changed_snapshots={len(changed)}")
    for baseline_row, tuned_row in changed:
        print("---")
        print(f"chain={tuned_row['chain']}")
        print(f"spot={tuned_row['spot']}")
        print(f"status: {baseline_row['trade_status']} -> {tuned_row['trade_status']}")
        print(
            f"composite: {tuned_row['runtime_composite_score']} / "
            f"min {tuned_row['min_composite_score_threshold']}"
        )
        print(f"message: {tuned_row['message']}")


if __name__ == "__main__":
    main()
