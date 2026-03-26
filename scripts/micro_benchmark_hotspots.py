from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.spot_history import load_spot_history
from features.event_features.aggregator import aggregate_event_features
from nlp.schemas.event_schema import validate_event_record


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if high == low:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _timed_call_ms(fn) -> float:
    start_ns = time.perf_counter_ns()
    fn()
    end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1_000_000.0


def _build_spot_history_fixture(base_dir: Path, *, symbol: str, days: int, rows_per_day: int) -> None:
    symbol_dir = base_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    base_ts = pd.Timestamp("2026-03-01T09:15:00+05:30")
    for day in range(days):
        day_ts = base_ts + pd.Timedelta(days=day)
        day_file = symbol_dir / f"{symbol}_{day_ts.strftime('%Y-%m-%d')}.csv"

        rows = ["timestamp,spot"]
        for i in range(rows_per_day):
            ts = day_ts + pd.Timedelta(minutes=i)
            spot = 22000.0 + day * 12.0 + i * 0.2
            rows.append(f"{ts.isoformat()},{spot:.2f}")

        # Insert deterministic duplicate timestamps to stress dedupe path.
        dup_ts = day_ts + pd.Timedelta(minutes=max(1, rows_per_day // 3))
        rows.append(f"{dup_ts.isoformat()},{(22000.0 + day * 12.0 + 999):.2f}")

        day_file.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _build_event_fixture(event_count: int) -> list:
    out = []
    base_ts = pd.Timestamp("2026-03-26T09:15:00+05:30")
    event_types = [
        "earnings_result",
        "guidance_revision",
        "regulatory_action",
        "macro_event_sector_index",
        "rumor_unconfirmed_report",
    ]
    scopes = ["single_stock", "sector", "index"]
    directions = ["bullish", "bearish", "mixed", "neutral"]
    vols = ["expansion", "compression", "mixed", "neutral"]

    for i in range(event_count):
        ts = base_ts + pd.Timedelta(minutes=(i % 180))
        payload = {
            "event_type": event_types[i % len(event_types)],
            "instrument_scope": scopes[i % len(scopes)],
            "expected_direction": directions[i % len(directions)],
            "directional_confidence": 0.45 + (i % 5) * 0.1,
            "vol_impact": vols[i % len(vols)],
            "vol_confidence": 0.4 + (i % 6) * 0.08,
            "event_strength": 0.35 + (i % 7) * 0.09,
            "uncertainty_score": 0.25 + (i % 4) * 0.15,
            "gap_risk_score": 0.2 + (i % 4) * 0.18,
            "time_horizon": "1_3_sessions" if i % 2 == 0 else "intraday",
            "catalyst_quality": "high" if i % 3 == 0 else "medium",
            "risk_flag": bool(i % 2),
            "summary": f"synthetic event {i}",
            "source": "benchmark",
            "symbols": ["NIFTY"] if i % 3 == 0 else ["RELIANCE"],
            "event_timestamp": ts.isoformat(),
        }
        out.append(validate_event_record(payload))
    return out


def _summarize(latencies_ms: list[float]) -> dict[str, float]:
    return {
        "count": float(len(latencies_ms)),
        "mean_ms": round(statistics.fmean(latencies_ms), 6) if latencies_ms else 0.0,
        "min_ms": round(min(latencies_ms), 6) if latencies_ms else 0.0,
        "p50_ms": round(_percentile(latencies_ms, 50), 6),
        "p95_ms": round(_percentile(latencies_ms, 95), 6),
        "max_ms": round(max(latencies_ms), 6) if latencies_ms else 0.0,
    }


def run_benchmarks(
    *,
    iterations: int,
    warmup: int,
    spot_days: int,
    spot_rows_per_day: int,
    event_count: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="oqe_bench_") as tmp:
        tmp_dir = Path(tmp)
        symbol = "NIFTY"
        _build_spot_history_fixture(
            tmp_dir,
            symbol=symbol,
            days=spot_days,
            rows_per_day=spot_rows_per_day,
        )

        events = _build_event_fixture(event_count)
        as_of = pd.Timestamp("2026-03-26T13:00:00+05:30")
        start_ts = pd.Timestamp("2026-03-10T09:15:00+05:30")
        end_ts = pd.Timestamp("2026-03-20T15:30:00+05:30")

        spot_latencies: list[float] = []
        event_latencies: list[float] = []

        for _ in range(warmup):
            load_spot_history(symbol, start_ts=start_ts, end_ts=end_ts, base_dir=tmp_dir)
            aggregate_event_features(
                events,
                direction_hint="CALL",
                underlying_symbol=symbol,
                as_of=as_of,
            )

        for _ in range(iterations):
            spot_latencies.append(
                _timed_call_ms(
                    lambda: load_spot_history(
                        symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        base_dir=tmp_dir,
                    )
                )
            )
            event_latencies.append(
                _timed_call_ms(
                    lambda: aggregate_event_features(
                        events,
                        direction_hint="CALL",
                        underlying_symbol=symbol,
                        as_of=as_of,
                    )
                )
            )

    return {
        "meta": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "iterations": iterations,
            "warmup": warmup,
            "spot_days": spot_days,
            "spot_rows_per_day": spot_rows_per_day,
            "event_count": event_count,
        },
        "spot_history_load": _summarize(spot_latencies),
        "event_aggregation": _summarize(event_latencies),
        "raw_latencies_ms": {
            "spot_history_load": spot_latencies,
            "event_aggregation": event_latencies,
        },
    }


def _write_artifacts(result: dict) -> tuple[Path, Path]:
    out_dir = ROOT / "debug_samples" / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"micro_benchmark_hotspots_{stamp}.json"
    csv_path = out_dir / f"micro_benchmark_hotspots_{stamp}_raw.csv"

    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "spot_history_load_ms", "event_aggregation_ms"])
        spot = result["raw_latencies_ms"]["spot_history_load"]
        event = result["raw_latencies_ms"]["event_aggregation"]
        for idx, (a, b) in enumerate(zip(spot, event), start=1):
            writer.writerow([idx, f"{a:.6f}", f"{b:.6f}"])

    return summary_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-benchmark core latency hotspots (p50/p95).")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--spot-days", type=int, default=20)
    parser.add_argument("--spot-rows-per-day", type=int, default=300)
    parser.add_argument("--event-count", type=int, default=300)
    args = parser.parse_args()

    result = run_benchmarks(
        iterations=max(1, args.iterations),
        warmup=max(0, args.warmup),
        spot_days=max(1, args.spot_days),
        spot_rows_per_day=max(5, args.spot_rows_per_day),
        event_count=max(10, args.event_count),
    )
    summary_path, csv_path = _write_artifacts(result)

    print(json.dumps({
        "summary": {
            "spot_history_load": result["spot_history_load"],
            "event_aggregation": result["event_aggregation"],
        },
        "artifacts": {
            "summary_json": str(summary_path),
            "raw_csv": str(csv_path),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
