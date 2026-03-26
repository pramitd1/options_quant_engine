from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = ROOT / "debug_samples" / "performance"

BENCH_PREFIX = "micro_benchmark_hotspots_"
BENCH_SUFFIX = ".json"

TARGET_SECTIONS = ["spot_history_load", "event_aggregation"]
TARGET_METRICS = ["mean_ms", "p50_ms", "p95_ms", "max_ms"]


def _is_benchmark_summary_file(path: Path) -> bool:
    name = path.name
    if not name.startswith(BENCH_PREFIX):
        return False
    if not name.endswith(BENCH_SUFFIX):
        return False
    if name.endswith("_raw.json"):
        return False
    if "_raw" in name:
        return False
    if "_comparison_" in name:
        return False
    return True


def _discover_latest_two() -> tuple[Path, Path]:
    if not PERF_DIR.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {PERF_DIR}")

    candidates = [p for p in PERF_DIR.glob(f"{BENCH_PREFIX}*{BENCH_SUFFIX}") if _is_benchmark_summary_file(p)]
    candidates = sorted(candidates)
    if len(candidates) < 2:
        raise FileNotFoundError("Need at least two benchmark JSON summaries to compare.")
    return candidates[-2], candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct_delta(base: float, current: float) -> float | None:
    if abs(base) < 1e-12:
        return None
    return ((current - base) / base) * 100.0


def _format_ms(value: float) -> str:
    return f"{value:.4f}"


def _format_delta(value: float) -> str:
    return f"{value:+.4f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _build_rows(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for section in TARGET_SECTIONS:
        base_section = baseline.get(section, {}) if isinstance(baseline.get(section), dict) else {}
        cur_section = current.get(section, {}) if isinstance(current.get(section), dict) else {}

        for metric in TARGET_METRICS:
            base_value = _safe_float(base_section.get(metric))
            cur_value = _safe_float(cur_section.get(metric))
            abs_delta = cur_value - base_value
            pct_delta = _pct_delta(base_value, cur_value)
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "baseline": base_value,
                    "current": cur_value,
                    "absolute_delta": abs_delta,
                    "percent_delta": pct_delta,
                }
            )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    section_order = {name: i for i, name in enumerate(TARGET_SECTIONS)}
    metric_order = {name: i for i, name in enumerate(TARGET_METRICS)}
    rows = sorted(rows, key=lambda r: (section_order.get(r["section"], 99), metric_order.get(r["metric"], 99)))

    header = (
        f"{'section':<20} {'metric':<10} {'baseline(ms)':>14} "
        f"{'current(ms)':>14} {'abs_delta(ms)':>14} {'pct_delta':>11}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['section']:<20} "
            f"{row['metric']:<10} "
            f"{_format_ms(row['baseline']):>14} "
            f"{_format_ms(row['current']):>14} "
            f"{_format_delta(row['absolute_delta']):>14} "
            f"{_format_pct(row['percent_delta']):>11}"
        )


def _write_comparison_artifact(*, baseline_path: Path, current_path: Path, rows: list[dict[str, Any]]) -> Path:
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = PERF_DIR / f"micro_benchmark_hotspots_comparison_{stamp}.json"

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "baseline_file": str(baseline_path),
        "current_file": str(current_path),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two micro-benchmark hotspot JSON summaries and print delta tables."
    )
    parser.add_argument("--baseline", type=Path, help="Path to baseline benchmark JSON")
    parser.add_argument("--current", type=Path, help="Path to current benchmark JSON")
    parser.add_argument(
        "--no-write-artifact",
        action="store_true",
        help="Skip writing comparison JSON artifact under debug_samples/performance",
    )
    args = parser.parse_args()

    if args.baseline and args.current:
        baseline_path = args.baseline
        current_path = args.current
    elif not args.baseline and not args.current:
        baseline_path, current_path = _discover_latest_two()
    else:
        raise ValueError("Provide both --baseline and --current, or neither for auto-discovery.")

    baseline_data = _load_json(baseline_path)
    current_data = _load_json(current_path)

    rows = _build_rows(baseline_data, current_data)

    print(f"baseline: {baseline_path}")
    print(f"current : {current_path}")
    print()
    _print_table(rows)

    if not args.no_write_artifact:
        artifact_path = _write_comparison_artifact(
            baseline_path=baseline_path,
            current_path=current_path,
            rows=rows,
        )
        print()
        print(f"comparison_artifact: {artifact_path}")


if __name__ == "__main__":
    main()
