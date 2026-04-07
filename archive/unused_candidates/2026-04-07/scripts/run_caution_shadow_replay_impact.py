import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine_runner import run_engine_snapshot


def _extract_day(name: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


def _extract_time(name: str) -> str:
    m = re.search(r"T(\d{2}-\d{2}-\d{2})", name)
    return m.group(1).replace("-", ":") if m else "00:00:00"


def discover_snapshot_pairs(days: set[str] | None = None) -> list[tuple[Path, Path]]:
    spot_dir = Path("debug_samples/spot_snapshots")
    chain_dir = Path("debug_samples/option_chain_snapshots")
    spots = sorted(spot_dir.glob("NIFTY_spot_snapshot_*.json"))
    chains = sorted(chain_dir.glob("NIFTY_*option_chain_snapshot_*.csv"))

    chain_by_day: dict[str, list[Path]] = {}
    for chain_file in chains:
        day = _extract_day(chain_file.name)
        if not day:
            continue
        if days and day not in days:
            continue
        chain_by_day.setdefault(day, []).append(chain_file)

    pairs: list[tuple[Path, Path]] = []
    for spot_file in spots:
        day = _extract_day(spot_file.name)
        if not day:
            continue
        if days and day not in days:
            continue
        candidate_chains = chain_by_day.get(day, [])
        if not candidate_chains:
            continue

        spot_time = _extract_time(spot_file.name)
        selected = None
        for chain_file in candidate_chains:
            if _extract_time(chain_file.name) >= spot_time:
                selected = chain_file
                break
        if selected is None:
            selected = candidate_chains[-1]
        pairs.append((spot_file, selected))

    return pairs


def normalize_status(value) -> str:
    status = str(value or "").strip().upper()
    if not status:
        return "NO_SIGNAL"
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay impact for conditional caution shadow pack")
    parser.add_argument("--shadow-pack", default="caution_conditional_shadow_v1")
    parser.add_argument("--baseline-pack", default="baseline_v1")
    parser.add_argument(
        "--days",
        default="2026-03-30,2026-04-01,2026-04-02,2026-04-06,2026-04-07",
        help="Comma-separated YYYY-MM-DD list",
    )
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all discovered pairs")
    args = parser.parse_args()

    days = {d.strip() for d in args.days.split(",") if d.strip()}
    pairs = discover_snapshot_pairs(days=days)
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    out_dir = Path("research/reviews/caution_shadow_replay_impact") / f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "generated_at": datetime.now().isoformat(),
        "baseline_pack": args.baseline_pack,
        "shadow_pack": args.shadow_pack,
        "days": sorted(days),
        "pairs_tested": len(pairs),
        "baseline_counts": {"TRADE": 0, "WATCHLIST": 0, "NO_SIGNAL": 0},
        "shadow_counts": {"TRADE": 0, "WATCHLIST": 0, "NO_SIGNAL": 0},
        "status_changed_count": 0,
        "watchlist_to_trade_count": 0,
        "no_signal_to_trade_count": 0,
        "trade_to_watchlist_count": 0,
        "errors": 0,
    }

    for idx, (spot_file, chain_file) in enumerate(pairs, start=1):
        day = _extract_day(spot_file.name) or "UNKNOWN"
        try:
            payload = run_engine_snapshot(
                symbol="NIFTY",
                mode="REPLAY",
                source="CAUTION_SHADOW_IMPACT",
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=65,
                max_capital=20000,
                replay_spot=str(spot_file),
                replay_chain=str(chain_file),
                replay_dir="debug_samples",
                capture_signal_evaluation=False,
                signal_capture_policy="ALL_SIGNALS",
                previous_chain=None,
                holding_profile="AUTO",
                headline_service=None,
                data_router=None,
                authoritative_pack_name=args.baseline_pack,
                shadow_pack_name=args.shadow_pack,
                enable_shadow_logging=False,
            )

            baseline_trade = payload.get("trade") or {}
            shadow_eval = payload.get("shadow_evaluation") or {}
            shadow_trade = (shadow_eval.get("trade") or {}) if isinstance(shadow_eval, dict) else {}

            baseline_status = normalize_status(baseline_trade.get("trade_status"))
            shadow_status = normalize_status(shadow_trade.get("trade_status"))
            changed = baseline_status != shadow_status

            summary["baseline_counts"][baseline_status] = summary["baseline_counts"].get(baseline_status, 0) + 1
            summary["shadow_counts"][shadow_status] = summary["shadow_counts"].get(shadow_status, 0) + 1

            if changed:
                summary["status_changed_count"] += 1
                if baseline_status == "WATCHLIST" and shadow_status == "TRADE":
                    summary["watchlist_to_trade_count"] += 1
                if baseline_status == "NO_SIGNAL" and shadow_status == "TRADE":
                    summary["no_signal_to_trade_count"] += 1
                if baseline_status == "TRADE" and shadow_status == "WATCHLIST":
                    summary["trade_to_watchlist_count"] += 1

            rows.append(
                {
                    "day": day,
                    "spot_file": spot_file.name,
                    "chain_file": chain_file.name,
                    "baseline_status": baseline_status,
                    "shadow_status": shadow_status,
                    "status_changed": int(changed),
                    "baseline_reason_code": baseline_trade.get("no_trade_reason_code"),
                    "shadow_reason_code": shadow_trade.get("no_trade_reason_code"),
                    "baseline_trade_strength": baseline_trade.get("trade_strength"),
                    "shadow_trade_strength": shadow_trade.get("trade_strength"),
                    "baseline_option_efficiency": baseline_trade.get("option_efficiency_score"),
                    "shadow_option_efficiency": shadow_trade.get("option_efficiency_score"),
                    "shadow_override_mode": shadow_trade.get("provider_health_override_mode"),
                }
            )

        except Exception as exc:
            summary["errors"] += 1
            rows.append(
                {
                    "day": day,
                    "spot_file": spot_file.name,
                    "chain_file": chain_file.name,
                    "baseline_status": "ERROR",
                    "shadow_status": "ERROR",
                    "status_changed": 0,
                    "baseline_reason_code": None,
                    "shadow_reason_code": None,
                    "baseline_trade_strength": None,
                    "shadow_trade_strength": None,
                    "baseline_option_efficiency": None,
                    "shadow_option_efficiency": None,
                    "shadow_override_mode": None,
                    "error": str(exc),
                }
            )

    summary["delta_trade_count"] = summary["shadow_counts"].get("TRADE", 0) - summary["baseline_counts"].get("TRADE", 0)
    summary["delta_watchlist_count"] = summary["shadow_counts"].get("WATCHLIST", 0) - summary["baseline_counts"].get("WATCHLIST", 0)

    day_rollup = {}
    for row in rows:
        day = row.get("day", "UNKNOWN")
        roll = day_rollup.setdefault(
            day,
            {
                "day": day,
                "pairs": 0,
                "baseline_trade": 0,
                "shadow_trade": 0,
                "watchlist_to_trade": 0,
                "status_changed": 0,
            },
        )
        roll["pairs"] += 1
        if row.get("baseline_status") == "TRADE":
            roll["baseline_trade"] += 1
        if row.get("shadow_status") == "TRADE":
            roll["shadow_trade"] += 1
        if row.get("baseline_status") == "WATCHLIST" and row.get("shadow_status") == "TRADE":
            roll["watchlist_to_trade"] += 1
        if row.get("status_changed"):
            roll["status_changed"] += 1

    day_rows = sorted(day_rollup.values(), key=lambda r: r["day"]) 

    (out_dir / "replay_impact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        with (out_dir / "replay_impact_rows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if day_rows:
        with (out_dir / "replay_impact_by_day.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(day_rows[0].keys()))
            writer.writeheader()
            writer.writerows(day_rows)

    print(out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
