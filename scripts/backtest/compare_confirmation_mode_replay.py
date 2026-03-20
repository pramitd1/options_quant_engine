"""
Short replay comparison: confirmation continuous vs discrete.

Runs recent replay snapshots twice (continuous mode and discrete mode), then
emits:
- detailed per-snapshot CSV
- summary JSON
- markdown report
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from app.engine_runner import run_preloaded_engine_snapshot
from config.policy_resolver import temporary_parameter_pack
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot
from backtest.replay_regression import (
    _find_chain_snapshots,
    _find_spot_snapshots,
    _nearest_spot_snapshot,
    _replay_global_market_snapshot,
)


def _is_trade(status: object) -> bool:
    return str(status or "").upper().strip() == "TRADE"


def _run_mode(*, symbol: str, source: str, replay_dir: str, limit: int, overrides: dict[str, object] | None) -> pd.DataFrame:
    chain_paths = _find_chain_snapshots(symbol, source, replay_dir)
    spot_paths = _find_spot_snapshots(symbol, replay_dir)

    if not chain_paths:
        raise ValueError(f"No replay option-chain snapshots found for {symbol}/{source} in {replay_dir}")

    if limit > 0:
        chain_paths = chain_paths[-limit:]

    rows: list[dict[str, object]] = []
    previous_chain = None
    global_market_cache = {}

    ctx = temporary_parameter_pack("baseline_v1", overrides=overrides) if overrides else nullcontext()

    with ctx:
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
                previous_chain=previous_chain,
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=1,
                max_capital=float("inf"),
                capture_signal_evaluation=False,
                enable_shadow_logging=False,
                global_market_snapshot=_replay_global_market_snapshot(symbol, spot_snapshot, _cache=global_market_cache),
            )
            if not signal_result.get("ok", False):
                raise ValueError(signal_result.get("error") or "Replay snapshot evaluation failed")

            previous_chain = option_chain

            trade = signal_result.get("execution_trade") or signal_result.get("trade") or {}
            trade_audit = signal_result.get("trade_audit") if isinstance(signal_result.get("trade_audit"), dict) else {}
            scoring_breakdown = (
                trade_audit.get("scoring_breakdown")
                if isinstance(trade_audit.get("scoring_breakdown"), dict)
                else (
                    trade.get("scoring_breakdown")
                    if isinstance(trade.get("scoring_breakdown"), dict)
                    else {}
                )
            )
            confirmation_breakdown = (
                trade_audit.get("confirmation_breakdown")
                if isinstance(trade_audit.get("confirmation_breakdown"), dict)
                else (
                    trade.get("confirmation_breakdown")
                    if isinstance(trade.get("confirmation_breakdown"), dict)
                    else {}
                )
            )

            rows.append(
                {
                    "chain_snapshot": str(chain_path),
                    "spot_snapshot": str(spot_path),
                    "timestamp": trade.get("timestamp") or spot_snapshot.get("timestamp"),
                    "trade_status": trade.get("trade_status"),
                    "direction": trade.get("direction"),
                    "direction_source": trade.get("direction_source"),
                    "decision_classification": trade.get("decision_classification"),
                    "confirmation_status": trade.get("confirmation_status") or trade_audit.get("confirmation_status"),
                    "confirmation_score": scoring_breakdown.get("confirmation_filter_score"),
                    "confirmation_open_score": confirmation_breakdown.get("open_alignment_score"),
                    "confirmation_prev_close_score": confirmation_breakdown.get("prev_close_alignment_score"),
                    "confirmation_range_score": confirmation_breakdown.get("range_expansion_score"),
                    "confirmation_move_prob_score": confirmation_breakdown.get("move_probability_confirmation_score"),
                    "trade_strength": trade.get("trade_strength"),
                    "no_trade_reason_code": trade.get("no_trade_reason_code"),
                }
            )

    return pd.DataFrame(rows)


def _build_summary(merged: pd.DataFrame, *, symbol: str, source: str, replay_dir: str, limit: int) -> dict:
    total = int(len(merged))

    conf_changed = merged["confirmation_score_changed"].fillna(False)
    trade_changed = merged["trade_status_changed"].fillna(False)

    cont_trade = merged["trade_cont"].fillna(False)
    disc_trade = merged["trade_disc"].fillna(False)

    both_trade = int((cont_trade & disc_trade).sum())
    both_no_trade = int((~cont_trade & ~disc_trade).sum())
    cont_trade_disc_no_trade = int((cont_trade & ~disc_trade).sum())
    disc_trade_cont_no_trade = int((disc_trade & ~cont_trade).sum())

    changed_status_counter = Counter(
        (
            str(a or "NONE").upper().strip(),
            str(b or "NONE").upper().strip(),
        )
        for a, b in zip(merged.get("trade_status_disc", []), merged.get("trade_status_cont", []))
        if str(a or "").upper().strip() != str(b or "").upper().strip()
    )

    conf_delta = pd.to_numeric(merged.get("confirmation_score_delta"), errors="coerce")

    return {
        "symbol": symbol,
        "source": source,
        "replay_dir": replay_dir,
        "limit": limit,
        "cases_compared": total,
        "confirmation_score_changed_count": int(conf_changed.sum()),
        "confirmation_score_changed_pct": round(float(conf_changed.mean() * 100.0), 2) if total else 0.0,
        "confirmation_score_delta_mean": round(float(conf_delta.mean()), 4) if total else 0.0,
        "confirmation_score_delta_median": round(float(conf_delta.median()), 4) if total else 0.0,
        "confirmation_score_delta_abs_mean": round(float(conf_delta.abs().mean()), 4) if total else 0.0,
        "trade_status_changed_count": int(trade_changed.sum()),
        "trade_status_changed_pct": round(float(trade_changed.mean() * 100.0), 2) if total else 0.0,
        "decision_impact_counts": {
            "both_trade": both_trade,
            "both_no_trade": both_no_trade,
            "continuous_trade_discrete_no_trade": cont_trade_disc_no_trade,
            "discrete_trade_continuous_no_trade": disc_trade_cont_no_trade,
        },
        "changed_trade_status_pairs": [
            {"from_discrete": k[0], "to_continuous": k[1], "count": int(v)}
            for k, v in sorted(changed_status_counter.items(), key=lambda x: x[1], reverse=True)
        ],
    }


def _write_markdown(path: Path, summary: dict, merged: pd.DataFrame) -> None:
    top_changes = merged[merged["trade_status_changed"] | merged["confirmation_score_changed"]].copy()
    top_changes = top_changes.sort_values(by=["confirmation_score_delta_abs"], ascending=False).head(20)

    lines = []
    lines.append("# Confirmation Mode Replay Delta Report")
    lines.append("")
    lines.append(f"- symbol: {summary['symbol']}")
    lines.append(f"- source: {summary['source']}")
    lines.append(f"- replay_dir: {summary['replay_dir']}")
    lines.append(f"- snapshots_compared: {summary['cases_compared']}")
    lines.append("")
    lines.append("## Score Delta")
    lines.append(f"- confirmation_score_changed_count: {summary['confirmation_score_changed_count']}")
    lines.append(f"- confirmation_score_changed_pct: {summary['confirmation_score_changed_pct']}%")
    lines.append(f"- confirmation_score_delta_mean (cont-disc): {summary['confirmation_score_delta_mean']}")
    lines.append(f"- confirmation_score_delta_median (cont-disc): {summary['confirmation_score_delta_median']}")
    lines.append(f"- confirmation_score_delta_abs_mean: {summary['confirmation_score_delta_abs_mean']}")
    lines.append("")
    lines.append("## Decision Impact")
    lines.append(f"- trade_status_changed_count: {summary['trade_status_changed_count']}")
    lines.append(f"- trade_status_changed_pct: {summary['trade_status_changed_pct']}%")
    impact = summary["decision_impact_counts"]
    lines.append(f"- both_trade: {impact['both_trade']}")
    lines.append(f"- both_no_trade: {impact['both_no_trade']}")
    lines.append(f"- continuous_trade_discrete_no_trade: {impact['continuous_trade_discrete_no_trade']}")
    lines.append(f"- discrete_trade_continuous_no_trade: {impact['discrete_trade_continuous_no_trade']}")
    lines.append("")
    lines.append("## Changed Trade-Status Pairs")
    pairs = summary.get("changed_trade_status_pairs") or []
    if not pairs:
        lines.append("- none")
    else:
        for item in pairs:
            lines.append(
                f"- {item['from_discrete']} -> {item['to_continuous']}: {item['count']}"
            )

    lines.append("")
    lines.append("## Top 20 Snapshot Deltas")
    lines.append("| chain_snapshot | trade_status_disc | trade_status_cont | conf_disc | conf_cont | delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in top_changes.iterrows():
        lines.append(
            "| "
            f"{row.get('chain_snapshot')} | "
            f"{row.get('trade_status_disc')} | "
            f"{row.get('trade_status_cont')} | "
            f"{row.get('confirmation_score_disc')} | "
            f"{row.get('confirmation_score_cont')} | "
            f"{row.get('confirmation_score_delta')} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare continuous vs discrete confirmation scoring on recent replay snapshots")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--source", default="ICICI")
    parser.add_argument("--replay-dir", default="debug_samples")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--output-dir",
        default="research/ml_evaluation/confirmation_mode_replay",
        help="Directory to write CSV/JSON/MD artifacts",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    continuous = _run_mode(
        symbol=args.symbol,
        source=args.source,
        replay_dir=args.replay_dir,
        limit=args.limit,
        overrides={
            "confirmation_filter.core.confirmation_scoring_mode": "continuous",
            "confirmation_filter.core.continuous_open_alignment": 1,
            "confirmation_filter.core.continuous_prev_close_alignment": 1,
            "confirmation_filter.core.continuous_range_expansion": 1,
            "confirmation_filter.core.continuous_move_probability": 1,
        },
    )
    discrete = _run_mode(
        symbol=args.symbol,
        source=args.source,
        replay_dir=args.replay_dir,
        limit=args.limit,
        overrides={"confirmation_filter.core.confirmation_scoring_mode": "discrete"},
    )

    cont_cols = {
        "trade_status": "trade_status_cont",
        "direction": "direction_cont",
        "confirmation_status": "confirmation_status_cont",
        "confirmation_score": "confirmation_score_cont",
        "trade_strength": "trade_strength_cont",
        "decision_classification": "decision_classification_cont",
        "no_trade_reason_code": "no_trade_reason_code_cont",
    }
    disc_cols = {
        "trade_status": "trade_status_disc",
        "direction": "direction_disc",
        "confirmation_status": "confirmation_status_disc",
        "confirmation_score": "confirmation_score_disc",
        "trade_strength": "trade_strength_disc",
        "decision_classification": "decision_classification_disc",
        "no_trade_reason_code": "no_trade_reason_code_disc",
    }

    continuous = continuous.rename(columns=cont_cols)
    discrete = discrete.rename(columns=disc_cols)

    keys = ["chain_snapshot", "spot_snapshot"]
    merged = discrete.merge(continuous, on=keys, how="inner")

    merged["confirmation_score_disc"] = pd.to_numeric(merged["confirmation_score_disc"], errors="coerce")
    merged["confirmation_score_cont"] = pd.to_numeric(merged["confirmation_score_cont"], errors="coerce")
    merged["confirmation_score_delta"] = merged["confirmation_score_cont"] - merged["confirmation_score_disc"]
    merged["confirmation_score_delta_abs"] = merged["confirmation_score_delta"].abs()
    merged["confirmation_score_changed"] = merged["confirmation_score_delta_abs"] > 1e-9

    merged["trade_disc"] = merged["trade_status_disc"].apply(_is_trade)
    merged["trade_cont"] = merged["trade_status_cont"].apply(_is_trade)
    merged["trade_status_changed"] = merged["trade_status_disc"].astype(str) != merged["trade_status_cont"].astype(str)

    summary = _build_summary(
        merged,
        symbol=args.symbol,
        source=args.source,
        replay_dir=args.replay_dir,
        limit=args.limit,
    )

    csv_path = out_dir / "confirmation_mode_replay_delta.csv"
    json_path = out_dir / "confirmation_mode_replay_summary.json"
    md_path = out_dir / "confirmation_mode_replay_report.md"

    merged.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary, merged)

    print("CONFIRMATION MODE REPLAY COMPARISON")
    print("-----------------------------------")
    print(f"cases_compared: {summary['cases_compared']}")
    print(f"confirmation_score_changed_count: {summary['confirmation_score_changed_count']}")
    print(f"trade_status_changed_count: {summary['trade_status_changed_count']}")
    print(f"continuous_trade_discrete_no_trade: {summary['decision_impact_counts']['continuous_trade_discrete_no_trade']}")
    print(f"discrete_trade_continuous_no_trade: {summary['decision_impact_counts']['discrete_trade_continuous_no_trade']}")
    print(f"saved_csv: {csv_path}")
    print(f"saved_json: {json_path}")
    print(f"saved_md: {md_path}")


if __name__ == "__main__":
    main()
