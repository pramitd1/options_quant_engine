from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.historical_snapshot import get_available_dates
from scripts.backtest.recent_policy_impact_analysis import RunConfig, _run_capture


@dataclass
class StickinessRunConfig:
    symbol: str = "NIFTY"
    max_expiries: int = 3
    min_quality_score: float = 40.0
    lookahead_snapshots: int = 3
    large_move_bps: float = 75.0


def _directional_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "signal_timestamp" in out.columns:
        out["signal_timestamp"] = pd.to_datetime(out["signal_timestamp"], errors="coerce")
        out = out.sort_values("signal_timestamp").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    out["spot_at_signal"] = pd.to_numeric(out.get("spot_at_signal"), errors="coerce")
    out = out[out["direction"].astype(str).str.upper().isin(["CALL", "PUT"])].copy().reset_index(drop=True)
    out["dir"] = out["direction"].astype(str).str.upper()
    out["is_trade"] = out["trade_status"].astype(str).str.upper().eq("TRADE")
    return out


def _reversal_metrics(df: pd.DataFrame, lookahead_snapshots: int, large_move_bps: float) -> dict[str, Any]:
    if df.empty or len(df) < 2:
        return {
            "directional_rows": int(len(df)),
            "reversal_events": 0,
            "reversal_latency_mean_snapshots": None,
            "reversal_latency_median_snapshots": None,
            "flip_back_rate": None,
            "missed_large_move_after_reversal": 0,
            "large_move_after_reversal_events": 0,
            "captured_large_move_after_reversal": 0,
        }

    df = df.copy()
    df["prev_dir"] = df["dir"].shift(1)
    df["is_reversal"] = df["prev_dir"].notna() & (df["dir"] != df["prev_dir"])
    reversal_idx = df.index[df["is_reversal"]].tolist()

    # Reversal latency proxy: run length before each flip.
    latencies = []
    run_len = 1
    for i in range(1, len(df)):
        if df.loc[i, "dir"] == df.loc[i - 1, "dir"]:
            run_len += 1
        else:
            latencies.append(run_len)
            run_len = 1

    # Flip-back rate: flips again within lookahead snapshots.
    flip_back_flags = []
    for i in reversal_idx:
        curr_dir = df.loc[i, "dir"]
        future_dirs = [
            df.loc[j, "dir"]
            for j in range(i + 1, min(len(df), i + 1 + max(1, lookahead_snapshots)))
        ]
        flip_back_flags.append(any(fd != curr_dir for fd in future_dirs))

    # Missed large move after first reversal signal using snapshot stream only.
    large_events = 0
    captured_large = 0
    missed_large = 0

    for i in reversal_idx:
        spot0 = df.loc[i, "spot_at_signal"]
        if pd.isna(spot0) or spot0 == 0:
            continue

        window = df.iloc[i + 1 : i + 1 + max(1, lookahead_snapshots)].copy()
        window = window[window["spot_at_signal"].notna()].copy()
        if window.empty:
            continue

        move_bps_series = (window["spot_at_signal"] - spot0) / spot0 * 10000.0
        if move_bps_series.empty:
            continue

        k = move_bps_series.abs().idxmax()
        realized_move_bps = float(move_bps_series.loc[k])
        if abs(realized_move_bps) < float(large_move_bps):
            continue

        large_events += 1
        realized_dir = "CALL" if realized_move_bps > 0 else "PUT"
        captured = bool(df.loc[i, "is_trade"] and df.loc[i, "dir"] == realized_dir)
        if captured:
            captured_large += 1
        else:
            missed_large += 1

    return {
        "directional_rows": int(len(df)),
        "reversal_events": int(len(reversal_idx)),
        "reversal_latency_mean_snapshots": round(float(pd.Series(latencies).mean()), 3) if latencies else None,
        "reversal_latency_median_snapshots": round(float(pd.Series(latencies).median()), 3) if latencies else None,
        "flip_back_rate": round(float(pd.Series(flip_back_flags).mean()), 4) if flip_back_flags else None,
        "missed_large_move_after_reversal": int(missed_large),
        "large_move_after_reversal_events": int(large_events),
        "captured_large_move_after_reversal": int(captured_large),
    }


def run_stickiness_tuning(
    *,
    run_cfg: StickinessRunConfig,
    start_date: date,
    end_date: date,
    variants: list[tuple[str, dict[str, Any]]],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    detailed = []

    for label, overrides in variants:
        cfg = RunConfig(
            label=label,
            symbol=run_cfg.symbol,
            start_date=start_date,
            end_date=end_date,
            max_expiries=run_cfg.max_expiries,
            prediction_method="blended",
            compute_iv=True,
            include_global_market=True,
            include_macro_events=True,
            min_quality_score=run_cfg.min_quality_score,
            evaluate_outcomes=True,
            overrides=overrides if overrides else None,
        )
        frame = _run_capture(cfg)
        d = _directional_frame(frame)
        metrics = _reversal_metrics(
            d,
            lookahead_snapshots=run_cfg.lookahead_snapshots,
            large_move_bps=run_cfg.large_move_bps,
        )
        row = {
            "variant": label,
            "rows": int(len(frame)),
            **metrics,
        }
        rows.append(row)
        if not d.empty:
            d["variant"] = label
            detailed.append(d)

    summary = pd.DataFrame(rows)

    # Multi-objective score: lower is better.
    # Heavy penalty on missed-large-move rate, then flip-back, then latency.
    def _objective(r: pd.Series) -> float | None:
        if pd.isna(r.get("flip_back_rate")) or pd.isna(r.get("reversal_latency_mean_snapshots")):
            return None
        large_events = float(r.get("large_move_after_reversal_events") or 0.0)
        missed_large = float(r.get("missed_large_move_after_reversal") or 0.0)
        missed_rate = (missed_large / large_events) if large_events > 0 else 1.0
        latency = float(r.get("reversal_latency_mean_snapshots") or 0.0)
        flip_back = float(r.get("flip_back_rate") or 0.0)
        return round((2.0 * missed_rate) + (1.0 * flip_back) + (0.35 * latency), 6)

    summary["objective_score"] = summary.apply(_objective, axis=1)
    summary = summary.sort_values(["objective_score", "variant"], na_position="last").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "stickiness_tuning_summary.csv", index=False)
    if detailed:
        pd.concat(detailed, ignore_index=True).to_csv(out_dir / "stickiness_tuning_rows_combined.csv", index=False)

    return summary


def main() -> None:
    symbol = "NIFTY"
    available = get_available_dates(symbol)
    selected = available[-260:-20] if len(available) > 300 else available[:-20]
    start_date, end_date = selected[0], selected[-1]

    run_cfg = StickinessRunConfig(
        symbol=symbol,
        max_expiries=3,
        min_quality_score=40.0,
        lookahead_snapshots=3,
        large_move_bps=75.0,
    )

    variants = [
        (
            "old_like",
            {
                "confirmation_filter.core.reversal_veto_steps": 1,
                "trade_strength.runtime_thresholds.reversal_breakout_override_move_probability_floor": 0.62,
                "trade_strength.runtime_thresholds.reversal_breakout_override_range_pct_floor": 0.35,
                "trade_strength.runtime_thresholds.reversal_breakout_override_requires_flow": 1,
                "trade_strength.runtime_thresholds.reversal_breakout_override_requires_hedging": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_min_signals": 2,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_evidence_threshold": 9.9,
            },
        ),
        ("new_default", {}),
        (
            "aggressive_handoff",
            {
                "confirmation_filter.core.reversal_veto_steps": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_move_probability_floor": 0.52,
                "trade_strength.runtime_thresholds.reversal_breakout_override_range_pct_floor": 0.20,
                "trade_strength.runtime_thresholds.reversal_breakout_override_requires_flow": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_requires_hedging": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_min_signals": 1,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_evidence_threshold": 0.8,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_margin_relief": 0.3,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_score_relief": 0.15,
                "trade_strength.runtime_thresholds.reversal_stage_early_size_mult": 0.70,
            },
        ),
        (
            "balanced_handoff",
            {
                "confirmation_filter.core.reversal_veto_steps": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_move_probability_floor": 0.55,
                "trade_strength.runtime_thresholds.reversal_breakout_override_range_pct_floor": 0.25,
                "trade_strength.runtime_thresholds.reversal_breakout_override_requires_flow": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_requires_hedging": 0,
                "trade_strength.runtime_thresholds.reversal_breakout_override_min_signals": 2,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_evidence_threshold": 1.0,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_margin_relief": 0.25,
                "trade_strength.runtime_thresholds.reversal_fast_handoff_score_relief": 0.12,
                "trade_strength.runtime_thresholds.reversal_stage_early_size_mult": 0.65,
            },
        ),
    ]

    out_dir = REPO_ROOT / "research/reviews/stickiness_tuning_2026-04-02"
    summary = run_stickiness_tuning(
        run_cfg=run_cfg,
        start_date=start_date,
        end_date=end_date,
        variants=variants,
        out_dir=out_dir,
    )

    best = None
    if not summary.empty and summary["objective_score"].notna().any():
        best = summary[summary["objective_score"].notna()].iloc[0].to_dict()

    report = {
        "symbol": symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "days_evaluated": len(selected),
        "max_expiries": run_cfg.max_expiries,
        "lookahead_snapshots": run_cfg.lookahead_snapshots,
        "large_move_bps": run_cfg.large_move_bps,
        "objective": "2.0*missed_large_move_rate + 1.0*flip_back_rate + 0.35*reversal_latency_mean",
        "best_variant": best,
    }
    (out_dir / "stickiness_tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Stickiness Objective Tuning Report",
        "",
        f"- Symbol: {symbol}",
        f"- Date window: {start_date} to {end_date} ({len(selected)} trading days)",
        f"- Max expiries per day: {run_cfg.max_expiries}",
        f"- Lookahead snapshots: {run_cfg.lookahead_snapshots}",
        f"- Large move threshold (bps): {run_cfg.large_move_bps}",
        "",
        "## Objective",
        "",
        "`2.0*missed_large_move_rate + 1.0*flip_back_rate + 0.35*reversal_latency_mean` (lower is better)",
        "",
        "## Results",
        "",
    ]
    headers = list(summary.columns)
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in summary.iterrows():
        vals = [str(row[h]) for h in headers]
        md_lines.append("| " + " | ".join(vals) + " |")

    (out_dir / "stickiness_tuning_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"OUTPUT_DIR {out_dir}")


if __name__ == "__main__":
    main()
