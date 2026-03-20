#!/usr/bin/env python3
"""
Full engine backtest comparison: discrete vs continuous trade-strength scoring.

Runs holistic backtests for all predictor methods under both scoring modes,
then writes CSV/JSON/Markdown artifacts with method-level deltas.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.holistic_backtest_runner import run_holistic_backtest
from config.policy_resolver import temporary_parameter_pack
from config.settings import DEFAULT_SYMBOL
from data.historical_snapshot import get_available_dates
from scripts.backtest.comparative_backtest_all_predictors import PREDICTION_METHODS


def _extract_metrics(result: dict[str, Any], method: str, mode: str) -> dict[str, Any]:
    if not result.get("ok"):
        return {
            "scoring_mode": mode,
            "method": method,
            "status": "FAILED",
            "error": result.get("message") or result.get("error") or "unknown",
        }

    metrics = result.get("metrics", {})
    avg_scores = metrics.get("avg_scores", {}) if isinstance(metrics.get("avg_scores"), dict) else {}
    return {
        "scoring_mode": mode,
        "method": method,
        "status": "SUCCESS",
        "total_signals": result.get("total_signals", 0),
        "evaluated_days": result.get("evaluated_days", 0),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
        "trade_signals": metrics.get("trade_signals", 0),
        "trade_rate": metrics.get("trade_rate", 0),
        "target_hit_rate": metrics.get("target_hit_rate", 0),
        "stop_loss_hit_rate": metrics.get("stop_loss_hit_rate", 0),
        "avg_trade_strength": metrics.get("avg_trade_strength", 0),
        "avg_composite_score": avg_scores.get("composite_signal_score", 0),
        "avg_direction_score": avg_scores.get("direction_score", 0),
        "avg_tradeability_score": avg_scores.get("tradeability_score", 0),
        "correct_1d": metrics.get("directional_accuracy", {}).get("correct_1d", 0),
        "correct_expiry": metrics.get("directional_accuracy", {}).get("correct_at_expiry", 0),
        "avg_mfe_bps": metrics.get("avg_eod_mfe_bps", 0),
        "avg_mae_bps": metrics.get("avg_eod_mae_bps", 0),
    }


def _run_mode(mode: str, *, symbol: str, start_date: str, end_date: str, max_expiries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    override = {"trade_strength.runtime_thresholds.trade_strength_scoring_mode": mode}

    with temporary_parameter_pack("baseline_v1", overrides=override):
        for method in PREDICTION_METHODS:
            print(f"  [{mode}] running method={method}...")
            result = run_holistic_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                max_expiries=max_expiries,
                evaluate_outcomes=True,
                prediction_method=method,
            )
            row = _extract_metrics(result, method, mode)
            rows.append(row)
            if row.get("status") == "SUCCESS":
                print(
                    f"    ok: signals={row.get('total_signals', 0)} "
                    f"trades={row.get('trade_signals', 0)} "
                    f"avg_strength={row.get('avg_trade_strength', 0):.2f}"
                )
            else:
                print(f"    failed: {row.get('error')}")
    return rows


def _build_delta(discrete_df: pd.DataFrame, continuous_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "total_signals",
        "evaluated_days",
        "elapsed_seconds",
        "trade_signals",
        "trade_rate",
        "target_hit_rate",
        "stop_loss_hit_rate",
        "avg_trade_strength",
        "avg_composite_score",
        "avg_direction_score",
        "avg_tradeability_score",
        "correct_1d",
        "correct_expiry",
        "avg_mfe_bps",
        "avg_mae_bps",
    ]

    left = discrete_df.set_index("method")
    right = continuous_df.set_index("method")

    rows: list[dict[str, Any]] = []
    for method in sorted(set(left.index).intersection(set(right.index))):
        l = left.loc[method]
        r = right.loc[method]
        row: dict[str, Any] = {
            "method": method,
            "discrete_status": l.get("status"),
            "continuous_status": r.get("status"),
            "changed": False,
        }

        if l.get("status") != "SUCCESS" or r.get("status") != "SUCCESS":
            row["changed"] = l.get("status") != r.get("status")
            rows.append(row)
            continue

        for c in metric_cols:
            lv = l.get(c)
            rv = r.get(c)
            row[f"discrete_{c}"] = lv
            row[f"continuous_{c}"] = rv
            try:
                dl = float(lv)
                dr = float(rv)
                d = dr - dl
            except Exception:
                d = None
            row[f"delta_{c}"] = d
            if d is not None and abs(d) > 1e-12:
                row["changed"] = True

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["changed", "method"], ascending=[False, True]).reset_index(drop=True)
    return out


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _render_report(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    max_expiries: int,
    discrete_df: pd.DataFrame,
    continuous_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Full Backtest: Discrete vs Continuous Trade-Strength Scoring")
    add("")
    add(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    add(f"Symbol: {symbol}")
    add(f"Date range: {start_date} to {end_date}")
    add(f"Max expiries/day: {max_expiries}")
    add("")

    changed = int(delta_df["changed"].sum()) if not delta_df.empty and "changed" in delta_df.columns else 0
    add("## Headline")
    add("")
    add(f"Methods with any metric change: {changed}/{len(PREDICTION_METHODS)}")
    add("")

    key_cols = [
        "method",
        "delta_trade_signals",
        "delta_trade_rate",
        "delta_avg_trade_strength",
        "delta_avg_composite_score",
        "delta_correct_expiry",
        "delta_target_hit_rate",
        "delta_stop_loss_hit_rate",
        "changed",
    ]
    present = [c for c in key_cols if c in delta_df.columns]

    add("## Delta Summary (continuous - discrete)")
    add("")
    add(_df_to_markdown(delta_df[present] if present else delta_df))
    add("")

    add("## Discrete Mode Metrics")
    add("")
    add(_df_to_markdown(discrete_df[[
        "method",
        "status",
        "total_signals",
        "trade_signals",
        "trade_rate",
        "avg_trade_strength",
        "avg_composite_score",
        "correct_expiry",
    ]]))
    add("")

    add("## Continuous Mode Metrics")
    add("")
    add(_df_to_markdown(continuous_df[[
        "method",
        "status",
        "total_signals",
        "trade_signals",
        "trade_rate",
        "avg_trade_strength",
        "avg_composite_score",
        "correct_expiry",
    ]]))
    add("")

    return "\n".join(lines)


def main() -> int:
    print("=" * 78)
    print("FULL BACKTEST COMPARISON: DISCRETE vs CONTINUOUS SCORING")
    print("=" * 78)

    symbol = DEFAULT_SYMBOL
    available_dates = get_available_dates(symbol)
    if not available_dates:
        raise RuntimeError(f"No historical dates available for symbol={symbol}")

    requested_start = date(2016, 1, 1)
    effective_start = max(requested_start, min(available_dates))
    effective_end = max(available_dates)

    start_date = str(effective_start)
    end_date = str(effective_end)
    max_expiries = 3

    print(f"Symbol: {symbol}")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Predictor methods: {len(PREDICTION_METHODS)}")

    print("\n[1/3] Running discrete mode...")
    discrete_rows = _run_mode(
        "discrete",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        max_expiries=max_expiries,
    )

    print("\n[2/3] Running continuous mode...")
    continuous_rows = _run_mode(
        "continuous",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        max_expiries=max_expiries,
    )

    print("\n[3/3] Building report artifacts...")
    discrete_df = pd.DataFrame(discrete_rows)
    continuous_df = pd.DataFrame(continuous_rows)
    delta_df = _build_delta(discrete_df, continuous_df)

    out_dir = Path(__file__).resolve().parent / "reports" / f"scoring_mode_full_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    discrete_csv = out_dir / "discrete_metrics.csv"
    continuous_csv = out_dir / "continuous_metrics.csv"
    delta_csv = out_dir / "delta_continuous_minus_discrete.csv"
    summary_json = out_dir / "summary.json"
    report_md = out_dir / "report.md"

    discrete_df.to_csv(discrete_csv, index=False)
    continuous_df.to_csv(continuous_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "predictor_methods": list(PREDICTION_METHODS),
        "methods_with_changes": int(delta_df["changed"].sum()) if not delta_df.empty and "changed" in delta_df.columns else 0,
        "artifacts": {
            "discrete_csv": str(discrete_csv),
            "continuous_csv": str(continuous_csv),
            "delta_csv": str(delta_csv),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report = _render_report(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        max_expiries=max_expiries,
        discrete_df=discrete_df,
        continuous_df=continuous_df,
        delta_df=delta_df,
    )
    report_md.write_text(report, encoding="utf-8")

    print("\nArtifacts:")
    print(f"  - {discrete_csv}")
    print(f"  - {continuous_csv}")
    print(f"  - {delta_csv}")
    print(f"  - {summary_json}")
    print(f"  - {report_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
