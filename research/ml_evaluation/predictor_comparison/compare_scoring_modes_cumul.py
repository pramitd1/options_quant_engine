#!/usr/bin/env python3
"""
Compare discrete vs continuous trade-strength scoring modes on the
Signal Cumulative dataset across all predictor methods.

Artifacts are written under:
  research/ml_evaluation/predictor_comparison/scoring_mode_comparison/

This script is research-only and does not modify production runtime behavior.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.policy_resolver import temporary_parameter_pack
from research.ml_evaluation.predictor_comparison.predictor_comparison_runner import (
    PREDICTORS,
    _evaluate_all,
    _load_cumulative,
    _prepare,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "scoring_mode_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"

    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"

    body = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + body)


def _evaluate_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    override = {
        "trade_strength.runtime_thresholds.trade_strength_scoring_mode": mode,
    }
    with temporary_parameter_pack("baseline_v1", overrides=override):
        rows = _evaluate_all(df, "cumulative")
    out = pd.DataFrame(rows)
    out["scoring_mode"] = mode
    return out


def _build_delta(discrete_df: pd.DataFrame, continuous_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["predictor"]
    left = discrete_df.copy().set_index(join_cols)
    right = continuous_df.copy().set_index(join_cols)

    metric_cols = [
        "evaluable",
        "n_trade",
        "n_no_trade",
        "retention_pct",
        "trade_hit_rate",
        "trade_avg_return_bps",
        "trade_cumulative_bps",
        "trade_max_dd_bps",
        "trade_volatility_bps",
        "trade_sharpe",
        "no_trade_hit_rate",
        "no_trade_avg_return_bps",
        "prob_return_corr",
    ]

    delta_rows: list[dict[str, Any]] = []
    for predictor in sorted(set(left.index).intersection(set(right.index))):
        row_l = left.loc[predictor]
        row_r = right.loc[predictor]
        payload: dict[str, Any] = {"predictor": predictor}
        any_delta = False

        for c in metric_cols:
            lv = row_l.get(c)
            rv = row_r.get(c)
            payload[f"discrete_{c}"] = lv
            payload[f"continuous_{c}"] = rv

            try:
                dlv = float(lv) if pd.notna(lv) else None
                drv = float(rv) if pd.notna(rv) else None
                if dlv is None or drv is None:
                    d = None
                else:
                    d = drv - dlv
                    if abs(d) > 1e-12:
                        any_delta = True
            except Exception:
                d = None
            payload[f"delta_{c}"] = d

        payload["changed"] = any_delta
        delta_rows.append(payload)

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(["changed", "predictor"], ascending=[False, True]).reset_index(drop=True)
    return delta_df


def _render_report(discrete_df: pd.DataFrame, continuous_df: pd.DataFrame, delta_df: pd.DataFrame) -> str:
    gen = datetime.now().isoformat(timespec="seconds")
    lines: list[str] = []
    add = lines.append

    add("# Scoring Mode Comparison Report")
    add("")
    add(f"Generated: {gen}")
    add("Dataset: Signal Cumulative (signals_dataset_cumul.csv)")
    add("Compared modes: discrete vs continuous")
    add("")

    add("## Predictor Coverage")
    add("")
    add(f"Predictors evaluated: {len(PREDICTORS)}")
    add("")
    add("| Predictor | Description |")
    add("|---|---|")
    for name, cfg in PREDICTORS.items():
        add(f"| {name} | {cfg['desc']} |")
    add("")

    changed = int(delta_df["changed"].sum()) if not delta_df.empty and "changed" in delta_df.columns else 0
    add("## Summary")
    add("")
    add(f"Predictors with any metric delta: {changed}")
    add(f"Predictors with no metric delta: {len(PREDICTORS) - changed}")
    add("")

    key_cols = [
        "predictor",
        "discrete_n_trade",
        "continuous_n_trade",
        "delta_n_trade",
        "discrete_trade_hit_rate",
        "continuous_trade_hit_rate",
        "delta_trade_hit_rate",
        "discrete_trade_avg_return_bps",
        "continuous_trade_avg_return_bps",
        "delta_trade_avg_return_bps",
        "discrete_trade_sharpe",
        "continuous_trade_sharpe",
        "delta_trade_sharpe",
        "changed",
    ]
    present_cols = [c for c in key_cols if c in delta_df.columns]

    add("## Delta Table (continuous - discrete)")
    add("")
    if delta_df.empty:
        add("No rows were produced.")
    else:
        show = delta_df[present_cols].copy()
        add(_df_to_markdown(show))
    add("")

    add("## Mode-specific Results")
    add("")
    add("### Discrete")
    add("")
    add(_df_to_markdown(discrete_df[[
        "predictor",
        "evaluable",
        "n_trade",
        "retention_pct",
        "trade_hit_rate",
        "trade_avg_return_bps",
        "trade_cumulative_bps",
        "trade_sharpe",
    ]]))
    add("")

    add("### Continuous")
    add("")
    add(_df_to_markdown(continuous_df[[
        "predictor",
        "evaluable",
        "n_trade",
        "retention_pct",
        "trade_hit_rate",
        "trade_avg_return_bps",
        "trade_cumulative_bps",
        "trade_sharpe",
    ]]))
    add("")

    add("## Notes")
    add("")
    add("This comparison uses the same signal-cumulative dataset and predictor definitions for both runs.")
    add("Only the runtime override trade_strength.runtime_thresholds.trade_strength_scoring_mode was changed.")

    return "\n".join(lines)


def main() -> int:
    print("=" * 72)
    print("Scoring Mode Comparison on Signal Cumulative Dataset")
    print("=" * 72)

    print("\n[1/5] Loading cumulative dataset...")
    cumul = _load_cumulative()
    print(f"  rows: {len(cumul):,}")

    print("\n[2/5] Preparing dataset (ML inference if required)...")
    cumul = _prepare(cumul)
    print("  done")

    print("\n[3/5] Evaluating predictors under discrete mode...")
    discrete_df = _evaluate_mode(cumul, "discrete")

    print("\n[4/5] Evaluating predictors under continuous mode...")
    continuous_df = _evaluate_mode(cumul, "continuous")

    discrete_df = _coerce_numeric(discrete_df, [
        "evaluable",
        "n_trade",
        "n_no_trade",
        "retention_pct",
        "trade_hit_rate",
        "trade_avg_return_bps",
        "trade_cumulative_bps",
        "trade_max_dd_bps",
        "trade_volatility_bps",
        "trade_sharpe",
        "no_trade_hit_rate",
        "no_trade_avg_return_bps",
        "prob_return_corr",
    ])
    continuous_df = _coerce_numeric(continuous_df, [
        "evaluable",
        "n_trade",
        "n_no_trade",
        "retention_pct",
        "trade_hit_rate",
        "trade_avg_return_bps",
        "trade_cumulative_bps",
        "trade_max_dd_bps",
        "trade_volatility_bps",
        "trade_sharpe",
        "no_trade_hit_rate",
        "no_trade_avg_return_bps",
        "prob_return_corr",
    ])

    print("\n[5/5] Building deltas and writing artifacts...")
    delta_df = _build_delta(discrete_df, continuous_df)

    discrete_csv = OUTPUT_DIR / "cumulative_discrete.csv"
    continuous_csv = OUTPUT_DIR / "cumulative_continuous.csv"
    delta_csv = OUTPUT_DIR / "cumulative_delta_continuous_minus_discrete.csv"
    report_md = OUTPUT_DIR / "scoring_mode_comparison_report.md"
    summary_json = OUTPUT_DIR / "scoring_mode_comparison_summary.json"

    discrete_df.to_csv(discrete_csv, index=False)
    continuous_df.to_csv(continuous_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)

    report = _render_report(discrete_df, continuous_df, delta_df)
    report_md.write_text(report, encoding="utf-8")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "signals_dataset_cumul.csv",
        "predictor_count": int(len(PREDICTORS)),
        "artifacts": {
            "discrete_csv": str(discrete_csv),
            "continuous_csv": str(continuous_csv),
            "delta_csv": str(delta_csv),
            "report_md": str(report_md),
        },
        "predictors_with_changes": int(delta_df["changed"].sum()) if not delta_df.empty and "changed" in delta_df.columns else 0,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nArtifacts written:")
    print(f"  - {discrete_csv}")
    print(f"  - {continuous_csv}")
    print(f"  - {delta_csv}")
    print(f"  - {report_md}")
    print(f"  - {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
