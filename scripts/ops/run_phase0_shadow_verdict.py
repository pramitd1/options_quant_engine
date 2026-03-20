#!/usr/bin/env python3
"""
Phase 0 Shadow Runner + Daily Promotion Verdict
===============================================

One-command script to evaluate `blended` vs `research_rank_gate` on the
cumulative dataset and emit a GO/NO-GO verdict using the exact KPI gates
in documentation/implementation_notes/ML_INFERENCE_GAP_FIX_COMPLETE.md.

Usage:
    .venv/bin/python scripts/ops/run_phase0_shadow_verdict.py
    .venv/bin/python scripts/ops/run_phase0_shadow_verdict.py --date 2026-03-20

Outputs:
    research/ml_evaluation/predictor_comparison/phase0_shadow/
      - phase0_shadow_verdict_<date>.json
      - phase0_shadow_verdict_<date>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.ml_evaluation.predictor_comparison.predictor_comparison_runner import (
    _evaluate_predictor,
    _load_cumulative,
    _prepare,
)

OUTPUT_DIR = ROOT / "research" / "ml_evaluation" / "predictor_comparison" / "phase0_shadow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HIT_COL = "correct_60m"
RETURN_COL = "signed_return_60m_bps"
MAE_COL = "mae_60m_bps"
TS_COL = "signal_timestamp"


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def _to_date_series(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df.get(TS_COL), errors="coerce")
    return ts.dt.date


def _safe(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _metric_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _gate_bool(expr: bool | None) -> bool:
    return bool(expr) if expr is not None else False


def _evaluate_window(df: pd.DataFrame) -> dict[str, Any]:
    blended = _evaluate_predictor(df, "blended", "hybrid_move_probability", threshold=0.50)
    rank_gate = _evaluate_predictor(df, "research_rank_gate", "pred_research_rank_gate", threshold=0.50)

    # Primary deltas
    hr_b = _safe(blended.get("trade_hit_rate"))
    hr_r = _safe(rank_gate.get("trade_hit_rate"))
    ret_b = _safe(blended.get("trade_avg_return_bps"))
    ret_r = _safe(rank_gate.get("trade_avg_return_bps"))
    sh_b = _safe(blended.get("trade_sharpe"))
    sh_r = _safe(rank_gate.get("trade_sharpe"))

    hr_delta = _metric_delta(hr_r, hr_b)
    ret_delta = _metric_delta(ret_r, ret_b)

    # MAE deterioration in bps magnitude
    mae_b = _safe(blended.get("trade_avg_mae_bps"))
    mae_r = _safe(rank_gate.get("trade_avg_mae_bps"))
    mae_deterioration = None
    if mae_b is not None and mae_r is not None:
        mae_deterioration = abs(mae_r) - abs(mae_b)

    # Coverage + integrity
    coverage_pct = _safe(rank_gate.get("retention_pct"))

    total_rows = len(df)
    has_ml = (df.get("ml_rank_score").notna() & df.get("ml_confidence_score").notna()) if total_rows > 0 else pd.Series(dtype=bool)
    ml_coverage_pct = float(has_ml.mean() * 100.0) if total_rows > 0 else 0.0
    inference_failure_pct = max(0.0, 100.0 - ml_coverage_pct)

    # Operational latency gate cannot be computed from current dataset schema.
    latency_gate_observable = False

    # Sample size gate
    n_trade_rank = int(rank_gate.get("n_trade") or 0)

    gates: list[GateResult] = []

    # Window gate
    gates.append(
        GateResult(
            "window_min_5d_and_100_outcomes",
            _gate_bool((len({d for d in _to_date_series(df).dropna().tolist()}) >= 5) and (n_trade_rank >= 100)),
            f"unique_days={len({d for d in _to_date_series(df).dropna().tolist()})}, rank_gate_n_trade={n_trade_rank}",
        )
    )

    # Primary gates
    gates.append(GateResult("hit_rate_delta_ge_0.05", _gate_bool(hr_delta is not None and hr_delta >= 0.05), f"hr_delta={hr_delta}"))
    gates.append(GateResult("return_delta_ge_3bps", _gate_bool(ret_delta is not None and ret_delta >= 3.0), f"ret_delta_bps={ret_delta}"))
    gates.append(GateResult("sharpe_nondegradation", _gate_bool(sh_r is not None and sh_b is not None and sh_r >= (sh_b - 0.05)), f"rank_sharpe={sh_r}, blended_sharpe={sh_b}"))

    # Risk gates
    gates.append(
        GateResult(
            "mae_not_worse_by_gt_20bps",
            _gate_bool(mae_deterioration is not None and mae_deterioration <= 20.0),
            f"mae_deterioration_bps={mae_deterioration}",
        )
    )
    gates.append(GateResult("coverage_ge_45pct", _gate_bool(coverage_pct is not None and coverage_pct >= 45.0), f"coverage_pct={coverage_pct}"))
    gates.append(GateResult("ml_coverage_ge_99pct", _gate_bool(ml_coverage_pct >= 99.0), f"ml_coverage_pct={ml_coverage_pct:.2f}"))

    # Operational gates
    gates.append(
        GateResult(
            "inference_failures_le_0.5pct",
            _gate_bool(inference_failure_pct <= 0.5),
            f"inference_failure_pct={inference_failure_pct:.2f}",
        )
    )
    gates.append(
        GateResult(
            "no_sustained_latency_or_ingestion_regressions",
            False,
            "Not observable from cumulative dataset; requires runtime telemetry/log hooks",
        )
    )

    verdict = "GO" if all(g.passed for g in gates) else "NO-GO"

    return {
        "verdict": verdict,
        "blended": blended,
        "research_rank_gate": rank_gate,
        "deltas": {
            "hit_rate_delta": hr_delta,
            "return_delta_bps": ret_delta,
            "mae_deterioration_bps": mae_deterioration,
            "ml_coverage_pct": ml_coverage_pct,
            "inference_failure_pct": inference_failure_pct,
        },
        "gates": [g.__dict__ for g in gates],
        "observability_notes": {
            "latency_gate_observable": latency_gate_observable,
            "latency_gate_reason": "No latency/ingestion telemetry columns in cumulative dataset",
        },
    }


def _render_markdown(report: dict[str, Any], eval_date: str, n_rows: int) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# Phase 0 Shadow Verdict — {eval_date}")
    a("")
    a(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    a(f"  ")
    a(f"**Rows in evaluation window:** {n_rows}")
    a("")
    a(f"## Verdict: **{report['verdict']}**")
    a("")

    b = report["blended"]
    r = report["research_rank_gate"]
    d = report["deltas"]

    a("## Side-by-side Metrics")
    a("")
    a("| Metric | blended | research_rank_gate | Delta (rank - blended) |")
    a("|---|---:|---:|---:|")
    a(f"| n_trade | {b.get('n_trade')} | {r.get('n_trade')} | {int((r.get('n_trade') or 0) - (b.get('n_trade') or 0))} |")
    a(f"| hit_rate | {b.get('trade_hit_rate')} | {r.get('trade_hit_rate')} | {d.get('hit_rate_delta')} |")
    a(f"| avg_return_bps | {b.get('trade_avg_return_bps')} | {r.get('trade_avg_return_bps')} | {d.get('return_delta_bps')} |")
    a(f"| sharpe | {b.get('trade_sharpe')} | {r.get('trade_sharpe')} | — |")
    a(f"| avg_mae_bps | {b.get('trade_avg_mae_bps')} | {r.get('trade_avg_mae_bps')} | {d.get('mae_deterioration_bps')} (deterioration) |")
    a(f"| retention_pct | {b.get('retention_pct')} | {r.get('retention_pct')} | — |")
    a(f"| ml_coverage_pct | — | {d.get('ml_coverage_pct')} | — |")
    a(f"| inference_failure_pct | — | {d.get('inference_failure_pct')} | — |")
    a("")

    a("## KPI Gates")
    a("")
    a("| Gate | Status | Detail |")
    a("|---|---|---|")
    for g in report["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        a(f"| {g['name']} | {status} | {g['detail']} |")

    a("")
    a("## Notes")
    a("")
    a("- This script applies the exact promotion gates from `documentation/implementation_notes/ML_INFERENCE_GAP_FIX_COMPLETE.md`.")
    a("- Operational latency/ingestion gate is marked FAIL when telemetry is unavailable, forcing conservative NO-GO.")
    a("- Use this as a daily guardrail during Phase 0 shadow.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 shadow verdict for blended vs research_rank_gate")
    parser.add_argument("--date", type=str, default=None, help="Evaluation end date (YYYY-MM-DD). Defaults to latest date with outcomes.")
    args = parser.parse_args()

    df = _load_cumulative()
    df = _prepare(df)

    if HIT_COL not in df.columns:
        raise ValueError(f"Missing required outcome column: {HIT_COL}")

    outcome_df = df[df[HIT_COL].notna()].copy()
    if outcome_df.empty:
        raise ValueError("No rows with 60m outcomes available.")

    outcome_df["_date"] = _to_date_series(outcome_df)

    if args.date:
        end_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        end_date = outcome_df["_date"].dropna().max()

    recent_days = sorted([d for d in outcome_df["_date"].dropna().unique() if d <= end_date])[-5:]
    window_df = outcome_df[outcome_df["_date"].isin(recent_days)].copy()

    report = _evaluate_window(window_df)

    out_stub = f"phase0_shadow_verdict_{end_date.isoformat()}"
    json_path = OUTPUT_DIR / f"{out_stub}.json"
    md_path = OUTPUT_DIR / f"{out_stub}.md"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_end_date": end_date.isoformat(),
        "evaluation_days": [d.isoformat() for d in recent_days],
        "window_rows": int(len(window_df)),
        "report": report,
    }

    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(report, end_date.isoformat(), len(window_df)))

    print("=" * 78)
    print("PHASE 0 SHADOW VERDICT")
    print("=" * 78)
    print(f"End date:         {end_date.isoformat()}")
    print(f"Window days:      {', '.join(d.isoformat() for d in recent_days)}")
    print(f"Window rows:      {len(window_df)}")
    print(f"Verdict:          {report['verdict']}")
    print("-")
    print(f"blended HR/ret:   {report['blended'].get('trade_hit_rate')} / {report['blended'].get('trade_avg_return_bps')} bps")
    print(f"rank_gate HR/ret: {report['research_rank_gate'].get('trade_hit_rate')} / {report['research_rank_gate'].get('trade_avg_return_bps')} bps")
    print(f"Delta HR:         {report['deltas'].get('hit_rate_delta')}")
    print(f"Delta return:     {report['deltas'].get('return_delta_bps')} bps")
    print("-")
    failed = [g for g in report["gates"] if not g["passed"]]
    if failed:
        print("Failed gates:")
        for g in failed:
            print(f"  - {g['name']}: {g['detail']}")
    else:
        print("All gates PASSED.")
    print("-")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
