#!/usr/bin/env python3
"""
Dry-run audit for ML inference backfill — never writes to signals_dataset_cumul.csv.
Outputs structured artifacts under research/ml_evaluation/ops_diagnostics/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

out_dir = ROOT / "research" / "ml_evaluation" / "ops_diagnostics"
out_dir.mkdir(parents=True, exist_ok=True)

cumul_path = ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"
df = pd.read_csv(cumul_path)

n_total = int(len(df))
rank_na = df.get("ml_rank_score", pd.Series(index=df.index, dtype="float64")).isna()
conf_na = df.get("ml_confidence_score", pd.Series(index=df.index, dtype="float64")).isna()
has_both = ~rank_na & ~conf_na
needs = rank_na | conf_na
n_needs = int(needs.sum())

sample_cols = [
    c for c in [
        "signal_timestamp", "signal_id", "symbol", "source", "trade_status",
        "direction", "gamma_regime", "final_flow_signal", "volatility_regime",
        "move_probability", "ml_rank_score", "ml_confidence_score",
    ] if c in df.columns
]
sample = df.loc[needs, sample_cols].head(1000)

by_status = df.groupby("trade_status").apply(
    lambda g: {
        "total": int(len(g)),
        "missing_ml": int((g.get("ml_rank_score", pd.Series(dtype=float)).isna() | g.get("ml_confidence_score", pd.Series(dtype=float)).isna()).sum()),
    }
).to_dict()

payload = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "mode": "dry_run",
    "dataset": str(cumul_path),
    "total_rows": n_total,
    "rows_needing_inference": n_needs,
    "rows_with_both_scores": int(has_both.sum()),
    "coverage_pct_both_scores": round(float(has_both.mean() * 100.0), 2) if n_total else 0.0,
    "would_backfill_pct": round(float(n_needs / n_total * 100.0), 2) if n_total else 0.0,
    "breakdown_by_trade_status": by_status,
}

json_path = out_dir / "backfill_dry_run_summary_20260320.json"
md_path = out_dir / "backfill_dry_run_summary_20260320.md"
csv_path = out_dir / "backfill_dry_run_sample_missing_rows_20260320.csv"

sample.to_csv(csv_path, index=False)
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

md_lines = [
    "# ML Inference Backfill — Dry-Run Summary (2026-03-20)",
    "",
    f"- Total rows: {n_total}",
    f"- Rows needing inference: {n_needs}",
    f"- Coverage (both scores present): {payload['coverage_pct_both_scores']}%",
    f"- Would-backfill share: {payload['would_backfill_pct']}%",
    "",
    "## Breakdown by trade_status",
    "",
]
for status, v in sorted(by_status.items()):
    md_lines.append(f"- {status}: total={v['total']}, missing_ml={v['missing_ml']}")

md_lines += [
    "",
    f"Sample missing rows CSV: {csv_path}",
    f"JSON summary: {json_path}",
]
md_path.write_text("\n".join(md_lines), encoding="utf-8")

print("DRY_RUN_BACKFILL_AUDIT_COMPLETE")
print(f"  json -> {json_path}")
print(f"  md   -> {md_path}")
print(f"  csv  -> {csv_path}")
print(f"  total_rows={n_total}  needs={n_needs}  coverage={payload['coverage_pct_both_scores']}%")
