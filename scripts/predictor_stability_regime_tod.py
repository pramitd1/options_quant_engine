#!/usr/bin/env python3
"""
Stability check for predictor reliability edge by regime and time-of-day.

Compares research_decision_policy vs blended on the same cumulative dataset
under the current calibrated selection policy.

Outputs:
  - documentation/daily_reports/predictor_stability_regime_tod_<date>.csv
  - documentation/daily_reports/predictor_stability_regime_tod_<date>.json
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, SIGNAL_DATASET_PATH, load_signals_dataset
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary
from tuning.objectives import apply_selection_policy
from scripts.predictor_comparative_report import PREDICTORS, SELECTION_POLICY

REPORT_DIR = ROOT / "documentation" / "daily_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()
BASELINE = "blended"
CANDIDATE = "research_decision_policy"


def _time_bucket(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "UNKNOWN"
    h = ts.hour + ts.minute / 60.0
    if h < 10.5:
        return "OPEN_915_1030"
    if h < 13.0:
        return "MIDDAY_1030_1300"
    if h < 14.75:
        return "AFTERNOON_1300_1445"
    return "LATE_1445_CLOSE"


def _safe_hit_rate(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return float("nan")
    vals = pd.to_numeric(frame[col], errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _safe_bps(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return float("nan")
    vals = pd.to_numeric(frame[col], errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _selected_frame(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    eff_prob = PREDICTORS[predictor](df)
    working = df.copy()
    working["hybrid_move_probability"] = pd.to_numeric(eff_prob, errors="coerce")
    return apply_selection_policy(working, thresholds=SELECTION_POLICY)


def _group_metrics(
    frame: pd.DataFrame,
    group_col: str,
    predictor: str,
    min_group_size: int = 8,
) -> list[dict]:
    rows: list[dict] = []
    if group_col not in frame.columns:
        return rows

    grouped = frame.groupby(group_col, dropna=False)
    for group, g in grouped:
        n = len(g)
        if n < min_group_size:
            continue
        hit_5m = _safe_hit_rate(g, "correct_5m")
        hit_60m = _safe_hit_rate(g, "correct_60m")
        hit_close = _safe_hit_rate(g, "correct_session_close")
        bps_5m = _safe_bps(g, "signed_return_5m_bps")
        bps_60m = _safe_bps(g, "signed_return_60m_bps")

        rows.append(
            {
                "predictor": predictor,
                "group_col": group_col,
                "group": str(group),
                "n": n,
                "hit_5m": hit_5m,
                "hit_60m": hit_60m,
                "hit_close": hit_close,
                "bps_5m": bps_5m,
                "bps_60m": bps_60m,
            }
        )
    return rows


def _edge_summary(base_df: pd.DataFrame, cand_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    base = pd.DataFrame(_group_metrics(base_df, group_col, BASELINE)).rename(
        columns={
            "n": "n_base",
            "hit_5m": "hit_5m_base",
            "hit_60m": "hit_60m_base",
            "hit_close": "hit_close_base",
            "bps_5m": "bps_5m_base",
            "bps_60m": "bps_60m_base",
        }
    )
    cand = pd.DataFrame(_group_metrics(cand_df, group_col, CANDIDATE)).rename(
        columns={
            "n": "n_cand",
            "hit_5m": "hit_5m_cand",
            "hit_60m": "hit_60m_cand",
            "hit_close": "hit_close_cand",
            "bps_5m": "bps_5m_cand",
            "bps_60m": "bps_60m_cand",
        }
    )

    if base.empty and cand.empty:
        return pd.DataFrame()

    key_cols = ["group_col", "group"]
    keep_base = [c for c in base.columns if c in key_cols or c.endswith("_base")]
    keep_cand = [c for c in cand.columns if c in key_cols or c.endswith("_cand")]

    merged = pd.merge(
        base[keep_base] if not base.empty else pd.DataFrame(columns=key_cols),
        cand[keep_cand] if not cand.empty else pd.DataFrame(columns=key_cols),
        on=key_cols,
        how="outer",
    )

    for col in ["n_base", "n_cand", "hit_5m_base", "hit_5m_cand", "hit_60m_base", "hit_60m_cand", "hit_close_base", "hit_close_cand", "bps_5m_base", "bps_5m_cand", "bps_60m_base", "bps_60m_cand"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["edge_hit_5m_pp"] = (merged["hit_5m_cand"] - merged["hit_5m_base"]) * 100.0
    merged["edge_hit_60m_pp"] = (merged["hit_60m_cand"] - merged["hit_60m_base"]) * 100.0
    merged["edge_hit_close_pp"] = (merged["hit_close_cand"] - merged["hit_close_base"]) * 100.0
    merged["edge_bps_5m"] = merged["bps_5m_cand"] - merged["bps_5m_base"]
    merged["edge_bps_60m"] = merged["bps_60m_cand"] - merged["bps_60m_base"]

    return merged.sort_values(["group_col", "group"]).reset_index(drop=True)


def _consistency_call(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"classification": "insufficient_data", "detail": "No overlapping groups."}

    w = np.minimum(pd.to_numeric(df["n_base"], errors="coerce").fillna(0), pd.to_numeric(df["n_cand"], errors="coerce").fillna(0))
    edge = pd.to_numeric(df["edge_hit_5m_pp"], errors="coerce")
    valid = edge.notna() & (w > 0)
    if valid.sum() == 0:
        return {"classification": "insufficient_data", "detail": "No valid weighted groups."}

    w = w[valid]
    edge = edge[valid]

    weighted_mean = float(np.average(edge, weights=w))
    positive_share = float((edge > 0).mean())
    negative_share = float((edge < 0).mean())

    if weighted_mean >= 2.0 and positive_share >= 0.70:
        cls = "consistent_edge"
    elif weighted_mean >= 0.5 and positive_share >= 0.50:
        cls = "mostly_consistent"
    elif positive_share <= 0.35:
        cls = "not_consistent"
    else:
        cls = "concentrated_or_mixed"

    return {
        "classification": cls,
        "weighted_mean_edge_5m_pp": round(weighted_mean, 3),
        "positive_group_share": round(positive_share, 3),
        "negative_group_share": round(negative_share, 3),
        "valid_group_count": int(valid.sum()),
    }


def main() -> int:
    dataset_path = CUMULATIVE_DATASET_PATH if CUMULATIVE_DATASET_PATH.exists() else SIGNAL_DATASET_PATH
    df = load_signals_dataset(dataset_path)

    if df.empty:
        print("Dataset is empty; nothing to analyze.")
        return 1

    quality_summary = label_quality_summary(df)
    df = apply_quality_label_view(df)

    ts = pd.to_datetime(df.get("signal_timestamp"), errors="coerce")
    df = df.copy()
    df["time_bucket"] = ts.apply(_time_bucket)

    base_sel = _selected_frame(df, BASELINE)
    cand_sel = _selected_frame(df, CANDIDATE)

    group_cols = ["time_bucket", "gamma_regime", "volatility_regime", "global_risk_state", "mode", "source"]
    summaries: dict[str, pd.DataFrame] = {}

    all_rows = []
    for gc in group_cols:
        edge_df = _edge_summary(base_sel, cand_sel, gc)
        summaries[gc] = edge_df
        if not edge_df.empty:
            all_rows.append(edge_df)

    all_edges = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Save artifacts
    csv_path = REPORT_DIR / f"predictor_stability_regime_tod_{TODAY}.csv"
    json_path = REPORT_DIR / f"predictor_stability_regime_tod_{TODAY}.json"

    if not all_edges.empty:
        all_edges.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=["group_col", "group"]).to_csv(csv_path, index=False)

    consistency_overall = _consistency_call(all_edges)
    consistency_by_dim = {gc: _consistency_call(df_gc) for gc, df_gc in summaries.items()}

    payload = {
        "date": TODAY,
        "dataset": str(dataset_path),
        "label_quality_summary": quality_summary,
        "baseline_predictor": BASELINE,
        "candidate_predictor": CANDIDATE,
        "selection_counts": {
            "baseline": int(len(base_sel)),
            "candidate": int(len(cand_sel)),
        },
        "consistency_overall": consistency_overall,
        "consistency_by_dimension": consistency_by_dim,
        "top_positive_edges_5m": (
            all_edges.sort_values("edge_hit_5m_pp", ascending=False)
            .head(12)
            .to_dict(orient="records")
            if not all_edges.empty else []
        ),
        "top_negative_edges_5m": (
            all_edges.sort_values("edge_hit_5m_pp", ascending=True)
            .head(12)
            .to_dict(orient="records")
            if not all_edges.empty else []
        ),
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Console summary
    print("=" * 88)
    print("Stability Check: research_decision_policy vs blended")
    print("=" * 88)
    print(f"Dataset: {dataset_path}")
    print(f"Selected counts -> blended={len(base_sel)}, research_decision_policy={len(cand_sel)}")
    print()

    print("Overall consistency:")
    for k, v in consistency_overall.items():
        print(f"  {k}: {v}")
    print()

    for gc, summary in summaries.items():
        if summary.empty:
            print(f"[{gc}] no overlapping groups with minimum sample size.")
            continue
        print(f"[{gc}] groups={len(summary)}")
        top = summary.sort_values("edge_hit_5m_pp", ascending=False).head(3)
        bot = summary.sort_values("edge_hit_5m_pp", ascending=True).head(3)
        print("  Top +edge (5m hit-rate pp):")
        for _, r in top.iterrows():
            print(
                f"    {r['group']}: {r['edge_hit_5m_pp']:+.2f}pp | n_base={int(r['n_base']) if pd.notna(r['n_base']) else 0}, n_cand={int(r['n_cand']) if pd.notna(r['n_cand']) else 0}"
            )
        print("  Top -edge (5m hit-rate pp):")
        for _, r in bot.iterrows():
            print(
                f"    {r['group']}: {r['edge_hit_5m_pp']:+.2f}pp | n_base={int(r['n_base']) if pd.notna(r['n_base']) else 0}, n_cand={int(r['n_cand']) if pd.notna(r['n_cand']) else 0}"
            )
        print()

    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
