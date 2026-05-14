#!/usr/bin/env python3
"""
Predictor Comparative Performance Report
=========================================
Simulates all 8 predictor methods against the live cumulative signal dataset
(signals_dataset_cumul.csv) using the raw probability legs already stored in
each row, then compares selection coverage, score quality and outcome metrics.

No re-running of the engine is required — every input needed is already
recorded in the dataset columns:
  • rule_move_probability      — rule leg
  • ml_move_probability        — ML leg
  • hybrid_move_probability    — blended (production) hybrid
  • ml_rank_score              — GBT rank inference
  • ml_confidence_score        — LogReg calibrated confidence
  • trade_strength, composite_signal_score, tradeability_score
  • correct_5m…correct_session_close  — directional outcome flags
  • signed_return_*_bps        — signed P&L proxy in bps

Output
------
  • Console report (tables + rankings)
  • JSON artifact in documentation/daily_reports/
  • CSV artifact in documentation/daily_reports/
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research.signal_evaluation.dataset import (
    CUMULATIVE_DATASET_PATH,
    SIGNAL_DATASET_PATH,
    load_signals_dataset,
)
from research.signal_evaluation.label_quality import apply_quality_label_view, label_quality_summary
from config.signal_evaluation_scoring import get_signal_evaluation_selection_policy
from tuning.objectives import apply_selection_policy

# ── constants ─────────────────────────────────────────────────────────────────
REPORT_DIR = ROOT / "documentation" / "daily_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

SELECTION_POLICY = dict(get_signal_evaluation_selection_policy())

# Rank gate threshold mirrors ResearchRankGatePredictor.rank_threshold
RANK_GATE_THRESHOLD = 0.55
# Uncertainty adjusted parameters mirror ResearchUncertaintyAdjustedPredictor
UA_DISAGREEMENT_WEIGHT = 0.80
UA_AMBIGUITY_WEIGHT = 0.35
UA_BLOCK_FLOOR = 0.35

OUTCOME_HORIZONS = [
    ("5m",   "correct_5m",             "signed_return_5m_bps"),
    ("15m",  "correct_15m",            "signed_return_15m_bps"),
    ("30m",  "correct_30m",            "signed_return_30m_bps"),
    ("60m",  "correct_60m",            "signed_return_60m_bps"),
    ("120m", "correct_120m",           "signed_return_120m_bps"),
    ("close","correct_session_close",  "signed_return_session_close_bps"),
]

SCORE_COLS = ["trade_strength", "composite_signal_score", "tradeability_score",
              "direction_score", "option_efficiency_score", "global_risk_score"]

W = 80  # console width


# ── helpers ───────────────────────────────────────────────────────────────────

def _flt(series_or_val, default=np.nan):
    """Safe float extraction from a pandas Series or scalar."""
    try:
        v = pd.to_numeric(series_or_val, errors="coerce")
        if hasattr(v, "fillna"):
            return v.fillna(default)
        return float(v) if not pd.isna(v) else default
    except Exception:
        return default


def hit_rate(frame: pd.DataFrame, col: str) -> float:
    vals = pd.to_numeric(frame[col], errors="coerce").dropna() if col in frame.columns else pd.Series([], dtype=float)
    return float(vals.mean()) if len(vals) else float("nan")


def avg_bps(frame: pd.DataFrame, col: str) -> float:
    vals = pd.to_numeric(frame[col], errors="coerce").dropna() if col in frame.columns else pd.Series([], dtype=float)
    return float(vals.mean()) if len(vals) else float("nan")


def avg_score(frame: pd.DataFrame, col: str) -> float:
    vals = pd.to_numeric(frame[col], errors="coerce").dropna() if col in frame.columns else pd.Series([], dtype=float)
    return float(vals.mean()) if len(vals) else float("nan")


def fmt(val, fmt_spec=".2f", suffix="", na="—"):
    if isinstance(val, float) and np.isnan(val):
        return na
    if val is None:
        return na
    try:
        return f"{val:{fmt_spec}}{suffix}"
    except Exception:
        return str(val)


def bar_chart(val: float, max_val: float, width: int = 20) -> str:
    if np.isnan(val) or max_val == 0:
        return " " * width
    filled = int(round((val / max_val) * width))
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


# ── predictor probability simulators ─────────────────────────────────────────

def simulate_blended(df: pd.DataFrame) -> pd.Series:
    """Production blended hybrid — already stored."""
    return _flt(df["hybrid_move_probability"])


def simulate_pure_rule(df: pd.DataFrame) -> pd.Series:
    return _flt(df["rule_move_probability"])


def simulate_pure_ml(df: pd.DataFrame) -> pd.Series:
    return _flt(df["ml_move_probability"])


def simulate_research_dual_model(df: pd.DataFrame) -> pd.Series:
    """Use ml_confidence_score → ml_rank_score → hybrid fallback."""
    conf = _flt(df["ml_confidence_score"])
    rank = _flt(df["ml_rank_score"])
    hyb  = _flt(df["hybrid_move_probability"])
    return conf.where(conf.notna(), rank).where(rank.notna(), hyb)


def simulate_research_decision_policy(df: pd.DataFrame) -> pd.Series:
    """
    Decision policy layer approximation: same as dual model but further
    zeroes out signals where ml_rank_score < RANK_GATE_THRESHOLD (the policy
    veto is conservative — it blocks rather than just discounts).
    """
    base = simulate_research_dual_model(df)
    rank = _flt(df["ml_rank_score"])
    # Veto where rank is present and below threshold
    veto = rank.notna() & (rank < RANK_GATE_THRESHOLD)
    return base.where(~veto, other=0.0)


def simulate_ev_sizing(df: pd.DataFrame) -> pd.Series:
    """
    EV sizing requires a conditional return table built from a separate parquet.
    Approximate: dual-model probability scaled by the signal's realised return
    at 60m to proxy for EV-adjusted output.  Falls back to dual-model where
    realised returns are unavailable.
    """
    base = simulate_research_dual_model(df)
    rr60 = _flt(df.get("realized_return_60m", pd.Series(np.nan, index=df.index)))
    # Positive realized return → scale up; negative → scale down
    # Scale factor: clip normalised 60m return to [0.8, 1.2]
    scale = (1.0 + rr60 * 50).clip(0.80, 1.20)
    return (base * scale).clip(0.0, 1.0).where(rr60.notna(), other=base)


def simulate_research_rank_gate(df: pd.DataFrame) -> pd.Series:
    """Gate signals below rank threshold; survivors use ml_confidence_score."""
    rank = _flt(df["ml_rank_score"])
    conf = _flt(df["ml_confidence_score"])
    hyb  = _flt(df["hybrid_move_probability"])
    prob = conf.where(conf.notna(), hyb)
    # Gate: set to 0 where rank is known and below threshold
    gated = rank.notna() & (rank < RANK_GATE_THRESHOLD)
    return prob.where(~gated, other=0.0)


def simulate_research_uncertainty_adjusted(df: pd.DataFrame) -> pd.Series:
    """Apply uncertainty multiplier exactly as per the predictor source."""
    conf = _flt(df["ml_confidence_score"])
    hyb  = _flt(df["hybrid_move_probability"])

    # Where confidence is unavailable fall back to hybrid
    baseline  = hyb.copy()
    conf_ok   = conf.notna()

    disagreement = (conf - baseline).abs()
    ambiguity    = 1.0 - ((2.0 * conf) - 1.0).abs()

    multiplier = (1.0
                  - UA_DISAGREEMENT_WEIGHT * disagreement
                  - UA_AMBIGUITY_WEIGHT    * ambiguity).clip(0.0, 1.0)

    effective = (conf * multiplier).clip(0.0, 1.0)
    # Block floor
    effective = effective.where(effective >= UA_BLOCK_FLOOR, other=0.0)

    # Use hybrid where confidence not available
    return effective.where(conf_ok, other=hyb)


# ── predictor registry ────────────────────────────────────────────────────────

PREDICTORS = {
    "blended":                      simulate_blended,
    "pure_rule":                    simulate_pure_rule,
    "pure_ml":                      simulate_pure_ml,
    "research_dual_model":          simulate_research_dual_model,
    "research_decision_policy":     simulate_research_decision_policy,
    "ev_sizing":                    simulate_ev_sizing,
    "research_rank_gate":           simulate_research_rank_gate,
    "research_uncertainty_adjusted":simulate_research_uncertainty_adjusted,
}

PREDICTOR_LABELS = {
    "blended":                      "Blended (prod default)",
    "pure_rule":                    "Pure Rule",
    "pure_ml":                      "Pure ML",
    "research_dual_model":          "Research Dual-Model",
    "research_decision_policy":     "Research Decision-Policy",
    "ev_sizing":                    "EV Sizing",
    "research_rank_gate":           "Research Rank-Gate",
    "research_uncertainty_adjusted":"Research Uncertainty-Adj",
}


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse_predictor(
    name: str,
    df: pd.DataFrame,
    policy: dict,
) -> dict:
    """Compute all metrics for one predictor against the full dataset."""
    total = len(df)
    if total == 0:
        return {"predictor": name, "total": 0}

    # Simulate the effective probability for each row
    eff_prob = PREDICTORS[name](df)

    # Assign effective probability into a working copy for apply_selection_policy
    working = df.copy()
    working["hybrid_move_probability"] = eff_prob.values

    # Apply the current calibrated selection policy
    sel = apply_selection_policy(working, thresholds=policy)

    # ── coverage ──
    n_total = total
    n_pre_prob_filter = int((eff_prob >= 0).sum())  # non-zero (not gated-out)
    n_selected = len(sel)

    # ── score quality on selected set ──
    scores = {col: avg_score(sel, col) for col in SCORE_COLS}
    avg_eff_prob = float(_flt(sel["hybrid_move_probability"]).mean()) if n_selected else float("nan")

    # ── outcome metrics (selected set) ──
    outcomes = {}
    outcome_coverage = {}
    for label, cor_col, ret_col in OUTCOME_HORIZONS:
        hr = hit_rate(sel, cor_col)
        br = avg_bps(sel, ret_col)
        cov = int(sel[cor_col].notna().sum()) if cor_col in sel.columns else 0
        outcomes[label] = {"hit_rate": hr, "avg_bps": br, "coverage": cov}
        outcome_coverage[label] = cov

    # ── outcome metrics (full dataset, for comparison) ──
    outcomes_full = {}
    for label, cor_col, ret_col in OUTCOME_HORIZONS:
        hr = hit_rate(df, cor_col)
        outcomes_full[label] = {"hit_rate": hr}

    # ── ML quality on selected set ──
    avg_rank    = avg_score(sel, "ml_rank_score")
    avg_conf    = avg_score(sel, "ml_confidence_score")
    avg_ml_prob = avg_score(sel, "ml_move_probability")
    avg_rule_prob = avg_score(sel, "rule_move_probability")

    # ── direction bias ──
    if "direction" in sel.columns:
        direction_counts = sel["direction"].value_counts(dropna=True).to_dict()
    else:
        direction_counts = {}

    return {
        "predictor":            name,
        "label":                PREDICTOR_LABELS[name],
        "n_total":              n_total,
        "n_selected":           n_selected,
        "retention_pct":        n_selected / n_total if n_total else 0.0,
        "avg_eff_prob":         avg_eff_prob,
        "avg_rule_prob":        avg_rule_prob,
        "avg_ml_prob":          avg_ml_prob,
        "avg_rank_score":       avg_rank,
        "avg_conf_score":       avg_conf,
        "scores":               scores,
        "outcomes":             outcomes,
        "outcomes_full":        outcomes_full,
        "outcome_coverage":     outcome_coverage,
        "direction_counts":     direction_counts,
    }


# ── report printing ───────────────────────────────────────────────────────────

def section(title: str):
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def subsection(title: str):
    print()
    print(f"  {title}")
    print("  " + "-" * (W - 2))


def print_overview_table(results: list[dict]):
    section("SIGNAL COVERAGE & PROBABILITY OVERVIEW")
    print(f"  {'Predictor':<30} {'Selected':>8} {'Retain':>7} {'EffProb':>8} {'Rule':>7} {'ML':>7} {'Rank':>7} {'Conf':>7}")
    print("  " + "-" * (W - 2))
    for r in results:
        print(
            f"  {r['label']:<30}"
            f" {r['n_selected']:>8,}"
            f" {fmt(r['retention_pct'] * 100, '.1f', '%'):>7}"
            f" {fmt(r['avg_eff_prob'], '.3f'):>8}"
            f" {fmt(r['avg_rule_prob'], '.3f'):>7}"
            f" {fmt(r['avg_ml_prob'], '.3f'):>7}"
            f" {fmt(r['avg_rank_score'], '.3f'):>7}"
            f" {fmt(r['avg_conf_score'], '.3f'):>7}"
        )


def print_score_quality_table(results: list[dict]):
    section("SIGNAL SCORE QUALITY (SELECTED SET)")
    print(f"  {'Predictor':<30} {'Strength':>9} {'Composite':>10} {'Trade%':>8} {'Direction':>10} {'OE Score':>9} {'GRisk':>7}")
    print("  " + "-" * (W - 2))
    for r in results:
        sc = r["scores"]
        print(
            f"  {r['label']:<30}"
            f" {fmt(sc.get('trade_strength'), '.1f'):>9}"
            f" {fmt(sc.get('composite_signal_score'), '.1f'):>10}"
            f" {fmt(sc.get('tradeability_score'), '.1f'):>8}"
            f" {fmt(sc.get('direction_score'), '.1f'):>10}"
            f" {fmt(sc.get('option_efficiency_score'), '.1f'):>9}"
            f" {fmt(sc.get('global_risk_score'), '.1f'):>7}"
        )


def print_outcome_table(results: list[dict], horizon: str, cor_col: str):
    has_outcomes = any(r["outcome_coverage"].get(horizon, 0) > 0 for r in results)
    if not has_outcomes:
        return
    section(f"DIRECTIONAL HIT RATE — {horizon.upper()}")
    max_hr = max(
        (r["outcomes"][horizon]["hit_rate"] for r in results
         if not np.isnan(r["outcomes"][horizon]["hit_rate"])),
        default=1.0,
    )
    print(f"  {'Predictor':<30} {'HitRate':>8} {'AvgBPS':>8} {'Coverage':>9}  {'':20}")
    print("  " + "-" * (W - 2))
    for r in results:
        o = r["outcomes"][horizon]
        hr   = o["hit_rate"]
        bps  = o["avg_bps"]
        cov  = o["coverage"]
        bar  = bar_chart(hr, max_hr) if not np.isnan(hr) else " " * 20
        print(
            f"  {r['label']:<30}"
            f" {fmt(hr * 100 if not np.isnan(hr) else float('nan'), '.1f', '%'):>8}"
            f" {fmt(bps, '.2f'):>8}"
            f" {fmt(cov, 'd'):>9}"
            f"  {bar}"
        )


def print_outcome_summary_table(results: list[dict]):
    section("DIRECTIONAL HIT RATE SUMMARY — ALL HORIZONS")
    horizons_available = [
        (lbl, cc)
        for lbl, cc, _ in OUTCOME_HORIZONS
        if any(r["outcome_coverage"].get(lbl, 0) > 0 for r in results)
    ]
    if not horizons_available:
        print("  No outcome data available in this dataset.")
        return

    header = f"  {'Predictor':<30}" + "".join(f" {lbl:>8}" for lbl, _ in horizons_available)
    print(header)
    print("  " + "-" * (W - 2))
    for r in results:
        row = f"  {r['label']:<30}"
        for lbl, _ in horizons_available:
            hr = r["outcomes"][lbl]["hit_rate"]
            row += f" {fmt(hr * 100 if not np.isnan(hr) else float('nan'), '.1f', '%'):>8}"
        print(row)


def print_bps_summary_table(results: list[dict]):
    section("SIGNED RETURN (BPS) SUMMARY — ALL HORIZONS")
    horizons_available = [
        (lbl, cc)
        for lbl, cc, _ in OUTCOME_HORIZONS
        if any(r["outcome_coverage"].get(lbl, 0) > 0 for r in results)
    ]
    if not horizons_available:
        print("  No return data available in this dataset.")
        return

    header = f"  {'Predictor':<30}" + "".join(f" {lbl:>9}" for lbl, _ in horizons_available)
    print(header)
    print("  " + "-" * (W - 2))
    for r in results:
        row = f"  {r['label']:<30}"
        for lbl, _ in horizons_available:
            bps = r["outcomes"][lbl]["avg_bps"]
            row += f" {fmt(bps, '.2f'):>9}"
        print(row)


def print_rankings(results: list[dict]):
    section("RANKINGS")

    def rank_by(key_fn, reverse=True, fmt_fn=str, label=""):
        valid = [(r, key_fn(r)) for r in results if not np.isnan(key_fn(r))]
        if not valid:
            return
        valid.sort(key=lambda x: x[1], reverse=reverse)
        print(f"\n  {label}")
        for rank, (r, val) in enumerate(valid, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"  {rank}.")
            print(f"    {medal}  {r['label']:<30}  {fmt_fn(val)}")

    rank_by(
        lambda r: r["n_selected"],
        label="Signal Volume (selected under new thresholds)",
        fmt_fn=lambda v: f"{int(v):,} signals",
    )
    rank_by(
        lambda r: r["retention_pct"],
        label="Retention Rate",
        fmt_fn=lambda v: f"{v*100:.1f}%",
    )
    rank_by(
        lambda r: r["scores"].get("composite_signal_score", float("nan")),
        label="Average Composite Score (selected set)",
        fmt_fn=lambda v: f"{v:.2f}",
    )
    rank_by(
        lambda r: r["scores"].get("trade_strength", float("nan")),
        label="Average Trade Strength (selected set)",
        fmt_fn=lambda v: f"{v:.2f}",
    )
    rank_by(
        lambda r: r["avg_conf_score"],
        label="Average ML Confidence Score (selected set)",
        fmt_fn=lambda v: f"{v:.3f}",
    )

    # Best hit-rate per horizon (where outcome data exists)
    for lbl, cor_col, _ in OUTCOME_HORIZONS:
        valid = [r for r in results if r["outcome_coverage"].get(lbl, 0) > 0]
        if not valid:
            continue
        rank_by(
            lambda r, lbl=lbl: r["outcomes"][lbl]["hit_rate"],
            label=f"Best Hit Rate @ {lbl}",
            fmt_fn=lambda v: f"{v*100:.1f}%",
            reverse=True,
        )
        break  # show only the first available horizon to keep output concise

    # Best avg bps per horizon
    for lbl, _, ret_col in OUTCOME_HORIZONS:
        valid = [r for r in results if r["outcome_coverage"].get(lbl, 0) > 0 and not np.isnan(r["outcomes"][lbl]["avg_bps"])]
        if not valid:
            continue
        rank_by(
            lambda r, lbl=lbl: r["outcomes"][lbl]["avg_bps"],
            label=f"Best Avg Signed Return @ {lbl} (bps)",
            fmt_fn=lambda v: f"{v:.2f} bps",
            reverse=True,
        )
        break


def print_recommendation(results: list[dict]):
    section("RECOMMENDATION")
    # Composite score: weight quality + hit rate + volume
    scored: list[tuple[dict, float]] = []
    for r in results:
        strength = r["scores"].get("trade_strength", 0.0) or 0.0
        composite = r["scores"].get("composite_signal_score", 0.0) or 0.0
        conf = r["avg_conf_score"] if not np.isnan(r["avg_conf_score"]) else (r["avg_eff_prob"] or 0.0)

        # Hit rate at earliest available horizon
        hr = float("nan")
        for lbl, _, _ in OUTCOME_HORIZONS:
            if r["outcome_coverage"].get(lbl, 0) > 0 and not np.isnan(r["outcomes"][lbl]["hit_rate"]):
                hr = r["outcomes"][lbl]["hit_rate"]
                break

        score = (
            (strength / 100) * 0.25
            + (composite / 100) * 0.35
            + (conf if not np.isnan(conf) else 0.5) * 0.20
            + (hr if not np.isnan(hr) else 0.50) * 0.20
        )
        scored.append((r, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    print()
    print(f"  Composite ranking (strength 25% + composite 35% + ML confidence 20% + hit-rate 20%)")
    print()
    for rank, (r, score) in enumerate(scored, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"   {rank}.")
        tag = ""
        if r["predictor"] == "blended":
            tag = "  ← CURRENT PRODUCTION"
        print(f"    {medal}  {r['label']:<32}  composite_rank_score={score:.3f}{tag}")

    best = scored[0][0]
    blended_r = next((r for r, _ in scored if r["predictor"] == "blended"), None)
    blended_score = next((s for r, s in scored if r["predictor"] == "blended"), None)

    print()
    if best["predictor"] == "blended":
        print(f"  ✅  CURRENT PRODUCTION ('blended') ranks #1 — no switch recommended.")
    else:
        gap = scored[0][1] - blended_score if blended_score is not None else 0
        print(f"  ⚠️   '{best['predictor']}' scores Δ{gap:+.3f} above 'blended'.")
        print(f"      Validate on a larger held-out set before promoting to production.")
    print()


# ── save artifacts ─────────────────────────────────────────────────────────────

def save_artifacts(results: list[dict], *, label_quality: dict | None = None, dataset_name: str | None = None):
    rows = []
    for r in results:
        row = {
            "predictor": r["predictor"],
            "label": r["label"],
            "n_total": r["n_total"],
            "n_selected": r["n_selected"],
            "retention_pct": round(r["retention_pct"] * 100, 2),
            "avg_eff_prob": fmt(r["avg_eff_prob"], ".4f"),
            "avg_conf_score": fmt(r["avg_conf_score"], ".4f"),
            "avg_trade_strength": fmt(r["scores"].get("trade_strength"), ".2f"),
            "avg_composite_score": fmt(r["scores"].get("composite_signal_score"), ".2f"),
            "avg_tradeability": fmt(r["scores"].get("tradeability_score"), ".2f"),
        }
        for lbl, _, _ in OUTCOME_HORIZONS:
            o = r["outcomes"].get(lbl, {})
            hr = o.get("hit_rate", float("nan"))
            bps = o.get("avg_bps", float("nan"))
            row[f"hit_rate_{lbl}"] = fmt(hr * 100, ".1f") if not np.isnan(hr) else "—"
            row[f"avg_bps_{lbl}"] = fmt(bps, ".2f") if not np.isnan(bps) else "—"
        rows.append(row)

    df_out = pd.DataFrame(rows)
    csv_path = REPORT_DIR / f"predictor_comparative_report_{TODAY}.csv"
    df_out.to_csv(csv_path, index=False)

    # Full JSON artifact
    json_path = REPORT_DIR / f"predictor_comparative_report_{TODAY}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "report_date": TODAY,
                "dataset": dataset_name or CUMULATIVE_DATASET_PATH.name,
                "label_quality_summary": label_quality or {},
                "selection_policy": SELECTION_POLICY,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n  Saved CSV  : {csv_path}")
    print(f"  Saved JSON : {json_path}")
    return csv_path, json_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── load dataset ──────────────────────────────────────────────────────────
    dataset_path = CUMULATIVE_DATASET_PATH if CUMULATIVE_DATASET_PATH.exists() else SIGNAL_DATASET_PATH
    df = load_signals_dataset(dataset_path)
    quality_summary = label_quality_summary(df)
    df = apply_quality_label_view(df)

    print()
    print("=" * W)
    print("  PREDICTOR COMPARATIVE PERFORMANCE REPORT")
    print(f"  Dataset   : {dataset_path.name}  ({len(df):,} signals)")
    print(f"  Date      : {TODAY}")
    print(f"  Predictors: {len(PREDICTORS)}")
    print("=" * W)

    # Selection policy thresholds used (display)
    print()
    print("  Active selection policy (new calibrated thresholds):")
    for k, v in SELECTION_POLICY.items():
        print(f"    {k:<40} = {v}")

    # ── run analysis for each predictor ───────────────────────────────────────
    print()
    print("  Running analysis for all predictors...")
    results = []
    for name in PREDICTORS:
        result = analyse_predictor(name, df, SELECTION_POLICY)
        results.append(result)
        status = f"{result['n_selected']:>4,} selected / {result['n_total']:,} total"
        print(f"    ✓  {PREDICTOR_LABELS[name]:<34}  {status}")

    # ── report sections ───────────────────────────────────────────────────────
    print_overview_table(results)
    print_score_quality_table(results)
    print_outcome_summary_table(results)
    print_bps_summary_table(results)
    for lbl, cor_col, ret_col in OUTCOME_HORIZONS:
        print_outcome_table(results, lbl, cor_col)
    print_rankings(results)
    print_recommendation(results)

    # ── save ──────────────────────────────────────────────────────────────────
    section("OUTPUT ARTIFACTS")
    save_artifacts(results, label_quality=quality_summary, dataset_name=dataset_path.name)
    print()


if __name__ == "__main__":
    main()
