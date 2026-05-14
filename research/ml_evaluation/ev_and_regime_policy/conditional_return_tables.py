"""
Conditional Return Tables
==========================
Builds historical conditional lookup tables that estimate expected gain
and expected loss for each signal, grouped by combinations of:

  • GBT rank bucket          (Q1_lowest … Q5_highest)
  • LogReg confidence bucket (Q1_lowest … Q5_highest)
  • Regime composite         (gamma_regime × volatility_regime)

Design philosophy
-----------------
*  Hierarchical back-off:  if a fine-grained bucket has too few samples,
   fall back to a broader parent group average.
*  Smoothing:  Bayesian-style shrinkage toward the global mean when the
   local sample count is small.
*  Sample-support tracking:  every cell carries its raw sample count so
   downstream consumers can gauge reliability.

Author: Pramit Dutta
Organization: Quant Engines

RESEARCH ONLY — never imported by production engine paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from research.signal_evaluation.label_quality import apply_quality_label_view

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────

MIN_BUCKET_SAMPLES: int = 10
"""Minimum observations in a bucket before it can stand alone."""

SMOOTHING_WEIGHT: float = 0.30
"""Weight applied to the global prior when blending with local estimate.
   local_estimate = (1 - w) * local_mean + w * global_mean
   Set to 0 to disable smoothing."""

RANK_BUCKET_LABELS = ["Q1_lowest", "Q2_low", "Q3_mid", "Q4_high", "Q5_highest"]
CONFIDENCE_BUCKET_LABELS = ["Q1_lowest", "Q2_low", "Q3_mid", "Q4_high", "Q5_highest"]

# Regime columns used for groupby — values come from the signal dataset.
REGIME_GROUP_COL = "gamma_regime"
SECONDARY_REGIME_COL = "volatility_regime"


# ── Data structures ──────────────────────────────────────────────────

@dataclass(frozen=True)
class BucketStats:
    """Conditional statistics for a single bucket cell."""
    n: int
    hit_rate: float
    avg_positive_return_bps: float
    avg_negative_return_bps: float   # always ≤ 0
    avg_mfe_bps: float
    avg_mae_bps: float               # always ≤ 0
    avg_return_bps: float
    smoothed: bool = False
    backed_off: bool = False
    parent_label: str = ""


@dataclass
class ConditionalReturnTable:
    """Complete lookup table indexed by (rank_bucket, confidence_bucket, regime)."""
    cells: dict[tuple[str, str, str], BucketStats] = field(default_factory=dict)
    global_stats: BucketStats | None = None
    parent_tables: dict[str, dict[str, BucketStats]] = field(default_factory=dict)
    build_meta: dict[str, Any] = field(default_factory=dict)


# ── Internal helpers ─────────────────────────────────────────────────

def _safe_mean(s: pd.Series, default: float = 0.0) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else default


def _compute_cell(sub: pd.DataFrame, hit_col: str, ret_col: str,
                  mfe_col: str, mae_col: str) -> BucketStats:
    """Compute raw stats from a subset DataFrame."""
    n = len(sub)
    hit = pd.to_numeric(sub[hit_col], errors="coerce")
    ret = pd.to_numeric(sub[ret_col], errors="coerce").dropna()
    mfe = pd.to_numeric(sub[mfe_col], errors="coerce").dropna() if mfe_col in sub.columns else pd.Series(dtype=float)
    mae = pd.to_numeric(sub[mae_col], errors="coerce").dropna() if mae_col in sub.columns else pd.Series(dtype=float)

    pos = ret[ret > 0]
    neg = ret[ret <= 0]

    return BucketStats(
        n=n,
        hit_rate=float(hit.mean()) if len(hit.dropna()) > 0 else 0.5,
        avg_positive_return_bps=float(pos.mean()) if len(pos) > 0 else 0.0,
        avg_negative_return_bps=float(neg.mean()) if len(neg) > 0 else 0.0,
        avg_mfe_bps=float(mfe.mean()) if len(mfe) > 0 else 0.0,
        avg_mae_bps=float(mae.mean()) if len(mae) > 0 else 0.0,
        avg_return_bps=float(ret.mean()) if len(ret) > 0 else 0.0,
    )


def _smooth(local: BucketStats, glob: BucketStats, weight: float) -> BucketStats:
    """Bayesian-style shrinkage toward the global prior."""
    w = weight
    return BucketStats(
        n=local.n,
        hit_rate=(1 - w) * local.hit_rate + w * glob.hit_rate,
        avg_positive_return_bps=(1 - w) * local.avg_positive_return_bps + w * glob.avg_positive_return_bps,
        avg_negative_return_bps=(1 - w) * local.avg_negative_return_bps + w * glob.avg_negative_return_bps,
        avg_mfe_bps=(1 - w) * local.avg_mfe_bps + w * glob.avg_mfe_bps,
        avg_mae_bps=(1 - w) * local.avg_mae_bps + w * glob.avg_mae_bps,
        avg_return_bps=(1 - w) * local.avg_return_bps + w * glob.avg_return_bps,
        smoothed=True,
    )


# ── Public API ───────────────────────────────────────────────────────

def build_conditional_return_table(
    df: pd.DataFrame,
    *,
    hit_col: str = "correct_60m",
    ret_col: str = "signed_return_60m_bps",
    mfe_col: str = "mfe_60m_bps",
    mae_col: str = "mae_60m_bps",
    rank_col: str = "ml_rank_bucket",
    confidence_col: str = "ml_confidence_bucket",
    regime_col: str = REGIME_GROUP_COL,
    min_samples: int = MIN_BUCKET_SAMPLES,
    smoothing: float = SMOOTHING_WEIGHT,
) -> ConditionalReturnTable:
    """
    Build a hierarchical conditional return table from the signal dataset.

    Parameters
    ----------
    df : DataFrame with columns for rank/confidence buckets, regime, outcomes.
    hit_col, ret_col, mfe_col, mae_col : outcome column names.
    rank_col, confidence_col, regime_col : grouping column names.
    min_samples : minimum samples before a cell is used without back-off.
    smoothing : shrinkage weight toward global prior (0 = no smoothing).

    Returns
    -------
    ConditionalReturnTable with per-cell BucketStats, global stats, and
    parent-level (single-dimension) fallback tables.
    """
    if hit_col == "correct_60m" and ret_col == "signed_return_60m_bps":
        df = apply_quality_label_view(df)
    else:
        df = df.copy()

    # Ensure outcome columns are numeric
    for col in [hit_col, ret_col, mfe_col, mae_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to rows with valid outcomes
    valid = df.dropna(subset=[hit_col, ret_col])
    if len(valid) == 0:
        logger.warning("No valid outcome rows — returning empty table.")
        return ConditionalReturnTable(build_meta={"n_total": 0, "n_valid": 0})

    # ── Global stats ────────────────────────────────────────────────
    glob = _compute_cell(valid, hit_col, ret_col, mfe_col, mae_col)

    # ── Parent-level tables (single dimension) ──────────────────────
    parent_tables: dict[str, dict[str, BucketStats]] = {}
    for dim_name, dim_col in [("rank", rank_col), ("confidence", confidence_col), ("regime", regime_col)]:
        if dim_col not in valid.columns:
            continue
        tbl: dict[str, BucketStats] = {}
        for label, grp in valid.groupby(dim_col, observed=True):
            if len(grp) >= 3:
                tbl[str(label)] = _compute_cell(grp, hit_col, ret_col, mfe_col, mae_col)
        parent_tables[dim_name] = tbl

    # ── Full 3-way cells ────────────────────────────────────────────
    cells: dict[tuple[str, str, str], BucketStats] = {}
    group_cols = [c for c in [rank_col, confidence_col, regime_col] if c in valid.columns]

    if len(group_cols) == 0:
        logger.warning("No grouping columns found — only global stats will be available.")
        return ConditionalReturnTable(
            cells={},
            global_stats=glob,
            parent_tables=parent_tables,
            build_meta={"n_total": len(df), "n_valid": len(valid), "n_cells": 0},
        )

    for key, grp in valid.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        # Pad to 3-element key
        padded = tuple(str(k) for k in key) + ("ALL",) * (3 - len(key))

        raw = _compute_cell(grp, hit_col, ret_col, mfe_col, mae_col)
        if raw.n >= min_samples:
            cell = _smooth(raw, glob, smoothing) if smoothing > 0 else raw
        else:
            # Back off: try rank-only parent
            rank_label = padded[0] if rank_col in group_cols else "ALL"
            parent = parent_tables.get("rank", {}).get(rank_label)
            if parent is not None and parent.n >= min_samples:
                cell = BucketStats(
                    n=raw.n,
                    hit_rate=parent.hit_rate,
                    avg_positive_return_bps=parent.avg_positive_return_bps,
                    avg_negative_return_bps=parent.avg_negative_return_bps,
                    avg_mfe_bps=parent.avg_mfe_bps,
                    avg_mae_bps=parent.avg_mae_bps,
                    avg_return_bps=parent.avg_return_bps,
                    smoothed=False,
                    backed_off=True,
                    parent_label=f"rank={rank_label}",
                )
            else:
                # Fall back to global
                cell = BucketStats(
                    n=raw.n,
                    hit_rate=glob.hit_rate,
                    avg_positive_return_bps=glob.avg_positive_return_bps,
                    avg_negative_return_bps=glob.avg_negative_return_bps,
                    avg_mfe_bps=glob.avg_mfe_bps,
                    avg_mae_bps=glob.avg_mae_bps,
                    avg_return_bps=glob.avg_return_bps,
                    smoothed=False,
                    backed_off=True,
                    parent_label="global",
                )
        cells[padded] = cell

    logger.info(
        "Built conditional return table: %d cells, %d backed-off, %d smoothed",
        len(cells),
        sum(1 for c in cells.values() if c.backed_off),
        sum(1 for c in cells.values() if c.smoothed),
    )

    return ConditionalReturnTable(
        cells=cells,
        global_stats=glob,
        parent_tables=parent_tables,
        build_meta={
            "n_total": len(df),
            "n_valid": len(valid),
            "n_cells": len(cells),
            "n_backed_off": sum(1 for c in cells.values() if c.backed_off),
            "n_smoothed": sum(1 for c in cells.values() if c.smoothed),
            "min_samples": min_samples,
            "smoothing_weight": smoothing,
        },
    )


def lookup(
    table: ConditionalReturnTable,
    rank_bucket: str,
    confidence_bucket: str,
    regime: str,
) -> BucketStats:
    """
    Look up the conditional stats for a given (rank, confidence, regime) triplet.

    Falls back through:
      1. exact 3-way match
      2. (rank, confidence, ALL)
      3. rank parent table
      4. global stats
    """
    # Exact match
    key = (rank_bucket, confidence_bucket, regime)
    if key in table.cells:
        return table.cells[key]

    # Drop regime
    key_no_regime = (rank_bucket, confidence_bucket, "ALL")
    if key_no_regime in table.cells:
        return table.cells[key_no_regime]

    # Rank parent
    rank_parent = table.parent_tables.get("rank", {}).get(rank_bucket)
    if rank_parent is not None:
        return rank_parent

    # Global fallback
    if table.global_stats is not None:
        return table.global_stats

    # Absolute fallback
    return BucketStats(
        n=0, hit_rate=0.5,
        avg_positive_return_bps=0.0, avg_negative_return_bps=0.0,
        avg_mfe_bps=0.0, avg_mae_bps=0.0, avg_return_bps=0.0,
        backed_off=True, parent_label="empty_fallback",
    )


def table_to_records(table: ConditionalReturnTable) -> list[dict[str, Any]]:
    """Serialize all cells to a flat list of dicts for CSV / JSON export."""
    rows: list[dict[str, Any]] = []
    for (rank_b, conf_b, regime), cell in sorted(table.cells.items()):
        rows.append({
            "rank_bucket": rank_b,
            "confidence_bucket": conf_b,
            "regime": regime,
            "n": cell.n,
            "hit_rate": round(cell.hit_rate, 4),
            "avg_positive_return_bps": round(cell.avg_positive_return_bps, 2),
            "avg_negative_return_bps": round(cell.avg_negative_return_bps, 2),
            "avg_mfe_bps": round(cell.avg_mfe_bps, 2),
            "avg_mae_bps": round(cell.avg_mae_bps, 2),
            "avg_return_bps": round(cell.avg_return_bps, 2),
            "smoothed": cell.smoothed,
            "backed_off": cell.backed_off,
            "parent_label": cell.parent_label,
        })
    return rows
