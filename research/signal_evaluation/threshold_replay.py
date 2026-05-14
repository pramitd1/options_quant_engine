"""Backtest-style threshold replay for signal research.

This module evaluates historical signal rows under advisory score/probability
thresholds.  It does not alter runtime decisions or execution behavior.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.signal_evaluation.confidence import outcome_confidence_fields
from research.signal_evaluation.label_quality import apply_quality_label_view
from utils.timestamp_helpers import coerce_timestamp_series


SCORE_THRESHOLD_GRID: dict[str, list[float]] = {
    "composite_signal_score": [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0],
    "tradeability_score": [40.0, 50.0, 60.0, 65.0, 70.0, 75.0, 80.0],
    "trade_strength": [40.0, 50.0, 60.0, 65.0, 70.0, 75.0, 80.0],
    "move_probability": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "hybrid_move_probability": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "ml_rank_score": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "ml_confidence_score": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
}

DEFAULT_REGIME_FIELDS = (
    "macro_regime",
    "gamma_regime",
    "volatility_regime",
    "global_risk_state",
)
DEFAULT_WALK_FORWARD_TRAIN_DAYS = 60
DEFAULT_WALK_FORWARD_HOLDOUT_DAYS = 20
DEFAULT_WALK_FORWARD_STEP_DAYS = 20


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = apply_quality_label_view(frame if frame is not None else pd.DataFrame()).copy()
    if working.empty:
        return working
    if "signal_timestamp" in working.columns:
        working["signal_timestamp"] = coerce_timestamp_series(working["signal_timestamp"])
        working = working.sort_values("signal_timestamp", kind="mergesort")
    for column in set(SCORE_THRESHOLD_GRID) | {"correct_60m", "signed_return_60m_bps"}:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    if "direction" in working.columns:
        direction = working["direction"].astype(str).str.upper().str.strip()
        working = working.loc[direction.isin({"CALL", "PUT"})].copy()
    return working.reset_index(drop=True)


def _thresholds_for_field(series: pd.Series, thresholds: list[float]) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return []
    max_value = float(values.max())
    if max_value > 1.5 and thresholds and max(thresholds) <= 1.0:
        return [round(value * 100.0, 4) for value in thresholds]
    return thresholds


def _max_drawdown_bps(returns: pd.Series) -> float | None:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return None
    cumulative = series.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return _round_or_none(drawdown.min(), 4)


def _objective_score(metrics: dict[str, Any], *, min_label_sample: int) -> float | None:
    label_count = int(metrics.get("label_count_60m") or 0)
    if label_count < int(min_label_sample):
        return None
    avg_return = metrics.get("avg_signed_return_60m_bps")
    if avg_return is None:
        return None
    return_floor = metrics.get("return_ci_low_bps")
    hit_floor = metrics.get("hit_rate_ci_low")
    drawdown = metrics.get("max_drawdown_bps")
    conservative_return = float(return_floor if return_floor is not None else avg_return)
    conservative_hit_bonus = 50.0 * (float(hit_floor if hit_floor is not None else 0.5) - 0.5)
    drawdown_penalty = abs(float(drawdown)) / max(label_count, 1) if drawdown is not None else 0.0
    return _round_or_none(conservative_return + conservative_hit_bonus - (0.10 * drawdown_penalty), 4)


def _metrics_for_subset(
    subset: pd.DataFrame,
    *,
    eligible_count: int,
    min_label_sample: int,
    strong_label_sample: int,
) -> dict[str, Any]:
    hit = pd.to_numeric(subset.get("correct_60m", pd.Series(dtype=float)), errors="coerce")
    returns = pd.to_numeric(subset.get("signed_return_60m_bps", pd.Series(dtype=float)), errors="coerce")
    hit_labeled = hit.dropna()
    return_labeled = returns.dropna()
    label_count = int(max(hit_labeled.count(), return_labeled.count(), 0))
    wins = return_labeled[return_labeled > 0]
    losses = return_labeled[return_labeled <= 0]
    metrics: dict[str, Any] = {
        "signal_count": int(len(subset)),
        "label_count_60m": label_count,
        "retention_ratio": _round_or_none(len(subset) / max(int(eligible_count), 1), 4),
        "hit_rate_60m": _round_or_none(hit_labeled.mean(), 4),
        "avg_signed_return_60m_bps": _round_or_none(return_labeled.mean(), 4),
        "median_signed_return_60m_bps": _round_or_none(return_labeled.median(), 4),
        "sum_signed_return_60m_bps": _round_or_none(return_labeled.sum(), 4),
        "max_drawdown_bps": _max_drawdown_bps(return_labeled),
        "win_rate_by_return": _round_or_none((return_labeled > 0).mean(), 4) if not return_labeled.empty else None,
        "avg_win_bps": _round_or_none(wins.mean(), 4) if not wins.empty else None,
        "avg_loss_bps": _round_or_none(losses.mean(), 4) if not losses.empty else None,
    }
    metrics.update(
        outcome_confidence_fields(
            hit_labeled,
            return_labeled,
            sample_count=label_count,
            min_sample=min_label_sample,
            strong_sample=strong_label_sample,
        )
    )
    metrics["objective_score"] = _objective_score(metrics, min_label_sample=min_label_sample)
    return metrics


def _split_frame(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    split_at = int(len(frame) * float(train_fraction))
    split_at = min(max(split_at, 1), max(len(frame) - 1, 1))
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def _stability_status(train_metrics: dict[str, Any], holdout_metrics: dict[str, Any]) -> str:
    if train_metrics.get("sample_quality") in {"NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"} or holdout_metrics.get("sample_quality") in {
        "NO_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
    }:
        return "INSUFFICIENT_HOLDOUT"
    train_return = train_metrics.get("avg_signed_return_60m_bps")
    holdout_return = holdout_metrics.get("avg_signed_return_60m_bps")
    train_hit = train_metrics.get("hit_rate_60m")
    holdout_hit = holdout_metrics.get("hit_rate_60m")
    if holdout_return is not None and float(holdout_return) < 0:
        return "HOLDOUT_DECAY"
    if holdout_hit is not None and float(holdout_hit) < 0.5:
        return "HOLDOUT_DECAY"
    if train_return is not None and holdout_return is not None and float(train_return) > 0 and float(holdout_return) > 0:
        return "STABLE"
    if train_hit is not None and holdout_hit is not None and float(train_hit) >= 0.5 and float(holdout_hit) >= 0.5:
        return "STABLE"
    return "MIXED"


def _candidate_row(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float | None,
    eligible_count: int,
    min_label_sample: int,
    strong_label_sample: int,
    train_fraction: float,
) -> dict[str, Any]:
    if threshold_value is None:
        selected = frame.copy()
    else:
        selected = frame.loc[pd.to_numeric(frame[threshold_field], errors="coerce") >= float(threshold_value)].copy()

    train, holdout = _split_frame(selected, train_fraction)
    full_metrics = _metrics_for_subset(
        selected,
        eligible_count=eligible_count,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    train_metrics = _metrics_for_subset(
        train,
        eligible_count=max(len(train), 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    holdout_metrics = _metrics_for_subset(
        holdout,
        eligible_count=max(len(holdout), 1),
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
    )
    row = {
        "threshold_field": threshold_field,
        "threshold_value": threshold_value,
        **full_metrics,
        "train_label_count_60m": train_metrics["label_count_60m"],
        "train_hit_rate_60m": train_metrics["hit_rate_60m"],
        "train_avg_signed_return_60m_bps": train_metrics["avg_signed_return_60m_bps"],
        "train_sample_quality": train_metrics["sample_quality"],
        "holdout_label_count_60m": holdout_metrics["label_count_60m"],
        "holdout_hit_rate_60m": holdout_metrics["hit_rate_60m"],
        "holdout_avg_signed_return_60m_bps": holdout_metrics["avg_signed_return_60m_bps"],
        "holdout_sample_quality": holdout_metrics["sample_quality"],
        "stability_status": _stability_status(train_metrics, holdout_metrics),
    }
    return row


def _select_threshold_subset(
    frame: pd.DataFrame,
    *,
    threshold_field: str,
    threshold_value: float | None,
) -> tuple[pd.DataFrame, int]:
    if threshold_field == "ALL_SIGNALS" or threshold_value is None:
        return frame.copy(), int(len(frame))
    if threshold_field not in frame.columns:
        return frame.iloc[0:0].copy(), 0
    values = pd.to_numeric(frame[threshold_field], errors="coerce")
    eligible = frame.loc[values.notna()].copy()
    selected = eligible.loc[pd.to_numeric(eligible[threshold_field], errors="coerce") >= float(threshold_value)].copy()
    return selected, int(len(eligible))


def _window_splits(
    frame: pd.DataFrame,
    *,
    train_window_days: int,
    holdout_window_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    if frame.empty or "signal_timestamp" not in frame.columns:
        return []
    day_key = coerce_timestamp_series(frame["signal_timestamp"]).dt.normalize()
    unique_days = sorted(day_key.dropna().unique())
    train_days = max(int(train_window_days), 1)
    holdout_days = max(int(holdout_window_days), 1)
    step = max(int(step_days), 1)
    total_window = train_days + holdout_days
    if len(unique_days) < total_window:
        return []

    splits: list[dict[str, Any]] = []
    for start in range(0, len(unique_days) - total_window + 1, step):
        train_day_values = unique_days[start : start + train_days]
        holdout_day_values = unique_days[start + train_days : start + total_window]
        train_mask = day_key.isin(train_day_values).fillna(False)
        holdout_mask = day_key.isin(holdout_day_values).fillna(False)
        splits.append(
            {
                "split_id": len(splits) + 1,
                "train": frame.loc[train_mask].copy(),
                "holdout": frame.loc[holdout_mask].copy(),
                "train_start": min(train_day_values).isoformat(),
                "train_end": max(train_day_values).isoformat(),
                "holdout_start": min(holdout_day_values).isoformat(),
                "holdout_end": max(holdout_day_values).isoformat(),
            }
        )
    return splits


def _walk_forward_split_status(
    train_candidate: dict[str, Any],
    holdout_metrics: dict[str, Any],
    *,
    min_holdout_labels: int,
) -> str:
    if train_candidate.get("objective_score") is None:
        return "INSUFFICIENT_TRAIN"
    if int(holdout_metrics.get("label_count_60m") or 0) < int(min_holdout_labels):
        return "INSUFFICIENT_HOLDOUT"
    holdout_return = holdout_metrics.get("avg_signed_return_60m_bps")
    holdout_hit = holdout_metrics.get("hit_rate_60m")
    if holdout_return is not None and float(holdout_return) > 0 and (holdout_hit is None or float(holdout_hit) >= 0.5):
        return "PASS"
    if holdout_return is not None and float(holdout_return) < 0:
        return "FAIL"
    if holdout_hit is not None and float(holdout_hit) < 0.5:
        return "FAIL"
    return "MIXED"


def _walk_forward_summary(rows: list[dict[str, Any]], *, available_split_count: int) -> dict[str, Any]:
    if available_split_count <= 0:
        return {
            "split_count": 0,
            "evaluated_split_count": 0,
            "robustness_status": "INSUFFICIENT_HISTORY",
            "positive_holdout_rate": None,
            "avg_holdout_return_60m_bps": None,
            "avg_holdout_hit_rate_60m": None,
            "worst_holdout_return_60m_bps": None,
            "selection_consistency": None,
            "most_selected_threshold": None,
        }

    evaluated = [
        row
        for row in rows
        if row.get("split_status") not in {"INSUFFICIENT_TRAIN", "INSUFFICIENT_HOLDOUT"}
    ]
    holdout_returns = pd.to_numeric(pd.Series([row.get("holdout_avg_signed_return_60m_bps") for row in evaluated]), errors="coerce").dropna()
    holdout_hits = pd.to_numeric(pd.Series([row.get("holdout_hit_rate_60m") for row in evaluated]), errors="coerce").dropna()
    positive_count = int((holdout_returns > 0).sum()) if not holdout_returns.empty else 0
    positive_rate = _round_or_none(positive_count / max(len(holdout_returns), 1), 4) if not holdout_returns.empty else None

    selections = [
        f"{row.get('threshold_field')}>={row.get('threshold_value')}"
        for row in rows
        if row.get("threshold_field") and row.get("split_status") != "INSUFFICIENT_TRAIN"
    ]
    selection_counts = pd.Series(selections, dtype="object").value_counts() if selections else pd.Series(dtype=int)
    most_selected = None
    consistency = None
    if not selection_counts.empty:
        most_selected = {
            "threshold": str(selection_counts.index[0]),
            "count": int(selection_counts.iloc[0]),
        }
        consistency = _round_or_none(selection_counts.iloc[0] / max(len(selections), 1), 4)

    if not evaluated:
        robustness = "INSUFFICIENT_HOLDOUT"
    elif positive_rate is not None and positive_rate >= 0.70 and _round_or_none(holdout_returns.mean(), 4) is not None and float(holdout_returns.mean()) > 0:
        robustness = "ROBUST"
    elif positive_rate is not None and positive_rate >= 0.50:
        robustness = "MIXED"
    else:
        robustness = "UNSTABLE"

    return {
        "split_count": int(available_split_count),
        "evaluated_split_count": int(len(evaluated)),
        "pass_count": int(sum(1 for row in rows if row.get("split_status") == "PASS")),
        "fail_count": int(sum(1 for row in rows if row.get("split_status") == "FAIL")),
        "positive_holdout_rate": positive_rate,
        "avg_holdout_return_60m_bps": _round_or_none(holdout_returns.mean(), 4) if not holdout_returns.empty else None,
        "avg_holdout_hit_rate_60m": _round_or_none(holdout_hits.mean(), 4) if not holdout_hits.empty else None,
        "worst_holdout_return_60m_bps": _round_or_none(holdout_returns.min(), 4) if not holdout_returns.empty else None,
        "selection_consistency": consistency,
        "most_selected_threshold": most_selected,
        "robustness_status": robustness,
    }


def run_threshold_replay(
    frame: pd.DataFrame,
    *,
    threshold_grid: dict[str, list[float]] | None = None,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    train_fraction: float = 0.70,
    top_n: int | None = 20,
    include_baseline: bool = True,
) -> pd.DataFrame:
    """Evaluate advisory threshold candidates on historical signal outcomes."""
    working = _prepare_frame(frame)
    if working.empty:
        return pd.DataFrame()

    grid = threshold_grid or SCORE_THRESHOLD_GRID
    rows: list[dict[str, Any]] = []
    eligible_count = int(len(working))
    if include_baseline:
        rows.append(
            _candidate_row(
                working,
                threshold_field="ALL_SIGNALS",
                threshold_value=None,
                eligible_count=eligible_count,
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
                train_fraction=train_fraction,
            )
        )

    for field, thresholds in grid.items():
        if field not in working.columns:
            continue
        field_values = pd.to_numeric(working[field], errors="coerce")
        if field_values.dropna().empty:
            continue
        for threshold in _thresholds_for_field(field_values, thresholds):
            rows.append(
                _candidate_row(
                    working.loc[field_values.notna()].copy(),
                    threshold_field=field,
                    threshold_value=float(threshold),
                    eligible_count=int(field_values.notna().sum()),
                    min_label_sample=min_label_sample,
                    strong_label_sample=strong_label_sample,
                    train_fraction=train_fraction,
                )
            )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["is_advisory_candidate"] = result["objective_score"].notna()
    result = result.sort_values(
        ["is_advisory_candidate", "objective_score", "label_count_60m", "signal_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    result["candidate_rank"] = range(1, len(result) + 1)
    if top_n is not None:
        result = result.head(int(top_n)).reset_index(drop=True)
    return result


def run_regime_threshold_replay(
    frame: pd.DataFrame,
    *,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    threshold_grid: dict[str, list[float]] | None = None,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    train_fraction: float = 0.70,
    top_n: int = 20,
    top_n_per_regime: int = 3,
) -> pd.DataFrame:
    """Evaluate threshold candidates within major market-regime buckets."""
    working = _prepare_frame(frame)
    if working.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for field in regime_fields:
        if field not in working.columns:
            continue
        for value, group in working.dropna(subset=[field]).groupby(field, dropna=False):
            replay = run_threshold_replay(
                group,
                threshold_grid=threshold_grid,
                min_label_sample=min_label_sample,
                strong_label_sample=strong_label_sample,
                train_fraction=train_fraction,
                top_n=top_n_per_regime,
                include_baseline=False,
            )
            if replay.empty:
                continue
            replay.insert(0, "regime_value", str(value))
            replay.insert(0, "regime_field", field)
            rows.append(replay)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(
        ["is_advisory_candidate", "objective_score", "label_count_60m", "signal_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    result["candidate_rank"] = range(1, len(result) + 1)
    return result.head(int(top_n)).reset_index(drop=True)


def run_walk_forward_threshold_validation(
    frame: pd.DataFrame,
    *,
    threshold_grid: dict[str, list[float]] | None = None,
    train_window_days: int = DEFAULT_WALK_FORWARD_TRAIN_DAYS,
    holdout_window_days: int = DEFAULT_WALK_FORWARD_HOLDOUT_DAYS,
    step_days: int = DEFAULT_WALK_FORWARD_STEP_DAYS,
    min_train_labels: int = 30,
    min_holdout_labels: int = 10,
    strong_label_sample: int = 100,
    top_n_splits: int | None = 20,
) -> dict[str, Any]:
    """Select thresholds on rolling train windows and score the next holdout window."""
    working = _prepare_frame(frame)
    splits = _window_splits(
        working,
        train_window_days=train_window_days,
        holdout_window_days=holdout_window_days,
        step_days=step_days,
    )
    rows: list[dict[str, Any]] = []
    for split in splits:
        train = split["train"]
        holdout = split["holdout"]
        train_replay = run_threshold_replay(
            train,
            threshold_grid=threshold_grid,
            min_label_sample=min_train_labels,
            strong_label_sample=strong_label_sample,
            top_n=1,
            include_baseline=True,
        )
        if train_replay.empty or not bool(train_replay.iloc[0].get("is_advisory_candidate", False)):
            rows.append(
                {
                    "split_id": split["split_id"],
                    "train_start": split["train_start"],
                    "train_end": split["train_end"],
                    "holdout_start": split["holdout_start"],
                    "holdout_end": split["holdout_end"],
                    "split_status": "INSUFFICIENT_TRAIN",
                    "train_signal_count": int(len(train)),
                    "holdout_signal_count": int(len(holdout)),
                    "threshold_field": None,
                    "threshold_value": None,
                    "train_label_count_60m": 0,
                    "holdout_label_count_60m": 0,
                }
            )
            continue

        candidate = train_replay.iloc[0].to_dict()
        selected_holdout, holdout_eligible_count = _select_threshold_subset(
            holdout,
            threshold_field=str(candidate.get("threshold_field")),
            threshold_value=candidate.get("threshold_value"),
        )
        holdout_metrics = _metrics_for_subset(
            selected_holdout,
            eligible_count=max(holdout_eligible_count, 1),
            min_label_sample=min_holdout_labels,
            strong_label_sample=max(strong_label_sample, min_holdout_labels + 1),
        )
        rows.append(
            {
                "split_id": split["split_id"],
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "holdout_start": split["holdout_start"],
                "holdout_end": split["holdout_end"],
                "split_status": _walk_forward_split_status(
                    candidate,
                    holdout_metrics,
                    min_holdout_labels=min_holdout_labels,
                ),
                "threshold_field": candidate.get("threshold_field"),
                "threshold_value": candidate.get("threshold_value"),
                "train_signal_count": int(candidate.get("signal_count") or 0),
                "train_label_count_60m": int(candidate.get("label_count_60m") or 0),
                "train_hit_rate_60m": candidate.get("hit_rate_60m"),
                "train_avg_signed_return_60m_bps": candidate.get("avg_signed_return_60m_bps"),
                "train_objective_score": candidate.get("objective_score"),
                "holdout_signal_count": int(holdout_metrics.get("signal_count") or 0),
                "holdout_label_count_60m": int(holdout_metrics.get("label_count_60m") or 0),
                "holdout_hit_rate_60m": holdout_metrics.get("hit_rate_60m"),
                "holdout_avg_signed_return_60m_bps": holdout_metrics.get("avg_signed_return_60m_bps"),
                "holdout_return_ci_low_bps": holdout_metrics.get("return_ci_low_bps"),
                "holdout_return_ci_high_bps": holdout_metrics.get("return_ci_high_bps"),
                "holdout_sample_quality": holdout_metrics.get("sample_quality"),
                "holdout_retention_ratio": holdout_metrics.get("retention_ratio"),
            }
        )

    summary = _walk_forward_summary(rows, available_split_count=len(splits))
    visible_rows = rows[: int(top_n_splits)] if top_n_splits is not None else rows
    return {
        "summary": summary,
        "splits": visible_rows,
        "config": {
            "train_window_days": int(train_window_days),
            "holdout_window_days": int(holdout_window_days),
            "step_days": int(step_days),
            "min_train_labels": int(min_train_labels),
            "min_holdout_labels": int(min_holdout_labels),
            "strong_label_sample": int(strong_label_sample),
        },
    }


def build_threshold_replay_summary(
    frame: pd.DataFrame,
    *,
    min_label_sample: int = 30,
    strong_label_sample: int = 100,
    train_fraction: float = 0.70,
    top_n: int = 10,
    walk_forward_train_days: int = DEFAULT_WALK_FORWARD_TRAIN_DAYS,
    walk_forward_holdout_days: int = DEFAULT_WALK_FORWARD_HOLDOUT_DAYS,
    walk_forward_step_days: int = DEFAULT_WALK_FORWARD_STEP_DAYS,
    walk_forward_min_train_labels: int = 30,
    walk_forward_min_holdout_labels: int = 10,
) -> dict[str, Any]:
    """Build JSON-friendly threshold replay diagnostics."""
    candidates = run_threshold_replay(
        frame,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
        train_fraction=train_fraction,
        top_n=top_n,
    )
    regime_candidates = run_regime_threshold_replay(
        frame,
        min_label_sample=min_label_sample,
        strong_label_sample=strong_label_sample,
        train_fraction=train_fraction,
        top_n=top_n,
    )
    walk_forward = run_walk_forward_threshold_validation(
        frame,
        train_window_days=walk_forward_train_days,
        holdout_window_days=walk_forward_holdout_days,
        step_days=walk_forward_step_days,
        min_train_labels=walk_forward_min_train_labels,
        min_holdout_labels=walk_forward_min_holdout_labels,
        strong_label_sample=strong_label_sample,
        top_n_splits=top_n,
    )
    return {
        "threshold_replay_candidates": candidates.to_dict(orient="records"),
        "regime_threshold_replay_candidates": regime_candidates.to_dict(orient="records"),
        "walk_forward_validation": walk_forward,
        "config": {
            "min_label_sample": int(min_label_sample),
            "strong_label_sample": int(strong_label_sample),
            "train_fraction": float(train_fraction),
            "objective": "conservative_return_ci_low_plus_hit_rate_floor_minus_drawdown_penalty",
        },
    }
