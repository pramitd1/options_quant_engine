#!/usr/bin/env python3
"""Train and persist runtime score calibrator from signal_cumul dataset."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset
from strategy.score_calibration import (
    CALIBRATION_SELECTOR_FIELDS,
    RuntimeScoreCalibrator,
    ScoreCalibrator,
    create_calibration_segment_key,
    normalize_calibration_context,
)


MIN_SEGMENT_OBSERVATIONS = 60
CALIBRATION_TARGET_CORRECTNESS_WEIGHT = max(
    0.0,
    min(1.0, float(os.getenv("CALIBRATION_TARGET_CORRECTNESS_WEIGHT", "0.20"))),
)
CALIBRATION_TARGET_UTILITY_WEIGHT = 1.0 - CALIBRATION_TARGET_CORRECTNESS_WEIGHT
MAX_SINGLETON_SEGMENT_ABS_GAP = {
    "direction": float(os.getenv("MAX_DIRECTION_SINGLETON_ABS_GAP", "0.16")),
    "gamma_regime": float(os.getenv("MAX_GAMMA_SINGLETON_ABS_GAP", "0.22")),
    "vol_regime": float(os.getenv("MAX_VOL_SINGLETON_ABS_GAP", "0.25")),
}


def _pick_return_column(columns):
    cols = [str(c) for c in columns]
    preferred = [
        "return_60m_bps",
        "pnl_60m_bps",
        "realized_return_60m_bps",
        "return_60m_pct",
        "pnl_60m_pct",
    ]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        lc = c.lower()
        if "60m" in lc and ("return" in lc or "pnl" in lc):
            return c
    return None


def _fit_isotonic_calibrator(raw_scores, hit_flags):
    calibrator = ScoreCalibrator(method="isotonic", n_bins=10)
    report = calibrator.fit(raw_scores, hit_flags)
    return calibrator, report


def _segment_axis(segment_key: str) -> str | None:
    if not segment_key or "|" in segment_key or "=" not in segment_key:
        return None
    return segment_key.split("=", 1)[0].strip()


def _clone_calibrator(calibrator: ScoreCalibrator) -> ScoreCalibrator:
    return ScoreCalibrator.from_state(calibrator.to_state())


def _build_segment_frame(df: pd.DataFrame) -> pd.DataFrame:
    segment_frame = pd.DataFrame(index=df.index)

    if "direction" in df.columns:
        segment_frame["direction"] = df["direction"].map(
            lambda value: normalize_calibration_context({"direction": value}).get("direction", "UNKNOWN")
        )
    if "gamma_regime" in df.columns:
        segment_frame["gamma_regime"] = df["gamma_regime"].map(
            lambda value: normalize_calibration_context({"gamma_regime": value}).get("gamma_regime", "UNKNOWN")
        )

    vol_source = None
    if "vol_regime" in df.columns:
        vol_source = "vol_regime"
    elif "volatility_regime" in df.columns:
        vol_source = "volatility_regime"
    if vol_source is not None:
        segment_frame["vol_regime"] = df[vol_source].map(
            lambda value: normalize_calibration_context({"vol_regime": value}).get("vol_regime", "UNKNOWN")
        )

    return segment_frame


def _compute_calibration_objective_metrics(raw_scores, targets, calibrator: ScoreCalibrator) -> dict:
    raw = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 100.0)
    y = np.clip(np.asarray(targets, dtype=float), 0.0, 1.0)
    if raw.size == 0 or y.size == 0:
        return {
            "objective_score": None,
            "brier_score": None,
            "ece": None,
            "top_decile_overconfidence": None,
        }

    calibrated = np.clip(np.asarray(calibrator.calibrate_batch(raw.tolist()), dtype=float) / 100.0, 0.0, 1.0)
    brier = float(np.mean((calibrated - y) ** 2))

    frame = pd.DataFrame({"p": calibrated, "y": y})
    n_bins = min(10, int(frame["p"].nunique()))
    if n_bins >= 2:
        binned = frame.assign(bin=pd.qcut(frame["p"], q=n_bins, duplicates="drop"))
        grouped = binned.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(drop=True)
        ece = float((grouped["n"] * (grouped["actual"] - grouped["pred"]).abs()).sum() / grouped["n"].sum())
        top = grouped.iloc[-1]
        top_decile_overconfidence = float(max(float(top["pred"]) - float(top["actual"]), 0.0))
    else:
        ece = 0.0
        top_decile_overconfidence = 0.0

    # Higher is better; bounded by 1.0 in the ideal calibrated case.
    objective = float(1.0 - (0.50 * brier + 0.35 * ece + 0.15 * top_decile_overconfidence))
    return {
        "objective_score": round(objective, 8),
        "brier_score": round(brier, 8),
        "ece": round(float(ece), 8),
        "top_decile_overconfidence": round(float(top_decile_overconfidence), 8),
    }


def _compute_calibration_drift_metrics(df: pd.DataFrame, raw_scores, targets, calibrator: ScoreCalibrator) -> dict:
    if len(raw_scores) < 200:
        return {
            "drift_samples_recent": 0,
            "drift_samples_prior": 0,
            "calibration_gap_recent": None,
            "calibration_gap_prior": None,
            "calibration_gap_abs_delta": None,
            "brier_recent": None,
            "brier_prior": None,
            "brier_delta": None,
        }

    frame = df.copy()
    frame["_raw_score"] = np.asarray(raw_scores, dtype=float)
    frame["_target"] = np.clip(np.asarray(targets, dtype=float), 0.0, 1.0)
    if "signal_timestamp" in frame.columns:
        frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], errors="coerce", format="mixed")
        frame = frame.sort_values("signal_timestamp")

    split_idx = max(int(len(frame) * 0.80), 1)
    prior = frame.iloc[:split_idx].copy()
    recent = frame.iloc[split_idx:].copy()
    if prior.empty or recent.empty:
        return {
            "drift_samples_recent": 0,
            "drift_samples_prior": 0,
            "calibration_gap_recent": None,
            "calibration_gap_prior": None,
            "calibration_gap_abs_delta": None,
            "brier_recent": None,
            "brier_prior": None,
            "brier_delta": None,
        }

    prior_p = np.clip(np.asarray(calibrator.calibrate_batch(prior["_raw_score"].tolist()), dtype=float) / 100.0, 0.0, 1.0)
    recent_p = np.clip(np.asarray(calibrator.calibrate_batch(recent["_raw_score"].tolist()), dtype=float) / 100.0, 0.0, 1.0)

    prior_gap = float(np.mean(prior["_target"].to_numpy(dtype=float) - prior_p))
    recent_gap = float(np.mean(recent["_target"].to_numpy(dtype=float) - recent_p))
    prior_brier = float(np.mean((prior_p - prior["_target"].to_numpy(dtype=float)) ** 2))
    recent_brier = float(np.mean((recent_p - recent["_target"].to_numpy(dtype=float)) ** 2))

    return {
        "drift_samples_recent": int(len(recent)),
        "drift_samples_prior": int(len(prior)),
        "calibration_gap_recent": round(recent_gap, 8),
        "calibration_gap_prior": round(prior_gap, 8),
        "calibration_gap_abs_delta": round(abs(recent_gap - prior_gap), 8),
        "brier_recent": round(recent_brier, 8),
        "brier_prior": round(prior_brier, 8),
        "brier_delta": round(recent_brier - prior_brier, 8),
    }


def main() -> int:
    df = load_signals_dataset(CUMULATIVE_DATASET_PATH)
    if df.empty:
        raise RuntimeError("Cumulative dataset is empty")

    req_cols = ["composite_signal_score", "correct_60m"]
    for col in req_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    raw_scores = pd.to_numeric(df["composite_signal_score"], errors="coerce").fillna(50.0).astype(float).tolist()
    hit_flags = (pd.to_numeric(df["correct_60m"], errors="coerce").fillna(0.0) > 0).astype(float)

    # Utility-aware target: blend directional correctness with realized return quality
    # when return columns are available in the cumulative dataset.
    target_source = "correct_60m"
    return_col = _pick_return_column(df.columns)
    if return_col is not None:
        r = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0).astype(float)
        scale = float(max(r.abs().quantile(0.75), 1e-6))
        utility = 1.0 / (1.0 + np.exp(-(r / scale)))
        calibration_target = (
            (CALIBRATION_TARGET_CORRECTNESS_WEIGHT * hit_flags)
            + (CALIBRATION_TARGET_UTILITY_WEIGHT * utility)
        )
        target_source = (
            f"blend(correct_60m:{CALIBRATION_TARGET_CORRECTNESS_WEIGHT:.2f},"
            f"{return_col}:{CALIBRATION_TARGET_UTILITY_WEIGHT:.2f})"
        )
    else:
        calibration_target = hit_flags

    hit_flags = calibration_target.tolist()

    default_calibrator, report = _fit_isotonic_calibrator(raw_scores, hit_flags)
    report["target_source"] = target_source
    report["tuning_objective"] = _compute_calibration_objective_metrics(raw_scores, hit_flags, default_calibrator)
    report["calibration_drift"] = _compute_calibration_drift_metrics(df, raw_scores, hit_flags, default_calibrator)
    report["segmentation"] = {
        "selector_fields": list(CALIBRATION_SELECTOR_FIELDS),
        "min_segment_observations": int(MIN_SEGMENT_OBSERVATIONS),
        "trained_segments": [],
    }

    segment_frame = _build_segment_frame(df)
    available_fields = [
        field for field in CALIBRATION_SELECTOR_FIELDS
        if field in segment_frame.columns
    ]
    runtime_calibrator = RuntimeScoreCalibrator(
        default_calibrator=default_calibrator,
        selector_fields=list(CALIBRATION_SELECTOR_FIELDS),
    )

    if available_fields:
        raw_scores_series = pd.Series(raw_scores, index=df.index, dtype=float)
        hit_flags_series = pd.Series(hit_flags, index=df.index, dtype=float)
        trained_segments = []

        for combo_size in range(1, len(available_fields) + 1):
            for field_combo in combinations(available_fields, combo_size):
                mask = pd.Series(True, index=df.index)
                for field in field_combo:
                    mask &= segment_frame[field].notna() & (segment_frame[field] != "UNKNOWN")
                if not bool(mask.any()):
                    continue

                grouped = segment_frame.loc[mask, list(field_combo)].groupby(list(field_combo), dropna=True)
                for group_values, group_df in grouped:
                    values_tuple = group_values if isinstance(group_values, tuple) else (group_values,)
                    segment_context = {
                        field: value
                        for field, value in zip(field_combo, values_tuple)
                    }
                    segment_key = create_calibration_segment_key(segment_context)
                    if segment_key == "default" or segment_key in runtime_calibrator.segments:
                        continue

                    segment_index = group_df.index
                    observation_count = int(len(segment_index))
                    if observation_count < MIN_SEGMENT_OBSERVATIONS:
                        continue

                    segment_calibrator, segment_report = _fit_isotonic_calibrator(
                        raw_scores_series.loc[segment_index].tolist(),
                        hit_flags_series.loc[segment_index].tolist(),
                    )
                    trained_gap = float(segment_report.get("overall_calibration_gap", 0.0) or 0.0)
                    deployed_gap = trained_gap
                    deployed_calibrator = segment_calibrator
                    fallback_reason = None

                    axis = _segment_axis(segment_key)
                    threshold = MAX_SINGLETON_SEGMENT_ABS_GAP.get(axis)
                    if threshold is not None and abs(trained_gap) > float(threshold):
                        deployed_calibrator = _clone_calibrator(default_calibrator)
                        deployed_gap = float(report.get("overall_calibration_gap", trained_gap) or trained_gap)
                        fallback_reason = (
                            f"singleton_gap_above_threshold:{axis}:"
                            f"{abs(trained_gap):.6f}>{float(threshold):.6f}"
                        )

                    runtime_calibrator.segments[segment_key] = deployed_calibrator
                    trained_segments.append(
                        {
                            "segment_key": segment_key,
                            "segment_context": segment_context,
                            "n_observations": observation_count,
                            "overall_calibration_gap": deployed_gap,
                            "trained_overall_calibration_gap": trained_gap,
                            "fallback_to_default": bool(fallback_reason),
                            "fallback_reason": fallback_reason,
                            "applied_max_abs_gap": threshold,
                        }
                    )

        report["segmentation"]["trained_segments"] = trained_segments
        report["segmentation"]["singleton_max_abs_gap"] = dict(MAX_SINGLETON_SEGMENT_ABS_GAP)
        report["segmentation"]["fallback_segments"] = int(
            sum(1 for row in trained_segments if bool(row.get("fallback_to_default")))
        )

    out_model = PROJECT_ROOT / "models_store" / "runtime_score_calibrator.json"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    runtime_calibrator.save_to_file(str(out_model))

    out_report = PROJECT_ROOT / "documentation" / "improvement_reports" / f"runtime_score_calibrator_train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    def _json_safe(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        raise TypeError(f"Object of type {type(v).__name__} is not JSON serializable")

    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_safe)

    print(f"Saved calibrator: {out_model}")
    print(f"Saved report: {out_report}")
    print(f"Calibration gap: {report.get('overall_calibration_gap')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
