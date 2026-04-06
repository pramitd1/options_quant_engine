#!/usr/bin/env python3
"""Train and persist calibrator for direction probability head outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset
from strategy.direction_probability_head import compute_direction_probability_head
from strategy.score_calibration import ScoreCalibrator


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models_store" / "direction_probability_calibrator.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "research" / "reviews" / "direction_probability_head"


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _to_upper(value) -> str:
    return str(value or "").upper().strip()


def _pick_target(df: pd.DataFrame) -> pd.Series:
    if "realized_return_60m" in df.columns:
        target = (pd.to_numeric(df["realized_return_60m"], errors="coerce") > 0).astype(float)
        return target
    if "signed_return_60m_bps" in df.columns:
        target = (pd.to_numeric(df["signed_return_60m_bps"], errors="coerce") > 0).astype(float)
        return target
    if "correct_60m" in df.columns:
        target = (pd.to_numeric(df["correct_60m"], errors="coerce") > 0).astype(float)
        return target
    raise KeyError("Dataset missing realized_return_60m/signed_return_60m_bps/correct_60m target columns")


def _vote_bull_probability(row: pd.Series) -> float:
    if "bull_probability" in row and pd.notna(row.get("bull_probability")):
        value = _safe_float(row.get("bull_probability"), 0.5)
        return max(0.0, min(1.0, float(value if value is not None else 0.5)))
    direction = _to_upper(row.get("direction"))
    if direction == "CALL":
        return 0.62
    if direction == "PUT":
        return 0.38
    return 0.5


def _build_head_probability_frame(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    target = _pick_target(df)

    for idx, row in df.iterrows():
        out = compute_direction_probability_head(
            final_flow_signal=row.get("final_flow_signal"),
            spot_vs_flip=row.get("spot_vs_flip"),
            hedging_bias=row.get("hedging_bias"),
            gamma_event=row.get("gamma_event"),
            gamma_regime=row.get("gamma_regime"),
            oi_velocity_score=row.get("oi_velocity_score"),
            rr_value=row.get("rr_value"),
            rr_momentum=row.get("rr_momentum"),
            volume_pcr_atm=row.get("volume_pcr_atm"),
            gamma_flip_drift=row.get("gamma_flip_drift"),
            hybrid_move_probability=row.get("hybrid_move_probability"),
            vote_bull_probability=_vote_bull_probability(row),
            provider_health_summary=row.get("provider_health_status") or row.get("provider_health_summary"),
            provider_health_blocking_status=row.get("provider_health_blocking_status"),
            core_effective_priced_ratio=row.get("core_effective_priced_ratio"),
            core_one_sided_quote_ratio=row.get("core_one_sided_quote_ratio"),
            core_quote_integrity_health=row.get("core_quote_integrity_health"),
            apply_calibration=False,
        )

        records.append(
            {
                "index": idx,
                "signal_id": row.get("signal_id"),
                "source": row.get("source"),
                "gamma_regime": row.get("gamma_regime"),
                "volatility_regime": row.get("volatility_regime") or row.get("vol_regime"),
                "target_up": float(_safe_float(target.get(idx), 0.0) or 0.0),
                "probability_up_raw": _safe_float(out.get("probability_up_raw"), 0.5),
            }
        )

    frame = pd.DataFrame(records)
    frame = frame.loc[frame["probability_up_raw"].notna()].copy()
    return frame


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train direction probability head calibrator")
    parser.add_argument("--dataset", type=Path, default=CUMULATIVE_DATASET_PATH, help="Input cumulative dataset path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Calibrator output JSON path")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Report output directory")
    parser.add_argument("--min-samples", type=int, default=500, help="Minimum rows required for fitting")
    args = parser.parse_args()

    df = load_signals_dataset(args.dataset)
    if df.empty:
        raise RuntimeError("Dataset is empty; cannot train direction probability calibrator")

    work = _build_head_probability_frame(df)
    if len(work) < max(50, int(args.min_samples)):
        raise RuntimeError(f"Insufficient samples for calibrator fit: {len(work)} < {args.min_samples}")

    raw_prob = np.clip(work["probability_up_raw"].to_numpy(dtype=float), 0.0, 1.0)
    y = np.clip(work["target_up"].to_numpy(dtype=float), 0.0, 1.0)

    calibrator = ScoreCalibrator(method="isotonic", n_bins=10)
    fit_report = calibrator.fit((raw_prob * 100.0).tolist(), y.tolist())
    calibrated_prob = np.clip(
        np.asarray(calibrator.calibrate_batch((raw_prob * 100.0).tolist()), dtype=float) / 100.0,
        0.0,
        1.0,
    )

    pre_brier = _brier(y, raw_prob)
    post_brier = _brier(y, calibrated_prob)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save_to_file(str(args.output))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = args.report_dir / f"direction_probability_calibrator_train_{run_id}.json"

    by_source = (
        work.assign(
            calibrated_prob=calibrated_prob,
            pre_brier=(work["target_up"].to_numpy(dtype=float) - raw_prob) ** 2,
            post_brier=(work["target_up"].to_numpy(dtype=float) - calibrated_prob) ** 2,
        )
        .groupby("source", dropna=False)
        .agg(
            samples=("target_up", "size"),
            avg_target_up=("target_up", "mean"),
            avg_prob_raw=("probability_up_raw", "mean"),
            avg_prob_calibrated=("calibrated_prob", "mean"),
            pre_brier=("pre_brier", "mean"),
            post_brier=("post_brier", "mean"),
        )
        .reset_index()
    )

    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_path": str(args.dataset),
        "calibrator_output_path": str(args.output),
        "samples": int(len(work)),
        "target_definition": "realized_return_60m > 0 (fallback: signed_return_60m_bps/correct_60m)",
        "metrics": {
            "pre_brier": round(pre_brier, 8),
            "post_brier": round(post_brier, 8),
            "brier_improvement": round(pre_brier - post_brier, 8),
        },
        "fit_report": fit_report,
        "by_source": by_source.to_dict(orient="records"),
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "calibrator": str(args.output),
        "report": str(report_path),
        "samples": int(len(work)),
        "pre_brier": round(pre_brier, 6),
        "post_brier": round(post_brier, 6),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
