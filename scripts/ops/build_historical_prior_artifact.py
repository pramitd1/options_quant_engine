#!/usr/bin/env python3
"""
Build the live historical-prior artifact from the latest historical insight run.

The artifact is intentionally compact: it carries bucket thresholds and
interaction rows needed by the live engine, while the larger research tables
remain in the historical-insights output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "research" / "ml_research" / "historical_insights" / "latest_historical_insight_report.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "research" / "ml_research" / "historical_priors" / "latest_historical_prior_artifact.json"
ARTIFACT_VERSION = "historical_prior_artifact_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if pd.isna(number):
        return default
    return number


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _bucket_thresholds(series: pd.Series, labels: list[str]) -> list[dict]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return [{"label": label, "upper": None if idx == len(labels) - 1 else None} for idx, label in enumerate(labels)]
    quantiles = values.quantile([idx / len(labels) for idx in range(1, len(labels))]).tolist()
    rows = []
    for idx, label in enumerate(labels):
        upper = None if idx == len(labels) - 1 else round(float(quantiles[idx]), 4)
        rows.append({"label": label, "upper": upper})
    return rows


def _interaction_rows(frame: pd.DataFrame, *, row_field: str, col_field: str) -> dict:
    if frame.empty:
        return {}
    rows = {}
    for item in frame.to_dict(orient="records"):
        row_value = str(item.get("row_value") or "").strip()
        col_value = str(item.get("col_value") or "").strip()
        if not row_value or not col_value or row_value.lower() == "nan" or col_value.lower() == "nan":
            continue
        key = f"{row_value}|{col_value}"
        rows[key] = {
            "n": _safe_int(item.get("n")),
            "mean_bps": round(_safe_float(item.get("mean_bps"), 0.0), 4),
            "hit_up": round(_safe_float(item.get("hit_up"), 0.0), 4),
            "abs_mean_bps": round(_safe_float(item.get("abs_mean_bps"), 0.0), 4),
            row_field: row_value,
            col_field: col_value,
        }
    return rows


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def build_historical_prior_artifact(*, report_path: Path = DEFAULT_REPORT_PATH) -> dict:
    report = _read_json(report_path)
    paths = report.get("paths") if isinstance(report.get("paths"), dict) else {}
    run_dir = PROJECT_ROOT / paths.get("run_dir", "")
    panel_path = PROJECT_ROOT / paths.get("daily_feature_panel", run_dir / "historical_daily_feature_panel.csv")
    panel = _read_csv(panel_path)

    expiry_x_pcr = _read_csv(PROJECT_ROOT / paths.get("interaction_expiry_x_pcr.csv", run_dir / "interaction_expiry_x_pcr.csv"))
    india_vix_x_trend = _read_csv(
        PROJECT_ROOT / paths.get("interaction_india_vix_x_trend.csv", run_dir / "interaction_india_vix_x_trend.csv")
    )
    weekday_x_vix = _read_csv(PROJECT_ROOT / paths.get("interaction_weekday_x_vix.csv", run_dir / "interaction_weekday_x_vix.csv"))

    return {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "historical_insight_mining",
        "source_report_path": str(report_path.relative_to(PROJECT_ROOT)),
        "source_run_id": report.get("run_id"),
        "source_run_dir": paths.get("run_dir"),
        "baseline": {
            "daily_range_bps_mean": 143.6159,
            "daily_abs_return_bps_mean": 86.0521,
        },
        "bucket_thresholds": {
            "pcr_oi": _bucket_thresholds(panel.get("pcr_oi", pd.Series(dtype=float)), ["low", "q2", "q3", "q4", "high"]),
            "india_vix": _bucket_thresholds(
                panel.get("india_vix_level", pd.Series(dtype=float)),
                ["low", "q2", "q3", "q4", "high"],
            ),
            "trend_20d": [
                {"label": "selloff", "upper": -500.0},
                {"label": "weak", "upper": -150.0},
                {"label": "flat", "upper": 150.0},
                {"label": "strong", "upper": 500.0},
                {"label": "surge", "upper": None},
            ],
            "expiry": [
                {"label": "0-1d", "upper": 1.0},
                {"label": "2-3d", "upper": 3.0},
                {"label": "4-7d", "upper": 7.0},
                {"label": "8-14d", "upper": 14.0},
                {"label": "15d+", "upper": None},
            ],
        },
        "interactions": {
            "expiry_x_pcr": {
                "target": "fwd_ret_1d_bps",
                "key_fields": ["expiry_bucket", "pcr_oi_bucket"],
                "rows": _interaction_rows(expiry_x_pcr, row_field="expiry_bucket", col_field="pcr_oi_bucket"),
            },
            "india_vix_x_trend": {
                "target": "fwd_ret_1d_bps",
                "key_fields": ["india_vix_bucket", "trend_20d_bucket"],
                "rows": _interaction_rows(india_vix_x_trend, row_field="india_vix_bucket", col_field="trend_20d_bucket"),
            },
            "weekday_x_vix": {
                "target": "next_day_range_bps",
                "key_fields": ["weekday", "india_vix_bucket"],
                "rows": _interaction_rows(weekday_x_vix, row_field="weekday", col_field="india_vix_bucket"),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    artifact = build_historical_prior_artifact(report_path=args.report)
    _atomic_write_json(args.output, artifact)
    print(json.dumps({
        "artifact_path": str(args.output),
        "artifact_version": artifact.get("artifact_version"),
        "source_run_id": artifact.get("source_run_id"),
        "interaction_counts": {
            name: len((section.get("rows") or {}))
            for name, section in (artifact.get("interactions") or {}).items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
