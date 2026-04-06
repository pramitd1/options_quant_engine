#!/usr/bin/env python3
"""
Directional Stickiness Health Dashboard
======================================

Builds a compact directional-stickiness dashboard with red-line alerts:
1) Excess stickiness
2) Direction imbalance
3) Flip-lag penalty versus short-horizon hit quality

Outputs are saved as Markdown, JSON, and CSV artifacts for auditability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research" / "reviews" / "stickiness_health_dashboard"


@dataclass
class AlertResult:
    name: str
    status: str
    threshold: float
    value: float | None
    note: str


def _safe_ratio(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _directional_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "direction" not in frame.columns:
        raise ValueError("Dataset missing required column: direction")

    working = frame.copy()
    if "signal_timestamp" in working.columns:
        working["signal_timestamp"] = pd.to_datetime(working["signal_timestamp"], errors="coerce")
        working = working.sort_values("signal_timestamp")

    working["dir"] = working["direction"].astype(str).str.upper()
    directional = working[working["dir"].isin(["CALL", "PUT"])].copy()
    return directional


def _stickiness_metrics(directional: pd.DataFrame) -> dict[str, Any]:
    if directional.empty:
        return {
            "directional_rows": 0,
            "stickiness_1step": None,
            "flip_rate_1step": None,
            "stickiness_2step": None,
            "run_count": 0,
            "run_length_mean": None,
            "run_length_median": None,
            "run_length_p75": None,
            "run_length_p90": None,
            "runs_ge3_share": None,
            "runs_ge5_share": None,
            "call_share": None,
            "put_share": None,
        }

    same_prev = directional["dir"].eq(directional["dir"].shift(1))
    stick_1 = _safe_ratio(same_prev.iloc[1:].mean()) if len(directional) > 1 else None
    flip_1 = None if stick_1 is None else 1.0 - stick_1

    same_2back = directional["dir"].eq(directional["dir"].shift(2))
    stick_2 = _safe_ratio(same_2back.iloc[2:].mean()) if len(directional) > 2 else None

    runs: list[int] = []
    current_dir = None
    current_len = 0
    for value in directional["dir"]:
        if value == current_dir:
            current_len += 1
        else:
            if current_dir is not None:
                runs.append(current_len)
            current_dir = value
            current_len = 1
    if current_dir is not None:
        runs.append(current_len)

    run_series = pd.Series(runs, dtype=float)

    mix = directional["dir"].value_counts(normalize=True)

    return {
        "directional_rows": int(len(directional)),
        "stickiness_1step": _round(stick_1),
        "flip_rate_1step": _round(flip_1),
        "stickiness_2step": _round(stick_2),
        "run_count": int(len(runs)),
        "run_length_mean": _round(_safe_ratio(run_series.mean() if not run_series.empty else None), 3),
        "run_length_median": _round(_safe_ratio(run_series.median() if not run_series.empty else None), 3),
        "run_length_p75": _round(_safe_ratio(run_series.quantile(0.75) if not run_series.empty else None), 3),
        "run_length_p90": _round(_safe_ratio(run_series.quantile(0.90) if not run_series.empty else None), 3),
        "runs_ge3_share": _round(_safe_ratio((run_series >= 3).mean() if not run_series.empty else None)),
        "runs_ge5_share": _round(_safe_ratio((run_series >= 5).mean() if not run_series.empty else None)),
        "call_share": _round(_safe_ratio(mix.get("CALL"))),
        "put_share": _round(_safe_ratio(mix.get("PUT"))),
    }


def _horizon_hit_rates(directional: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for horizon in ["5m", "15m", "30m", "60m", "120m", "session_close"]:
        col = f"correct_{horizon}"
        if col in directional.columns:
            series = pd.to_numeric(directional[col], errors="coerce")
            out[horizon] = _round(_safe_ratio(series.mean() if series.notna().any() else None))
        else:
            out[horizon] = None
    return out


def _transition_matrix(directional: pd.DataFrame) -> dict[str, float | None]:
    if len(directional) <= 1:
        return {
            "call_to_call": None,
            "call_to_put": None,
            "put_to_put": None,
            "put_to_call": None,
        }

    previous = directional["dir"].shift(1)
    matrix = pd.crosstab(previous, directional["dir"], normalize="index").fillna(0.0)

    def _cell(r: str, c: str) -> float | None:
        if r not in matrix.index or c not in matrix.columns:
            return None
        return _round(_safe_ratio(matrix.loc[r, c]))

    return {
        "call_to_call": _cell("CALL", "CALL"),
        "call_to_put": _cell("CALL", "PUT"),
        "put_to_put": _cell("PUT", "PUT"),
        "put_to_call": _cell("PUT", "CALL"),
    }


def _recent_window(directional: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if directional.empty or "signal_timestamp" not in directional.columns:
        return directional.iloc[0:0].copy()
    if directional["signal_timestamp"].notna().sum() == 0:
        return directional.iloc[0:0].copy()

    local = directional.copy()
    local["signal_date"] = local["signal_timestamp"].dt.date
    unique_days = sorted(local["signal_date"].dropna().unique())
    if not unique_days:
        return local.iloc[0:0].copy()

    selected_days = set(unique_days[-lookback_days:])
    return local[local["signal_date"].isin(selected_days)].copy()


def _flip_lag_penalty(stickiness_1step: float | None, hit_rates: dict[str, float | None]) -> float | None:
    if stickiness_1step is None:
        return None

    short_horizons = [hit_rates.get("5m"), hit_rates.get("15m"), hit_rates.get("30m")]
    valid = [value for value in short_horizons if value is not None]
    if not valid:
        return None

    reference_hit = max(valid)
    return _round(stickiness_1step - reference_hit)


def _build_alerts(
    *,
    stickiness_1step: float | None,
    direction_imbalance: float | None,
    flip_lag_penalty: float | None,
    max_stickiness: float,
    max_imbalance: float,
    max_flip_lag_penalty: float,
) -> list[AlertResult]:
    alerts: list[AlertResult] = []

    def _status(value: float | None, threshold: float) -> str:
        if value is None:
            return "UNKNOWN"
        return "RED" if value > threshold else "GREEN"

    alerts.append(
        AlertResult(
            name="stickiness_above_threshold",
            status=_status(stickiness_1step, max_stickiness),
            threshold=max_stickiness,
            value=stickiness_1step,
            note="Directional persistence is too high and may suppress adaptive flips.",
        )
    )
    alerts.append(
        AlertResult(
            name="direction_imbalance_above_threshold",
            status=_status(direction_imbalance, max_imbalance),
            threshold=max_imbalance,
            value=direction_imbalance,
            note="CALL/PUT mix drift indicates one-sided directional bias.",
        )
    )
    alerts.append(
        AlertResult(
            name="flip_lag_penalty_above_threshold",
            status=_status(flip_lag_penalty, max_flip_lag_penalty),
            threshold=max_flip_lag_penalty,
            value=flip_lag_penalty,
            note="Persistence exceeds short-horizon hit quality and may indicate late flips.",
        )
    )

    return alerts


def _render_markdown(payload: dict[str, Any]) -> str:
    generated_at = payload["metadata"]["generated_at"]
    dataset_path = payload["metadata"]["dataset_path"]
    total_rows = payload["metadata"]["total_rows"]
    as_of_date = payload["metadata"]["as_of_date"]

    overall = payload["overall"]
    recent = payload["recent_window"]

    alert_rows = payload["alerts"]

    lines: list[str] = []
    lines.append("# Directional Stickiness Health Dashboard")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append(f"- Dataset: {dataset_path}")
    lines.append(f"- As-of date: {as_of_date}")
    lines.append(f"- Total rows: {total_rows}")
    lines.append("")

    lines.append("## Alert Board")
    lines.append("")
    lines.append("| Alert | Status | Value | Threshold |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in alert_rows:
        value = "NA" if row["value"] is None else f"{row['value']:.4f}"
        lines.append(f"| {row['name']} | {row['status']} | {value} | {row['threshold']:.4f} |")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in [
        "directional_rows",
        "stickiness_1step",
        "flip_rate_1step",
        "stickiness_2step",
        "run_count",
        "run_length_mean",
        "run_length_median",
        "run_length_p75",
        "run_length_p90",
        "runs_ge3_share",
        "runs_ge5_share",
        "call_share",
        "put_share",
        "direction_imbalance",
        "flip_lag_penalty",
    ]:
        value = overall.get(key)
        if value is None:
            rendered = "NA"
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")

    lines.append("### Overall Transition Matrix")
    lines.append("")
    lines.append("| Transition | Probability |")
    lines.append("| --- | ---: |")
    for key in ["call_to_call", "call_to_put", "put_to_put", "put_to_call"]:
        value = overall["transition_matrix"].get(key)
        rendered = "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")

    lines.append("### Overall Horizon Hit Rates")
    lines.append("")
    lines.append("| Horizon | Hit Rate |")
    lines.append("| --- | ---: |")
    for horizon, value in overall["horizon_hit_rates"].items():
        rendered = "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {horizon} | {rendered} |")
    lines.append("")

    lines.append(f"## Recent Window ({recent['lookback_days']} trading days)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in [
        "directional_rows",
        "stickiness_1step",
        "flip_rate_1step",
        "stickiness_2step",
        "call_share",
        "put_share",
        "direction_imbalance",
        "flip_lag_penalty",
    ]:
        value = recent.get(key)
        if value is None:
            rendered = "NA"
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")

    lines.append("### Recent Transition Matrix")
    lines.append("")
    lines.append("| Transition | Probability |")
    lines.append("| --- | ---: |")
    for key in ["call_to_call", "call_to_put", "put_to_put", "put_to_call"]:
        value = recent["transition_matrix"].get(key)
        rendered = "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def _daily_metrics_csv(directional: pd.DataFrame, output_csv: Path) -> None:
    if directional.empty or "signal_timestamp" not in directional.columns:
        pd.DataFrame().to_csv(output_csv, index=False)
        return

    working = directional.copy()
    working["signal_date"] = working["signal_timestamp"].dt.date

    rows: list[dict[str, Any]] = []
    for signal_date, group in working.groupby("signal_date"):
        if pd.isna(signal_date):
            continue
        same_prev = group["dir"].eq(group["dir"].shift(1))
        stickiness = _safe_ratio(same_prev.iloc[1:].mean()) if len(group) > 1 else None
        mix = group["dir"].value_counts(normalize=True)

        horizon = {}
        for h in ["5m", "15m", "30m"]:
            col = f"correct_{h}"
            if col in group.columns:
                s = pd.to_numeric(group[col], errors="coerce")
                horizon[h] = _safe_ratio(s.mean() if s.notna().any() else None)
            else:
                horizon[h] = None

        call_share = _safe_ratio(mix.get("CALL"))
        put_share = _safe_ratio(mix.get("PUT"))
        direction_imbalance = None
        if call_share is not None and put_share is not None:
            direction_imbalance = abs(put_share - call_share)

        flip_lag_penalty = _flip_lag_penalty(stickiness, {k: _safe_ratio(v) for k, v in horizon.items()})

        rows.append(
            {
                "signal_date": str(signal_date),
                "directional_rows": int(len(group)),
                "stickiness_1step": _round(stickiness),
                "flip_rate_1step": _round(None if stickiness is None else 1.0 - stickiness),
                "call_share": _round(call_share),
                "put_share": _round(put_share),
                "direction_imbalance": _round(direction_imbalance),
                "hit_5m": _round(_safe_ratio(horizon.get("5m"))),
                "hit_15m": _round(_safe_ratio(horizon.get("15m"))),
                "hit_30m": _round(_safe_ratio(horizon.get("30m"))),
                "flip_lag_penalty": _round(flip_lag_penalty),
            }
        )

    out = pd.DataFrame(rows).sort_values("signal_date")
    out.to_csv(output_csv, index=False)


def build_dashboard(
    *,
    dataset_path: Path,
    output_dir: Path,
    lookback_days: int,
    max_stickiness: float,
    max_imbalance: float,
    max_flip_lag_penalty: float,
) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path)
    directional = _directional_frame(frame)
    recent = _recent_window(directional, lookback_days=lookback_days)

    overall_metrics = _stickiness_metrics(directional)
    overall_hit_rates = _horizon_hit_rates(directional)
    overall_transitions = _transition_matrix(directional)

    recent_metrics = _stickiness_metrics(recent)
    recent_hit_rates = _horizon_hit_rates(recent)
    recent_transitions = _transition_matrix(recent)

    overall_imbalance = None
    if overall_metrics["call_share"] is not None and overall_metrics["put_share"] is not None:
        overall_imbalance = _round(abs(overall_metrics["put_share"] - overall_metrics["call_share"]))

    recent_imbalance = None
    if recent_metrics["call_share"] is not None and recent_metrics["put_share"] is not None:
        recent_imbalance = _round(abs(recent_metrics["put_share"] - recent_metrics["call_share"]))

    overall_flip_lag = _flip_lag_penalty(overall_metrics["stickiness_1step"], overall_hit_rates)
    recent_flip_lag = _flip_lag_penalty(recent_metrics["stickiness_1step"], recent_hit_rates)

    overall_metrics["direction_imbalance"] = overall_imbalance
    overall_metrics["flip_lag_penalty"] = overall_flip_lag
    overall_metrics["horizon_hit_rates"] = overall_hit_rates
    overall_metrics["transition_matrix"] = overall_transitions

    recent_metrics["direction_imbalance"] = recent_imbalance
    recent_metrics["flip_lag_penalty"] = recent_flip_lag
    recent_metrics["horizon_hit_rates"] = recent_hit_rates
    recent_metrics["transition_matrix"] = recent_transitions

    alerts = _build_alerts(
        stickiness_1step=overall_metrics["stickiness_1step"],
        direction_imbalance=overall_imbalance,
        flip_lag_penalty=overall_flip_lag,
        max_stickiness=max_stickiness,
        max_imbalance=max_imbalance,
        max_flip_lag_penalty=max_flip_lag_penalty,
    )

    as_of_date = None
    if "signal_timestamp" in directional.columns and directional["signal_timestamp"].notna().any():
        as_of_date = str(directional["signal_timestamp"].max().date())

    payload = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "dataset_path": str(dataset_path),
            "total_rows": int(len(frame)),
            "directional_rows": int(len(directional)),
            "as_of_date": as_of_date,
            "lookback_days": int(lookback_days),
            "thresholds": {
                "max_stickiness": max_stickiness,
                "max_imbalance": max_imbalance,
                "max_flip_lag_penalty": max_flip_lag_penalty,
            },
        },
        "overall": overall_metrics,
        "recent_window": {
            "lookback_days": int(lookback_days),
            **recent_metrics,
        },
        "alerts": [asdict(alert) for alert in alerts],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    md_path = output_dir / f"stickiness_health_dashboard_{ts_tag}.md"
    json_path = output_dir / f"stickiness_health_dashboard_{ts_tag}.json"
    csv_path = output_dir / f"stickiness_daily_metrics_{ts_tag}.csv"

    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _daily_metrics_csv(directional, csv_path)

    payload["artifacts"] = {
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate directional stickiness health dashboard artifacts.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET), help="Path to signal dataset CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output artifacts.")
    parser.add_argument("--lookback-days", type=int, default=5, help="Recent trading-day lookback for secondary metrics.")
    parser.add_argument("--max-stickiness", type=float, default=0.90, help="Red-line threshold for 1-step stickiness.")
    parser.add_argument("--max-imbalance", type=float, default=0.20, help="Red-line threshold for absolute CALL/PUT imbalance.")
    parser.add_argument(
        "--max-flip-lag-penalty",
        type=float,
        default=0.35,
        help="Red-line threshold for stickiness minus best short-horizon hit-rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = build_dashboard(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        lookback_days=args.lookback_days,
        max_stickiness=args.max_stickiness,
        max_imbalance=args.max_imbalance,
        max_flip_lag_penalty=args.max_flip_lag_penalty,
    )

    print("Directional stickiness health dashboard generated.")
    print(f"dataset_path: {args.dataset_path}")
    print(f"total_rows: {payload['metadata']['total_rows']}")
    print(f"directional_rows: {payload['metadata']['directional_rows']}")
    print("alerts:")
    for row in payload["alerts"]:
        value = "NA" if row["value"] is None else f"{row['value']:.4f}"
        print(f"  - {row['name']}: {row['status']} (value={value}, threshold={row['threshold']:.4f})")
    print(f"markdown_path: {payload['artifacts']['markdown_path']}")
    print(f"json_path: {payload['artifacts']['json_path']}")
    print(f"csv_path: {payload['artifacts']['csv_path']}")


if __name__ == "__main__":
    main()
