from __future__ import annotations

from scripts._bootstrap import ensure_project_root_on_path
ensure_project_root_on_path(__file__)

import argparse
import json
from pathlib import Path

import pandas as pd

from config.signal_policy import get_trade_runtime_thresholds


def _extract_signal_date(frame: pd.DataFrame) -> pd.Series:
    if "signal_date" in frame.columns:
        return pd.to_datetime(frame["signal_date"], errors="coerce").dt.date

    for candidate in ["signal_timestamp", "timestamp", "as_of", "valuation_time", "signal_time"]:
        if candidate in frame.columns:
            return pd.to_datetime(frame[candidate], errors="coerce").dt.date

    raise ValueError("No valid date column found for daily readiness dashboard")


def summarize_daily_readiness(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    frame = frame.copy()
    frame["signal_date"] = _extract_signal_date(frame)
    frame = frame.dropna(subset=["signal_date"])
    frame["direction"] = frame.get("direction", "UNKNOWN").astype(str).str.upper().fillna("UNKNOWN")
    frame["trade_status"] = frame.get("trade_status", "UNKNOWN").astype(str).str.upper().fillna("UNKNOWN")

    daily_rows: list[dict[str, object]] = []
    for signal_date, group in frame.groupby("signal_date"):
        total_signals = len(group)
        qualified_signals = group[group["trade_status"] == "TRADE"]
        qualified_count = len(qualified_signals)
        suppression_rate = 100.0 * max(0.0, 1.0 - qualified_count / max(total_signals, 1))
        composite_75th = float(group["runtime_composite_score"].dropna().quantile(0.75)) if "runtime_composite_score" in group else 0.0
        call_count = len(group[group["direction"] == "CALL"])
        put_count = len(group[group["direction"] == "PUT"])
        directional_balance = float(call_count - put_count) / max(total_signals, 1)
        call_put_ratio = float(call_count / max(call_count + put_count, 1))
        average_confidence = float(group["confidence_score"].dropna().mean()) if "confidence_score" in group else 0.0

        daily_rows.append(
            {
                "signal_date": signal_date.isoformat(),
                "total_signals": total_signals,
                "qualified_signals": qualified_count,
                "suppression_rate_pct": round(suppression_rate, 2),
                "runtime_composite_score_75th_pctile": round(composite_75th, 2),
                "directional_balance": round(directional_balance, 4),
                "call_put_ratio": round(call_put_ratio, 4),
                "average_confidence_score": round(average_confidence, 2),
            }
        )

    daily_frame = pd.DataFrame(daily_rows).sort_values("signal_date")
    return daily_frame


def detect_daily_readiness_anomalies(daily_frame: pd.DataFrame, thresholds: dict[str, float]) -> list[dict[str, object]]:
    anomalies: list[dict[str, object]] = []
    for _, row in daily_frame.iterrows():
        flags: list[str] = []
        if row["qualified_signals"] < int(thresholds.get("daily_min_qualified_signals", 6)):
            flags.append("qualified_signal_volume_below_threshold")
        if row["runtime_composite_score_75th_pctile"] < float(thresholds.get("daily_min_composite_score_75th_pctl", 65.0)):
            flags.append("low_composite_score_tail")
        if row["suppression_rate_pct"] > float(thresholds.get("daily_min_suppression_rate_pct", 85.0)):
            flags.append("suppression_rate_excessive")
        if abs(row["directional_balance"]) > float(thresholds.get("daily_call_put_ratio_target", 0.50)) + 0.10:
            flags.append("directional_imbalance")
        if row["average_confidence_score"] < float(thresholds.get("daily_min_average_confidence", 50.0)):
            flags.append("average_confidence_below_threshold")
        if flags:
            anomalies.append({"signal_date": row["signal_date"], "issues": flags})
    return anomalies


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a daily readiness dashboard from cumulative signal output.")
    parser.add_argument("--input-file", required=True, help="Path to the signal dataset CSV or parquet file.")
    parser.add_argument("--output-dir", default=".", help="Directory for dashboard artifacts.")
    parser.add_argument("--write-json", action="store_true", help="Write summary and anomalies to JSON files.")
    args = parser.parse_args()

    thresholds = get_trade_runtime_thresholds()
    path = Path(args.input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)

    summary = summarize_daily_readiness(frame, thresholds)
    anomalies = detect_daily_readiness_anomalies(summary, thresholds)

    if args.write_json:
        summary_path = Path(args.output_dir) / "daily_readiness_summary.json"
        anomalies_path = Path(args.output_dir) / "daily_readiness_anomalies.json"
        summary.to_json(summary_path, orient="records", date_format="iso")
        with open(anomalies_path, "w", encoding="utf-8") as handle:
            json.dump(anomalies, handle, indent=2)
        print(f"Wrote daily readiness summary to {summary_path}")
        print(f"Wrote daily readiness anomalies to {anomalies_path}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_file = output_path / "daily_readiness_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Wrote daily readiness summary CSV to {summary_file}")

    report_file = output_path / "daily_readiness_report.txt"
    with report_file.open("w", encoding="utf-8") as report:
        report.write(f"Daily readiness summary written to: {summary_file}\n")
        report.write(f"Anomalies count: {len(anomalies)}\n")
        for anomaly in anomalies:
            report.write(f"{anomaly['signal_date']}: {', '.join(anomaly['issues'])}\n")

    if args.write_json:
        summary_path = output_path / "daily_readiness_summary.json"
        anomalies_path = output_path / "daily_readiness_anomalies.json"
        summary.to_json(summary_path, orient="records", date_format="iso")
        with open(anomalies_path, "w", encoding="utf-8") as handle:
            json.dump(anomalies, handle, indent=2)
        print(f"Wrote daily readiness summary to {summary_path}")
        print(f"Wrote daily readiness anomalies to {anomalies_path}")

    print(summary.to_string(index=False))
    if anomalies:
        print("\nDetected anomalies:")
        for anomaly in anomalies:
            print(anomaly)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
