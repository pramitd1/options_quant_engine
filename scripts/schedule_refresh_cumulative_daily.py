#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
import time

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(Path(__file__))

TREND_DIR = PROJECT_ROOT / "research" / "signal_evaluation" / "readiness_trend"
TREND_JSONL = TREND_DIR / "readiness_trend.jsonl"
TREND_CSV = TREND_DIR / "readiness_trend.csv"
LATEST_JSON = TREND_DIR / "latest_readiness.json"
ERROR_JSONL = TREND_DIR / "readiness_errors.jsonl"


def _parse_stdout_metrics(raw: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in (raw or "").splitlines():
        token = line.strip()
        if not token or ":" not in token:
            continue
        key, value = token.split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def _coerce_metric(value: str) -> object:
    token = str(value).strip()
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    try:
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            return int(token)
        return float(token)
    except Exception:
        return token


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _append_csv(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: payload.get(k) for k in sorted(payload.keys())}
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_refresh(dataset_path: str | None) -> dict[str, object]:
    script_path = PROJECT_ROOT / "scripts" / "refresh_cumulative_signal_dataset.py"
    cmd = [sys.executable, str(script_path)]
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    started_at = datetime.now(UTC)
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    finished_at = datetime.now(UTC)

    event: dict[str, object] = {
        "timestamp_utc": finished_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "exit_code": int(completed.returncode),
    }

    stdout_metrics = _parse_stdout_metrics(completed.stdout)
    for key, value in stdout_metrics.items():
        event[key] = _coerce_metric(value)

    if completed.returncode != 0:
        event["error"] = (completed.stderr or completed.stdout or "refresh command failed").strip()
        raise RuntimeError(event["error"])

    return event


def _run_high_only_review_artifacts(
    *,
    dataset_path: str | None,
    promote_high_confidence_repairs: bool,
) -> dict[str, object]:
    script_path = PROJECT_ROOT / "scripts" / "backfill_signal_contract_fields.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--apply-repair-proposals",
        "--emit-audit",
        "--emit-repair-proposals",
        "--high-confidence-only",
    ]
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    if promote_high_confidence_repairs:
        cmd.append("--promote-repaired-dataset")
    else:
        cmd.append("--dry-run")

    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    metrics = _parse_stdout_metrics(completed.stdout)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "high-only review artifact generation failed").strip()
        raise RuntimeError(detail)

    normalized: dict[str, object] = {}
    for key, value in metrics.items():
        normalized[f"high_only_{key}"] = _coerce_metric(value)
    normalized["high_only_enabled"] = True
    normalized["high_only_promote_mode"] = bool(promote_high_confidence_repairs)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative refresh on a daily cadence and persist readiness trend artifacts "
            "under research/signal_evaluation/readiness_trend/."
        )
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--interval-seconds", type=int, default=86400)
    parser.add_argument("--run-forever", action="store_true")
    parser.add_argument(
        "--emit-high-only-review-artifacts",
        action="store_true",
        help="Generate HIGH-only repair proposals and review-queue artifacts after each refresh run.",
    )
    parser.add_argument(
        "--promote-high-confidence-repairs",
        action="store_true",
        help="When high-only artifact mode is enabled, also promote HIGH repairs into the source dataset.",
    )
    return parser.parse_args()


def _record_success(event: dict[str, object]) -> None:
    TREND_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(event, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _append_jsonl(TREND_JSONL, event)
    _append_csv(TREND_CSV, event)


def _record_error(message: str) -> None:
    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "error": message,
    }
    _append_jsonl(ERROR_JSONL, payload)


def main() -> int:
    args = parse_args()

    while True:
        try:
            event = _run_refresh(args.dataset_path)
            if args.emit_high_only_review_artifacts:
                event.update(
                    _run_high_only_review_artifacts(
                        dataset_path=args.dataset_path,
                        promote_high_confidence_repairs=args.promote_high_confidence_repairs,
                    )
                )
            else:
                event["high_only_enabled"] = False
            _record_success(event)
            print("Scheduler run succeeded")
            print(json.dumps(event, ensure_ascii=True))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            _record_error(detail)
            print(f"Scheduler run failed: {detail}", file=sys.stderr)
            traceback.print_exc()
            if not args.run_forever:
                return 1

        if not args.run_forever:
            return 0

        time.sleep(max(int(args.interval_seconds), 1))


if __name__ == "__main__":
    raise SystemExit(main())
