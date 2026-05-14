"""Reproducibility manifests for signal research reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.label_quality import label_quality_summary
from utils.timestamp_helpers import coerce_timestamp_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_metadata(path: Path) -> dict[str, Any]:
    exists = path.exists()
    metadata: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(path.resolve()) if exists else None,
        "exists": bool(exists),
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
    }
    if not exists or not path.is_file():
        return metadata

    stat = path.stat()
    metadata.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "sha256": _file_sha256(path),
        }
    )
    return metadata


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        commit = None

    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
    except Exception:
        status = []

    return {
        "commit": commit,
        "dirty": bool(status),
        "status_entry_count": int(len(status)),
    }


def _timestamp_parse_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "signal_timestamp" not in frame.columns:
        return {
            "column": "signal_timestamp",
            "row_count": int(len(frame)),
            "parsed_count": 0,
            "failed_count": 0,
            "min": None,
            "max": None,
            "trading_days": 0,
        }

    parsed = coerce_timestamp_series(frame["signal_timestamp"]).dropna()
    return {
        "column": "signal_timestamp",
        "row_count": int(len(frame)),
        "parsed_count": int(parsed.count()),
        "failed_count": int(len(frame) - parsed.count()),
        "min": parsed.min().isoformat() if not parsed.empty else None,
        "max": parsed.max().isoformat() if not parsed.empty else None,
        "trading_days": int(parsed.dt.normalize().nunique()) if not parsed.empty else 0,
    }


def build_report_reproducibility_manifest(
    *,
    report_path: str | Path,
    dataset_path: str | Path | None,
    frame: pd.DataFrame,
    report_kind: str,
    report_date: date | str | None = None,
    mode: str | None = None,
    run_evaluation: bool | None = None,
    narrative: bool | None = None,
) -> dict[str, Any]:
    report = Path(report_path)
    dataset = Path(dataset_path) if dataset_path is not None else None
    columns = [str(column) for column in frame.columns]

    manifest: dict[str, Any] = {
        "manifest_type": "signal_report_reproducibility_manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "report_kind": report_kind,
        "report_date": report_date.isoformat() if isinstance(report_date, date) else report_date,
        "mode": mode,
        "run_evaluation": run_evaluation,
        "narrative": narrative,
        "command": list(sys.argv),
        "git": _git_metadata(),
        "report": _path_metadata(report),
        "dataset": _path_metadata(dataset) if dataset is not None else None,
        "frame": {
            "row_count": int(len(frame)),
            "column_count": int(len(columns)),
            "columns": columns,
            "columns_sha256": _json_sha256(columns),
        },
        "timestamp_parse": _timestamp_parse_summary(frame),
        "label_quality_summary": label_quality_summary(frame),
    }
    return manifest


def write_report_reproducibility_manifest(
    *,
    report_path: str | Path,
    dataset_path: str | Path | None,
    frame: pd.DataFrame,
    report_kind: str,
    report_date: date | str | None = None,
    mode: str | None = None,
    run_evaluation: bool | None = None,
    narrative: bool | None = None,
) -> Path:
    report = Path(report_path)
    manifest_path = report.with_suffix(".manifest.json")
    manifest = build_report_reproducibility_manifest(
        report_path=report,
        dataset_path=dataset_path,
        frame=frame,
        report_kind=report_kind,
        report_date=report_date,
        mode=mode,
        run_evaluation=run_evaluation,
        narrative=narrative,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return manifest_path
