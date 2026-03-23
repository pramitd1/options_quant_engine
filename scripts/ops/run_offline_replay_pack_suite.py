#!/usr/bin/env python3
"""
Offline Replay Pack Suite
=========================

Purpose:
    Evaluate baseline and candidate parameter packs on historical signal data
    across rolling windows, with resumable checkpoints and auditable artifacts.

Safety:
    This script is offline-only. It does not mutate live runtime state, does not
    enable regime auto-switching, and only writes under research outputs.

Usage examples:
    .venv/bin/python scripts/ops/run_offline_replay_pack_suite.py
    .venv/bin/python scripts/ops/run_offline_replay_pack_suite.py --candidates macro_overlay_v1 overnight_focus_v1
    .venv/bin/python scripts/ops/run_offline_replay_pack_suite.py --windows all 30 60 90
    .venv/bin/python scripts/ops/run_offline_replay_pack_suite.py --resume-dir research/parameter_tuning/offline_replay_runs/suite_20260323_190000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.signal_evaluation_scoring import get_signal_evaluation_selection_policy
from research.signal_evaluation.dataset import SIGNAL_DATASET_PATH, load_signals_dataset
from tuning.objectives import apply_selection_policy, compute_frame_metrics
from tuning.runtime import temporary_parameter_pack
from tuning.validation import summarize_metrics_by_regime


OUTPUT_ROOT = ROOT / "research" / "parameter_tuning" / "offline_replay_runs"


@dataclass(frozen=True)
class WindowSpec:
    label: str
    days: int | None


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run_id_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_pack_slug(name: str) -> str:
    return str(name).strip().replace("+", "__")


def _parse_windows(values: list[str] | None) -> list[WindowSpec]:
    requested = values or ["all", "30", "60", "90"]
    specs: list[WindowSpec] = []
    seen: set[str] = set()
    for raw in requested:
        text = str(raw).strip().lower()
        if text == "all":
            key = "all"
            if key not in seen:
                specs.append(WindowSpec(label="all", days=None))
                seen.add(key)
            continue
        days = int(text)
        if days <= 0:
            raise ValueError(f"Window must be positive days: {raw}")
        key = f"{days}d"
        if key not in seen:
            specs.append(WindowSpec(label=key, days=days))
            seen.add(key)
    return specs


def _filter_window(frame: pd.DataFrame, spec: WindowSpec) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if spec.days is None:
        return frame.copy().reset_index(drop=True)

    ts = pd.to_datetime(frame.get("signal_timestamp"), errors="coerce")
    valid = frame.loc[ts.notna()].copy()
    if valid.empty:
        return valid

    valid_ts = pd.to_datetime(valid["signal_timestamp"], errors="coerce")
    end_ts = valid_ts.max()
    start_ts = end_ts - pd.Timedelta(days=spec.days)
    selected = valid.loc[(valid_ts >= start_ts) & (valid_ts <= end_ts)].copy()
    return selected.reset_index(drop=True)


def _selection_policy_for_pack(pack_name: str) -> dict[str, Any]:
    with temporary_parameter_pack(pack_name):
        return dict(get_signal_evaluation_selection_policy())


def _evaluate_pack_on_frame(
    frame: pd.DataFrame,
    *,
    pack_name: str,
    minimum_regime_sample_count: int,
) -> dict[str, Any]:
    with temporary_parameter_pack(pack_name):
        thresholds = dict(get_signal_evaluation_selection_policy())
        selected = apply_selection_policy(frame, thresholds=thresholds)
        aggregate = compute_frame_metrics(selected, len(frame))
        regime_summary = summarize_metrics_by_regime(
            selected,
            minimum_regime_sample_count=minimum_regime_sample_count,
        )

    return {
        "generated_at": _utc_now(),
        "pack_name": pack_name,
        "total_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selection_thresholds": thresholds,
        "aggregate_metrics": aggregate,
        "regime_summary": regime_summary,
    }


def _regime_delta_rows(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_regime = dict(baseline_summary.get("regime_summary") or {})
    candidate_regime = dict(candidate_summary.get("regime_summary") or {})

    for regime_col, candidate_rows in candidate_regime.items():
        baseline_map = {str(item.get("regime_label")): item for item in baseline_regime.get(regime_col, [])}
        for item in candidate_rows:
            label = str(item.get("regime_label"))
            base_item = baseline_map.get(label, {})
            b_hr = float((base_item.get("metrics") or {}).get("direction_hit_rate", 0.0) or 0.0)
            c_hr = float((item.get("metrics") or {}).get("direction_hit_rate", 0.0) or 0.0)
            rows.append(
                {
                    "regime_column": regime_col,
                    "regime_label": label,
                    "baseline_sample_count": int(base_item.get("sample_count", 0) or 0),
                    "candidate_sample_count": int(item.get("sample_count", 0) or 0),
                    "baseline_direction_hit_rate": round(b_hr, 6),
                    "candidate_direction_hit_rate": round(c_hr, 6),
                    "direction_hit_rate_delta": round(c_hr - b_hr, 6),
                }
            )
    return rows


def _aggregate_deltas(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> dict[str, float]:
    b = dict(baseline_summary.get("aggregate_metrics") or {})
    c = dict(candidate_summary.get("aggregate_metrics") or {})
    keys = sorted((set(b.keys()) | set(c.keys())) - {"selected_count"})
    return {k: round(float(c.get(k, 0.0)) - float(b.get(k, 0.0)), 6) for k in keys}


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_csv_rows(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        return
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, rows], ignore_index=True)
        combined.to_csv(path, index=False)
        return
    rows.to_csv(path, index=False)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": [], "started_at": _utc_now(), "updated_at": _utc_now()}


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _utc_now()
    _write_json(path, payload)


def _task_key(window_label: str, baseline_pack: str, candidate_pack: str) -> str:
    return f"{window_label}|{baseline_pack}|{candidate_pack}"


def _run_suite(
    *,
    baseline_pack: str,
    candidate_packs: list[str],
    windows: list[WindowSpec],
    minimum_regime_sample_count: int,
    dataset_path: Path,
    out_dir: Path,
    force: bool,
    subrun_id: str,
    invocation_started_at: str,
) -> dict[str, Any]:
    frame = load_signals_dataset(dataset_path)
    checkpoint_path = out_dir / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    completed = set(str(x) for x in checkpoint.get("completed", []))

    baseline_policy = _selection_policy_for_pack(baseline_pack)
    candidate_policy_map = {name: _selection_policy_for_pack(name) for name in candidate_packs}

    policy_audit_rows: list[dict[str, Any]] = []
    for candidate in candidate_packs:
        candidate_policy = candidate_policy_map[candidate]
        keys = sorted(set(baseline_policy.keys()) | set(candidate_policy.keys()))
        for key in keys:
            b_val = baseline_policy.get(key)
            c_val = candidate_policy.get(key)
            policy_audit_rows.append(
                {
                    "baseline_pack": baseline_pack,
                    "candidate_pack": candidate,
                    "policy_key": key,
                    "baseline_value": b_val,
                    "candidate_value": c_val,
                    "changed": b_val != c_val,
                }
            )

    _write_json(out_dir / "selection_policy_audit.json", policy_audit_rows)
    pd.DataFrame(policy_audit_rows).to_csv(out_dir / "selection_policy_audit.csv", index=False)

    run_rows: list[dict[str, Any]] = []

    for window_spec in windows:
        window_frame = _filter_window(frame, window_spec)
        window_dir = out_dir / "windows" / window_spec.label
        window_dir.mkdir(parents=True, exist_ok=True)

        baseline_summary = _evaluate_pack_on_frame(
            window_frame,
            pack_name=baseline_pack,
            minimum_regime_sample_count=minimum_regime_sample_count,
        )
        _write_json(window_dir / f"pack_{_safe_pack_slug(baseline_pack)}.json", baseline_summary)

        for candidate in candidate_packs:
            task_key = _task_key(window_spec.label, baseline_pack, candidate)
            comparison_path = window_dir / f"comparison_{_safe_pack_slug(baseline_pack)}_vs_{_safe_pack_slug(candidate)}.json"
            regime_csv_path = window_dir / f"regime_deltas_{_safe_pack_slug(baseline_pack)}_vs_{_safe_pack_slug(candidate)}.csv"

            if (not force) and (task_key in completed) and comparison_path.exists() and regime_csv_path.exists():
                run_rows.append(
                    {
                        "subrun_id": subrun_id,
                        "run_started_at": invocation_started_at,
                        "window": window_spec.label,
                        "baseline_pack": baseline_pack,
                        "candidate_pack": candidate,
                        "status": "skipped_completed",
                    }
                )
                continue

            candidate_summary = _evaluate_pack_on_frame(
                window_frame,
                pack_name=candidate,
                minimum_regime_sample_count=minimum_regime_sample_count,
            )
            _write_json(window_dir / f"pack_{_safe_pack_slug(candidate)}.json", candidate_summary)

            regime_rows = _regime_delta_rows(baseline_summary, candidate_summary)
            pd.DataFrame(regime_rows).to_csv(regime_csv_path, index=False)
            _write_json(
                window_dir / f"regime_deltas_{_safe_pack_slug(baseline_pack)}_vs_{_safe_pack_slug(candidate)}.json",
                regime_rows,
            )

            comparison_payload = {
                "generated_at": _utc_now(),
                "window": window_spec.label,
                "window_days": window_spec.days,
                "baseline_pack": baseline_pack,
                "candidate_pack": candidate,
                "total_rows": int(len(window_frame)),
                "baseline_selected_rows": int(baseline_summary.get("selected_rows", 0)),
                "candidate_selected_rows": int(candidate_summary.get("selected_rows", 0)),
                "aggregate_deltas": _aggregate_deltas(baseline_summary, candidate_summary),
                "artifact_paths": {
                    "baseline_summary": str((window_dir / f"pack_{_safe_pack_slug(baseline_pack)}.json").relative_to(out_dir)),
                    "candidate_summary": str((window_dir / f"pack_{_safe_pack_slug(candidate)}.json").relative_to(out_dir)),
                    "regime_deltas_csv": str(regime_csv_path.relative_to(out_dir)),
                },
            }
            _write_json(comparison_path, comparison_payload)

            completed.add(task_key)
            checkpoint["completed"] = sorted(completed)
            _save_checkpoint(checkpoint_path, checkpoint)

            run_rows.append(
                {
                    "subrun_id": subrun_id,
                    "run_started_at": invocation_started_at,
                    "window": window_spec.label,
                    "baseline_pack": baseline_pack,
                    "candidate_pack": candidate,
                    "status": "completed",
                    "total_rows": int(len(window_frame)),
                    "baseline_selected_rows": int(baseline_summary.get("selected_rows", 0)),
                    "candidate_selected_rows": int(candidate_summary.get("selected_rows", 0)),
                    "direction_hit_rate_delta": comparison_payload["aggregate_deltas"].get("direction_hit_rate"),
                    "return_60m_bps_delta": comparison_payload["aggregate_deltas"].get("average_realized_return_60m_bps"),
                }
            )

    summary_df = pd.DataFrame(run_rows)
    subrun_dir = out_dir / "subruns" / subrun_id
    subrun_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(subrun_dir / "run_summary.csv", index=False)
    _write_json(
        subrun_dir / "run_meta.json",
        {
            "subrun_id": subrun_id,
            "generated_at": _utc_now(),
            "tasks_recorded": int(len(summary_df)),
        },
    )

    # Backward-compatible latest summary in suite root.
    summary_df.to_csv(out_dir / "run_summary.csv", index=False)
    _append_csv_rows(out_dir / "run_history.csv", summary_df)

    manifest = {
        "generated_at": _utc_now(),
        "baseline_pack": baseline_pack,
        "candidate_packs": list(candidate_packs),
        "windows": [{"label": w.label, "days": w.days} for w in windows],
        "dataset_path": str(dataset_path),
        "minimum_regime_sample_count": int(minimum_regime_sample_count),
        "total_dataset_rows": int(len(frame)),
        "resume_checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "notes": [
            "Offline-only suite. No live runtime mutation.",
            "Resume by passing --resume-dir to this script.",
            "Each invocation writes a dated sub-run folder under subruns/.",
            "run_history.csv is append-only and preserves prior invocations.",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)

    return {
        "out_dir": str(out_dir),
        "tasks_total": int(len(windows) * len(candidate_packs)),
        "tasks_recorded": int(len(run_rows)),
        "completed_count": int((summary_df["status"] == "completed").sum()) if not summary_df.empty else 0,
        "skipped_completed_count": int((summary_df["status"] == "skipped_completed").sum()) if not summary_df.empty else 0,
        "subrun_id": subrun_id,
        "subrun_summary": str((subrun_dir / "run_summary.csv").relative_to(out_dir)),
        "run_history": str((out_dir / "run_history.csv").relative_to(out_dir)),
    }


def _resolve_output_dir(resume_dir: str | None) -> Path:
    if resume_dir:
        path = Path(resume_dir)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline replay comparison suite with resume support.")
    parser.add_argument("--baseline", default="baseline_v1", help="Baseline pack name")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["macro_overlay_v1", "overnight_focus_v1"],
        help="Candidate pack names",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["all", "30", "60", "90"],
        help="Window list: all and/or positive day values",
    )
    parser.add_argument(
        "--minimum-regime-sample-count",
        type=int,
        default=5,
        help="Minimum regime sample count for insufficient-sample tagging",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(SIGNAL_DATASET_PATH),
        help="Signal dataset path",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Existing suite directory to resume",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute tasks even if checkpoint marks them complete",
    )
    args = parser.parse_args()

    windows = _parse_windows(args.windows)
    out_dir = _resolve_output_dir(args.resume_dir)
    subrun_id = _run_id_now()
    invocation_started_at = _utc_now()

    result = _run_suite(
        baseline_pack=str(args.baseline).strip(),
        candidate_packs=[str(name).strip() for name in args.candidates if str(name).strip()],
        windows=windows,
        minimum_regime_sample_count=max(int(args.minimum_regime_sample_count), 1),
        dataset_path=Path(args.dataset_path),
        out_dir=out_dir,
        force=bool(args.force),
        subrun_id=subrun_id,
        invocation_started_at=invocation_started_at,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
