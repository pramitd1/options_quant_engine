"""One-command automated threshold shadow-mode workflow.

The workflow runs the full research-only shadow chain:

1. Threshold governance
2. Threshold policy experiment sandbox
3. Threshold shadow simulation
4. Shadow promotion-readiness review
5. Manual promotion package, post-promotion monitor, and adoption reconciliation

It writes advisory artifacts only. Runtime signal-generation configuration and
execution behavior are never changed.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_governance import (
    DEFAULT_THRESHOLD_GOVERNANCE_DIR,
    PROMOTE_TO_REVIEW,
    write_threshold_governance_report,
)
from research.signal_evaluation.threshold_adoption_reconciliation import (
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    write_threshold_adoption_reconciliation_report,
)
from research.signal_evaluation.threshold_policy_experiment import (
    DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR,
    write_threshold_policy_experiment_report,
    write_threshold_policy_experiment_skip,
)
from research.signal_evaluation.threshold_post_promotion_monitor import (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    write_threshold_post_promotion_monitor_report,
)
from research.signal_evaluation.threshold_promotion_review import (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    write_threshold_promotion_review_package,
)
from research.signal_evaluation.threshold_shadow_review import (
    DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR,
    write_threshold_shadow_review_report,
)
from research.signal_evaluation.threshold_shadow_simulation import (
    DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR,
    write_threshold_shadow_simulation_report,
    write_threshold_shadow_simulation_skip,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "research" / "signal_evaluation" / "signals_dataset.csv"
DEFAULT_CUMULATIVE_DATASET_PATH = PROJECT_ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"
DEFAULT_THRESHOLD_SHADOW_MODE_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_shadow_mode_runs"
)
THRESHOLD_SHADOW_MODE_JSON_FILENAME = "latest_threshold_shadow_mode_run.json"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    return f"threshold_shadow_mode_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _output_dir(explicit_root: Path | None, name: str, default_dir: Path) -> Path:
    return explicit_root / name if explicit_root is not None else default_dir


def _is_concrete_candidate(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("threshold_field")) and candidate.get("threshold_value") is not None


def _artifact_status(artifact: dict[str, Any], key: str) -> Any:
    if artifact.get("error"):
        return None
    return (artifact.get("report", {}) or {}).get(key)


def run_threshold_shadow_mode(
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
) -> dict[str, Any]:
    """Run the full automated shadow-mode artifact chain."""
    run_id = report_name or _run_id()
    generated_at = _utc_now()
    ds_path = Path(dataset_path) if dataset_path is not None else _default_dataset_path()
    explicit_root = Path(output_dir) if output_dir is not None else None
    run_dir = _output_dir(explicit_root, "threshold_shadow_mode_runs", DEFAULT_THRESHOLD_SHADOW_MODE_DIR)
    governance_dir = _output_dir(explicit_root, "threshold_governance", DEFAULT_THRESHOLD_GOVERNANCE_DIR)
    experiment_dir = _output_dir(
        explicit_root,
        "threshold_policy_experiments",
        DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR,
    )
    simulation_dir = _output_dir(
        explicit_root,
        "threshold_shadow_simulation",
        DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR,
    )
    review_dir = _output_dir(explicit_root, "threshold_shadow_review", DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR)
    promotion_review_dir = _output_dir(
        explicit_root,
        "threshold_promotion_review",
        DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    )
    post_promotion_monitor_dir = _output_dir(
        explicit_root,
        "threshold_post_promotion_monitoring",
        DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    )
    adoption_reconciliation_dir = _output_dir(
        explicit_root,
        "threshold_adoption_reconciliation",
        DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": generated_at,
        "completed_at": None,
        "status": "RUNNING",
        "dataset_path": str(ds_path),
        "output_dir": str(explicit_root) if explicit_root is not None else None,
        "runtime_config_changed": False,
        "governance_status": None,
        "policy_experiment_status": None,
        "shadow_status": None,
        "shadow_review_status": None,
        "promotion_review_status": None,
        "post_promotion_monitor_status": None,
        "adoption_reconciliation_status": None,
        "manual_promotion_review_required": False,
        "recommended_next_action": None,
        "threshold_governance": {},
        "threshold_policy_experiment": {},
        "threshold_shadow_simulation": {},
        "threshold_shadow_review": {},
        "threshold_promotion_review": {},
        "threshold_post_promotion_monitor": {},
        "threshold_adoption_reconciliation": {},
        "checkpoint_json": str(run_dir / f"{run_id}.json"),
        "latest_checkpoint_json": str(run_dir / THRESHOLD_SHADOW_MODE_JSON_FILENAME),
        "error": None,
    }

    try:
        frame = pd.read_csv(ds_path)
        governance_artifact = write_threshold_governance_report(
            frame,
            dataset_path=str(ds_path),
            output_dir=governance_dir,
            report_name=f"{run_id}_threshold_governance",
            write_latest=True,
        )
        governance_report = governance_artifact.get("report", {}) or {}
        top_candidate = governance_report.get("top_candidate_review", {}) or {}
        governance_report_path = governance_artifact.get("latest_json_path") or governance_artifact.get("json_path")

        if top_candidate.get("governance_status") == PROMOTE_TO_REVIEW and _is_concrete_candidate(top_candidate):
            experiment_artifact = write_threshold_policy_experiment_report(
                frame,
                governance_report=governance_report,
                candidate_review=top_candidate,
                dataset_path=str(ds_path),
                governance_report_path=governance_report_path,
                output_dir=experiment_dir,
                report_name=f"{run_id}_threshold_policy_experiment",
                write_latest=True,
            )
        else:
            experiment_artifact = write_threshold_policy_experiment_skip(
                reason=(
                    "No promoted concrete threshold candidate is available; "
                    f"governance status is {top_candidate.get('governance_status') or governance_report.get('overall_status') or 'UNKNOWN'}."
                ),
                candidate_review=top_candidate,
                dataset_path=str(ds_path),
                governance_report_path=governance_report_path,
                output_dir=experiment_dir,
                report_name=f"{run_id}_threshold_policy_experiment",
                write_latest=True,
            )

        experiment_report = experiment_artifact.get("report", {}) or {}
        experiment_report_path = experiment_artifact.get("latest_json_path") or experiment_artifact.get("json_path")
        if experiment_report.get("experiment_status") == "APPROVED_FOR_POLICY_EXPERIMENT":
            simulation_artifact = write_threshold_shadow_simulation_report(
                frame,
                policy_experiment_report=experiment_report,
                dataset_path=str(ds_path),
                policy_experiment_report_path=experiment_report_path,
                output_dir=simulation_dir,
                report_name=f"{run_id}_threshold_shadow_simulation",
                write_latest=True,
            )
        else:
            simulation_artifact = write_threshold_shadow_simulation_skip(
                reason=(
                    "Policy experiment is not approved for shadow simulation; "
                    f"status is {experiment_report.get('experiment_status') or 'UNKNOWN'}."
                ),
                policy_experiment_report=experiment_report,
                dataset_path=str(ds_path),
                policy_experiment_report_path=experiment_report_path,
                output_dir=simulation_dir,
                report_name=f"{run_id}_threshold_shadow_simulation",
                write_latest=True,
            )

        simulation_report = simulation_artifact.get("report", {}) or {}
        simulation_report_path = simulation_artifact.get("latest_json_path") or simulation_artifact.get("json_path")
        review_artifact = write_threshold_shadow_review_report(
            simulation_report,
            shadow_simulation_report_path=simulation_report_path,
            output_dir=review_dir,
            report_name=f"{run_id}_threshold_shadow_review",
            write_latest=True,
        )
        review_report = review_artifact.get("report", {}) or {}
        review_report_path = review_artifact.get("latest_json_path") or review_artifact.get("json_path")
        promotion_artifact = write_threshold_promotion_review_package(
            review_report,
            shadow_review_report_path=review_report_path,
            output_dir=promotion_review_dir,
            report_name=f"{run_id}_threshold_promotion_review",
            write_latest=True,
        )
        promotion_report = promotion_artifact.get("report", {}) or {}
        post_promotion_artifact = write_threshold_post_promotion_monitor_report(
            frame,
            promotion_package_report=promotion_report,
            promotion_package_report_path=promotion_artifact.get("latest_json_path") or promotion_artifact.get("json_path"),
            ledger_path=promotion_artifact.get("review_ledger_path"),
            dataset_path=str(ds_path),
            output_dir=post_promotion_monitor_dir,
            report_name=f"{run_id}_threshold_post_promotion_monitor",
            write_latest=True,
        )
        adoption_reconciliation_artifact = write_threshold_adoption_reconciliation_report(
            promotion_package_report=promotion_report,
            promotion_package_report_path=promotion_artifact.get("latest_json_path") or promotion_artifact.get("json_path"),
            ledger_path=promotion_artifact.get("review_ledger_path"),
            post_promotion_monitor_report=post_promotion_artifact.get("report", {}) or {},
            post_promotion_monitor_report_path=post_promotion_artifact.get("latest_json_path") or post_promotion_artifact.get("json_path"),
            output_dir=adoption_reconciliation_dir,
            report_name=f"{run_id}_threshold_adoption_reconciliation",
            write_latest=True,
        )

        payload.update(
            {
                "status": "SUCCESS",
                "governance_status": _artifact_status(governance_artifact, "overall_status"),
                "policy_experiment_status": _artifact_status(experiment_artifact, "experiment_status"),
                "shadow_status": _artifact_status(simulation_artifact, "shadow_status"),
                "shadow_review_status": _artifact_status(review_artifact, "review_status"),
                "promotion_review_status": _artifact_status(promotion_artifact, "promotion_review_status"),
                "post_promotion_monitor_status": _artifact_status(post_promotion_artifact, "monitor_status"),
                "adoption_reconciliation_status": _artifact_status(adoption_reconciliation_artifact, "adoption_status"),
                "manual_promotion_review_required": bool(promotion_report.get("manual_review_required")),
                "recommended_next_action": promotion_report.get("recommended_next_action"),
                "threshold_governance": {key: value for key, value in governance_artifact.items() if key != "report"},
                "threshold_policy_experiment": {key: value for key, value in experiment_artifact.items() if key != "report"},
                "threshold_shadow_simulation": {key: value for key, value in simulation_artifact.items() if key != "report"},
                "threshold_shadow_review": {key: value for key, value in review_artifact.items() if key != "report"},
                "threshold_promotion_review": {key: value for key, value in promotion_artifact.items() if key != "report"},
                "threshold_post_promotion_monitor": {
                    key: value for key, value in post_promotion_artifact.items() if key != "report"
                },
                "threshold_adoption_reconciliation": {
                    key: value for key, value in adoption_reconciliation_artifact.items() if key != "report"
                },
            }
        )
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["error"] = str(exc)

    payload["completed_at"] = _utc_now()
    checkpoint_path = Path(payload["checkpoint_json"])
    latest_path = Path(payload["latest_checkpoint_json"])
    _atomic_write_text(checkpoint_path, json.dumps(payload, indent=2, sort_keys=True, default=str))
    _atomic_write_text(latest_path, json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload
