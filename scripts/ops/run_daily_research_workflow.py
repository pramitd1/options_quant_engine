#!/usr/bin/env python3
"""
Daily Research Workflow Orchestrator
====================================

Runs the research-only daily signal report workflow in one command and records
an append-only checkpoint/history trail.

Outputs:
  - documentation/daily_reports/signal_research_report_YYYYMMDD.md
  - research/signal_evaluation/reports/drift_monitoring/latest_signal_drift.json
  - research/signal_evaluation/reports/drift_monitoring/latest_signal_drift_trend.md
  - research/signal_evaluation/reports/drift_monitoring/latest_signal_drift_alert.json
  - research/signal_evaluation/reports/threshold_governance/latest_threshold_governance.json
  - research/signal_evaluation/reports/threshold_governance/latest_threshold_governance.md
  - research/signal_evaluation/reports/threshold_policy_experiments/latest_threshold_policy_experiment.json
  - research/signal_evaluation/reports/threshold_policy_experiments/latest_threshold_policy_experiment.md
  - research/signal_evaluation/reports/threshold_shadow_simulation/latest_threshold_shadow_simulation.json
  - research/signal_evaluation/reports/threshold_shadow_simulation/latest_threshold_shadow_simulation.md
  - research/signal_evaluation/reports/threshold_shadow_review/latest_threshold_shadow_review.json
  - research/signal_evaluation/reports/threshold_shadow_review/latest_threshold_shadow_review.md
  - research/signal_evaluation/reports/threshold_promotion_review/latest_threshold_promotion_review.json
  - research/signal_evaluation/reports/threshold_promotion_review/latest_threshold_promotion_review.md
  - research/signal_evaluation/reports/threshold_post_promotion_monitoring/latest_threshold_post_promotion_monitor.json
  - research/signal_evaluation/reports/threshold_post_promotion_monitoring/latest_threshold_post_promotion_monitor.md
  - research/signal_evaluation/reports/threshold_adoption_reconciliation/latest_threshold_adoption_reconciliation.json
  - research/signal_evaluation/reports/threshold_adoption_reconciliation/latest_threshold_adoption_reconciliation.md
  - research/signal_evaluation/reports/daily_ops_runs/<run_id>.json
  - research/signal_evaluation/reports/daily_ops_runs/run_history.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.daily_research_report import (  # noqa: E402
    DEFAULT_CUMULATIVE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_DRIFT_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    generate_daily_report,
)
from research.signal_evaluation.drift_alerts import (  # noqa: E402
    DRIFT_ALERT_JSON_FILENAME,
    DRIFT_ALERT_MARKDOWN_FILENAME,
    DRIFT_REVIEW_LEDGER_FILENAME,
    run_signal_drift_alert_workflow,
)
from research.signal_evaluation.threshold_governance import (  # noqa: E402
    DEFAULT_THRESHOLD_GOVERNANCE_DIR,
    THRESHOLD_GOVERNANCE_CANDIDATES_FILENAME,
    THRESHOLD_GOVERNANCE_JSON_FILENAME,
    THRESHOLD_GOVERNANCE_MARKDOWN_FILENAME,
    THRESHOLD_GOVERNANCE_REVIEW_LEDGER_FILENAME,
)
from research.signal_evaluation.threshold_adoption_reconciliation import (  # noqa: E402
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    THRESHOLD_ADOPTION_RECONCILIATION_COMPARISON_FILENAME,
    THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME,
    THRESHOLD_ADOPTION_RECONCILIATION_MARKDOWN_FILENAME,
)
from research.signal_evaluation.threshold_policy_experiment import (  # noqa: E402
    DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR,
    THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME,
    THRESHOLD_POLICY_EXPERIMENT_MARKDOWN_FILENAME,
    THRESHOLD_POLICY_EXPERIMENT_POLICY_PACK_FILENAME,
    THRESHOLD_POLICY_EXPERIMENT_QUALITY_BUCKETS_FILENAME,
    THRESHOLD_POLICY_EXPERIMENT_REGIMES_FILENAME,
    THRESHOLD_POLICY_EXPERIMENT_SPLITS_FILENAME,
)
from research.signal_evaluation.threshold_post_promotion_monitor import (  # noqa: E402
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME,
    THRESHOLD_POST_PROMOTION_MONITOR_MARKDOWN_FILENAME,
    THRESHOLD_POST_PROMOTION_MONITOR_SEGMENTS_FILENAME,
)
from research.signal_evaluation.threshold_promotion_review import (  # noqa: E402
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME,
    THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME,
    THRESHOLD_PROMOTION_REVIEW_MARKDOWN_FILENAME,
)
from research.signal_evaluation.threshold_shadow_simulation import (  # noqa: E402
    DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR,
    THRESHOLD_SHADOW_SIMULATION_BUCKETS_FILENAME,
    THRESHOLD_SHADOW_SIMULATION_JSON_FILENAME,
    THRESHOLD_SHADOW_SIMULATION_MARKDOWN_FILENAME,
    THRESHOLD_SHADOW_SIMULATION_REGIMES_FILENAME,
    THRESHOLD_SHADOW_SIMULATION_RETAINED_FILENAME,
    THRESHOLD_SHADOW_SIMULATION_SUPPRESSED_FILENAME,
)
from research.signal_evaluation.threshold_shadow_review import (  # noqa: E402
    DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR,
    THRESHOLD_SHADOW_REVIEW_JSON_FILENAME,
    THRESHOLD_SHADOW_REVIEW_MARKDOWN_FILENAME,
    THRESHOLD_SHADOW_REVIEW_SEGMENTS_FILENAME,
)


DEFAULT_HISTORY_DIR = ROOT / "research" / "signal_evaluation" / "reports" / "daily_ops_runs"

HISTORY_FIELDS = [
    "run_id",
    "generated_at",
    "completed_at",
    "status",
    "report_date",
    "dataset_path",
    "daily_report_path",
    "cumulative_report_path",
    "drift_latest_json",
    "drift_trend_history_csv",
    "drift_trend_dashboard_json",
    "drift_trend_dashboard_markdown",
    "drift_alert_json",
    "drift_alert_markdown",
    "drift_alert_status",
    "drift_review_ledger_csv",
    "threshold_governance_json",
    "threshold_governance_markdown",
    "threshold_governance_candidates_csv",
    "threshold_governance_status",
    "threshold_governance_review_ledger_csv",
    "threshold_policy_experiment_json",
    "threshold_policy_experiment_markdown",
    "threshold_policy_experiment_policy_pack_json",
    "threshold_policy_experiment_splits_csv",
    "threshold_policy_experiment_regimes_csv",
    "threshold_policy_experiment_quality_buckets_csv",
    "threshold_policy_experiment_status",
    "threshold_shadow_simulation_json",
    "threshold_shadow_simulation_markdown",
    "threshold_shadow_simulation_retained_signals_csv",
    "threshold_shadow_simulation_suppressed_signals_csv",
    "threshold_shadow_simulation_regimes_csv",
    "threshold_shadow_simulation_buckets_csv",
    "threshold_shadow_simulation_status",
    "threshold_shadow_review_json",
    "threshold_shadow_review_markdown",
    "threshold_shadow_review_segments_csv",
    "threshold_shadow_review_status",
    "threshold_promotion_review_json",
    "threshold_promotion_review_markdown",
    "threshold_promotion_review_ledger_csv",
    "threshold_promotion_review_status",
    "threshold_post_promotion_monitor_json",
    "threshold_post_promotion_monitor_markdown",
    "threshold_post_promotion_monitor_segments_csv",
    "threshold_post_promotion_monitor_status",
    "threshold_adoption_reconciliation_json",
    "threshold_adoption_reconciliation_markdown",
    "threshold_adoption_reconciliation_comparison_csv",
    "threshold_adoption_reconciliation_status",
    "drift_status",
    "drift_warning_count",
    "checkpoint_json",
    "error",
]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    return f"daily_research_workflow_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _parse_report_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append_history_row(history_dir: Path, row: dict[str, Any]) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "run_history.csv"
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in HISTORY_FIELDS})
    return history_path


def _write_checkpoint(history_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{run_id}.json"
    payload["checkpoint_json"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _drift_latest_path(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    return _drift_output_dir(explicit_output_dir=explicit_output_dir, report_output_dir=report_output_dir) / "latest_signal_drift.json"


def _drift_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "drift_monitoring"
    return DEFAULT_DRIFT_OUTPUT_DIR


def _threshold_governance_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_governance"
    return DEFAULT_THRESHOLD_GOVERNANCE_DIR


def _threshold_policy_experiment_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_policy_experiments"
    return DEFAULT_THRESHOLD_POLICY_EXPERIMENT_DIR


def _threshold_shadow_simulation_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_shadow_simulation"
    return DEFAULT_THRESHOLD_SHADOW_SIMULATION_DIR


def _threshold_shadow_review_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_shadow_review"
    return DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR


def _threshold_promotion_review_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_promotion_review"
    return DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR


def _threshold_post_promotion_monitor_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_post_promotion_monitoring"
    return DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR


def _threshold_adoption_reconciliation_output_dir(*, explicit_output_dir: bool, report_output_dir: Path) -> Path:
    if explicit_output_dir:
        return report_output_dir / "threshold_adoption_reconciliation"
    return DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR


def run_daily_research_workflow(
    *,
    report_date: date | None = None,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    history_dir: str | Path | None = None,
    include_cumulative: bool = False,
    narrative: bool = False,
    run_evaluation: bool = True,
) -> dict[str, Any]:
    """Run daily research report generation and persist an auditable checkpoint."""
    run_id = _run_id()
    generated_at = _utc_now()
    ds_path = Path(dataset_path) if dataset_path is not None else _default_dataset_path()
    explicit_output_dir = output_dir is not None
    report_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    workflow_history_dir = Path(history_dir) if history_dir is not None else DEFAULT_HISTORY_DIR
    drift_dir = _drift_output_dir(explicit_output_dir=explicit_output_dir, report_output_dir=report_output_dir)
    threshold_dir = _threshold_governance_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )
    threshold_experiment_dir = _threshold_policy_experiment_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )
    threshold_shadow_dir = _threshold_shadow_simulation_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )
    threshold_shadow_review_dir = _threshold_shadow_review_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )
    threshold_promotion_review_dir = _threshold_promotion_review_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )
    threshold_post_promotion_monitor_dir = _threshold_post_promotion_monitor_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )
    threshold_adoption_reconciliation_dir = _threshold_adoption_reconciliation_output_dir(
        explicit_output_dir=explicit_output_dir,
        report_output_dir=report_output_dir,
    )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": generated_at,
        "completed_at": None,
        "status": "RUNNING",
        "report_date": report_date.isoformat() if report_date is not None else None,
        "dataset_path": str(ds_path),
        "output_dir": str(report_output_dir),
        "history_dir": str(workflow_history_dir),
        "include_cumulative": bool(include_cumulative),
        "narrative": bool(narrative),
        "run_evaluation": bool(run_evaluation),
        "daily_report_path": None,
        "cumulative_report_path": None,
        "drift_latest_json": str(_drift_latest_path(explicit_output_dir=explicit_output_dir, report_output_dir=report_output_dir)),
        "drift_trend_history_csv": str(drift_dir / "signal_drift_trend_history.csv"),
        "drift_trend_dashboard_json": str(drift_dir / "latest_signal_drift_trend.json"),
        "drift_trend_dashboard_markdown": str(drift_dir / "latest_signal_drift_trend.md"),
        "drift_alert_json": str(drift_dir / DRIFT_ALERT_JSON_FILENAME),
        "drift_alert_markdown": str(drift_dir / DRIFT_ALERT_MARKDOWN_FILENAME),
        "drift_alert_status": None,
        "drift_review_ledger_csv": str(drift_dir / DRIFT_REVIEW_LEDGER_FILENAME),
        "threshold_governance_json": str(threshold_dir / THRESHOLD_GOVERNANCE_JSON_FILENAME),
        "threshold_governance_markdown": str(threshold_dir / THRESHOLD_GOVERNANCE_MARKDOWN_FILENAME),
        "threshold_governance_candidates_csv": str(threshold_dir / THRESHOLD_GOVERNANCE_CANDIDATES_FILENAME),
        "threshold_governance_status": None,
        "threshold_governance_review_ledger_csv": str(threshold_dir / THRESHOLD_GOVERNANCE_REVIEW_LEDGER_FILENAME),
        "threshold_policy_experiment_json": str(threshold_experiment_dir / THRESHOLD_POLICY_EXPERIMENT_JSON_FILENAME),
        "threshold_policy_experiment_markdown": str(threshold_experiment_dir / THRESHOLD_POLICY_EXPERIMENT_MARKDOWN_FILENAME),
        "threshold_policy_experiment_policy_pack_json": str(threshold_experiment_dir / THRESHOLD_POLICY_EXPERIMENT_POLICY_PACK_FILENAME),
        "threshold_policy_experiment_splits_csv": str(threshold_experiment_dir / THRESHOLD_POLICY_EXPERIMENT_SPLITS_FILENAME),
        "threshold_policy_experiment_regimes_csv": str(threshold_experiment_dir / THRESHOLD_POLICY_EXPERIMENT_REGIMES_FILENAME),
        "threshold_policy_experiment_quality_buckets_csv": str(threshold_experiment_dir / THRESHOLD_POLICY_EXPERIMENT_QUALITY_BUCKETS_FILENAME),
        "threshold_policy_experiment_status": None,
        "threshold_shadow_simulation_json": str(threshold_shadow_dir / THRESHOLD_SHADOW_SIMULATION_JSON_FILENAME),
        "threshold_shadow_simulation_markdown": str(threshold_shadow_dir / THRESHOLD_SHADOW_SIMULATION_MARKDOWN_FILENAME),
        "threshold_shadow_simulation_retained_signals_csv": str(threshold_shadow_dir / THRESHOLD_SHADOW_SIMULATION_RETAINED_FILENAME),
        "threshold_shadow_simulation_suppressed_signals_csv": str(threshold_shadow_dir / THRESHOLD_SHADOW_SIMULATION_SUPPRESSED_FILENAME),
        "threshold_shadow_simulation_regimes_csv": str(threshold_shadow_dir / THRESHOLD_SHADOW_SIMULATION_REGIMES_FILENAME),
        "threshold_shadow_simulation_buckets_csv": str(threshold_shadow_dir / THRESHOLD_SHADOW_SIMULATION_BUCKETS_FILENAME),
        "threshold_shadow_simulation_status": None,
        "threshold_shadow_review_json": str(threshold_shadow_review_dir / THRESHOLD_SHADOW_REVIEW_JSON_FILENAME),
        "threshold_shadow_review_markdown": str(threshold_shadow_review_dir / THRESHOLD_SHADOW_REVIEW_MARKDOWN_FILENAME),
        "threshold_shadow_review_segments_csv": str(threshold_shadow_review_dir / THRESHOLD_SHADOW_REVIEW_SEGMENTS_FILENAME),
        "threshold_shadow_review_status": None,
        "threshold_promotion_review_json": str(threshold_promotion_review_dir / THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME),
        "threshold_promotion_review_markdown": str(threshold_promotion_review_dir / THRESHOLD_PROMOTION_REVIEW_MARKDOWN_FILENAME),
        "threshold_promotion_review_ledger_csv": str(threshold_promotion_review_dir / THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME),
        "threshold_promotion_review_status": None,
        "threshold_post_promotion_monitor_json": str(threshold_post_promotion_monitor_dir / THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME),
        "threshold_post_promotion_monitor_markdown": str(threshold_post_promotion_monitor_dir / THRESHOLD_POST_PROMOTION_MONITOR_MARKDOWN_FILENAME),
        "threshold_post_promotion_monitor_segments_csv": str(threshold_post_promotion_monitor_dir / THRESHOLD_POST_PROMOTION_MONITOR_SEGMENTS_FILENAME),
        "threshold_post_promotion_monitor_status": None,
        "threshold_adoption_reconciliation_json": str(threshold_adoption_reconciliation_dir / THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME),
        "threshold_adoption_reconciliation_markdown": str(threshold_adoption_reconciliation_dir / THRESHOLD_ADOPTION_RECONCILIATION_MARKDOWN_FILENAME),
        "threshold_adoption_reconciliation_comparison_csv": str(threshold_adoption_reconciliation_dir / THRESHOLD_ADOPTION_RECONCILIATION_COMPARISON_FILENAME),
        "threshold_adoption_reconciliation_status": None,
        "drift_status": None,
        "drift_warning_count": None,
        "checkpoint_json": None,
        "history_csv": None,
        "error": None,
    }

    try:
        daily_report = generate_daily_report(
            report_date=report_date,
            dataset_path=ds_path,
            output_dir=output_dir,
            narrative=narrative,
            mode="daily",
            run_evaluation=run_evaluation,
        )
        payload["daily_report_path"] = str(daily_report)

        if include_cumulative:
            cumulative_report = generate_daily_report(
                report_date=report_date,
                dataset_path=ds_path,
                output_dir=output_dir,
                narrative=narrative,
                mode="cumulative",
                run_evaluation=False,
            )
            payload["cumulative_report_path"] = str(cumulative_report)

        drift_payload = _load_json_if_exists(Path(payload["drift_latest_json"]))
        payload["drift_status"] = drift_payload.get("monitor_status")
        payload["drift_warning_count"] = len(drift_payload.get("warnings", [])) if drift_payload else None
        threshold_payload = _load_json_if_exists(Path(payload["threshold_governance_json"]))
        payload["threshold_governance_status"] = threshold_payload.get("overall_status")
        threshold_experiment_payload = _load_json_if_exists(Path(payload["threshold_policy_experiment_json"]))
        payload["threshold_policy_experiment_status"] = threshold_experiment_payload.get("experiment_status")
        threshold_shadow_payload = _load_json_if_exists(Path(payload["threshold_shadow_simulation_json"]))
        payload["threshold_shadow_simulation_status"] = threshold_shadow_payload.get("shadow_status")
        threshold_shadow_review_payload = _load_json_if_exists(Path(payload["threshold_shadow_review_json"]))
        payload["threshold_shadow_review_status"] = threshold_shadow_review_payload.get("review_status")
        threshold_promotion_review_payload = _load_json_if_exists(Path(payload["threshold_promotion_review_json"]))
        payload["threshold_promotion_review_status"] = threshold_promotion_review_payload.get("promotion_review_status")
        threshold_post_promotion_monitor_payload = _load_json_if_exists(Path(payload["threshold_post_promotion_monitor_json"]))
        payload["threshold_post_promotion_monitor_status"] = threshold_post_promotion_monitor_payload.get("monitor_status")
        threshold_adoption_reconciliation_payload = _load_json_if_exists(Path(payload["threshold_adoption_reconciliation_json"]))
        payload["threshold_adoption_reconciliation_status"] = threshold_adoption_reconciliation_payload.get("adoption_status")
        alert_artifact = run_signal_drift_alert_workflow(
            trend_dashboard_path=payload["drift_trend_dashboard_json"],
            output_dir=drift_dir,
            review_ledger_path=payload["drift_review_ledger_csv"],
        )
        payload["drift_alert_json"] = alert_artifact["alert_json_path"]
        payload["drift_alert_markdown"] = alert_artifact["alert_markdown_path"]
        payload["drift_alert_status"] = alert_artifact["alert_summary"].get("ops_status")
        payload["status"] = "SUCCESS"
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["error"] = str(exc)

    payload["completed_at"] = _utc_now()
    checkpoint_json = _write_checkpoint(workflow_history_dir, run_id, payload)
    payload["checkpoint_json"] = str(checkpoint_json)
    history_csv = _append_history_row(workflow_history_dir, payload)
    payload["history_csv"] = str(history_csv)
    checkpoint_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the daily research report + drift monitor workflow.")
    parser.add_argument("--date", dest="report_date", default=None, help="Report date in YYYY-MM-DD format.")
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset path. Defaults to cumulative when present.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Daily report output directory.")
    parser.add_argument("--history-dir", type=Path, default=None, help="Run history/checkpoint directory.")
    parser.add_argument("--include-cumulative", action="store_true", help="Also generate the cumulative research report.")
    parser.add_argument("--narrative", action="store_true", help="Enable AI narrative sections when configured.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip outcome backfill before report generation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_daily_research_workflow(
        report_date=_parse_report_date(args.report_date),
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        history_dir=args.history_dir,
        include_cumulative=args.include_cumulative,
        narrative=args.narrative,
        run_evaluation=not args.skip_evaluation,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
