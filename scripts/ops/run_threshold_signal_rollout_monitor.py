#!/usr/bin/env python3
"""Run signal-only monitoring for the adopted threshold candidate pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.daily_research_report import DEFAULT_CUMULATIVE_DATASET_PATH, DEFAULT_DATASET_PATH  # noqa: E402
from research.signal_evaluation.threshold_adoption_reconciliation import (  # noqa: E402
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME,
)
from research.signal_evaluation.threshold_post_promotion_monitor import (  # noqa: E402
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME,
)
from research.signal_evaluation.threshold_runtime_activation import (  # noqa: E402
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR,
    THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME,
)
from research.signal_evaluation.threshold_signal_rollout_monitor import (  # noqa: E402
    CANDIDATE_SIGNAL_ROLLOUT_BLOCKED,
    DEFAULT_BASELINE_PARAMETER_PACK,
    DEFAULT_CANDIDATE_PARAMETER_PACK,
    DEFAULT_CONFIG_HINT,
    DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR,
    write_threshold_signal_rollout_monitor_report,
)


DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH = (
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR / THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME
)
DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH = (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR / THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME
)
DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH = (
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR / THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME
)


def _default_dataset_path() -> Path:
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor candidate threshold rollout from the signal dataset only. "
            "This command does not run the engine or touch execution."
        )
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Signal dataset CSV path.")
    parser.add_argument("--baseline-pack", default=DEFAULT_BASELINE_PARAMETER_PACK)
    parser.add_argument("--candidate-pack", default=DEFAULT_CANDIDATE_PARAMETER_PACK)
    parser.add_argument("--config-hint", default=DEFAULT_CONFIG_HINT)
    parser.add_argument("--approved-threshold-value", type=float, default=None)
    parser.add_argument("--adoption-start-at", default=None)
    parser.add_argument("--adoption-reconciliation", type=Path, default=DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH)
    parser.add_argument("--post-promotion-monitor", type=Path, default=DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH)
    parser.add_argument("--runtime-activation-at", default=None)
    parser.add_argument("--runtime-activation-marker", type=Path, default=DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR)
    parser.add_argument("--report-name", default=None)
    parser.add_argument(
        "--strict-candidate-pack",
        action="store_true",
        help="Block if post-adoption rows include non-candidate parameter-pack signals.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit with status 2 when rollout status is CANDIDATE_SIGNAL_ROLLOUT_BLOCKED.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or _default_dataset_path()
    frame = pd.read_csv(dataset_path, low_memory=False)
    adoption = _load_json(args.adoption_reconciliation)
    post_monitor = _load_json(args.post_promotion_monitor)
    activation_marker = _load_json(args.runtime_activation_marker)
    artifact = write_threshold_signal_rollout_monitor_report(
        frame,
        dataset_path=str(dataset_path),
        baseline_pack_name=args.baseline_pack,
        candidate_pack_name=args.candidate_pack,
        config_hint=args.config_hint,
        approved_threshold_value=args.approved_threshold_value,
        adoption_start_at=args.adoption_start_at,
        adoption_reconciliation_report=adoption,
        adoption_reconciliation_report_path=args.adoption_reconciliation,
        post_promotion_monitor_report=post_monitor,
        post_promotion_monitor_report_path=args.post_promotion_monitor,
        runtime_activation_at=args.runtime_activation_at,
        runtime_activation_marker=activation_marker,
        runtime_activation_marker_path=args.runtime_activation_marker,
        strict_candidate_pack_required=args.strict_candidate_pack,
        output_dir=args.output_dir,
        report_name=args.report_name,
        write_latest=True,
    )
    report = artifact["report"]
    comparison = report.get("rollout_comparison", {}) or {}
    traceability = report.get("post_adoption_traceability", {}) or {}
    payload = {key: value for key, value in artifact.items() if key != "report"}
    payload["rollout_status"] = report.get("rollout_status")
    payload["runtime_config_changed"] = report.get("runtime_config_changed")
    payload["execution_behavior_changed"] = report.get("execution_behavior_changed")
    payload["candidate_runtime_value"] = report.get("candidate_runtime_value")
    payload["baseline_signal_count"] = comparison.get("baseline_signal_count")
    payload["candidate_signal_count"] = comparison.get("candidate_signal_count")
    payload["post_adoption_signal_count"] = traceability.get("post_adoption_signal_count")
    payload["candidate_pack_signal_count"] = traceability.get("candidate_pack_signal_count")
    payload["runtime_activation_timestamp"] = report.get("runtime_activation_timestamp")
    payload["traceability_window_start_timestamp"] = report.get("traceability_window_start_timestamp")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.fail_on_blocked and report.get("rollout_status") == CANDIDATE_SIGNAL_ROLLOUT_BLOCKED:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
