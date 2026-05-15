#!/usr/bin/env python3
"""Run the one-command post-activation threshold rollout verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.threshold_adoption_history import (  # noqa: E402
    DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR,
    THRESHOLD_ADOPTION_HISTORY_CSV_FILENAME,
)
from research.signal_evaluation.threshold_post_activation_verification import (  # noqa: E402
    DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH,
    DEFAULT_THRESHOLD_POST_ACTIVATION_VERIFICATION_DIR,
    POST_ACTIVATION_VERIFICATION_CLEAN,
    default_signal_dataset_path,
    run_threshold_post_activation_verification_from_paths,
)
from research.signal_evaluation.threshold_signal_rollout_monitor import (  # noqa: E402
    DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH,
    DEFAULT_BASELINE_PARAMETER_PACK,
    DEFAULT_CANDIDATE_PARAMETER_PACK,
    DEFAULT_CONFIG_HINT,
    DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH,
    DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR,
)


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify post-activation threshold rollout from existing signal/research artifacts. "
            "This command does not run the engine, change parameter packs, or touch execution."
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
    parser.add_argument("--rollout-output-dir", type=Path, default=DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR)
    parser.add_argument("--rollout-report-name", default=None)
    parser.add_argument("--history-output-dir", type=Path, default=DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR)
    parser.add_argument("--history-filename", default=THRESHOLD_ADOPTION_HISTORY_CSV_FILENAME)
    parser.add_argument("--lookback-runs", type=int, default=20)
    parser.add_argument("--verification-output-dir", type=Path, default=DEFAULT_THRESHOLD_POST_ACTIVATION_VERIFICATION_DIR)
    parser.add_argument("--verification-report-name", default=None)
    parser.add_argument("--min-candidate-label-count", type=int, default=1)
    parser.add_argument(
        "--allow-awaiting-labels",
        action="store_true",
        help="Do not block solely because candidate outcome labels are not ready yet.",
    )
    parser.add_argument(
        "--allow-mixed-pack-signals",
        action="store_true",
        help="Do not make the rollout monitor block on mixed candidate/non-candidate post-activation signals.",
    )
    parser.add_argument(
        "--allow-unadopted-reconciliation",
        action="store_true",
        help="Do not block solely because adoption reconciliation is not ADOPTED_MANUALLY.",
    )
    parser.add_argument(
        "--no-fail-on-not-clean",
        action="store_true",
        help="Print blocked verification output but exit with status 0.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = args.dataset or default_signal_dataset_path()
    adoption = _load_json(args.adoption_reconciliation)
    post_monitor = _load_json(args.post_promotion_monitor)
    activation_marker = _load_json(args.runtime_activation_marker)
    artifact = run_threshold_post_activation_verification_from_paths(
        dataset_path=dataset_path,
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
        rollout_output_dir=args.rollout_output_dir,
        rollout_report_name=args.rollout_report_name,
        history_output_dir=args.history_output_dir,
        history_filename=args.history_filename,
        lookback_runs=args.lookback_runs,
        verification_output_dir=args.verification_output_dir,
        verification_report_name=args.verification_report_name,
        min_candidate_label_count=args.min_candidate_label_count,
        require_candidate_labels=not args.allow_awaiting_labels,
        require_adopted_reconciliation=not args.allow_unadopted_reconciliation,
        strict_candidate_pack_required=not args.allow_mixed_pack_signals,
    )
    report = artifact.get("verification_report", {}) or {}
    rollout = artifact.get("rollout_artifact", {}) or {}
    history = artifact.get("adoption_history_artifact", {}) or {}
    history_row = history.get("history_row", {}) or {}
    payload = {
        "verification_status": report.get("verification_status"),
        "verification_reasons": report.get("verification_reasons"),
        "verification_json_path": artifact.get("verification_json_path"),
        "verification_markdown_path": artifact.get("verification_markdown_path"),
        "latest_verification_json_path": artifact.get("latest_verification_json_path"),
        "latest_verification_markdown_path": artifact.get("latest_verification_markdown_path"),
        "rollout_status": (rollout.get("report", {}) or {}).get("rollout_status"),
        "rollout_json_path": rollout.get("json_path"),
        "latest_rollout_json_path": rollout.get("latest_json_path"),
        "history_path": history.get("history_path"),
        "adoption_lifecycle_status": history_row.get("adoption_lifecycle_status"),
        "runtime_signal_status": history_row.get("runtime_signal_status"),
        "runtime_config_changed": report.get("runtime_config_changed"),
        "execution_behavior_changed": report.get("execution_behavior_changed"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if not args.no_fail_on_not_clean and report.get("verification_status") != POST_ACTIVATION_VERIFICATION_CLEAN:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
