from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_adoption_history import (
    LIFECYCLE_ADOPTED_ACTIVE,
    LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING,
    LIFECYCLE_MISMATCHED,
    RUNTIME_CANDIDATE_SIGNALING,
    RUNTIME_NON_CANDIDATE_SIGNALING,
    build_threshold_adoption_history_dashboard,
    build_threshold_adoption_history_row,
    write_threshold_adoption_history,
)
from research.signal_evaluation.threshold_signal_rollout_monitor import (
    CANDIDATE_SIGNAL_ROLLOUT_BLOCKED,
    CANDIDATE_SIGNAL_ROLLOUT_HEALTHY,
    CANDIDATE_SIGNAL_ROLLOUT_WATCH,
)


def _rollout_report(
    *,
    rollout_status: str = CANDIDATE_SIGNAL_ROLLOUT_WATCH,
    candidate_count: int = 0,
    non_candidate_count: int = 1,
    missing_count: int = 0,
    traceability_status: str = "NO_CANDIDATE_PACK_SIGNALS_YET",
) -> dict:
    return {
        "generated_at": "2026-05-15T04:00:00+00:00",
        "dataset_path": "research/signal_evaluation/signals_dataset_cumul.csv",
        "baseline_pack_name": "baseline_v1",
        "candidate_pack_name": "candidate_v1",
        "config_hint": "evaluation_thresholds.selection.composite_signal_score_floor",
        "approved_threshold_value": 85.0,
        "baseline_runtime_value": 75.0,
        "candidate_runtime_value": 85.0,
        "adoption_start_timestamp": "2026-05-15T03:21:43+00:00",
        "adoption_reconciliation_status": "ADOPTED_MANUALLY",
        "post_promotion_monitor_status": "POST_PROMOTION_HEALTHY",
        "rollout_status": rollout_status,
        "runtime_config_changed": False,
        "execution_behavior_changed": False,
        "post_adoption_traceability": {
            "traceability_status": traceability_status,
            "post_adoption_signal_count": candidate_count + non_candidate_count + missing_count,
            "candidate_pack_signal_count": candidate_count,
            "non_candidate_pack_signal_count": non_candidate_count,
            "missing_parameter_pack_count": missing_count,
            "parameter_pack_values": ["candidate_v1"] if candidate_count else ["baseline_v1"],
        },
        "candidate_label_readiness": {
            "label_count_60m": 0,
            "outcome_monitoring_status": "AWAITING_OUTCOME_LABELS",
        },
        "rollout_comparison": {
            "baseline_signal_count": 214,
            "candidate_signal_count": 117,
            "signal_count_delta": -97,
        },
        "execution_side_effects": {
            "execution_side_effect_check_passed": True,
            "orders_submitted": False,
        },
    }


def test_adoption_history_row_flags_adopted_but_not_signaling():
    row = build_threshold_adoption_history_row(_rollout_report())

    assert row["runtime_signal_status"] == RUNTIME_NON_CANDIDATE_SIGNALING
    assert row["adoption_lifecycle_status"] == LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING
    assert row["candidate_pack_signal_count"] == 0
    assert row["non_candidate_pack_signal_count"] == 1


def test_adoption_history_row_flags_active_candidate_signaling():
    row = build_threshold_adoption_history_row(
        _rollout_report(
            rollout_status=CANDIDATE_SIGNAL_ROLLOUT_HEALTHY,
            candidate_count=2,
            non_candidate_count=0,
            traceability_status="ALL_POST_ADOPTION_SIGNALS_CANDIDATE_PACK",
        )
    )

    assert row["runtime_signal_status"] == RUNTIME_CANDIDATE_SIGNALING
    assert row["adoption_lifecycle_status"] == LIFECYCLE_ADOPTED_ACTIVE


def test_adoption_history_row_flags_blocked_as_mismatched():
    row = build_threshold_adoption_history_row(
        _rollout_report(
            rollout_status=CANDIDATE_SIGNAL_ROLLOUT_BLOCKED,
            candidate_count=1,
            non_candidate_count=0,
            traceability_status="ALL_POST_ADOPTION_SIGNALS_CANDIDATE_PACK",
        )
    )

    assert row["adoption_lifecycle_status"] == LIFECYCLE_MISMATCHED


def test_adoption_history_writer_appends_and_writes_dashboard(tmp_path: Path):
    rollout_path = tmp_path / "rollout.json"
    adoption_path = tmp_path / "adoption.json"
    post_path = tmp_path / "post.json"
    rollout_path.write_text(json.dumps(_rollout_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps({"adoption_status": "ADOPTED_MANUALLY"}), encoding="utf-8")
    post_path.write_text(json.dumps({"monitor_status": "POST_PROMOTION_HEALTHY"}), encoding="utf-8")

    first = write_threshold_adoption_history(
        rollout_report_path=rollout_path,
        adoption_reconciliation_report_path=adoption_path,
        post_promotion_monitor_report_path=post_path,
        output_dir=tmp_path,
        observed_at="2026-05-15T04:00:00+00:00",
    )
    rollout_path.write_text(
        json.dumps(
            _rollout_report(
                rollout_status=CANDIDATE_SIGNAL_ROLLOUT_HEALTHY,
                candidate_count=2,
                non_candidate_count=0,
                traceability_status="ALL_POST_ADOPTION_SIGNALS_CANDIDATE_PACK",
            )
        ),
        encoding="utf-8",
    )
    second = write_threshold_adoption_history(
        rollout_report_path=rollout_path,
        adoption_reconciliation_report_path=adoption_path,
        post_promotion_monitor_report_path=post_path,
        output_dir=tmp_path,
        observed_at="2026-05-15T05:00:00+00:00",
    )

    history = pd.read_csv(second["history_path"])
    assert len(history) == 2
    assert history["adoption_lifecycle_status"].tolist() == [
        LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING,
        LIFECYCLE_ADOPTED_ACTIVE,
    ]
    assert Path(first["history_dashboard_json_path"]).exists()
    assert Path(second["history_dashboard_markdown_path"]).exists()
    assert second["history_dashboard"]["trend_assessment"] == "ACTIVE"


def test_adoption_history_dashboard_handles_empty_history():
    dashboard = build_threshold_adoption_history_dashboard(pd.DataFrame())

    assert dashboard["trend_assessment"] == "NO_HISTORY"
    assert dashboard["run_count"] == 0
