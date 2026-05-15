from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.policy_resolver import get_active_parameter_pack, temporary_parameter_pack
from research.signal_evaluation.threshold_adoption_replay_gate import (
    ADOPTION_REPLAY_BLOCKED,
    ADOPTION_REPLAY_READY,
    build_threshold_adoption_replay_gate_report,
    write_threshold_adoption_replay_gate_report,
)


CONFIG_HINT = "evaluation_thresholds.selection.composite_signal_score_floor"


def _adoption_plan(value: float = 85.0, *, status: str = "ADOPTION_PLAN_READY") -> dict:
    return {
        "report_type": "threshold_adoption_plan",
        "plan_status": status,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "proposed_change": {
            "config_hint": CONFIG_HINT,
            "current_active_value": 75.0,
            "current_active_source": "active_selection_policy",
            "current_target_pack_value": None,
            "candidate_value": value,
            "candidate_value_source": "promotion_candidate.overrides",
            "default_runtime_value": 75.0,
            "operation": "add",
        },
        "parameter_pack_patch": {
            "op": "add",
            "path": f"/overrides/{CONFIG_HINT}",
            "from": None,
            "value": value,
        },
        "target_parameter_pack_before": {
            "name": "candidate_v1",
            "overrides": {},
        },
        "target_parameter_pack_after": {
            "name": "candidate_v1",
            "overrides": {CONFIG_HINT: value},
        },
    }


def _frame() -> pd.DataFrame:
    rows = []
    for idx, composite in enumerate([80.0, 90.0, 70.0], start=1):
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": f"2026-02-0{idx}T09:20:00+05:30",
                "source": "unit",
                "requested_option_source": "nse",
                "option_source": "nse",
                "spot_source": "yfinance",
                "market_data_provenance_status": "OK",
                "trade_strength": 70.0,
                "composite_signal_score": composite,
                "tradeability_score": 70.0,
                "hybrid_move_probability": 0.70,
                "option_efficiency_score": 50.0,
                "global_risk_score": 20.0,
                "overnight_hold_allowed": True,
                "correct_60m": 1.0 if composite >= 80 else 0.0,
                "signed_return_60m_bps": 20.0 if composite >= 80 else -10.0,
            }
        )
    return pd.DataFrame(rows)


def test_adoption_replay_gate_ready_for_selection_only_threshold_change():
    with temporary_parameter_pack("baseline_v1"):
        active_before = get_active_parameter_pack()
        report = build_threshold_adoption_replay_gate_report(
            _frame(),
            adoption_plan_report=_adoption_plan(85.0),
            dataset_path="unit.csv",
        )
        active_after = get_active_parameter_pack()

    assert report["replay_status"] == ADOPTION_REPLAY_READY
    assert report["runtime_config_changed"] is False
    assert report["parameter_pack_file_changed"] is False
    assert report["policy_delta"] == {"composite_signal_score_floor": {"baseline": 75.0, "candidate": 85.0}}
    assert report["replay_comparison"]["baseline_signal_count"] == 2
    assert report["replay_comparison"]["candidate_signal_count"] == 1
    assert report["replay_comparison"]["expected_relationship"] == "candidate_subset_of_baseline"
    assert report["output_structure_guard"]["candidate_matches_input"] is True
    assert report["provenance_guard"]["provenance_check_passed"] is True
    assert active_before == active_after


def test_adoption_replay_gate_blocks_unsupported_config_hint():
    plan = _adoption_plan(85.0)
    plan["proposed_change"]["config_hint"] = "research_only.hybrid_move_probability_floor"
    plan["target_parameter_pack_after"]["overrides"] = {
        "research_only.hybrid_move_probability_floor": 85.0,
    }

    with temporary_parameter_pack("baseline_v1"):
        report = build_threshold_adoption_replay_gate_report(
            _frame(),
            adoption_plan_report=plan,
            dataset_path="unit.csv",
        )

    assert report["replay_status"] == ADOPTION_REPLAY_BLOCKED
    assert "not a supported selection-policy override" in report["replay_reasons"][0]


def test_adoption_replay_gate_blocks_unplanned_parameter_pack_override():
    plan = _adoption_plan(85.0)
    plan["target_parameter_pack_after"]["overrides"]["evaluation_thresholds.selection.trade_strength_floor"] = 65.0

    with temporary_parameter_pack("baseline_v1"):
        report = build_threshold_adoption_replay_gate_report(
            _frame(),
            adoption_plan_report=plan,
            dataset_path="unit.csv",
        )

    assert report["replay_status"] == ADOPTION_REPLAY_BLOCKED
    assert "target parameter-pack diff contains override changes" in " ".join(report["replay_reasons"])
    assert report["plan_override_guard"]["changed_override_keys"] == [
        CONFIG_HINT,
        "evaluation_thresholds.selection.trade_strength_floor",
    ]


def test_adoption_replay_gate_writer_outputs_latest_artifacts(tmp_path: Path):
    with temporary_parameter_pack("baseline_v1"):
        artifact = write_threshold_adoption_replay_gate_report(
            _frame(),
            adoption_plan_report=_adoption_plan(85.0),
            dataset_path="unit.csv",
            output_dir=tmp_path,
            report_name="unit_threshold_adoption_replay_gate",
        )

    for key in [
        "json_path",
        "markdown_path",
        "comparison_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_comparison_csv_path",
    ]:
        assert Path(artifact[key]).exists()

    assert artifact["report"]["replay_status"] == ADOPTION_REPLAY_READY
