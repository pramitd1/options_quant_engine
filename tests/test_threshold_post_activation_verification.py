from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.policy_resolver import temporary_parameter_pack
from research.signal_evaluation.threshold_post_activation_verification import (
    POST_ACTIVATION_VERIFICATION_BLOCKED,
    POST_ACTIVATION_VERIFICATION_CLEAN,
    run_threshold_post_activation_verification,
)
from research.signal_evaluation.threshold_signal_rollout_monitor import DEFAULT_CONFIG_HINT


def _frame(
    *,
    pack_name: str = "baseline_v1",
    include_labels: bool = True,
    mixed_pack: bool = False,
) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-02-02T09:20:00+05:30")
    for idx, composite in enumerate([80.0, 90.0, 70.0], start=1):
        row = {
            "signal_id": f"sig-{idx}",
            "signal_timestamp": (base + pd.Timedelta(minutes=idx)).isoformat(),
            "source": "unit",
            "parameter_pack_name": pack_name,
            "requested_option_source": "nse",
            "option_source": "nse",
            "spot_source": "yfinance",
            "market_data_source_consistency": "MATCHED",
            "market_data_provenance_status": "OK",
            "market_data_trade_blocking_status": "CLEAR",
            "market_data_timestamp_status": "FRESH",
            "trade_strength": 70.0,
            "composite_signal_score": composite,
            "tradeability_score": 70.0,
            "hybrid_move_probability": 0.70,
            "option_efficiency_score": 50.0,
            "global_risk_score": 20.0,
            "overnight_hold_allowed": True,
            "signed_return_60m_bps": 20.0 if composite >= 80 else -10.0,
        }
        row["correct_60m"] = 1.0 if include_labels and composite >= 80 else None
        rows.append(row)
    if mixed_pack:
        rows[0]["parameter_pack_name"] = "baseline_writer"
    return pd.DataFrame(rows)


def _adoption_report() -> dict:
    return {
        "adoption_status": "ADOPTED_MANUALLY",
        "comparison": {
            "config_hint": DEFAULT_CONFIG_HINT,
            "candidate_value": 85.0,
            "observed_runtime_value": 85.0,
        },
        "approval_decision": {
            "reviewed_at": "2026-02-01T00:00:00Z",
        },
    }


def _marker(pack_name: str = "baseline_v1") -> dict:
    return {
        "candidate_pack_name": pack_name,
        "activated_at": "2026-02-01T00:00:00Z",
        "runtime_config_changed": False,
        "execution_behavior_changed": False,
    }


def _run(tmp_path: Path, frame: pd.DataFrame, *, marker: dict | None = None):
    adoption_path = tmp_path / "adoption.json"
    post_monitor_path = tmp_path / "post_monitor.json"
    adoption_path.write_text(json.dumps({"adoption_status": "ADOPTED_MANUALLY"}), encoding="utf-8")
    post_monitor_path.write_text(json.dumps({"monitor_status": "POST_PROMOTION_HEALTHY"}), encoding="utf-8")

    with temporary_parameter_pack("baseline_v1"):
        return run_threshold_post_activation_verification(
            frame,
            dataset_path=tmp_path / "signals.csv",
            baseline_pack_name="baseline_v1",
            candidate_pack_name="baseline_v1",
            candidate_overrides={DEFAULT_CONFIG_HINT: 85.0},
            adoption_reconciliation_report=_adoption_report(),
            adoption_reconciliation_report_path=adoption_path,
            post_promotion_monitor_report={"monitor_status": "POST_PROMOTION_HEALTHY"},
            post_promotion_monitor_report_path=post_monitor_path,
            runtime_activation_marker=marker if marker is not None else _marker(),
            runtime_activation_marker_path=tmp_path / "activation.json",
            rollout_output_dir=tmp_path / "rollout",
            history_output_dir=tmp_path / "history",
            verification_output_dir=tmp_path / "verification",
        )


def test_post_activation_verification_marks_clean_rollout_and_writes_artifacts(tmp_path: Path):
    artifact = _run(tmp_path, _frame())
    report = artifact["verification_report"]

    assert report["verification_status"] == POST_ACTIVATION_VERIFICATION_CLEAN
    assert report["active_pack_matches_marker"] is True
    assert report["checked_conditions"]["candidate_label_count_60m"] == 2
    assert Path(artifact["verification_json_path"]).exists()
    assert Path(artifact["rollout_artifact"]["latest_json_path"]).exists()
    assert Path(artifact["adoption_history_artifact"]["history_path"]).exists()


def test_post_activation_verification_blocks_active_pack_marker_mismatch(tmp_path: Path):
    artifact = _run(tmp_path, _frame(), marker=_marker("candidate_v1"))
    report = artifact["verification_report"]

    assert report["verification_status"] == POST_ACTIVATION_VERIFICATION_BLOCKED
    assert report["active_pack_matches_marker"] is False
    assert "does not match runtime activation marker" in " ".join(report["verification_reasons"])


def test_post_activation_verification_blocks_missing_candidate_labels(tmp_path: Path):
    artifact = _run(tmp_path, _frame(include_labels=False))
    report = artifact["verification_report"]

    assert report["verification_status"] == POST_ACTIVATION_VERIFICATION_BLOCKED
    assert "Candidate outcome labels are below requirement" in " ".join(report["verification_reasons"])


def test_post_activation_verification_blocks_mixed_pack_capture(tmp_path: Path):
    artifact = _run(tmp_path, _frame(mixed_pack=True))
    report = artifact["verification_report"]

    assert report["verification_status"] == POST_ACTIVATION_VERIFICATION_BLOCKED
    assert report["checked_conditions"]["non_candidate_pack_signal_count"] == 1
    assert "non-candidate parameter-pack signals" in " ".join(report["verification_reasons"])
