from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.signal_evaluation.dataset import write_signals_dataset
from tuning.governance import run_controlled_tuning_workflow
from tuning.promotion import (
    load_promotion_state,
    promote_candidate,
    record_manual_approval,
    update_pack_state,
)


def _build_dataset_frame():
    return pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "signal_timestamp": "2026-01-02T09:20:00+05:30",
                "symbol": "NIFTY",
                "direction": "CALL",
                "trade_strength": 58,
                "composite_signal_score": 68,
                "tradeability_score": 62,
                "hybrid_move_probability": 0.58,
                "move_probability": 0.58,
                "option_efficiency_score": 65,
                "global_risk_score": 42,
                "overnight_hold_allowed": True,
                "correct_5m": 1,
                "correct_15m": 1,
                "correct_30m": 1,
                "correct_60m": 1,
                "mae_60m_bps": -35,
                "target_reachability_score": 72,
                "signal_regime": "DIRECTIONAL_BIAS",
                "macro_regime": "RISK_ON",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "direction_score": 72,
                "magnitude_score": 69,
                "timing_score": 66,
                "realized_return_5m": 0.0007,
                "realized_return_15m": 0.0012,
                "realized_return_30m": 0.0018,
                "realized_return_60m": 0.0024,
                "signed_return_5m_bps": 7.0,
                "signed_return_15m_bps": 12.0,
                "signed_return_30m_bps": 18.0,
                "signed_return_60m_bps": 24.0,
                "outcome_status": "COMPLETE",
            },
            {
                "signal_id": "sig-2",
                "signal_timestamp": "2026-01-03T09:20:00+05:30",
                "symbol": "NIFTY",
                "direction": "PUT",
                "trade_strength": 43,
                "composite_signal_score": 54,
                "tradeability_score": 51,
                "hybrid_move_probability": 0.46,
                "move_probability": 0.46,
                "option_efficiency_score": 48,
                "global_risk_score": 55,
                "overnight_hold_allowed": True,
                "correct_5m": 0,
                "correct_15m": 0,
                "correct_30m": 0,
                "correct_60m": 0,
                "mae_60m_bps": -60,
                "target_reachability_score": 49,
                "signal_regime": "BALANCED",
                "macro_regime": "RISK_OFF",
                "gamma_regime": "LONG_GAMMA_ZONE",
                "direction_score": 42,
                "magnitude_score": 41,
                "timing_score": 38,
                "realized_return_5m": -0.0002,
                "realized_return_15m": -0.0004,
                "realized_return_30m": -0.0008,
                "realized_return_60m": -0.0011,
                "signed_return_5m_bps": -2.0,
                "signed_return_15m_bps": -4.0,
                "signed_return_30m_bps": -8.0,
                "signed_return_60m_bps": -11.0,
                "outcome_status": "COMPLETE",
            },
            {
                "signal_id": "sig-3",
                "signal_timestamp": "2026-01-04T09:20:00+05:30",
                "symbol": "BANKNIFTY",
                "direction": "CALL",
                "trade_strength": 39,
                "composite_signal_score": 44,
                "tradeability_score": 35,
                "hybrid_move_probability": 0.32,
                "move_probability": 0.32,
                "option_efficiency_score": 30,
                "global_risk_score": 88,
                "overnight_hold_allowed": False,
                "correct_5m": 0,
                "correct_15m": 0,
                "correct_30m": 0,
                "correct_60m": 0,
                "mae_60m_bps": -120,
                "target_reachability_score": 25,
                "signal_regime": "CONFLICTED",
                "macro_regime": "RISK_OFF",
                "gamma_regime": "LONG_GAMMA_ZONE",
                "direction_score": 28,
                "magnitude_score": 22,
                "timing_score": 18,
                "realized_return_5m": -0.0005,
                "realized_return_15m": -0.0008,
                "realized_return_30m": -0.0015,
                "realized_return_60m": -0.0025,
                "signed_return_5m_bps": -5.0,
                "signed_return_15m_bps": -8.0,
                "signed_return_30m_bps": -15.0,
                "signed_return_60m_bps": -25.0,
                "outcome_status": "COMPLETE",
            },
            {
                "signal_id": "sig-4",
                "signal_timestamp": "2026-01-05T09:20:00+05:30",
                "symbol": "BANKNIFTY",
                "direction": "CALL",
                "trade_strength": 81,
                "composite_signal_score": 84,
                "tradeability_score": 78,
                "hybrid_move_probability": 0.73,
                "move_probability": 0.73,
                "option_efficiency_score": 82,
                "global_risk_score": 38,
                "overnight_hold_allowed": True,
                "correct_5m": 1,
                "correct_15m": 1,
                "correct_30m": 1,
                "correct_60m": 1,
                "mae_60m_bps": -20,
                "target_reachability_score": 88,
                "signal_regime": "EXPANSION_BIAS",
                "macro_regime": "RISK_ON",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "direction_score": 86,
                "magnitude_score": 81,
                "timing_score": 78,
                "realized_return_5m": 0.0010,
                "realized_return_15m": 0.0018,
                "realized_return_30m": 0.0028,
                "realized_return_60m": 0.0035,
                "signed_return_5m_bps": 10.0,
                "signed_return_15m_bps": 18.0,
                "signed_return_30m_bps": 28.0,
                "signed_return_60m_bps": 35.0,
                "outcome_status": "COMPLETE",
            },
        ]
    )


def _write_pack(pack_dir: Path, name: str, *, parent: str | None = None, overrides: dict | None = None) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "version": "1.0.0",
        "description": f"Pack {name}",
        "parent": parent,
        "tags": ["test"],
        "metadata": {"owner": "tests"},
        "overrides": dict(overrides or {}),
    }
    (pack_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_controlled_tuning_workflow_creates_candidate_without_touching_live(tmp_path, monkeypatch):
    import tuning.packs as packs_module

    config_packs_dir = tmp_path / "config_packs"
    candidate_packs_dir = tmp_path / "candidate_packs"
    reports_dir = tmp_path / "reports"
    dataset_path = tmp_path / "signals.csv"
    state_path = tmp_path / "promotion_state.json"
    ledger_path = tmp_path / "promotion_ledger.jsonl"

    _write_pack(config_packs_dir, "baseline_v1")
    monkeypatch.setattr(
        packs_module,
        "DEFAULT_PARAMETER_PACK_DIRS",
        (config_packs_dir, candidate_packs_dir),
    )

    write_signals_dataset(_build_dataset_frame(), dataset_path)
    workflow = run_controlled_tuning_workflow(
        dataset_path=dataset_path,
        production_pack_name="baseline_v1",
        groups=["trade_strength"],
        walk_forward_config={
            "split_type": "rolling",
            "train_window_days": 2,
            "validation_window_days": 1,
            "step_size_days": 1,
            "minimum_train_rows": 1,
            "minimum_validation_rows": 1,
        },
        seed=11,
        created_by="researcher",
        notes="unit test campaign",
        persist=False,
        state_path=state_path,
        ledger_path=ledger_path,
        reports_dir=reports_dir,
        candidate_packs_dir=candidate_packs_dir,
    )

    state = load_promotion_state(state_path)
    assert workflow["production_pack_name"] == "baseline_v1"
    assert state["live"] == "baseline_v1"
    assert state["candidate"] == workflow["candidate_pack_name"]
    assert Path(workflow["candidate_pack_path"]).exists()
    assert Path(workflow["candidate_report_paths"]["json_path"]).exists()
    assert Path(workflow["signal_evaluation_report"]["json_path"]).exists()
    assert workflow["candidate_report"]["parameter_change_table"]


def test_promotion_requires_recorded_manual_approval(tmp_path):
    state_path = tmp_path / "promotion_state.json"
    ledger_path = tmp_path / "promotion_ledger.jsonl"

    update_pack_state(
        state_name="candidate",
        pack_name="candidate_under_review",
        reason="candidate_created",
        assigned_by="researcher",
        path=state_path,
        ledger_path=ledger_path,
    )

    with pytest.raises(PermissionError):
        promote_candidate(
            "candidate_under_review",
            approved_by="pm",
            path=state_path,
            ledger_path=ledger_path,
        )


def test_approved_candidate_promotion_updates_live_and_logs_expected_improvement(tmp_path):
    state_path = tmp_path / "promotion_state.json"
    ledger_path = tmp_path / "promotion_ledger.jsonl"

    update_pack_state(
        state_name="candidate",
        pack_name="candidate_under_review",
        reason="candidate_created",
        assigned_by="researcher",
        path=state_path,
        ledger_path=ledger_path,
    )
    record_manual_approval(
        pack_name="candidate_under_review",
        approved=True,
        reviewer="pm",
        notes="review complete",
        path=state_path,
        ledger_path=ledger_path,
    )

    promote_candidate(
        "candidate_under_review",
        approved_by="pm",
        source_experiment_id="exp_123",
        expected_improvement_summary={"objective_score_delta": 0.031, "hit_rate_delta": 0.02},
        path=state_path,
        ledger_path=ledger_path,
    )

    state = load_promotion_state(state_path)
    assert state["live"] == "candidate_under_review"
    assert state["previous_live"] == "baseline_v1"

    ledger_records = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    promotion_events = [row for row in ledger_records if row.get("event_type") == "candidate_promoted_to_live"]
    assert promotion_events
    assert promotion_events[-1]["metadata"]["expected_improvement_summary"]["objective_score_delta"] == 0.031
