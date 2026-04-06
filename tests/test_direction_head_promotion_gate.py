from __future__ import annotations

from scripts.ops.run_direction_head_promotion_matrix import _evaluate_matrix_gate


def _base_row() -> dict:
    return {
        "scenario": "strict_balanced",
        "matched_rows": 120,
        "direction_changed_rows": 120,
        "direction_accuracy_60m_delta": 0.25,
        "avg_directional_return_60m_bps_delta": 12.0,
        "trade_count_off": 0,
        "trade_count_on": 0,
    }


def _thresholds(min_trade_evidence_rows_pass: int) -> dict:
    return {
        "min_matched_rows_block": 60,
        "min_direction_changed_share_pass": 0.10,
        "min_direction_accuracy_delta_pass": 0.02,
        "min_direction_return_delta_bps_pass": 2.0,
        "min_trade_evidence_rows_pass": min_trade_evidence_rows_pass,
        "min_direction_accuracy_delta_block": -0.02,
        "min_direction_return_delta_bps_block": -5.0,
    }


def test_gate_passes_when_trade_evidence_not_required():
    row = _base_row()
    gate = _evaluate_matrix_gate([row], _thresholds(min_trade_evidence_rows_pass=0))

    assert gate["overall_status"] == "PASS"
    scenario = gate["scenario_results"]["strict_balanced"]
    assert scenario["status"] == "PASS"
    assert "trade_evidence_not_required" in scenario["pass_checks"]


def test_gate_is_caution_when_trade_evidence_required_but_missing():
    row = _base_row()
    gate = _evaluate_matrix_gate([row], _thresholds(min_trade_evidence_rows_pass=10))

    assert gate["overall_status"] == "CAUTION"
    scenario = gate["scenario_results"]["strict_balanced"]
    assert scenario["status"] == "CAUTION"
    assert "insufficient_trade_evidence" in scenario["caution_reasons"]
