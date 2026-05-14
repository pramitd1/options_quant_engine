from __future__ import annotations

from engine.runtime_metadata import build_trader_view, split_trade_payload


def test_split_trade_payload_keeps_trust_scalars_execution_facing():
    trade = {
        "signal_id": "sig-1",
        "market_data_provenance": {"status": "CAUTION", "reasons": ["mixed_spot_option_source"]},
        "market_data_provenance_status": "CAUTION",
        "market_data_source_consistency": "MIXED_SPOT_OPTION_SOURCE",
        "requested_option_source": "ICICI",
        "option_source": "ICICI",
        "spot_source": "YFINANCE_INTRADAY",
        "option_chain_validation": {"is_valid": True},
        "option_chain_validation_status": "GOOD",
        "signal_confidence_calibration_guardrail": {"status": "CAUTION", "sample_size": 12},
        "signal_confidence_recalibration_guards": ["thin_calibration_history"],
    }

    execution_trade, trade_audit = split_trade_payload(trade)

    assert execution_trade["market_data_provenance_status"] == "CAUTION"
    assert execution_trade["market_data_source_consistency"] == "MIXED_SPOT_OPTION_SOURCE"
    assert execution_trade["requested_option_source"] == "ICICI"
    assert execution_trade["signal_confidence_calibration_guardrail"]["status"] == "CAUTION"
    assert execution_trade["signal_confidence_recalibration_guards"] == ["thin_calibration_history"]
    assert trade_audit["market_data_provenance"]["status"] == "CAUTION"
    assert trade_audit["option_chain_validation"]["is_valid"] is True


def test_build_trader_view_falls_back_to_full_trade_for_operator_fields():
    trade = {
        "signal_id": "sig-2",
        "execution_trade": {"signal_id": "sig-2"},
        "hybrid_move_probability": 0.71,
        "data_quality_status": "GOOD",
        "market_data_provenance_status": "GOOD",
    }

    view = build_trader_view(trade)

    assert view["signal_id"] == "sig-2"
    assert view["hybrid_move_probability"] == 0.71
    assert view["data_quality_status"] == "GOOD"
    assert view["market_data_provenance_status"] == "GOOD"
