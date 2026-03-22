"""Tests for sink failure isolation and error tracking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from app.runtime_sinks import DefaultShadowEvaluationSink


def test_shadow_evaluation_sink_handles_evaluation_failure_gracefully():
    """When shadow evaluation raises, sink records failure status but does not crash."""
    sink = DefaultShadowEvaluationSink()
    result_payload = {
        "spot_summary": {"timestamp": "2026-03-22T10:00:00+05:30"},
    }

    def failing_evaluator(**kwargs):
        raise ValueError("Shadow pack evaluation failed")

    sink.apply(
        result_payload=result_payload,
        shadow_pack_name="test_pack",
        symbol="NIFTY",
        mode="replay",
        source="mock",
        spot=23000,
        option_chain=MagicMock(),
        previous_chain=None,
        day_high=23100,
        day_low=22900,
        day_open=23050,
        prev_close=22950,
        lookback_avg_range_pct=0.5,
        spot_validation={},
        option_chain_validation={},
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=75,
        max_capital=1000000,
        macro_event_state={},
        headline_state=None,
        global_market_snapshot={},
        holding_profile="intraday",
        spot_timestamp="2026-03-22T10:00:00+05:30",
        baseline_pack_name="baseline",
        enable_shadow_logging=False,
        backtest_mode=False,
        target_profit_percent=2.0,
        stop_loss_percent=1.0,
        evaluate_snapshot_for_pack=failing_evaluator,
    )

    assert result_payload["shadow_mode_active"] is True
    assert result_payload["shadow_pack_name"] == "test_pack"
    assert result_payload["shadow_evaluation_failed"] is True
    assert "failed" in result_payload["shadow_evaluation_error"].lower()


def test_shadow_evaluation_sink_handles_comparison_failure_gracefully():
    """When shadow comparison raises, sink records failure status but does not crash."""
    sink = DefaultShadowEvaluationSink()
    result_payload = {
        "spot_summary": {"timestamp": "2026-03-22T10:00:00+05:30"},
    }

    def succeeding_evaluator(**kwargs):
        return {
            "parameter_pack_name": "test_pack",
            "trade": None,
            "macro_news_state": {},
            "global_risk_state": {},
        }

    def failing_comparator(*args, **kwargs):
        raise RuntimeError("Comparison logic failed")

    with patch("app.runtime_sinks.compare_shadow_trade_outputs", side_effect=failing_comparator):
        sink.apply(
            result_payload=result_payload,
            shadow_pack_name="test_pack",
            symbol="NIFTY",
            mode="replay",
            source="mock",
            spot=23000,
            option_chain=MagicMock(),
            previous_chain=None,
            day_high=23100,
            day_low=22900,
            day_open=23050,
            prev_close=22950,
            lookback_avg_range_pct=0.5,
            spot_validation={},
            option_chain_validation={},
            apply_budget_constraint=False,
            requested_lots=1,
            lot_size=75,
            max_capital=1000000,
            macro_event_state={},
            headline_state=None,
            global_market_snapshot={},
            holding_profile="intraday",
            spot_timestamp="2026-03-22T10:00:00+05:30",
            baseline_pack_name="baseline",
            enable_shadow_logging=False,
            backtest_mode=False,
            target_profit_percent=2.0,
            stop_loss_percent=1.0,
            evaluate_snapshot_for_pack=succeeding_evaluator,
        )

    assert result_payload["shadow_comparison_failed"] is True
    assert "Comparison" in result_payload["shadow_comparison_error"]


def test_shadow_sink_uses_explicit_status_flags():
    """Shadow sink populates shadow_log_status field explicitly."""
    sink = DefaultShadowEvaluationSink()
    result_payload = {
        "spot_summary": {"timestamp": "2026-03-22T10:00:00+05:30"},
    }

    def succeeding_evaluator(**kwargs):
        return {
            "parameter_pack_name": "test_pack",
            "trade": None,
            "macro_news_state": {},
            "global_risk_state": {},
        }

    with patch("app.runtime_sinks.compare_shadow_trade_outputs", return_value={}):
        with patch("app.runtime_sinks.append_shadow_log", side_effect=IOError("Disk full")):
            sink.apply(
                result_payload=result_payload,
                shadow_pack_name="test_pack",
                symbol="NIFTY",
                mode="replay",
                source="mock",
                spot=23000,
                option_chain=MagicMock(),
                previous_chain=None,
                day_high=23100,
                day_low=22900,
                day_open=23050,
                prev_close=22950,
                lookback_avg_range_pct=0.5,
                spot_validation={},
                option_chain_validation={},
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=75,
                max_capital=1000000,
                macro_event_state={},
                headline_state=None,
                global_market_snapshot={},
                holding_profile="intraday",
                spot_timestamp="2026-03-22T10:00:00+05:30",
                baseline_pack_name="baseline",
                enable_shadow_logging=True,
                backtest_mode=False,
                target_profit_percent=2.0,
                stop_loss_percent=1.0,
                evaluate_snapshot_for_pack=succeeding_evaluator,
            )

    assert result_payload["shadow_log_status"].startswith("FAILED")
    assert "IOError" in result_payload["shadow_log_status"] or "OSError" in result_payload["shadow_log_status"]
