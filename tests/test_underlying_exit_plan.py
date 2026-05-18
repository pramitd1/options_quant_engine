from __future__ import annotations

from engine.signal_engine import _build_underlying_exit_plan


def test_underlying_exit_plan_for_call_maps_profit_above_and_stop_below():
    plan = _build_underlying_exit_plan(
        spot=23378.55,
        direction="CALL",
        entry_price=100.0,
        target=130.0,
        stop_loss=85.0,
        option_delta=0.50,
        expected_move_points=240.0,
        support_wall=23350.0,
        resistance_wall=23450.0,
        liquidity_levels=[23300.0, 23400.0, 23500.0],
        runtime_thresholds={},
    )

    assert plan["underlying_profit_booking_level"] > 23378.55
    assert plan["underlying_stop_loss_level"] < 23378.55
    assert plan["underlying_profit_booking_lower"] <= plan["underlying_profit_booking_level"]
    assert plan["underlying_profit_booking_upper"] >= plan["underlying_profit_booking_level"]
    assert plan["underlying_stop_loss_lower"] <= plan["underlying_stop_loss_level"]
    assert plan["underlying_stop_loss_upper"] >= plan["underlying_stop_loss_level"]
    assert plan["underlying_exit_plan_confidence"] == "HIGH"


def test_underlying_exit_plan_for_put_maps_profit_below_and_stop_above():
    plan = _build_underlying_exit_plan(
        spot=23378.55,
        direction="PUT",
        entry_price=118.5,
        target=154.05,
        stop_loss=100.73,
        option_delta=-0.46,
        expected_move_points=270.0,
        support_wall=23200.0,
        resistance_wall=23400.0,
        gamma_flip=23442.0,
        liquidity_levels=[23000.0, 23200.0, 23300.0, 23400.0],
        runtime_thresholds={},
    )

    assert plan["underlying_profit_booking_level"] < 23378.55
    assert plan["underlying_stop_loss_level"] > 23378.55
    assert plan["underlying_profit_booking_lower"] <= plan["underlying_profit_booking_level"]
    assert plan["underlying_profit_booking_upper"] >= plan["underlying_profit_booking_level"]
    assert plan["underlying_stop_loss_lower"] <= plan["underlying_stop_loss_level"]
    assert plan["underlying_stop_loss_upper"] >= plan["underlying_stop_loss_level"]
    assert "DELTA_PROJECTED_OPTION_EXIT" in plan["underlying_exit_plan_basis"]


def test_underlying_exit_plan_uses_fallback_delta_when_greeks_are_missing():
    plan = _build_underlying_exit_plan(
        spot=22000.0,
        direction="CALL",
        entry_price=100.0,
        target=130.0,
        stop_loss=85.0,
        option_delta=None,
        expected_move_points=180.0,
        support_wall=21950.0,
        resistance_wall=22100.0,
        runtime_thresholds={"underlying_exit_plan_fallback_delta": 0.40},
    )

    assert plan["underlying_exit_plan_confidence"] == "LOW"
    assert "fallback_delta_used" in plan["underlying_exit_plan_reasons"]
    assert plan["underlying_exit_plan"]["projection_delta"] == 0.4
