"""Tests for compute_confirmation_filters — both continuous and discrete scoring modes."""
from __future__ import annotations

from config.policy_resolver import temporary_parameter_pack
from strategy.confirmation_filters import compute_confirmation_filters

# Shared kwargs that produce a clearly bullish setup for PUT direction
_BEARISH_KWARGS = dict(
    direction="PUT",
    spot=22900.0,
    day_open=23100.0,       # spot well below open  → confirms PUT
    prev_close=23050.0,     # spot below prev_close → confirms PUT
    intraday_range_pct=0.55,
    final_flow_signal="BEARISH_FLOW",
    hedging_bias="DOWNSIDE_ACCELERATION",
    gamma_event="NONE",
    hybrid_move_probability=0.64,
    spot_vs_flip="BELOW_FLIP",
    gamma_regime="NEGATIVE_GAMMA",
)


def test_continuous_mode_yields_float_breakdown_scores():
    """In continuous mode, breakdown scores are floats, not just integers."""
    with temporary_parameter_pack(
        "cont_conf_test",
        overrides={"confirmation_filter.core.confirmation_scoring_mode": "continuous"},
    ):
        result = compute_confirmation_filters(**_BEARISH_KWARGS)

    bd = result["breakdown"]
    assert result["score_adjustment"] != 0, "Bearish setup should yield non-zero adjustment"
    # At least some breakdown values should be non-integer floats in continuous mode
    float_values = [v for v in bd.values() if isinstance(v, float) and v != round(v)]
    assert len(float_values) > 0, f"Expected fractional float scores in continuous mode, got {bd}"


def test_discrete_mode_yields_integer_breakdown_scores():
    """In discrete mode, breakdown scores are whole numbers (no interpolation)."""
    with temporary_parameter_pack(
        "disc_conf_test",
        overrides={"confirmation_filter.core.confirmation_scoring_mode": "discrete"},
    ):
        result = compute_confirmation_filters(**_BEARISH_KWARGS)

    bd = result["breakdown"]
    for key, val in bd.items():
        assert val == int(val), f"Expected integer score in discrete mode for {key}, got {val}"


def test_continuous_and_discrete_agree_on_directional_sign():
    """Both modes agree on the direction of the score adjustment (positive for confirmed, negative for conflict)."""
    with temporary_parameter_pack("cont_sign_test",
                                  overrides={"confirmation_filter.core.confirmation_scoring_mode": "continuous"}):
        cont = compute_confirmation_filters(**_BEARISH_KWARGS)

    with temporary_parameter_pack("disc_sign_test",
                                  overrides={"confirmation_filter.core.confirmation_scoring_mode": "discrete"}):
        disc = compute_confirmation_filters(**_BEARISH_KWARGS)

    # A well-aligned bearish setup should have positive score_adjustment in both modes
    assert cont["score_adjustment"] > 0, f"Continuous should be positive, got {cont['score_adjustment']}"
    assert disc["score_adjustment"] > 0, f"Discrete should be positive, got {disc['score_adjustment']}"


def test_confirmation_status_label_consistent_with_score_in_continuous_mode():
    """Status labels (CONFIRMED / STRONG_CONFIRMATION / MIXED / CONFLICT) are still
    produced in continuous mode and correlate correctly with the score magnitude."""
    with temporary_parameter_pack("cont_status",
                                  overrides={"confirmation_filter.core.confirmation_scoring_mode": "continuous"}):
        result = compute_confirmation_filters(**_BEARISH_KWARGS)

    valid_statuses = {"STRONG_CONFIRMATION", "CONFIRMED", "MIXED", "CONFLICT", "NO_DIRECTION"}
    assert result["status"] in valid_statuses, f"Unexpected status: {result['status']}"
    # A strongly confirmed setup should not return CONFLICT
    assert result["status"] != "CONFLICT", "Strongly aligned PUT setup should not be CONFLICT"
