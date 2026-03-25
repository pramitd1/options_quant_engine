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


def test_direction_change_penalty_reduces_confirmation_score_on_reversal():
    with temporary_parameter_pack(
        "no_reversal_penalty",
        overrides={"confirmation_filter.core.direction_change_penalty": 0.0},
    ):
        baseline = compute_confirmation_filters(**_BEARISH_KWARGS, previous_direction="CALL")

    with temporary_parameter_pack(
        "with_reversal_penalty",
        overrides={"confirmation_filter.core.direction_change_penalty": 4.0},
    ):
        penalized = compute_confirmation_filters(**_BEARISH_KWARGS, previous_direction="CALL")

    assert penalized["score_adjustment"] == baseline["score_adjustment"] - 4.0
    assert penalized["breakdown"]["direction_change_penalty"] == -4.0
    assert "direction_change_penalty_applied" in penalized["reasons"]


def test_direction_change_penalty_is_clamped_to_bounded_range():
    with temporary_parameter_pack(
        "clamped_reversal_penalty",
        overrides={"confirmation_filter.core.direction_change_penalty": 99.0},
    ):
        penalized = compute_confirmation_filters(**_BEARISH_KWARGS, previous_direction="CALL")

    assert penalized["breakdown"]["direction_change_penalty"] == -6.0


def test_post_reversal_decay_applies_at_step_1():
    """After a direction flip (reversal_age=1), a decayed penalty reduces the score."""
    with temporary_parameter_pack(
        "no_decay",
        overrides={
            "confirmation_filter.core.direction_change_penalty": 4.0,
            "confirmation_filter.core.direction_change_decay_steps": 0,
        },
    ):
        no_decay = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=1)

    with temporary_parameter_pack(
        "with_decay",
        overrides={
            "confirmation_filter.core.direction_change_penalty": 4.0,
            "confirmation_filter.core.direction_change_decay_steps": 3,
            "confirmation_filter.core.direction_change_decay_factor": 0.5,
        },
    ):
        decayed = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=1)

    # decay at step-1 = 4.0 * 0.5^1 = 2.0
    assert decayed["breakdown"]["direction_change_decay_penalty"] == -2.0
    assert decayed["score_adjustment"] == no_decay["score_adjustment"] - 2.0
    assert "direction_change_decay_applied" in decayed["reasons"]


def test_post_reversal_decay_stops_after_decay_steps():
    """Decay penalty is not applied when reversal_age exceeds direction_change_decay_steps."""
    with temporary_parameter_pack(
        "beyond_window",
        overrides={
            "confirmation_filter.core.direction_change_penalty": 4.0,
            "confirmation_filter.core.direction_change_decay_steps": 2,
            "confirmation_filter.core.direction_change_decay_factor": 0.5,
        },
    ):
        beyond = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=3)

    assert beyond["breakdown"]["direction_change_decay_penalty"] == 0.0
    assert "direction_change_decay_applied" not in beyond["reasons"]


def test_decay_factor_clamped_between_0_and_1():
    """direction_change_decay_factor outside [0,1] is silently clamped."""
    with temporary_parameter_pack(
        "decay_clamp",
        overrides={
            "confirmation_filter.core.direction_change_penalty": 4.0,
            "confirmation_filter.core.direction_change_decay_steps": 2,
            "confirmation_filter.core.direction_change_decay_factor": 5.0,  # should clamp to 1.0
        },
    ):
        result = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=1)

    # decay_factor clamped to 1.0 → effective penalty = 4.0 * 1.0^1 = 4.0
    assert result["breakdown"]["direction_change_decay_penalty"] == -4.0


def test_reversal_veto_forces_mixed_on_reversal_snapshot():
    """When reversal_veto_steps > 0 and reversal_age=0 (the flip), status is forced to MIXED."""
    with temporary_parameter_pack(
        "with_veto",
        overrides={"confirmation_filter.core.reversal_veto_steps": 2},
    ):
        result = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=0)

    # Even though the baseline is STRONG_CONFIRMATION, the veto forces MIXED
    assert result["status"] == "MIXED"
    assert "reversal_grace_period_active" in result["reasons"]


def test_reversal_veto_forces_mixed_during_grace_period():
    """Veto applies to post-reversal snapshots within the grace window."""
    with temporary_parameter_pack(
        "with_veto",
        overrides={"confirmation_filter.core.reversal_veto_steps": 3},
    ):
        at_step_1 = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=1)
        at_step_2 = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=2)
        at_step_3 = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=3)

    assert at_step_1["status"] == "MIXED"
    assert at_step_2["status"] == "MIXED"
    assert at_step_3["status"] != "MIXED"  # beyond grace window
    assert "reversal_grace_period_active" in at_step_1["reasons"]
    assert "reversal_grace_period_active" in at_step_2["reasons"]


def test_reversal_veto_inactive_when_disabled():
    """With reversal_veto_steps=0, no veto is applied."""
    with temporary_parameter_pack(
        "veto_disabled",
        overrides={"confirmation_filter.core.reversal_veto_steps": 0},
    ):
        result = compute_confirmation_filters(**_BEARISH_KWARGS, reversal_age=0)

    # Should not be forced to MIXED
    assert result["status"] != "MIXED"
    assert "reversal_grace_period_active" not in result["reasons"]


def test_pcr_alignment_adds_confirmation_score_for_bearish_setup():
    result = compute_confirmation_filters(
        **_BEARISH_KWARGS,
        volume_pcr_atm=1.35,
    )
    assert result["breakdown"]["pcr_alignment"] > 0
    assert "pcr_confirms_direction" in result["reasons"]


def test_pcr_alignment_can_be_switched_off_via_runtime_flag():
    with temporary_parameter_pack(
        "disable_pcr_confirmation",
        overrides={"trade_strength.runtime_thresholds.use_pcr_in_confirmation": 0},
    ):
        result = compute_confirmation_filters(
            **_BEARISH_KWARGS,
            volume_pcr_atm=1.35,
        )

    assert result["breakdown"]["pcr_alignment"] == 0.0
