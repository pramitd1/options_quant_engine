from __future__ import annotations

import json

from config.policy_resolver import (
    evaluate_regime_pack_switch,
    get_active_parameter_pack,
    set_active_parameter_pack,
    suggest_regime_pack,
    temporary_parameter_pack,
)


def test_suggest_regime_pack_prefers_most_specific_then_priority(tmp_path):
    map_path = tmp_path / "regime_auto_pack_map.json"
    map_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "fallback_pack": "baseline_v1",
                "map": [
                    {
                        "gamma_regime": "*",
                        "vol_regime": "*",
                        "pack": "baseline_v1",
                        "priority": 0,
                    },
                    {
                        "gamma_regime": "NEGATIVE_GAMMA",
                        "vol_regime": "HIGH_VOL",
                        "pack": "macro_overlay_v1",
                        "priority": 0,
                    },
                    {
                        "gamma_regime": "NEGATIVE_GAMMA",
                        "vol_regime": "HIGH_VOL",
                        "global_risk_state": "RISK_OFF",
                        "pack": "overnight_focus_v1",
                        "priority": 5,
                    },
                ],
            }
        )
    )

    suggested = suggest_regime_pack(
        "NEGATIVE_GAMMA",
        "HIGH_VOL",
        global_risk_state="RISK_OFF",
        config_path=str(map_path),
    )

    assert suggested == "overnight_focus_v1"


def test_suggest_regime_pack_returns_none_when_disabled(tmp_path):
    map_path = tmp_path / "regime_auto_pack_map.json"
    map_path.write_text(json.dumps({"enabled": False, "map": []}))

    suggested = suggest_regime_pack(
        "NEGATIVE_GAMMA",
        "HIGH_VOL",
        config_path=str(map_path),
    )

    assert suggested is None


def test_suggest_regime_pack_returns_shadow_candidate_when_shadow_mode_enabled(tmp_path):
    map_path = tmp_path / "regime_auto_pack_map.json"
    map_path.write_text(
        json.dumps(
            {
                "enabled": False,
                "shadow_enabled": True,
                "fallback_pack": "baseline_v1",
                "map": [
                    {
                        "gamma_regime": "POSITIVE_GAMMA",
                        "vol_regime": "NORMAL_VOL",
                        "pack": "experimental_v1",
                        "priority": 10,
                    }
                ],
            }
        )
    )

    suggested = suggest_regime_pack(
        "POSITIVE_GAMMA",
        "NORMAL_VOL",
        config_path=str(map_path),
        evaluation_mode="shadow",
    )

    assert suggested == "experimental_v1"


def test_evaluate_regime_pack_switch_requires_consecutive_hits():
    state = None

    first = evaluate_regime_pack_switch(
        suggested_pack="macro_overlay_v1",
        current_pack="baseline_v1",
        regime_signature="NEGATIVE_GAMMA|HIGH_VOL",
        switch_state=state,
        required_consecutive=2,
        cooldown_seconds=0,
        min_dwell_seconds=0,
        now_ts=100.0,
    )
    assert first["apply"] is False
    assert first["reason"] == "insufficient_consecutive"

    second = evaluate_regime_pack_switch(
        suggested_pack="macro_overlay_v1",
        current_pack="baseline_v1",
        regime_signature="NEGATIVE_GAMMA|HIGH_VOL",
        switch_state=first["state"],
        required_consecutive=2,
        cooldown_seconds=0,
        min_dwell_seconds=0,
        now_ts=101.0,
    )
    assert second["apply"] is True
    assert second["reason"] == "switch_approved"


def test_evaluate_regime_pack_switch_respects_cooldown_and_dwell():
    applied = evaluate_regime_pack_switch(
        suggested_pack="macro_overlay_v1",
        current_pack="baseline_v1",
        regime_signature="NEGATIVE_GAMMA|HIGH_VOL",
        switch_state=None,
        required_consecutive=1,
        cooldown_seconds=60,
        min_dwell_seconds=300,
        now_ts=100.0,
    )
    assert applied["apply"] is True

    cooldown_blocked = evaluate_regime_pack_switch(
        suggested_pack="overnight_focus_v1",
        current_pack="macro_overlay_v1",
        regime_signature="ANY|ANY",
        switch_state=applied["state"],
        required_consecutive=1,
        cooldown_seconds=60,
        min_dwell_seconds=300,
        now_ts=120.0,
    )
    assert cooldown_blocked["apply"] is False
    assert cooldown_blocked["reason"] == "cooldown_active"

    dwell_blocked = evaluate_regime_pack_switch(
        suggested_pack="overnight_focus_v1",
        current_pack="macro_overlay_v1",
        regime_signature="ANY|ANY",
        switch_state=cooldown_blocked["state"],
        required_consecutive=1,
        cooldown_seconds=60,
        min_dwell_seconds=300,
        now_ts=200.0,
    )
    assert dwell_blocked["apply"] is False
    assert dwell_blocked["reason"] == "min_dwell_active"

    approved = evaluate_regime_pack_switch(
        suggested_pack="overnight_focus_v1",
        current_pack="macro_overlay_v1",
        regime_signature="ANY|ANY",
        switch_state=dwell_blocked["state"],
        required_consecutive=1,
        cooldown_seconds=60,
        min_dwell_seconds=300,
        now_ts=500.0,
    )
    assert approved["apply"] is True


def test_evaluate_regime_pack_switch_confidence_and_noop_guards():
    low_confidence = evaluate_regime_pack_switch(
        suggested_pack="macro_overlay_v1",
        current_pack="baseline_v1",
        regime_signature="NEGATIVE_GAMMA|HIGH_VOL",
        switch_state=None,
        required_consecutive=1,
        cooldown_seconds=0,
        min_dwell_seconds=0,
        regime_confidence=0.4,
        min_regime_confidence=0.7,
        now_ts=100.0,
    )
    assert low_confidence["apply"] is False
    assert low_confidence["reason"] == "confidence_below_floor"

    noop = evaluate_regime_pack_switch(
        suggested_pack="baseline_v1",
        current_pack="baseline_v1",
        regime_signature="NEUTRAL_GAMMA|NORMAL_VOL",
        switch_state=None,
        required_consecutive=1,
        cooldown_seconds=0,
        min_dwell_seconds=0,
        now_ts=100.0,
    )
    assert noop["apply"] is False
    assert noop["reason"] == "already_active"


def test_suggest_regime_pack_supports_pack_layers_payload(tmp_path):
    map_path = tmp_path / "regime_auto_pack_map.json"
    map_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "fallback_pack": "baseline_v1",
                "map": [
                    {
                        "gamma_regime": "NEGATIVE_GAMMA",
                        "vol_regime": "HIGH_VOL",
                        "pack_layers": ["baseline_v1", "macro_overlay_v1"],
                    }
                ],
            }
        )
    )

    suggested = suggest_regime_pack(
        "NEGATIVE_GAMMA",
        "HIGH_VOL",
        config_path=str(map_path),
    )
    assert suggested == "baseline_v1+macro_overlay_v1"


def test_set_active_parameter_pack_supports_overlay_layers():
    with temporary_parameter_pack("baseline_v1"):
        state = set_active_parameter_pack(["baseline_v1", "macro_overlay_v1"])
        assert state["name"] == "baseline_v1+macro_overlay_v1"
        assert state["layers"] == ["baseline_v1", "macro_overlay_v1"]
        assert (
            state["overrides"].get("macro_news.adjustment.lockdown_adjustment_score")
            == -14
        )
        # Should expose the same layer metadata through the read API.
        active = get_active_parameter_pack()
        assert active["layers"] == ["baseline_v1", "macro_overlay_v1"]
