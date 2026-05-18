from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from config.policy_resolver import ParameterPackResolutionError
from config.signal_policy import get_trade_runtime_thresholds
from research.signal_evaluation.dataset import write_signals_dataset
from tuning.experiments import run_parameter_experiment
from tuning.objectives import apply_selection_policy, compute_objective
from tuning.packs import list_parameter_packs, load_parameter_pack, resolve_parameter_pack
from tuning.promotion import evaluate_promotion, load_promotion_state
from tuning.registry import get_parameter_registry
from tuning.campaigns import default_group_tuning_plans, run_group_tuning_campaign
from tuning.runtime import temporary_parameter_pack
from tuning.search import (
    run_coordinate_descent_search,
    run_grid_search,
    run_latin_hypercube_search,
    run_random_search,
)


def _build_dataset_frame():
    return pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "signal_timestamp": "2026-01-02T09:20:00+05:30",
                "trade_strength": 58,
                "composite_signal_score": 68,
                "tradeability_score": 62,
                "hybrid_move_probability": 0.58,
                "option_efficiency_score": 65,
                "global_risk_score": 42,
                "overnight_hold_allowed": True,
                "correct_60m": 1,
                "mae_60m_bps": -35,
                "target_reachability_score": 72,
                "signal_regime": "DIRECTIONAL_BIAS",
            },
            {
                "signal_id": "sig-2",
                "signal_timestamp": "2026-01-03T09:20:00+05:30",
                "trade_strength": 43,
                "composite_signal_score": 54,
                "tradeability_score": 51,
                "hybrid_move_probability": 0.46,
                "option_efficiency_score": 48,
                "global_risk_score": 55,
                "overnight_hold_allowed": True,
                "correct_60m": 0,
                "mae_60m_bps": -60,
                "target_reachability_score": 49,
                "signal_regime": "BALANCED",
            },
            {
                "signal_id": "sig-3",
                "signal_timestamp": "2026-01-04T09:20:00+05:30",
                "trade_strength": 39,
                "composite_signal_score": 44,
                "tradeability_score": 35,
                "hybrid_move_probability": 0.32,
                "option_efficiency_score": 30,
                "global_risk_score": 88,
                "overnight_hold_allowed": False,
                "correct_60m": 0,
                "mae_60m_bps": -120,
                "target_reachability_score": 25,
                "signal_regime": "CONFLICTED",
            },
            {
                "signal_id": "sig-4",
                "signal_timestamp": "2026-01-05T09:20:00+05:30",
                "trade_strength": 81,
                "composite_signal_score": 84,
                "tradeability_score": 78,
                "hybrid_move_probability": 0.73,
                "option_efficiency_score": 82,
                "global_risk_score": 38,
                "overnight_hold_allowed": True,
                "correct_60m": 1,
                "mae_60m_bps": -20,
                "target_reachability_score": 88,
                "signal_regime": "EXPANSION_BIAS",
            },
        ]
    )


def test_parameter_registry_exposes_key_groups():
    registry = get_parameter_registry()

    assert "trade_strength.scoring.flow_call_bullish" in registry.keys()
    assert "confirmation_filter.core.flow_support" in registry.keys()
    assert "signal_engine.data_quality.invalid_spot_penalty" in registry.keys()
    assert "signal_engine.probability.vacuum_breakout_strength" in registry.keys()
    assert "signal_engine.trade_strength_continuous.hybrid_probability_floor" in registry.keys()
    assert "signal_engine.exit_timing.peak_alpha_minutes" in registry.keys()
    assert "signal_engine.consistency.default_trade_escalation_min_severity" in registry.keys()
    assert "tradable_data_layer.analytics.min_rows" in registry.keys()
    assert "option_chain_validation.provider_health.core_window_points" in registry.keys()
    assert "analytics.flow_imbalance.bullish_threshold" in registry.keys()
    assert "analytics.gamma_flip.neutral_band_pct" in registry.keys()
    assert "analytics.technical_analysis.sma_fast_window" in registry.keys()
    assert "analytics.mean_reversion.zscore_threshold" in registry.keys()
    assert "analytics.volume_pcr.bearish_threshold" in registry.keys()
    assert "signal_drift.monitor.recent_days" in registry.keys()
    assert "headline_rules.geopolitics.vol_weight" in registry.keys()
    assert "macro_news.adjustment.lockdown_adjustment_score" in registry.keys()
    assert "global_risk.core.risk_adjustment_extreme" in registry.keys()
    assert "strike_selection.core.strike_window_steps" in registry.keys()
    assert "large_move_probability.core.base_probability" in registry.keys()
    assert "event_windows.core.pre_event_warning_minutes" in registry.keys()
    assert "keyword_category.impact.geopolitics" in registry.keys()
    assert "evaluation_thresholds.selection.trade_strength_floor" in registry.keys()


def test_static_runtime_policy_keys_are_registered():
    root = Path(__file__).resolve().parents[1]
    scan_dirs = ("config", "data", "analytics", "engine", "features", "macro", "research")
    registry_keys = set(get_parameter_registry().keys())
    missing: list[str] = []

    for dirname in scan_dirs:
        for path in sorted((root / dirname).rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else func.id
                    if isinstance(func, ast.Name)
                    else None
                )
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                first_arg = node.args[0].value
                if not isinstance(first_arg, str):
                    continue

                if name in {"resolve_mapping", "resolve_dataclass_config"}:
                    if not any(
                        key == first_arg or key.startswith(f"{first_arg}.")
                        for key in registry_keys
                    ):
                        missing.append(f"{first_arg} ({path.relative_to(root)}:{node.lineno})")
                elif name == "get_parameter_value" and first_arg not in registry_keys:
                    missing.append(f"{first_arg} ({path.relative_to(root)}:{node.lineno})")

    assert missing == []


def test_parameter_packs_resolve_and_override_defaults():
    pack_names = list_parameter_packs()
    assert "baseline_v1" in pack_names
    assert load_parameter_pack("baseline_v1").name == "baseline_v1"

    resolved = resolve_parameter_pack("overnight_focus_v1")
    assert resolved.overrides["global_risk.core.overnight_gap_block_threshold"] == 64
    assert resolved.overrides["evaluation_thresholds.selection.require_overnight_hold_allowed"] is True


def test_runtime_pack_temporarily_overrides_thresholds():
    baseline = get_trade_runtime_thresholds()
    baseline_min_trade_strength = baseline["min_trade_strength"]

    with temporary_parameter_pack("experimental_v1"):
        experimental = get_trade_runtime_thresholds()
        assert experimental["min_trade_strength"] == 42

    restored = get_trade_runtime_thresholds()
    assert restored["min_trade_strength"] == baseline_min_trade_strength


def test_runtime_pack_context_restores_nested_overrides():
    baseline_pack = get_trade_runtime_thresholds()["min_trade_strength"]

    with temporary_parameter_pack("experimental_v1"):
        assert get_trade_runtime_thresholds()["min_trade_strength"] == 42

        with temporary_parameter_pack(
            "baseline_v1",
            overrides={"trade_strength.runtime_thresholds.min_trade_strength": 41},
        ):
            assert get_trade_runtime_thresholds()["min_trade_strength"] == 41

        assert get_trade_runtime_thresholds()["min_trade_strength"] == 42

    assert get_trade_runtime_thresholds()["min_trade_strength"] == baseline_pack


def test_strict_runtime_pack_rejects_unknown_pack(monkeypatch):
    monkeypatch.setenv("OQE_STRICT_PARAMETER_PACK", "1")

    with pytest.raises(ParameterPackResolutionError):
        with temporary_parameter_pack("missing_pack_for_strict_resolution_test"):
            pass


def test_objective_framework_uses_selection_thresholds():
    frame = _build_dataset_frame()
    selected = apply_selection_policy(
        frame,
        thresholds={
            "trade_strength_floor": 45,
            "composite_signal_score_floor": 55,
            "tradeability_score_floor": 50,
            "move_probability_floor": 0.40,
            "option_efficiency_score_floor": 35,
            "global_risk_score_cap": 85,
            "require_overnight_hold_allowed": False,
        },
    )
    assert list(selected["signal_id"]) == ["sig-1", "sig-4"]

    objective = compute_objective(
        frame,
        thresholds={
            "trade_strength_floor": 42,
            "composite_signal_score_floor": 52,
            "tradeability_score_floor": 50,
            "move_probability_floor": 0.40,
            "option_efficiency_score_floor": 35,
            "global_risk_score_cap": 85,
            "require_overnight_hold_allowed": False,
        },
        parameter_count=3,
    )
    assert "direction_hit_rate" in objective.metrics
    assert "stability_gap" in objective.safeguards


def test_experiment_runner_persists_results(tmp_path):
    dataset_path = tmp_path / "signals.csv"
    write_signals_dataset(_build_dataset_frame(), dataset_path)

    result = run_parameter_experiment(
        "experimental_v1",
        dataset_path=dataset_path,
        persist=False,
    )

    assert result.parameter_pack_name == "experimental_v1"
    assert result.sample_count == 4
    assert "metrics" in result.objective_metrics
    assert "evaluation_thresholds.selection.trade_strength_floor" in result.parameter_overrides


def test_search_strategies_return_reproducible_results(tmp_path):
    dataset_path = tmp_path / "signals.csv"
    write_signals_dataset(_build_dataset_frame(), dataset_path)

    grid_results = run_grid_search(
        "baseline_v1",
        grid={"trade_strength.direction_thresholds.min_score": [1.6, 1.75]},
        dataset_path=dataset_path,
        persist=False,
        selection_thresholds={"trade_strength_floor": 45},
    )
    assert len(grid_results) == 2

    random_results = run_random_search(
        "baseline_v1",
        parameter_keys=[
            "trade_strength.direction_thresholds.min_score",
            "trade_strength.direction_thresholds.min_margin",
            "macro_news.adjustment.lockdown_adjustment_score",
        ],
        dataset_path=dataset_path,
        iterations=2,
        seed=11,
        persist=False,
    )
    assert len(random_results) == 2
    assert all("objective_score" in row for row in random_results)

    lhs_results = run_latin_hypercube_search(
        "baseline_v1",
        parameter_keys=[
            "trade_strength.direction_thresholds.min_score",
            "trade_strength.direction_thresholds.min_margin",
            "global_risk.core.risk_adjustment_extreme",
        ],
        dataset_path=dataset_path,
        iterations=3,
        seed=13,
        persist=False,
    )
    assert len(lhs_results) == 3

    coord_results = run_coordinate_descent_search(
        "baseline_v1",
        parameter_keys=[
            "trade_strength.direction_thresholds.min_score",
            "trade_strength.direction_thresholds.min_margin",
        ],
        dataset_path=dataset_path,
        initial_overrides={"trade_strength.direction_thresholds.min_score": 1.75},
        passes=1,
        persist=False,
    )
    assert len(coord_results) >= 1
    assert all("parameter_overrides" in row for row in coord_results)


def test_group_tuning_campaign_builds_registry_driven_plans(tmp_path):
    plans = default_group_tuning_plans(allow_live_unsafe=True)
    groups = {plan.group for plan in plans}
    assert "trade_strength" in groups
    assert "strike_selection" in groups
    assert "large_move_probability" in groups
    assert "keyword_category" in groups

    dataset_path = tmp_path / "signals.csv"
    write_signals_dataset(_build_dataset_frame(), dataset_path)

    campaign = run_group_tuning_campaign(
        "baseline_v1",
        dataset_path=dataset_path,
        groups=["trade_strength", "option_efficiency"],
        allow_live_unsafe=False,
        walk_forward_config={
            "split_type": "rolling",
            "train_window_days": 2,
            "validation_window_days": 1,
            "minimum_train_rows": 1,
            "minimum_validation_rows": 1,
        },
        persist=False,
    )
    assert len(campaign["steps"]) == 2
    assert "final_overrides" in campaign


def test_promotion_workflow_requires_more_than_single_win(tmp_path):
    state = load_promotion_state(tmp_path / "promotion_state.json")
    assert state["live"] == "baseline_v1"

    baseline_result = {
        "parameter_pack_name": "baseline_v1",
        "sample_count": 40,
        "objective_score": 0.45,
        "objective_metrics": {
            "metrics": {"signal_frequency": 0.20},
            "safeguards": {"minimum_sample_ok": True},
        },
    }
    candidate_result = {
        "parameter_pack_name": "candidate_v1",
        "sample_count": 40,
        "objective_score": 0.49,
        "objective_metrics": {
            "metrics": {"signal_frequency": 0.18},
            "safeguards": {"minimum_sample_ok": True, "stability_gap": 0.03},
        },
    }
    decision = evaluate_promotion(
        baseline_result=baseline_result,
        candidate_result=candidate_result,
    )
    assert decision.approved is True
    assert decision.reason == "candidate_meets_promotion_thresholds"
