from __future__ import annotations

import pandas as pd

from research.signal_evaluation.dataset import write_signals_dataset
from tuning.experiments import run_parameter_experiment
from tuning.promotion import evaluate_promotion
from tuning.regimes import label_validation_regimes
from tuning.validation import compare_validation_results, run_walk_forward_validation
from tuning.walk_forward import build_walk_forward_splits


def _validation_frame():
    rows = []
    for idx in range(12):
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (pd.Timestamp("2026-01-01", tz="Asia/Kolkata") + pd.Timedelta(days=idx)).isoformat(),
                "trade_strength": 42 + idx * 4,
                "composite_signal_score": 50 + idx * 3,
                "tradeability_score": 45 + idx * 2,
                "hybrid_move_probability": min(0.35 + idx * 0.04, 0.9),
                "option_efficiency_score": 35 + idx * 3,
                "global_risk_score": 25 + (idx % 4) * 15,
                "overnight_hold_allowed": idx % 3 != 0,
                "correct_60m": 1 if idx % 2 == 0 else 0,
                "mae_60m_bps": -20 - idx * 3,
                "signed_return_60m_bps": 18 + idx * 4,
                "signed_return_session_close_bps": 10 + idx * 3,
                "target_reachability_score": 48 + idx * 3,
                "signal_regime": "EXPANSION_BIAS" if idx % 4 == 0 else "DIRECTIONAL_BIAS",
                "volatility_regime": ["VOL_SUPPRESSION", "NORMAL_VOL", "VOL_EXPANSION"][idx % 3],
                "gamma_regime": ["LONG_GAMMA_ZONE", "SHORT_GAMMA_ZONE", "NEUTRAL_GAMMA"][idx % 3],
                "macro_regime": ["MACRO_NEUTRAL", "RISK_OFF", "RISK_ON", "EVENT_LOCKDOWN"][idx % 4],
                "global_risk_state": ["GLOBAL_NEUTRAL", "RISK_OFF", "RISK_ON", "VOL_SHOCK"][idx % 4],
                "squeeze_risk_state": [
                    "LOW_ACCELERATION_RISK",
                    "MODERATE_ACCELERATION_RISK",
                    "HIGH_ACCELERATION_RISK",
                    "EXTREME_ACCELERATION_RISK",
                ][idx % 4],
                "dealer_flow_state": [
                    "HEDGING_NEUTRAL",
                    "UPSIDE_HEDGING_ACCELERATION",
                    "DOWNSIDE_HEDGING_ACCELERATION",
                    "PINNING_DOMINANT",
                ][idx % 4],
                "macro_event_risk_score": [15, 38, 52, 78][idx % 4],
                "trade_status": "TRADE",
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_split_engine_is_deterministic_and_leak_free():
    frame = _validation_frame()
    splits = build_walk_forward_splits(
        frame,
        split_type="rolling",
        train_window_days=4,
        validation_window_days=2,
        step_size_days=2,
        minimum_train_rows=4,
        minimum_validation_rows=2,
    )

    assert len(splits) == 4
    assert splits[0].split_id == "rolling_000"
    assert pd.Timestamp(splits[0].train_end) < pd.Timestamp(splits[0].validation_start)


def test_regime_labeling_assigns_expected_buckets():
    frame = _validation_frame().iloc[[1, 3]].copy()
    labeled = label_validation_regimes(frame)

    assert labeled.iloc[0]["vol_regime_bucket"] == "NORMAL_VOL"
    assert labeled.iloc[0]["gamma_regime_bucket"] == "SHORT_GAMMA"
    assert labeled.iloc[0]["macro_regime_bucket"] == "RISK_OFF"
    assert labeled.iloc[1]["event_risk_bucket"] == "HIGH_EVENT_RISK"
    assert labeled.iloc[1]["overnight_bucket"] in {"OVERNIGHT_ALLOWED", "OVERNIGHT_BLOCKED"}


def test_walk_forward_validation_returns_split_regime_and_robustness_outputs():
    frame = _validation_frame()
    result = run_walk_forward_validation(
        frame,
        selection_thresholds={
            "trade_strength_floor": 40,
            "composite_signal_score_floor": 50,
            "tradeability_score_floor": 45,
            "move_probability_floor": 0.35,
            "option_efficiency_score_floor": 35,
            "global_risk_score_cap": 85,
            "require_overnight_hold_allowed": False,
        },
        walk_forward_config={
            "split_type": "rolling",
            "train_window_days": 4,
            "validation_window_days": 2,
            "step_size_days": 2,
            "minimum_train_rows": 4,
            "minimum_validation_rows": 2,
        },
        parameter_count=3,
        minimum_regime_sample_count=1,
    )

    assert result["validation_type"] == "walk_forward_regime_aware"
    assert len(result["split_results"]) == 4
    assert "aggregate_out_of_sample_metrics" in result
    assert "vol_regime_bucket" in result["regime_summary"]
    assert "robustness_score" in result["robustness_metrics"]


def test_experiment_runner_integrates_walk_forward_and_comparison(tmp_path):
    dataset_path = tmp_path / "signals.csv"
    write_signals_dataset(_validation_frame(), dataset_path)

    result = run_parameter_experiment(
        "experimental_v1",
        dataset_path=dataset_path,
        walk_forward_config={
            "split_type": "rolling",
            "train_window_days": 4,
            "validation_window_days": 2,
            "step_size_days": 2,
            "minimum_train_rows": 4,
            "minimum_validation_rows": 2,
        },
        comparison_baseline_pack="baseline_v1",
        persist=False,
    )

    assert result.validation_results["validation_type"] == "walk_forward_regime_aware"
    assert "robustness_score" in result.robustness_metrics
    assert result.comparison_summary["baseline_pack_name"] == "baseline_v1"
    assert "aggregate_comparison" in result.comparison_summary


def test_compare_validation_results_highlights_regime_deltas():
    baseline = {
        "aggregate_out_of_sample_score": 0.12,
        "aggregate_out_of_sample_metrics": {"direction_hit_rate": 0.52},
        "robustness_metrics": {"robustness_score": 0.61},
        "split_results": [{"split_id": "rolling_000", "validation_objective_score": 0.11}],
        "regime_summary": {
            "gamma_regime_bucket": [
                {"regime_label": "SHORT_GAMMA", "sample_count": 10, "metrics": {"direction_hit_rate": 0.55}}
            ]
        },
    }
    candidate = {
        "aggregate_out_of_sample_score": 0.18,
        "aggregate_out_of_sample_metrics": {"direction_hit_rate": 0.58},
        "robustness_metrics": {"robustness_score": 0.68},
        "split_results": [{"split_id": "rolling_000", "validation_objective_score": 0.17}],
        "regime_summary": {
            "gamma_regime_bucket": [
                {"regime_label": "SHORT_GAMMA", "sample_count": 10, "metrics": {"direction_hit_rate": 0.63}}
            ]
        },
    }

    comparison = compare_validation_results(
        baseline,
        candidate,
        baseline_pack_name="baseline_v1",
        candidate_pack_name="candidate_v1",
    )

    assert comparison["aggregate_comparison"]["out_of_sample_score_delta"] > 0
    assert comparison["robustness_comparison"]["robustness_score_delta"] > 0
    assert comparison["regime_comparison"]["gamma_regime_bucket"][0]["direction_hit_rate_delta"] > 0


def test_promotion_uses_validation_hooks():
    baseline_result = {
        "parameter_pack_name": "baseline_v1",
        "sample_count": 60,
        "objective_score": 0.40,
        "objective_metrics": {
            "metrics": {"signal_frequency": 0.20},
            "safeguards": {"minimum_sample_ok": True, "stability_gap": 0.03},
        },
        "validation_results": {"aggregate_out_of_sample_score": 0.19},
        "robustness_metrics": {"robustness_score": 0.60},
    }
    candidate_result = {
        "parameter_pack_name": "candidate_v1",
        "sample_count": 60,
        "objective_score": 0.46,
        "objective_metrics": {
            "metrics": {"signal_frequency": 0.18},
            "safeguards": {"minimum_sample_ok": True, "stability_gap": 0.03},
        },
        "validation_results": {"aggregate_out_of_sample_score": 0.17},
        "robustness_metrics": {"robustness_score": 0.28},
    }

    decision = evaluate_promotion(
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        minimum_robustness_score=0.35,
    )

    assert decision.approved is False
    assert decision.reason in {
        "candidate_out_of_sample_improvement_insufficient",
        "candidate_robustness_too_low",
    }
