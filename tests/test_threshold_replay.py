from __future__ import annotations

import pandas as pd

from research.signal_evaluation.threshold_replay import (
    build_threshold_replay_summary,
    run_regime_threshold_replay,
    run_threshold_replay,
    run_walk_forward_threshold_validation,
)


def _replay_frame() -> pd.DataFrame:
    rows = []
    for idx in range(80):
        high_score = idx >= 32
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (pd.Timestamp("2026-03-01 09:20:00+05:30") + pd.Timedelta(days=idx)).isoformat(),
                "direction": "CALL" if idx % 2 == 0 else "PUT",
                "composite_signal_score": 82.0 if high_score else 54.0,
                "tradeability_score": 78.0 if high_score else 48.0,
                "move_probability": 0.72 if high_score else 0.48,
                "macro_regime": "RISK_ON" if idx % 3 else "RISK_OFF",
                "correct_60m": 1.0 if high_score or idx % 5 == 0 else 0.0,
                "signed_return_60m_bps": 24.0 if high_score else -8.0,
                "calibration_label": 1.0 if high_score or idx % 5 == 0 else 0.0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 24.0 if high_score else -8.0,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_threshold_replay_ranks_candidate_thresholds_with_holdout_metrics():
    replay = run_threshold_replay(
        _replay_frame(),
        threshold_grid={"composite_signal_score": [50.0, 70.0, 80.0]},
        min_label_sample=10,
        strong_label_sample=20,
        top_n=5,
    )

    assert not replay.empty
    best = replay.iloc[0].to_dict()
    assert best["threshold_field"] == "composite_signal_score"
    assert best["threshold_value"] in {70.0, 80.0}
    assert bool(best["is_advisory_candidate"]) is True
    assert best["sample_quality"] == "RELIABLE"
    assert best["holdout_label_count_60m"] > 0
    assert best["stability_status"] in {"STABLE", "INSUFFICIENT_HOLDOUT"}


def test_regime_threshold_replay_adds_regime_context():
    replay = run_regime_threshold_replay(
        _replay_frame(),
        regime_fields=("macro_regime",),
        threshold_grid={"composite_signal_score": [70.0]},
        min_label_sample=5,
        strong_label_sample=10,
        top_n=5,
    )

    assert not replay.empty
    assert set(replay["regime_field"]) == {"macro_regime"}
    assert "regime_value" in replay.columns


def test_threshold_replay_summary_is_json_friendly():
    summary = build_threshold_replay_summary(
        _replay_frame(),
        min_label_sample=10,
        strong_label_sample=20,
        top_n=3,
    )

    assert summary["config"]["min_label_sample"] == 10
    assert summary["threshold_replay_candidates"]
    assert summary["regime_threshold_replay_candidates"]
    assert summary["walk_forward_validation"]["summary"]["split_count"] >= 1


def test_walk_forward_validation_selects_on_train_and_scores_holdout():
    validation = run_walk_forward_threshold_validation(
        _replay_frame(),
        threshold_grid={"composite_signal_score": [50.0, 70.0, 80.0]},
        train_window_days=40,
        holdout_window_days=20,
        step_days=20,
        min_train_labels=10,
        min_holdout_labels=5,
        strong_label_sample=20,
    )

    summary = validation["summary"]
    assert summary["split_count"] >= 2
    assert summary["evaluated_split_count"] >= 1
    assert summary["robustness_status"] in {"ROBUST", "MIXED", "UNSTABLE", "INSUFFICIENT_HOLDOUT"}
    first = validation["splits"][0]
    assert first["threshold_field"] in {"composite_signal_score", "ALL_SIGNALS"}
    assert first["holdout_label_count_60m"] >= 0
