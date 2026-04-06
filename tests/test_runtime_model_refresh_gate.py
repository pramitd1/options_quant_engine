from __future__ import annotations

from scripts.ops.run_runtime_model_refresh import _evaluate_gates


def test_refresh_gate_blocks_on_bad_singleton_vol_segment_gap(tmp_path):
    gate = _evaluate_gates(
        decay_payload={
            "regime_fits": [
                {"fit_mse": 0.10},
                {"fit_mse": 0.11},
                {"fit_mse": 0.12},
            ]
        },
        calibrator_payload={
            "overall_calibration_gap": -0.08,
            "calibration_drift": {
                "calibration_gap_abs_delta": 0.04,
                "brier_delta": 0.01,
            },
            "segmentation": {
                "trained_segments": [
                    {"segment_key": "direction=CALL", "overall_calibration_gap": -0.04},
                    {"segment_key": "direction=PUT", "overall_calibration_gap": -0.05},
                    {"segment_key": "gamma_regime=NEGATIVE_GAMMA", "overall_calibration_gap": -0.10},
                    {"segment_key": "gamma_regime=NEUTRAL_GAMMA", "overall_calibration_gap": -0.11},
                    {"segment_key": "gamma_regime=POSITIVE_GAMMA", "overall_calibration_gap": -0.12},
                    {"segment_key": "vol_regime=NORMAL_VOL", "overall_calibration_gap": -0.31},
                    {"segment_key": "vol_regime=VOL_EXPANSION", "overall_calibration_gap": -0.14},
                ]
            },
        },
        calibrator_model_path=tmp_path / "runtime_score_calibrator.json",
        max_decay_fit_mse=0.45,
        max_abs_calibration_gap=0.10,
        max_abs_direction_segment_gap=0.16,
        max_abs_gamma_regime_segment_gap=0.22,
        max_abs_vol_regime_segment_gap=0.25,
        min_direction_segments=2,
        min_gamma_regime_segments=3,
        min_vol_regime_segments=2,
        max_calibration_gap_abs_delta=0.12,
        max_brier_delta=0.03,
    )

    assert gate["gate_passed"] is False
    assert gate["checks"]["vol_regime_segment_gap_ok"] is False
    assert gate["metrics"]["worst_vol_regime_abs_gap"] == 0.31


def test_refresh_gate_passes_when_singleton_segments_are_covered_and_bounded(tmp_path):
    calibrator_path = tmp_path / "runtime_score_calibrator.json"
    calibrator_path.write_text("{}", encoding="utf-8")

    gate = _evaluate_gates(
        decay_payload={
            "regime_fits": [
                {"fit_mse": 0.10},
                {"fit_mse": 0.11},
                {"fit_mse": 0.12},
            ]
        },
        calibrator_payload={
            "overall_calibration_gap": -0.08,
            "calibration_drift": {
                "calibration_gap_abs_delta": 0.04,
                "brier_delta": 0.01,
            },
            "segmentation": {
                "trained_segments": [
                    {"segment_key": "direction=CALL", "overall_calibration_gap": -0.04},
                    {"segment_key": "direction=PUT", "overall_calibration_gap": -0.05},
                    {"segment_key": "gamma_regime=NEGATIVE_GAMMA", "overall_calibration_gap": -0.10},
                    {"segment_key": "gamma_regime=NEUTRAL_GAMMA", "overall_calibration_gap": -0.11},
                    {"segment_key": "gamma_regime=POSITIVE_GAMMA", "overall_calibration_gap": -0.12},
                    {"segment_key": "vol_regime=NORMAL_VOL", "overall_calibration_gap": -0.20},
                    {"segment_key": "vol_regime=VOL_EXPANSION", "overall_calibration_gap": -0.14},
                ]
            },
        },
        calibrator_model_path=calibrator_path,
        max_decay_fit_mse=0.45,
        max_abs_calibration_gap=0.10,
        max_abs_direction_segment_gap=0.16,
        max_abs_gamma_regime_segment_gap=0.22,
        max_abs_vol_regime_segment_gap=0.25,
        min_direction_segments=2,
        min_gamma_regime_segments=3,
        min_vol_regime_segments=2,
        max_calibration_gap_abs_delta=0.12,
        max_brier_delta=0.03,
    )

    assert gate["gate_passed"] is True
    assert gate["checks"]["direction_segment_gap_ok"] is True
    assert gate["checks"]["gamma_regime_segment_gap_ok"] is True
    assert gate["checks"]["vol_regime_segment_gap_ok"] is True


def test_refresh_gate_blocks_when_calibration_drift_exceeds_thresholds(tmp_path):
    calibrator_path = tmp_path / "runtime_score_calibrator.json"
    calibrator_path.write_text("{}", encoding="utf-8")

    gate = _evaluate_gates(
        decay_payload={
            "regime_fits": [
                {"fit_mse": 0.10},
                {"fit_mse": 0.11},
                {"fit_mse": 0.12},
            ]
        },
        calibrator_payload={
            "overall_calibration_gap": -0.08,
            "calibration_drift": {
                "calibration_gap_abs_delta": 0.20,
                "brier_delta": 0.05,
            },
            "segmentation": {
                "trained_segments": [
                    {"segment_key": "direction=CALL", "overall_calibration_gap": -0.04},
                    {"segment_key": "direction=PUT", "overall_calibration_gap": -0.05},
                    {"segment_key": "gamma_regime=NEGATIVE_GAMMA", "overall_calibration_gap": -0.10},
                    {"segment_key": "gamma_regime=NEUTRAL_GAMMA", "overall_calibration_gap": -0.11},
                    {"segment_key": "gamma_regime=POSITIVE_GAMMA", "overall_calibration_gap": -0.12},
                    {"segment_key": "vol_regime=NORMAL_VOL", "overall_calibration_gap": -0.20},
                    {"segment_key": "vol_regime=VOL_EXPANSION", "overall_calibration_gap": -0.14},
                ]
            },
        },
        calibrator_model_path=calibrator_path,
        max_decay_fit_mse=0.45,
        max_abs_calibration_gap=0.10,
        max_abs_direction_segment_gap=0.16,
        max_abs_gamma_regime_segment_gap=0.22,
        max_abs_vol_regime_segment_gap=0.25,
        min_direction_segments=2,
        min_gamma_regime_segments=3,
        min_vol_regime_segments=2,
        max_calibration_gap_abs_delta=0.12,
        max_brier_delta=0.03,
    )

    assert gate["gate_passed"] is False
    assert gate["checks"]["calibration_gap_drift_ok"] is False
    assert gate["checks"]["calibration_brier_drift_ok"] is False