import json
import time
from pathlib import Path

from strategy import score_calibration as sc


def _write_isotonic_artifact(path: Path, mapping: dict[str, float]) -> None:
    payload = {
        "method": "isotonic",
        "n_bins": 10,
        "config": {"method": "isotonic", "n_bins": 10},
        "calibration_mapping": mapping,
        "bin_edges": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_segmented_artifact(path: Path, segments: dict[str, dict[str, float]]) -> None:
    payload = {
        "artifact_type": "segmented_score_calibrator",
        "version": 2,
        "selector_fields": ["direction", "gamma_regime", "vol_regime"],
        "default_segment": "default",
        "segments": {
            segment_key: {
                "method": "isotonic",
                "n_bins": 10,
                "config": {"method": "isotonic", "n_bins": 10},
                "calibration_mapping": mapping,
                "bin_edges": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
            for segment_key, mapping in segments.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _reset_runtime_calibrator_state() -> None:
    sc._global_calibrator = None
    sc._calibration_autoload_attempted = False
    sc._loaded_calibrator_path = None
    sc._loaded_calibrator_mtime = None


def test_runtime_calibrator_reloads_when_artifact_changes(tmp_path):
    calibrator_path = tmp_path / "runtime_score_calibrator.json"

    # Version 1: map score 55 (bin center 55) to 10.
    _write_isotonic_artifact(
        calibrator_path,
        {
            "5": 0.0,
            "15": 0.0,
            "25": 0.0,
            "35": 0.0,
            "45": 0.0,
            "55": 0.10,
            "65": 0.10,
            "75": 0.10,
            "85": 0.10,
            "95": 0.10,
        },
    )

    # Reset module globals to mimic runtime start.
    _reset_runtime_calibrator_state()

    score_v1 = sc.apply_score_calibration(55.0, calibrator_path=str(calibrator_path))
    assert score_v1 == 10

    # Version 2: emulate refresh overwriting same file with new mapping.
    time.sleep(0.02)
    _write_isotonic_artifact(
        calibrator_path,
        {
            "5": 0.0,
            "15": 0.0,
            "25": 0.0,
            "35": 0.0,
            "45": 0.0,
            "55": 0.90,
            "65": 0.90,
            "75": 0.90,
            "85": 0.90,
            "95": 0.90,
        },
    )

    score_v2 = sc.apply_score_calibration(55.0, calibrator_path=str(calibrator_path))
    assert score_v2 == 90


def test_segmented_calibrator_uses_most_specific_matching_segment(tmp_path):
    calibrator_path = tmp_path / "runtime_score_calibrator.json"
    _write_segmented_artifact(
        calibrator_path,
        {
            "default": {"55": 0.20},
            "direction=CALL": {"55": 0.60},
            "direction=CALL|gamma_regime=NEGATIVE_GAMMA": {"55": 0.85},
        },
    )

    _reset_runtime_calibrator_state()

    score = sc.apply_score_calibration(
        55.0,
        calibrator_path=str(calibrator_path),
        calibration_context={"direction": "CALL", "gamma_regime": "SHORT_GAMMA"},
    )
    metadata = sc.get_calibrator_runtime_metadata(
        str(calibrator_path),
        calibration_context={"direction": "CALL", "gamma_regime": "SHORT_GAMMA"},
    )

    assert score == 85
    assert metadata.get("selected_segment_key") == "direction=CALL|gamma_regime=NEGATIVE_GAMMA"


def test_segmented_calibrator_falls_back_to_default_when_no_segment_matches(tmp_path):
    calibrator_path = tmp_path / "runtime_score_calibrator.json"
    _write_segmented_artifact(
        calibrator_path,
        {
            "default": {"55": 0.25},
            "direction=CALL": {"55": 0.70},
        },
    )

    _reset_runtime_calibrator_state()

    score = sc.apply_score_calibration(
        55.0,
        calibrator_path=str(calibrator_path),
        calibration_context={"direction": "PUT", "gamma_regime": "NEGATIVE_GAMMA"},
    )
    metadata = sc.get_calibrator_runtime_metadata(
        str(calibrator_path),
        calibration_context={"direction": "PUT", "gamma_regime": "NEGATIVE_GAMMA"},
    )

    assert score == 25
    assert metadata.get("selected_segment_key") == "default"


def test_isotonic_calibrator_mapping_is_monotonic_after_fit():
    calibrator = sc.ScoreCalibrator(method="isotonic", n_bins=5)
    # Intentionally non-monotonic raw bucket hit rates.
    raw_scores = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    hit_flags = [0.1, 0.2, 0.9, 0.8, 0.2, 0.3, 0.6, 0.7, 0.9]
    calibrator.fit(raw_scores, hit_flags)

    mapping = calibrator.backend.calibration_mapping
    keys = sorted(mapping)
    for k0, k1 in zip(keys, keys[1:]):
        assert mapping[k1] >= mapping[k0]
