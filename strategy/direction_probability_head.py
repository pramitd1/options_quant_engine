"""
Calibrated probabilistic direction head.

This module estimates P(up | X) from regime, flow, and execution-quality
features, then optionally applies a calibration map. It also produces an
uncertainty score used by runtime decision logic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from strategy.score_calibration import ScoreCalibrator


_DEFAULT_CALIBRATOR_PATH = Path("models_store") / "direction_probability_calibrator.json"
_CALIBRATOR_CACHE: dict[str, tuple[float, ScoreCalibrator]] = {}
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _as_upper(value: Any) -> str:
    return str(value or "").upper().strip()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_entropy(probability: float) -> float:
    p = _clip(float(probability), 1e-6, 1.0 - 1e-6)
    return (-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))) / math.log(2.0)


def _provider_status_score(status: str) -> float:
    mapping = {
        "GOOD": 0.0,
        "CAUTION": 0.5,
        "WEAK": 1.0,
    }
    return mapping.get(_as_upper(status), 0.5)


def _directional_signal_bias(
    *,
    final_flow_signal: Any,
    spot_vs_flip: Any,
    hedging_bias: Any,
    gamma_event: Any,
    gamma_regime: Any,
    oi_velocity_score: Any,
    rr_value: Any,
    rr_momentum: Any,
    volume_pcr_atm: Any,
    gamma_flip_drift: Any,
) -> float:
    score = 0.0

    flow = _as_upper(final_flow_signal)
    if flow == "BULLISH_FLOW":
        score += 1.30
    elif flow == "BEARISH_FLOW":
        score -= 1.30

    flip = _as_upper(spot_vs_flip)
    if flip == "ABOVE_FLIP":
        score += 0.85
    elif flip == "BELOW_FLIP":
        score -= 0.85

    hedge = _as_upper(hedging_bias)
    if hedge == "UPSIDE_ACCELERATION":
        score += 1.00
    elif hedge == "DOWNSIDE_ACCELERATION":
        score -= 1.00

    if _as_upper(gamma_event) == "GAMMA_SQUEEZE":
        if flip == "ABOVE_FLIP":
            score += 0.35
        elif flip == "BELOW_FLIP":
            score -= 0.35

    regime = _as_upper(gamma_regime)
    if regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}:
        score += 0.15 if flip == "ABOVE_FLIP" else -0.15 if flip == "BELOW_FLIP" else 0.0

    oi_velocity = _safe_float(oi_velocity_score, None)
    if oi_velocity is not None:
        score += _clip(oi_velocity, -0.7, 0.7) * 0.55

    rr = _safe_float(rr_value, None)
    if rr is not None:
        # Positive RR implies put skew dominance (bearish), negative RR bullish.
        score += _clip(-rr / 3.0, -0.6, 0.6)

    rr_m = _as_upper(rr_momentum)
    if rr_m == "RISING_PUT_SKEW":
        score -= 0.20
    elif rr_m == "FALLING_PUT_SKEW":
        score += 0.20

    pcr = _safe_float(volume_pcr_atm, None)
    if pcr is not None:
        score += _clip((1.0 - pcr) / 0.6, -0.35, 0.35)

    flip_drift = None
    if isinstance(gamma_flip_drift, dict):
        flip_drift = _safe_float(gamma_flip_drift.get("drift"), None)
    if flip_drift is not None:
        score += _clip(flip_drift / 250.0, -0.35, 0.35)

    return float(score)


def _microstructure_friction(
    *,
    provider_health_summary: Any,
    provider_health_blocking_status: Any,
    core_effective_priced_ratio: Any,
    core_one_sided_quote_ratio: Any,
    core_quote_integrity_health: Any,
) -> float:
    provider_component = _provider_status_score(_as_upper(provider_health_summary))
    blocking_component = 1.0 if _as_upper(provider_health_blocking_status) == "BLOCK" else 0.0

    quote_integrity = _as_upper(core_quote_integrity_health)
    quote_component = 1.0 if quote_integrity == "WEAK" else 0.5 if quote_integrity == "CAUTION" else 0.0

    priced_ratio = _safe_float(core_effective_priced_ratio, None)
    priced_component = 0.5
    if priced_ratio is not None:
        priced_component = _clip((0.60 - priced_ratio) / 0.60, 0.0, 1.0)

    one_sided_ratio = _safe_float(core_one_sided_quote_ratio, None)
    one_sided_component = 0.0 if one_sided_ratio is None else _clip(one_sided_ratio / 0.55, 0.0, 1.0)

    friction = (
        0.30 * provider_component
        + 0.30 * blocking_component
        + 0.20 * quote_component
        + 0.10 * priced_component
        + 0.10 * one_sided_component
    )
    return _clip(friction, 0.0, 1.0)


def _load_calibrator(calibrator_path: str | None) -> ScoreCalibrator | None:
    path = Path(calibrator_path or _DEFAULT_CALIBRATOR_PATH)
    if not path.is_absolute():
        # Resolve relative to repository root (stable across launch contexts),
        # not process CWD which depends on how the app was started.
        path = _REPO_ROOT / path
    if not path.exists():
        return None

    key = str(path.resolve())
    mtime = float(path.stat().st_mtime)
    cached = _CALIBRATOR_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        calibrator = ScoreCalibrator.load_from_file(str(path))
    except Exception:
        return None

    _CALIBRATOR_CACHE[key] = (mtime, calibrator)
    return calibrator


def compute_direction_probability_head(
    *,
    final_flow_signal: Any,
    spot_vs_flip: Any,
    hedging_bias: Any,
    gamma_event: Any,
    gamma_regime: Any,
    oi_velocity_score: Any = None,
    rr_value: Any = None,
    rr_momentum: Any = None,
    volume_pcr_atm: Any = None,
    gamma_flip_drift: Any = None,
    hybrid_move_probability: Any = None,
    vote_bull_probability: Any = 0.5,
    provider_health_summary: Any = None,
    provider_health_blocking_status: Any = None,
    core_effective_priced_ratio: Any = None,
    core_one_sided_quote_ratio: Any = None,
    core_quote_integrity_health: Any = None,
    calibrator_path: str | None = None,
    apply_calibration: bool = True,
) -> dict[str, Any]:
    directional_bias = _directional_signal_bias(
        final_flow_signal=final_flow_signal,
        spot_vs_flip=spot_vs_flip,
        hedging_bias=hedging_bias,
        gamma_event=gamma_event,
        gamma_regime=gamma_regime,
        oi_velocity_score=oi_velocity_score,
        rr_value=rr_value,
        rr_momentum=rr_momentum,
        volume_pcr_atm=volume_pcr_atm,
        gamma_flip_drift=gamma_flip_drift,
    )
    friction = _microstructure_friction(
        provider_health_summary=provider_health_summary,
        provider_health_blocking_status=provider_health_blocking_status,
        core_effective_priced_ratio=core_effective_priced_ratio,
        core_one_sided_quote_ratio=core_one_sided_quote_ratio,
        core_quote_integrity_health=core_quote_integrity_health,
    )

    move_probability = _safe_float(hybrid_move_probability, 0.5)
    vote_up = _clip(_safe_float(vote_bull_probability, 0.5) or 0.5, 0.0, 1.0)

    logit = (
        -0.05
        + 0.90 * directional_bias
        + 0.80 * (vote_up - 0.5)
        + 0.55 * ((move_probability or 0.5) - 0.5)
    )
    # Microstructure friction attenuates effective directional edge.
    logit *= (1.0 - 0.60 * friction)

    probability_up_raw = _sigmoid(logit)
    probability_up = probability_up_raw
    calibration_applied = False
    calibrator_loaded = False

    calibrator = _load_calibrator(calibrator_path)
    if apply_calibration and calibrator is not None:
        calibrator_loaded = True
        try:
            calibrated_score = calibrator.calibrate(float(probability_up_raw * 100.0))
            probability_up = _clip(float(calibrated_score) / 100.0, 0.0, 1.0)
            calibration_applied = True
        except Exception:
            probability_up = probability_up_raw

    disagreement = abs(probability_up - vote_up)
    entropy = _binary_entropy(probability_up)
    # Weights are normalised to sum exactly to 1.0:
    #   entropy      → 0.4783  (≈ 55/115)
    #   disagreement → 0.2609  (≈ 30/115)
    #   friction     → 0.2609  (≈ 30/115)
    # This prevents the composite from saturating at 1.0 and masking
    # individual component contributions in the output diagnostics.
    uncertainty = _clip(0.4783 * entropy + 0.2609 * disagreement + 0.2609 * friction, 0.0, 1.0)

    return {
        "probability_up_raw": round(float(probability_up_raw), 6),
        "probability_up": round(float(probability_up), 6),
        "probability_down": round(float(1.0 - probability_up), 6),
        "logit": round(float(logit), 6),
        "directional_bias_score": round(float(directional_bias), 6),
        "microstructure_friction_score": round(float(friction), 6),
        "disagreement_with_vote": round(float(disagreement), 6),
        "uncertainty": round(float(uncertainty), 6),
        "confidence": round(float(1.0 - uncertainty), 6),
        "calibrator_loaded": bool(calibrator_loaded),
        "calibration_applied": bool(calibration_applied),
    }
