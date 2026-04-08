"""
Expanded Feature Builder for ML MovePredictor
===============================================
Computes 33 features from the signal evaluation dataset using topological
ordering to eliminate circular dependencies.

DEPENDENCY TIERS (topological sort):
  Tier 0: Raw inputs — spot, OHLCV, prev_close (no computation)
  Tier 1: Market analytics — gamma, flow, vol, hedging, dealer (from option chain)
  Tier 2: Probability sub-features — flip distance, vacuum strength, IV percentile
  Tier 3: Global market — oil, vix, sp500, usdinr, etc. (independent)
  Tier 4: Macro events — event window, risk score (independent)
  Tier 5: Derived composites — intraday range, hist vol, moneyness

All 33 features are available BEFORE probability is computed and BEFORE
direction is decided, so there is zero circularity.

The model's prediction (move probability) then feeds into trade_strength
and confirmation_filters in the live pipeline.

TRAIN/SERVE ALIGNMENT:
  The following features are intentionally zeroed out during both training
  and inference because they are NOT available at probability computation
  time in the live engine:
    - hist_vol_5d / hist_vol_20d: requires trailing closes (backtest-only enrichment)
    - confirmation_status: computed AFTER probability (circular if used)
    - moneyness_pct: requires strike selection (happens after probability)
  Using defaults for these ensures the model never learns to depend on
  signals that won't be available at inference time.
"""
from __future__ import annotations

import numpy as np

# ── Feature names — the canonical ordering ──────────────────────────
FEATURE_NAMES = [
    # Tier 1: Market analytics (categorical → numeric)
    "gamma_regime_numeric",       # SHORT_GAMMA=1, NEUTRAL=0, LONG_GAMMA=-0.5
    "flow_signal_numeric",        # BULLISH=1, BEARISH=-1, NEUTRAL=0
    "vol_regime_numeric",         # LOW_VOL=0, NORMAL=1, VOL_EXPANSION=2
    "hedging_bias_numeric",       # UPSIDE_ACCEL=1, DOWNSIDE_ACCEL=-1, else 0
    "spot_vs_flip_numeric",       # ABOVE=1, BELOW=-1, AT=0
    "vacuum_state_numeric",       # BREAKOUT_ZONE=1, NORMAL=0
    "atm_iv_scaled",              # atm_iv / 100

    # Tier 2: Probability sub-features (continuous)
    "gamma_flip_distance_pct",    # |spot - flip| / spot * 100
    "vacuum_strength",            # 0-1 from vacuum zones
    "hedging_flow_ratio",         # normalized hedging flow
    "smart_money_flow_score",     # normalized smart money
    "atm_iv_percentile",          # 0-1 IV percentile
    "intraday_range_pct",         # today range / lookback avg range

    # Tier 0 + 5: Spot-derived features
    "lookback_avg_range_pct",     # trailing 10d avg range %
    "gap_pct",                    # (open - prev_close) / prev_close * 100
    "close_vs_prev_close_pct",    # (close - prev_close) / prev_close * 100
    "spot_in_day_range",          # (close - low) / (high - low)
    "hist_vol_5d",                # ZEROED: not available in live engine
    "hist_vol_20d",               # ZEROED: not available in live engine

    # Tier 1: Extended market analytics
    "dealer_position_numeric",    # Long_Gamma=1, Short_Gamma=-1
    "vanna_regime_numeric",       # VANNA_BULLISH=1, BEARISH=-1, NEUTRAL=0
    "charm_regime_numeric",       # CHARM_BULLISH=1, BEARISH=-1, NEUTRAL=0
    "confirmation_numeric",       # ZEROED: computed after probability (circular)

    # Tier 3: Global market (continuous)
    "india_vix_level",            # absolute India VIX
    "india_vix_change_24h",       # India VIX % change
    "oil_shock_score",            # oil risk score
    "commodity_risk_score",       # composite commodity risk
    "volatility_shock_score",     # VIX-based shock score

    # Tier 4: Macro events
    "macro_event_risk_score",     # 0-1 scheduled event risk
    "macro_event_numeric",        # NO_EVENT=0, POST=1, PRE=2, DURING=3

    # Tier 5: Temporal & structural
    "days_to_expiry",             # calendar days to selected expiry
    "moneyness_pct",              # ZEROED: strike not selected until after probability
    "weekday",                    # 0=Mon .. 4=Fri
]

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 33, f"Expected 33 features, got {N_FEATURES}"

LEGACY_FEATURE_COUNT = 7
FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
LIVE_UNAVAILABLE_FEATURE_DEFAULTS = {
    "hist_vol_5d": 0.0,
    "hist_vol_20d": 0.0,
    "confirmation_numeric": 1.0,
    "moneyness_pct": 0.0,
}

# ── Target columns ──────────────────────────────────────────────────
TARGET_COLUMNS = {
    "target_1d":          "correct_1d",
    "target_5d":          "correct_5d",
    "target_at_expiry":   "correct_at_expiry",
    "target_hit_binary":  "target_hit",
    "eod_mfe_bps":        "eod_mfe_bps",      # regression target
    "eod_mae_bps":        "eod_mae_bps",       # regression target
}

LEAKAGE_LABEL_COLUMNS = {
    "correct_1d",
    "correct_2d",
    "correct_3d",
    "correct_5d",
    "correct_at_expiry",
    "return_1d_bps",
    "return_2d_bps",
    "return_3d_bps",
    "return_5d_bps",
    "return_at_expiry_bps",
    "eod_mfe_bps",
    "eod_mae_bps",
    "spot_1d",
    "spot_2d",
    "spot_3d",
    "spot_5d",
    "spot_at_expiry",
}


def validate_no_post_signal_labels_in_features(feature_names: list[str]) -> list[str]:
    """Return any post-signal label columns accidentally present in features."""
    if not feature_names:
        return []
    normalized = {str(name).strip() for name in feature_names}
    return sorted(normalized.intersection(LEAKAGE_LABEL_COLUMNS))


# ── Extraction from signal evaluation row ───────────────────────────

def _safe_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _encode_categorical(value, mapping, default=0.0):
    return mapping.get(str(value).upper().strip() if value else "", default)


# Encoding maps
_GAMMA = {"SHORT_GAMMA": 1.0, "NEGATIVE_GAMMA": 1.0, "SHORT_GAMMA_ZONE": 1.0,
          "LONG_GAMMA": -0.5, "POSITIVE_GAMMA": -0.5, "LONG_GAMMA_ZONE": -0.5,
          "NEUTRAL_GAMMA": 0.0, "NEUTRAL": 0.0}

_FLOW = {"BULLISH_FLOW": 1.0, "BEARISH_FLOW": -1.0,
         "NEUTRAL_FLOW": 0.0, "MIXED_FLOW": 0.0, "NO_FLOW": 0.0}

_VOL = {"LOW_VOL": 0.0, "NORMAL_VOL": 1.0, "VOL_EXPANSION": 2.0, "HIGH_VOL": 2.0}

_HEDGING = {"UPSIDE_ACCELERATION": 1.0, "DOWNSIDE_ACCELERATION": -1.0,
            "PINNING": 0.0, "UPSIDE_PINNING": 0.3, "DOWNSIDE_PINNING": -0.3}

_FLIP = {"ABOVE_FLIP": 1.0, "BELOW_FLIP": -1.0, "AT_FLIP": 0.0}

_VACUUM = {"BREAKOUT_ZONE": 1.0, "NORMAL": 0.0}

_DEALER = {"LONG GAMMA": 1.0, "SHORT GAMMA": -1.0, "NEUTRAL": 0.0}

_VANNA = {"VANNA_BULLISH": 1.0, "VANNA_BEARISH": -1.0,
          "BULLISH_VANNA": 1.0, "BEARISH_VANNA": -1.0,
          "VANNA_NEUTRAL": 0.0, "NEUTRAL_VANNA": 0.0, "NEUTRAL": 0.0}

_CHARM = {"CHARM_BULLISH": 1.0, "CHARM_BEARISH": -1.0,
          "BULLISH_CHARM": 1.0, "BEARISH_CHARM": -1.0,
          "CHARM_NEUTRAL": 0.0, "NEUTRAL_CHARM": 0.0, "NEUTRAL": 0.0}

_CONFIRM = {"STRONG_CONFIRMATION": 3.0, "CONFIRMED": 2.0, "MIXED": 1.0, "CONFLICT": 0.0}

_MACRO_EVENT = {"NO_EVENT": 0.0, "POST_EVENT": 1.0, "PRE_EVENT": 2.0, "DURING_EVENT": 3.0}


def get_live_feature_contract() -> dict:
    """Return the canonical live inference contract for the expanded feature set."""
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_count": N_FEATURES,
        "legacy_feature_count": LEGACY_FEATURE_COUNT,
        "live_unavailable_feature_defaults": dict(LIVE_UNAVAILABLE_FEATURE_DEFAULTS),
    }


def validate_live_feature_vector(feature_vector, *, atol: float = 1e-9) -> dict:
    """Validate the expanded live feature vector against the train-serve contract."""
    arr = np.asarray(feature_vector, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            return {
                "valid": False,
                "feature_count": int(arr.shape[1]) if arr.ndim > 1 else int(arr.size),
                "violations": ["multirow_feature_vector_not_supported"],
                "legacy_vector": False,
            }
        arr = arr.reshape(-1)

    feature_count = int(arr.size)
    if feature_count == LEGACY_FEATURE_COUNT:
        return {
            "valid": True,
            "feature_count": feature_count,
            "violations": [],
            "legacy_vector": True,
        }

    if feature_count != N_FEATURES:
        return {
            "valid": False,
            "feature_count": feature_count,
            "violations": [f"unexpected_feature_count:{feature_count}"],
            "legacy_vector": False,
        }

    violations = []
    for feature_name, expected_default in LIVE_UNAVAILABLE_FEATURE_DEFAULTS.items():
        idx = FEATURE_INDEX[feature_name]
        actual = arr[idx]
        if np.isnan(actual) or abs(float(actual) - float(expected_default)) > atol:
            violations.append(
                {
                    "feature": feature_name,
                    "index": idx,
                    "expected": float(expected_default),
                    "actual": None if np.isnan(actual) else float(actual),
                }
            )

    return {
        "valid": len(violations) == 0,
        "feature_count": feature_count,
        "violations": violations,
        "legacy_vector": False,
    }


def enforce_live_feature_contract(feature_vector, *, atol: float = 1e-9) -> tuple[np.ndarray, dict]:
    """Sanitize an expanded feature vector so live-unavailable features stay fixed."""
    arr = np.asarray(feature_vector, dtype=np.float64)
    original_shape = arr.shape
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)

    validation = validate_live_feature_vector(arr, atol=atol)
    if validation.get("legacy_vector"):
        return np.asarray(feature_vector, dtype=np.float64), {
            **validation,
            "sanitized": False,
        }

    if validation["feature_count"] != N_FEATURES:
        return np.asarray(feature_vector, dtype=np.float64), {
            **validation,
            "sanitized": False,
        }

    sanitized = arr.copy()
    for feature_name, expected_default in LIVE_UNAVAILABLE_FEATURE_DEFAULTS.items():
        sanitized[FEATURE_INDEX[feature_name]] = float(expected_default)

    if len(original_shape) == 2 and original_shape[0] == 1:
        sanitized = sanitized.reshape(1, -1)

    post_validation = validate_live_feature_vector(sanitized, atol=atol)
    return sanitized, {
        **post_validation,
        "sanitized": len(validation["violations"]) > 0,
        "original_violations": validation["violations"],
    }


def extract_features(row: dict) -> np.ndarray:
    """Extract the 33-element feature vector from a signal evaluation row.

    This function operates purely on the fields already captured in the
    signal evaluation schema (136+ columns).  It does NOT call any
    analytics or probability functions, so it is safe to use in the
    training loop without introducing circularity.

    Parameters
    ----------
    row : dict
        One signal evaluation row from the backtest dataset.

    Returns
    -------
    np.ndarray of shape (33,)
    """
    f = np.zeros(N_FEATURES, dtype=np.float64)

    # Tier 1: Market analytics (categorical → numeric)
    # Prefer pre-computed numeric if available, else encode from categorical
    f[0] = _safe_float(row.get("gamma_regime_numeric"), _encode_categorical(row.get("gamma_regime"), _GAMMA))
    f[1] = _safe_float(row.get("flow_signal_numeric"), _encode_categorical(row.get("final_flow_signal"), _FLOW))
    f[2] = _safe_float(row.get("vol_regime_numeric"), _encode_categorical(row.get("volatility_regime"), _VOL))
    f[3] = _encode_categorical(row.get("dealer_hedging_bias"), _HEDGING)
    f[4] = _safe_float(row.get("spot_vs_flip_numeric"), _encode_categorical(row.get("spot_vs_flip"), _FLIP))
    f[5] = _encode_categorical(row.get("liquidity_vacuum_state"), _VACUUM)
    f[6] = _safe_float(row.get("move_probability"), 0.0)  # proxy for atm_iv_scaled if raw IV missing
    # Prefer direct atm_iv if captured in enrichment
    atm_iv_scaled = row.get("atm_iv_scaled")
    if atm_iv_scaled is not None and not (isinstance(atm_iv_scaled, float) and np.isnan(atm_iv_scaled)):
        f[6] = _safe_float(atm_iv_scaled)

    # Tier 2: Probability sub-features
    f[7]  = _safe_float(row.get("gamma_flip_distance_pct"))
    f[8]  = _safe_float(row.get("vacuum_strength"))
    f[9]  = _safe_float(row.get("hedging_flow_ratio"))
    f[10] = _safe_float(row.get("smart_money_flow_score"))
    f[11] = _safe_float(row.get("atm_iv_percentile"))
    f[12] = _safe_float(row.get("intraday_range_pct"))

    # Tier 0 + 5: Spot-derived
    f[13] = _safe_float(row.get("lookback_avg_range_pct"))
    f[14] = _safe_float(row.get("gap_pct"))
    f[15] = _safe_float(row.get("close_vs_prev_close_pct"))
    f[16] = _safe_float(row.get("spot_in_day_range"), 0.5)
    # hist_vol_5d/20d: zeroed — not available in live engine at probability time.
    # Backtest enrichment computes these, but the model must not depend on them.
    f[17] = 0.0
    f[18] = 0.0

    # Tier 1 extended
    f[19] = _encode_categorical(row.get("dealer_position"), _DEALER)
    f[20] = _encode_categorical(row.get("vanna_regime"), _VANNA)
    f[21] = _encode_categorical(row.get("charm_regime"), _CHARM)
    # confirmation_status: zeroed — computed AFTER probability in live engine,
    # so using it here would create a circular dependency.
    f[22] = 1.0  # MIXED default

    # Tier 3: Global market
    f[23] = _safe_float(row.get("india_vix_level"))
    f[24] = _safe_float(row.get("india_vix_change_24h"))
    f[25] = _safe_float(row.get("oil_shock_score"))
    f[26] = _safe_float(row.get("commodity_risk_score"))
    f[27] = _safe_float(row.get("volatility_shock_score"))

    # Tier 4: Macro events
    f[28] = _safe_float(row.get("macro_event_risk_score"))
    f[29] = _safe_float(row.get("macro_event_numeric"),
                        _encode_categorical(row.get("macro_regime"), _MACRO_EVENT))

    # Tier 5: Temporal & structural
    f[30] = _safe_float(row.get("days_to_expiry"), 7.0)
    # moneyness_pct: zeroed — strike not selected until after probability.
    f[31] = 0.0
    f[32] = _safe_float(row.get("weekday"), 2.0)

    return f


def extract_features_batch(rows: list[dict]) -> np.ndarray:
    """Extract feature matrix from a list of signal evaluation rows.

    Returns
    -------
    np.ndarray of shape (n_samples, 33)
    """
    return np.vstack([extract_features(r) for r in rows])


def extract_target(row: dict, target_name: str = "target_1d") -> float | None:
    """Extract a single target value from a signal evaluation row.

    Returns None if the target is unavailable.
    """
    source_key = TARGET_COLUMNS.get(target_name, target_name)
    v = row.get(source_key)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
