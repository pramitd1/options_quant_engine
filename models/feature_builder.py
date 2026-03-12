"""
Feature Builder
"""

import numpy as np


def build_features(
    option_chain,
    spot=None,
    gamma_regime=None,
    final_flow_signal=None,
    vol_regime=None,
    hedging_bias=None,
    spot_vs_flip=None,
    vacuum_state=None,
    atm_iv=None
):
    gamma_sign = 0.0
    if gamma_regime in ["NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"]:
        gamma_sign = 1.0
    elif gamma_regime in ["POSITIVE_GAMMA", "LONG_GAMMA_ZONE"]:
        gamma_sign = -0.5

    flow_bias = 0.0
    if final_flow_signal == "BULLISH_FLOW":
        flow_bias = 1.0
    elif final_flow_signal == "BEARISH_FLOW":
        flow_bias = -1.0

    vol_expansion = 1.0 if vol_regime == "VOL_EXPANSION" else 0.0

    hedging_bias_score = 0.0
    if hedging_bias == "UPSIDE_ACCELERATION":
        hedging_bias_score = 1.0
    elif hedging_bias == "DOWNSIDE_ACCELERATION":
        hedging_bias_score = -1.0

    spot_flip_score = 0.0
    if spot_vs_flip == "ABOVE_FLIP":
        spot_flip_score = 1.0
    elif spot_vs_flip == "BELOW_FLIP":
        spot_flip_score = -1.0

    vacuum_score = 1.0 if vacuum_state == "BREAKOUT_ZONE" else 0.0
    iv_level = float(atm_iv) / 100.0 if atm_iv is not None else 0.0

    features = np.array([
        gamma_sign,
        flow_bias,
        vol_expansion,
        hedging_bias_score,
        spot_flip_score,
        vacuum_score,
        iv_level
    ], dtype=float)

    return features.reshape(1, -1)