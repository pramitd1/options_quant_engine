import numpy as np


def large_move_probability(
        gamma_regime,
        vacuum_state,
        hedging_bias,
        smart_money_flow
):
    """
    Estimate probability of 150-300 point move
    """

    score = 0

    # Gamma regime impact
    if gamma_regime == "NEGATIVE_GAMMA":
        score += 2

    if gamma_regime == "POSITIVE_GAMMA":
        score -= 1

    # Liquidity vacuum
    if vacuum_state == "BREAKOUT_ZONE":
        score += 2

    # Dealer hedging
    if hedging_bias in [
        "UPSIDE_ACCELERATION",
        "DOWNSIDE_ACCELERATION"
    ]:
        score += 2

    # Institutional flow
    if smart_money_flow in [
        "BULLISH_FLOW",
        "BEARISH_FLOW"
    ]:
        score += 1

    probability = 1 / (1 + np.exp(-score))

    return round(probability, 2)