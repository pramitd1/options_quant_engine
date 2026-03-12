def build_signal(
    gamma_state,
    hedging,
    vol_regime
):

    if gamma_state == "SHORT_GAMMA":

        return "CALL"

    if gamma_state == "LONG_GAMMA":

        return "PUT"

    return None