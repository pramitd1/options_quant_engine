"""
Intraday Gamma Shift Tracker

Compares gamma exposure between two option-chain snapshots.
"""

from analytics.gamma_exposure import calculate_gamma_exposure


def compute_gamma_profile(option_chain, spot=None):
    """
    Compute total gamma exposure for a snapshot.
    """
    return calculate_gamma_exposure(option_chain, spot)


def detect_gamma_shift(previous_chain, current_chain, spot=None):
    """
    Detect change in gamma exposure between two snapshots.
    """

    prev_gamma = compute_gamma_profile(previous_chain, spot)
    curr_gamma = compute_gamma_profile(current_chain, spot)

    shift = curr_gamma - prev_gamma

    if abs(prev_gamma) > 0 and abs(shift) < abs(prev_gamma) * 0.05:
        return "NO_SHIFT"

    if shift > 0:
        return "GAMMA_INCREASE"

    return "GAMMA_DECREASE"


def gamma_shift_signal(previous_chain, current_chain, spot=None):
    """
    Convert gamma shift into a regime signal.
    """

    shift = detect_gamma_shift(previous_chain, current_chain, spot)

    if shift == "GAMMA_DECREASE":
        return "VOL_EXPANSION"

    if shift == "GAMMA_INCREASE":
        return "VOL_SUPPRESSION"

    return "NEUTRAL"