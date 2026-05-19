"""Research-only Heston stochastic-volatility diagnostics.

The live engine continues to use the existing Black-Scholes Greek engine.
This package supplies optional calibration and diagnostic features for signal
evaluation and research workflows.
"""

from .heston_pricer import HestonParams, black_scholes_price, heston_price
from .heston_calibration import HestonCalibrationResult, calibrate_heston_to_chain
from .heston_features import (
    HESTON_FEATURE_COLUMNS,
    build_heston_research_features,
    default_heston_research_features,
)

__all__ = [
    "HESTON_FEATURE_COLUMNS",
    "HestonCalibrationResult",
    "HestonParams",
    "black_scholes_price",
    "build_heston_research_features",
    "calibrate_heston_to_chain",
    "default_heston_research_features",
    "heston_price",
]
