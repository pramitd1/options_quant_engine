"""
Module: regime_conditional_thresholds.py

Purpose:
    Implement regime-conditional trading thresholds that adapt to market conditions.

Context:
    Cumulative analysis shows:
    - NEGATIVE_GAMMA: 44.1% hit rate (vs 53.6% in POSITIVE_GAMMA)
    - Gamma regime determines whether dealer hedging helps or hurts signals
    - Position sizing should scale inversely with regime difficulty
    
    This module:
    1. Adjusts minimum score thresholds based on regime
    2. Scales position sizes by regime favorability
    3. Dynamically sets maximum holding periods
    4. Applies regime-aware risk penalties

Key Outputs:
    effective_thresholds = {
        "min_composite_score": int,           # Adjusted threshold
        "min_trade_strength": int,
        "position_size_multiplier": float,    # 0.5–1.2x
        "max_holding_minutes": int,
        "confidence_multiplier": float,
        "regime_name": str,
        "adjustments": dict
    }

Downstream Usage:
    Consumed by signal_engine.py for trade qualification and position sizing.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_float(value, default=None):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


class RegimeAdaptiveThresholds:
    """
    Compute regime-conditional thresholds.
    
    Adjustment Strategy:
    
    POSITIVE_GAMMA (Favorable):
        - Relax composite score threshold (-3 points)
        - Relax trade strength threshold (-2 points)
        - Increase position size (+20%)
        - Longer holding periods (+60 minutes)
        - Higher confidence boost (+1.1x)
    
    NEGATIVE_GAMMA (Toxic):
        - Tighten composite score threshold (+5 points)
        - Tighten trade strength threshold (+3 points)
        - Reduce position size (-30%)
        - Shorter holding periods (-60 minutes)
        - Lower confidence boost (0.7x)
    
    NEUTRAL_GAMMA (Balanced):
        - No adjustment
    """
    
    def __init__(
        self,
        base_composite_score: int = 45,
        base_trade_strength: int = 25,
        base_max_holding_m: int = 240,
        base_position_size: float = 1.0,
        positive_gamma_composite_delta: int = -3,
        positive_gamma_strength_delta: int = -2,
        positive_gamma_holding_delta_m: int = 60,
        positive_gamma_position_size_mult: float = 1.2,
        negative_gamma_composite_delta: int = 5,
        negative_gamma_strength_delta: int = 3,
        negative_gamma_holding_delta_m: int = -60,
        negative_gamma_position_size_mult: float = 0.7,
        neutral_gamma_composite_delta: int = 0,
        neutral_gamma_strength_delta: int = 0,
        neutral_gamma_holding_delta_m: int = 0,
        neutral_gamma_position_size_mult: float = 1.0,
    ):
        self.base_composite_score = int(base_composite_score)
        self.base_trade_strength = int(base_trade_strength)
        self.base_max_holding_m = int(base_max_holding_m)
        self.base_position_size = float(base_position_size)
        
        # Define regime-specific adjustments
        self.regime_adjustments = {
            "POSITIVE_GAMMA": {
                "composite_delta": int(positive_gamma_composite_delta),
                "strength_delta": int(positive_gamma_strength_delta),
                "max_holding_delta_m": int(positive_gamma_holding_delta_m),
                "position_size_mult": float(positive_gamma_position_size_mult),
                "confidence_mult": 1.1,
                "rationale": "Configured POSITIVE_GAMMA threshold and sizing adjustment"
            },
            "NEGATIVE_GAMMA": {
                "composite_delta": int(negative_gamma_composite_delta),
                "strength_delta": int(negative_gamma_strength_delta),
                "max_holding_delta_m": int(negative_gamma_holding_delta_m),
                "position_size_mult": float(negative_gamma_position_size_mult),
                "confidence_mult": 0.7,
                "rationale": "Configured NEGATIVE_GAMMA threshold and sizing adjustment"
            },
            "NEUTRAL_GAMMA": {
                "composite_delta": int(neutral_gamma_composite_delta),
                "strength_delta": int(neutral_gamma_strength_delta),
                "max_holding_delta_m": int(neutral_gamma_holding_delta_m),
                "position_size_mult": float(neutral_gamma_position_size_mult),
                "confidence_mult": 1.0,
                "rationale": "Configured NEUTRAL_GAMMA threshold and sizing adjustment"
            }
        }
        
        # Volatility regime multipliers
        self.volatility_adjustments = {
            "NORMAL_VOL": {
                "composite_delta": 0,
                "position_size_mult": 1.0,
                "holding_delta_m": 0
            },
            "VOL_EXPANSION": {
                "composite_delta": 4,    # Tighten further in high vol
                "position_size_mult": 0.75,  # Reduce size
                "holding_delta_m": -45   # Shorten holds
            },
            "VOL_CONTRACTION": {
                "composite_delta": -1,   # Relax in low vol
                "position_size_mult": 1.1,   # Increase size
                "holding_delta_m": 30    # Lengthen holds
            }
        }
    
    def compute_thresholds(
        self,
        gamma_regime: str,
        volatility_regime: Optional[str] = None,
        spot_vs_flip: Optional[str] = None,
        dealer_position: Optional[str] = None
    ) -> Dict:
        """
        Compute effective thresholds for trading.
        
        Args:
            gamma_regime: POSITIVE_GAMMA, NEGATIVE_GAMMA, NEUTRAL_GAMMA
            volatility_regime: NORMAL_VOL, VOL_EXPANSION, VOL_CONTRACTION
            spot_vs_flip: AT_FLIP, ABOVE_FLIP, BELOW_FLIP
            dealer_position: Short Gamma, Long Gamma, etc.
        
        Returns:
            Dict with effective thresholds and regime adjustments
        """
        result = {
            "gamma_regime": gamma_regime,
            "volatility_regime": volatility_regime,
            "effective_composite_score": self.base_composite_score,
            "effective_trade_strength": self.base_trade_strength,
            "effective_max_holding_m": self.base_max_holding_m,
            "position_size_multiplier": self.base_position_size,
            "confidence_multiplier": 1.0,
            "adjustments": {},
            "rationale": []
        }
        
        # Apply gamma regime adjustments
        gamma_upper = (gamma_regime or "NEUTRAL_GAMMA").upper()
        if gamma_upper in self.regime_adjustments:
            adj = self.regime_adjustments[gamma_upper]
            
            result["effective_composite_score"] += adj["composite_delta"]
            result["effective_trade_strength"] += adj["strength_delta"]
            result["effective_max_holding_m"] += adj["max_holding_delta_m"]
            result["position_size_multiplier"] *= adj["position_size_mult"]
            result["confidence_multiplier"] *= adj["confidence_mult"]
            
            result["adjustments"]["gamma"] = adj
            result["rationale"].append(adj["rationale"])
        
        # Apply volatility regime adjustments
        vol_upper = (volatility_regime or "NORMAL_VOL").upper()
        if vol_upper in self.volatility_adjustments:
            adj = self.volatility_adjustments[vol_upper]
            
            result["effective_composite_score"] += adj["composite_delta"]
            result["effective_max_holding_m"] += adj["holding_delta_m"]
            result["position_size_multiplier"] *= adj["position_size_mult"]
            
            result["adjustments"]["volatility"] = adj
            if adj["composite_delta"] != 0:
                result["rationale"].append(
                    f"Adjusted for {vol_upper}: composite {adj['composite_delta']:+d}, "
                    f"size {adj['position_size_mult']:.2f}x"
                )
        
        # Apply spot vs flip adjustments
        spot_vs_flip_upper = (spot_vs_flip or "ABOVE_FLIP").upper()
        if spot_vs_flip_upper == "AT_FLIP":
            # At flip, tighten thresholds (squeeze risk)
            result["effective_composite_score"] += 3
            result["effective_trade_strength"] += 2
            result["position_size_multiplier"] *= 0.9
            result["rationale"].append("AT_FLIP: Tightened thresholds (squeeze risk)")
        
        # Bounds checking
        result["effective_composite_score"] = int(_clip(
            result["effective_composite_score"],
            20,  # Min threshold
            85   # Max threshold (no point raising above this)
        ))
        result["effective_trade_strength"] = int(_clip(
            result["effective_trade_strength"],
            10,
            60
        ))
        result["effective_max_holding_m"] = int(_clip(
            result["effective_max_holding_m"],
            30,   # Minimum 30m holding
            480   # Maximum 8-hour holding
        ))
        result["position_size_multiplier"] = float(_clip(
            result["position_size_multiplier"],
            0.5,   # Never smaller than 50% base
            1.3    # Never larger than 130% base
        ))
        
        return result
    
    def should_qualify_trade(
        self,
        raw_composite_score: float,
        raw_trade_strength: float,
        gamma_regime: str,
        volatility_regime: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Dict]:
        """
        Determine if trade passes regime-conditional thresholds.
        
        Returns:
            (qualification_passed, detail_dict)
        """
        thresholds = self.compute_thresholds(
            gamma_regime, volatility_regime, **kwargs
        )
        
        composite_ok = raw_composite_score >= thresholds["effective_composite_score"]
        strength_ok = raw_trade_strength >= thresholds["effective_trade_strength"]
        
        qualification_passed = composite_ok and strength_ok
        
        detail = {
            "passed": qualification_passed,
            "composite_ok": composite_ok,
            "strength_ok": strength_ok,
            "raw_composite": float(raw_composite_score),
            "required_composite": thresholds["effective_composite_score"],
            "composite_gap": float(raw_composite_score) - thresholds["effective_composite_score"],
            "raw_strength": float(raw_trade_strength),
            "required_strength": thresholds["effective_trade_strength"],
            "strength_gap": float(raw_trade_strength) - thresholds["effective_trade_strength"],
            "thresholds": thresholds,
            "reason": (
                "PASS" if qualification_passed
                else (
                    f"Composite {raw_composite_score:.0f} < {thresholds['effective_composite_score']}"
                    if not composite_ok
                    else f"Strength {raw_trade_strength:.0f} < {thresholds['effective_trade_strength']}"
                )
            )
        }
        
        return qualification_passed, detail


class RegimeMonitor:
    """Monitor regime transitions and alert when thresholds change significantly."""
    
    def __init__(self):
        self.last_regime = None
        self.last_vol_regime = None
        self.threshold_cache = {}
    
    def detect_regime_change(
        self,
        gamma_regime: str,
        volatility_regime: str
    ) -> Tuple[bool, str]:
        """
        Detect significant regime change.
        
        Returns:
            (changed, reason)
        """
        changed = False
        reasons = []
        
        if self.last_regime != gamma_regime:
            reasons.append(f"gamma: {self.last_regime} → {gamma_regime}")
            self.last_regime = gamma_regime
            changed = True
        
        if self.last_vol_regime != volatility_regime:
            reasons.append(f"vol: {self.last_vol_regime} → {volatility_regime}")
            self.last_vol_regime = volatility_regime
            changed = True
        
        reason_str = " | ".join(reasons) if reasons else "no_change"
        return changed, reason_str


# ============================================================================
# Convenience Functions
# ============================================================================

_global_regime_thresholds: Optional[RegimeAdaptiveThresholds] = None
_global_regime_monitor: Optional[RegimeMonitor] = None


def initialize_regime_thresholds(
    base_composite: int = 45,
    base_strength: int = 25,
    base_max_holding_m: int = 240,
    base_position_size: float = 1.0,
    **kwargs
) -> RegimeAdaptiveThresholds:
    """Initialize global regime-adaptive thresholds."""
    global _global_regime_thresholds, _global_regime_monitor
    _global_regime_thresholds = RegimeAdaptiveThresholds(
        base_composite_score=base_composite,
        base_trade_strength=base_strength,
        base_max_holding_m=base_max_holding_m,
        base_position_size=base_position_size,
        **kwargs
    )
    _global_regime_monitor = RegimeMonitor()
    return _global_regime_thresholds


def compute_regime_thresholds(
    gamma_regime: str,
    volatility_regime: str = "NORMAL_VOL",
    **kwargs
) -> Dict:
    """Compute effective thresholds using global instance."""
    global _global_regime_thresholds
    if _global_regime_thresholds is None:
        initialize_regime_thresholds()
    return _global_regime_thresholds.compute_thresholds(
        gamma_regime, volatility_regime, **kwargs
    )


def check_regime_qualification(
    composite: float,
    strength: float,
    gamma_regime: str,
    volatility_regime: str = "NORMAL_VOL",
    **kwargs
) -> Tuple[bool, Dict]:
    """Check if trade qualifies under regime-conditional thresholds."""
    global _global_regime_thresholds
    if _global_regime_thresholds is None:
        initialize_regime_thresholds()
    
    return _global_regime_thresholds.should_qualify_trade(
        composite, strength, gamma_regime, volatility_regime, **kwargs
    )


def get_regime_thresholds() -> Optional[RegimeAdaptiveThresholds]:
    """Get reference to global regime thresholds."""
    return _global_regime_thresholds


def get_regime_monitor() -> Optional[RegimeMonitor]:
    """Get reference to global regime monitor."""
    return _global_regime_monitor
