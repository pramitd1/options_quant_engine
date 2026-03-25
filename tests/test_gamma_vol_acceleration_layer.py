from __future__ import annotations

import unittest

from tuning.runtime import temporary_parameter_pack
from risk import build_gamma_vol_acceleration_state
from risk.gamma_vol_acceleration_features import build_gamma_vol_acceleration_features


class GammaVolAccelerationLayerTests(unittest.TestCase):
    def test_feature_model_detects_short_gamma_flip_instability(self):
        features = build_gamma_vol_acceleration_features(
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="AT_FLIP",
            gamma_flip_distance_pct=0.06,
            dealer_hedging_bias="UPSIDE_ACCELERATION",
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_range_pct=0.88,
            volatility_compression_score=0.72,
            volatility_shock_score=0.26,
            macro_event_risk_score=18,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.52,
        )

        self.assertGreater(features["normalized_acceleration"], 0.55)
        self.assertGreater(features["upside_squeeze_risk"], features["downside_airpocket_risk"])
        self.assertGreater(features["flip_proximity_score"], 0.8)

    def test_feature_model_degrades_gracefully_with_missing_inputs(self):
        features = build_gamma_vol_acceleration_features()

        self.assertEqual(features["normalized_acceleration"], 0.0)
        self.assertTrue(features["neutral_fallback"])
        self.assertIn("gamma_regime_missing", features["warnings"])

    def test_state_classifies_upside_squeeze(self):
        state = build_gamma_vol_acceleration_state(
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="ABOVE_FLIP",
            gamma_flip_distance_pct=0.08,
            dealer_hedging_bias="UPSIDE_ACCELERATION",
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_range_pct=0.95,
            volatility_compression_score=0.7,
            volatility_shock_score=0.25,
            macro_event_risk_score=10,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.48,
        )

        self.assertEqual(state["directional_convexity_state"], "UPSIDE_SQUEEZE_RISK")
        self.assertIn(state["squeeze_risk_state"], {"HIGH_ACCELERATION_RISK", "EXTREME_ACCELERATION_RISK"})
        self.assertGreater(state["gamma_vol_adjustment_score"], 0)

    def test_state_blocks_overnight_when_convexity_and_global_instability_combine(self):
        state = build_gamma_vol_acceleration_state(
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="AT_FLIP",
            gamma_flip_distance_pct=0.05,
            dealer_hedging_bias="DOWNSIDE_ACCELERATION",
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_range_pct=1.1,
            volatility_compression_score=0.68,
            volatility_shock_score=0.82,
            macro_event_risk_score=72,
            global_risk_state={
                "global_risk_state": "VOL_SHOCK",
                "holding_context": {"holding_profile": "OVERNIGHT", "overnight_relevant": True},
            },
            volatility_explosion_probability=0.84,
            holding_profile="OVERNIGHT",
        )

        self.assertFalse(state["overnight_hold_allowed"])
        self.assertGreaterEqual(state["overnight_convexity_penalty"], 7)
        self.assertGreater(state["overnight_convexity_risk"], 0.7)

    def test_state_dampens_long_gamma_pinned_market(self):
        state = build_gamma_vol_acceleration_state(
            gamma_regime="LONG_GAMMA_ZONE",
            spot_vs_flip="ABOVE_FLIP",
            gamma_flip_distance_pct=1.0,
            dealer_hedging_bias="PINNING",
            liquidity_vacuum_state="NO_VACUUM",
            intraday_range_pct=0.2,
            volatility_compression_score=0.1,
            volatility_shock_score=0.02,
            macro_event_risk_score=5,
            global_risk_state={"global_risk_state": "RISK_ON"},
            volatility_explosion_probability=0.03,
        )

        self.assertEqual(state["squeeze_risk_state"], "LOW_ACCELERATION_RISK")
        self.assertEqual(state["directional_convexity_state"], "NO_CONVEXITY_EDGE")
        self.assertLessEqual(state["gamma_vol_adjustment_score"], 0)

    def test_flip_drift_boosts_directional_alignment_when_enabled(self):
        base = build_gamma_vol_acceleration_features(
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="AT_FLIP",
            gamma_flip_distance_pct=0.08,
            dealer_hedging_bias="PINNING",
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_range_pct=0.75,
            volatility_compression_score=0.60,
            volatility_shock_score=0.25,
            macro_event_risk_score=12,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.45,
            gamma_flip_drift={"drift": 140.0},
        )

        with temporary_parameter_pack(
            "disable_flip_drift_overlay",
            overrides={"trade_strength.runtime_thresholds.use_flip_drift_in_overlays": 0},
        ):
            disabled = build_gamma_vol_acceleration_features(
                gamma_regime="SHORT_GAMMA_ZONE",
                spot_vs_flip="AT_FLIP",
                gamma_flip_distance_pct=0.08,
                dealer_hedging_bias="PINNING",
                liquidity_vacuum_state="BREAKOUT_ZONE",
                intraday_range_pct=0.75,
                volatility_compression_score=0.60,
                volatility_shock_score=0.25,
                macro_event_risk_score=12,
                global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
                volatility_explosion_probability=0.45,
                gamma_flip_drift={"drift": 140.0},
            )

        self.assertGreater(base["drift_up_boost"], 0.0)
        self.assertEqual(disabled["drift_up_boost"], 0.0)
        self.assertGreater(base["upside_squeeze_risk"], disabled["upside_squeeze_risk"])
