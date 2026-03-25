from __future__ import annotations

import unittest

from tuning.runtime import temporary_parameter_pack
from risk import build_dealer_hedging_pressure_state
from risk.dealer_hedging_pressure_features import build_dealer_hedging_pressure_features


class DealerHedgingPressureLayerTests(unittest.TestCase):
    def test_feature_model_detects_upside_hedging_pressure(self):
        features = build_dealer_hedging_pressure_features(
            spot=22000,
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="ABOVE_FLIP",
            gamma_flip_distance_pct=0.08,
            dealer_position="Short Gamma",
            dealer_hedging_bias="UPSIDE_ACCELERATION",
            dealer_hedging_flow="BUY_FUTURES",
            gamma_clusters=[21940, 22040],
            liquidity_levels=[21930, 22060],
            support_wall=21950,
            resistance_wall=22180,
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_gamma_state="VOL_EXPANSION",
            intraday_range_pct=0.9,
            flow_signal="BULLISH_FLOW",
            smart_money_flow="BULLISH_FLOW",
            macro_event_risk_score=12,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.42,
            gamma_vol_acceleration_score=70,
        )

        self.assertGreater(features["upside_hedging_pressure"], features["downside_hedging_pressure"])
        self.assertGreater(features["normalized_pressure"], 0.55)

    def test_feature_model_gracefully_handles_missing_inputs(self):
        features = build_dealer_hedging_pressure_features()

        self.assertEqual(features["normalized_pressure"], 0.0)
        self.assertTrue(features["neutral_fallback"])
        self.assertIn("gamma_regime_missing", features["warnings"])

    def test_state_classifies_pinning_dominant_environment(self):
        state = build_dealer_hedging_pressure_state(
            spot=22000,
            gamma_regime="LONG_GAMMA_ZONE",
            spot_vs_flip="ABOVE_FLIP",
            gamma_flip_distance_pct=0.34,
            dealer_position="Long Gamma",
            dealer_hedging_bias="PINNING",
            dealer_hedging_flow="BUY_FUTURES",
            gamma_clusters=[21980, 22020, 22030],
            liquidity_levels=[21990, 22010],
            support_wall=21985,
            resistance_wall=22025,
            liquidity_vacuum_state="NO_VACUUM",
            intraday_gamma_state="VOL_SUPPRESSION",
            intraday_range_pct=0.24,
            flow_signal="NEUTRAL_FLOW",
            smart_money_flow="NEUTRAL_FLOW",
            macro_event_risk_score=5,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.08,
            gamma_vol_acceleration_score=16,
        )

        self.assertEqual(state["dealer_flow_state"], "PINNING_DOMINANT")
        self.assertLessEqual(state["dealer_pressure_adjustment_score"], 0)
        self.assertGreater(state["pinning_pressure_score"], 0.6)

    def test_state_blocks_overnight_when_pressure_and_macro_risk_combine(self):
        state = build_dealer_hedging_pressure_state(
            spot=22000,
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="BELOW_FLIP",
            gamma_flip_distance_pct=0.09,
            dealer_position="Short Gamma",
            dealer_hedging_bias="DOWNSIDE_ACCELERATION",
            dealer_hedging_flow="SELL_FUTURES",
            gamma_clusters=[22120],
            liquidity_levels=[22140],
            support_wall=21750,
            resistance_wall=22080,
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_gamma_state="VOL_EXPANSION",
            intraday_range_pct=1.02,
            flow_signal="BEARISH_FLOW",
            smart_money_flow="BEARISH_FLOW",
            macro_event_risk_score=78,
            global_risk_state={
                "global_risk_state": "VOL_SHOCK",
                "holding_context": {"holding_profile": "OVERNIGHT", "overnight_relevant": True},
            },
            volatility_explosion_probability=0.84,
            gamma_vol_acceleration_score=82,
            holding_profile="OVERNIGHT",
        )

        self.assertFalse(state["overnight_hold_allowed"])
        self.assertGreaterEqual(state["overnight_dealer_pressure_penalty"], 7)
        self.assertGreater(state["overnight_hedging_risk"], 0.7)

    def test_max_pain_overlay_increases_pinning_near_expiry(self):
        base = build_dealer_hedging_pressure_features(
            spot=22000,
            gamma_regime="LONG_GAMMA_ZONE",
            spot_vs_flip="AT_FLIP",
            gamma_flip_distance_pct=0.12,
            dealer_position="Long Gamma",
            dealer_hedging_bias="PINNING",
            intraday_gamma_state="VOL_SUPPRESSION",
            max_pain_dist=30,
            max_pain_zone="AT_MAX_PAIN",
            days_to_expiry=1,
        )

        with temporary_parameter_pack(
            "disable_max_pain_overlay",
            overrides={"trade_strength.runtime_thresholds.use_max_pain_expiry_overlay": 0},
        ):
            disabled = build_dealer_hedging_pressure_features(
                spot=22000,
                gamma_regime="LONG_GAMMA_ZONE",
                spot_vs_flip="AT_FLIP",
                gamma_flip_distance_pct=0.12,
                dealer_position="Long Gamma",
                dealer_hedging_bias="PINNING",
                intraday_gamma_state="VOL_SUPPRESSION",
                max_pain_dist=30,
                max_pain_zone="AT_MAX_PAIN",
                days_to_expiry=1,
            )

        self.assertGreater(base["max_pain_pinning_boost"], 0.0)
        self.assertEqual(disabled["max_pain_pinning_boost"], 0.0)
        self.assertGreater(base["pinning_base"], disabled["pinning_base"])
