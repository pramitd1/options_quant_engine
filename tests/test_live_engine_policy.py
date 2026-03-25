from __future__ import annotations

import unittest

import pandas as pd

from tuning.runtime import temporary_parameter_pack
from data.provider_normalization import normalize_live_option_chain
from data.option_chain_validation import validate_option_chain
from engine.trading_engine import (
    classify_execution_regime,
    classify_signal_regime,
    classify_spot_vs_flip_for_symbol,
    derive_dealer_pressure_trade_modifiers,
    decide_direction,
    derive_gamma_vol_trade_modifiers,
    derive_global_risk_trade_modifiers,
    derive_option_efficiency_trade_modifiers,
)
from risk import (
    build_dealer_hedging_pressure_state,
    build_gamma_vol_acceleration_state,
    build_option_efficiency_state,
)
from models.large_move_probability import large_move_probability
from risk.global_risk_layer import evaluate_global_risk_layer


class LiveEnginePolicyTests(unittest.TestCase):
    def test_provider_normalization_adds_metadata_and_dedupes(self):
        option_chain = pd.DataFrame(
            [
                {"strikePrice": 22000, "OPTION_TYP": "ce", "lastPrice": "101.5", "openInterest": "1200", "EXPIRY_DT": "2026-03-26"},
                {"strikePrice": 22000, "OPTION_TYP": "ce", "lastPrice": "102.0", "openInterest": "1300", "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22000, "OPTION_TYP": "PE", "LAST_PRICE": "98", "OPEN_INT": "1400", "EXPIRY_DT": "2026-03-26"},
            ]
        )

        normalized = normalize_live_option_chain(option_chain, source="nse", symbol="nifty")

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["source"].iloc[0], "NSE")
        self.assertEqual(normalized["underlying_symbol"].iloc[0], "NIFTY")
        self.assertEqual(set(normalized["OPTION_TYP"].tolist()), {"CE", "PE"})
        self.assertTrue(pd.api.types.is_numeric_dtype(normalized["lastPrice"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(normalized["openInterest"]))

    def test_weighted_direction_policy_accepts_aligned_call(self):
        direction, source = decide_direction(
            final_flow_signal="BULLISH_FLOW",
            dealer_pos="Short Gamma",
            vol_regime="VOL_EXPANSION",
            spot_vs_flip="ABOVE_FLIP",
            gamma_regime="SHORT_GAMMA_ZONE",
            hedging_bias="UPSIDE_ACCELERATION",
            gamma_event="GAMMA_SQUEEZE",
            vanna_regime="POSITIVE_VANNA",
            charm_regime="POSITIVE_CHARM",
        )

        self.assertEqual(direction, "CALL")
        self.assertIn("FLOW", source)
        self.assertIn("HEDGING_BIAS", source)

    def test_weighted_direction_policy_rejects_conflicted_setup(self):
        direction, source = decide_direction(
            final_flow_signal="BULLISH_FLOW",
            dealer_pos="Short Gamma",
            vol_regime="NORMAL_VOL",
            spot_vs_flip="BELOW_FLIP",
            gamma_regime="SHORT_GAMMA_ZONE",
            hedging_bias="UPSIDE_ACCELERATION",
            gamma_event="GAMMA_SQUEEZE",
            vanna_regime=None,
            charm_regime=None,
        )

        self.assertIsNone(direction)
        self.assertIsNone(source)

    def test_direction_policy_uses_rr_oi_pcr_and_flip_drift_votes(self):
        direction, source = decide_direction(
            final_flow_signal=None,
            dealer_pos=None,
            vol_regime=None,
            spot_vs_flip="AT_FLIP",
            gamma_regime=None,
            hedging_bias=None,
            gamma_event=None,
            vanna_regime=None,
            charm_regime=None,
            oi_velocity_score=0.35,
            rr_value=-1.2,
            rr_momentum="FALLING_PUT_SKEW",
            volume_pcr_atm=0.70,
            gamma_flip_drift={"drift": 120.0},
        )

        self.assertEqual(direction, "CALL")
        self.assertIn("OI_VELOCITY", source)
        self.assertIn("RR_SKEW", source)
        self.assertIn("PCR_ATM", source)
        self.assertIn("FLIP_DRIFT", source)

    def test_direction_policy_toggle_can_disable_new_votes(self):
        with temporary_parameter_pack(
            "disable_new_direction_votes",
            overrides={
                "trade_strength.runtime_thresholds.use_oi_velocity_in_direction": 0,
                "trade_strength.runtime_thresholds.use_rr_in_direction": 0,
                "trade_strength.runtime_thresholds.gamma_flip_drift_pts_vote_on": 1,
            },
        ):
            direction, source = decide_direction(
                final_flow_signal=None,
                dealer_pos=None,
                vol_regime=None,
                spot_vs_flip="AT_FLIP",
                gamma_regime=None,
                hedging_bias=None,
                gamma_event=None,
                vanna_regime=None,
                charm_regime=None,
                oi_velocity_score=0.50,
                rr_value=-1.0,
                rr_momentum="FALLING_PUT_SKEW",
                volume_pcr_atm=1.0,
                gamma_flip_drift=None,
            )

        self.assertIsNone(direction)
        self.assertIsNone(source)

    def test_symbol_aware_flip_buffer_changes_classification(self):
        self.assertEqual(classify_spot_vs_flip_for_symbol("NIFTY", 22010, 22030), "AT_FLIP")
        self.assertEqual(classify_spot_vs_flip_for_symbol("RELIANCE", 1502, 1510), "AT_FLIP")
        self.assertEqual(classify_spot_vs_flip_for_symbol("RELIANCE", 1498, 1510), "BELOW_FLIP")

    def test_provider_health_summary_exposes_weak_pairing(self):
        option_chain = pd.DataFrame(
            [
                {"strikePrice": 22000, "OPTION_TYP": "CE", "lastPrice": 100, "source": "NSE"},
                {"strikePrice": 22100, "OPTION_TYP": "CE", "lastPrice": 90, "source": "NSE"},
                {"strikePrice": 22200, "OPTION_TYP": "PE", "lastPrice": 80, "source": "NSE"},
                {"strikePrice": 22300, "OPTION_TYP": "PE", "lastPrice": 70, "source": "NSE"},
                {"strikePrice": 22400, "OPTION_TYP": "CE", "lastPrice": 60, "source": "NSE"},
                {"strikePrice": 22500, "OPTION_TYP": "PE", "lastPrice": 50, "source": "NSE"},
                {"strikePrice": 22600, "OPTION_TYP": "CE", "lastPrice": 40, "source": "NSE"},
                {"strikePrice": 22700, "OPTION_TYP": "PE", "lastPrice": 30, "source": "NSE"},
                {"strikePrice": 22800, "OPTION_TYP": "CE", "lastPrice": 20, "source": "NSE"},
                {"strikePrice": 22900, "OPTION_TYP": "PE", "lastPrice": 10, "source": "NSE"},
                {"strikePrice": 23000, "OPTION_TYP": "CE", "lastPrice": 9, "source": "NSE"},
                {"strikePrice": 23100, "OPTION_TYP": "PE", "lastPrice": 8, "source": "NSE"},
                {"strikePrice": 23200, "OPTION_TYP": "CE", "lastPrice": 7, "source": "NSE"},
                {"strikePrice": 23300, "OPTION_TYP": "PE", "lastPrice": 6, "source": "NSE"},
                {"strikePrice": 23400, "OPTION_TYP": "CE", "lastPrice": 5, "source": "NSE"},
                {"strikePrice": 23500, "OPTION_TYP": "PE", "lastPrice": 4, "source": "NSE"},
                {"strikePrice": 23600, "OPTION_TYP": "CE", "lastPrice": 3, "source": "NSE"},
                {"strikePrice": 23700, "OPTION_TYP": "PE", "lastPrice": 2, "source": "NSE"},
                {"strikePrice": 23800, "OPTION_TYP": "CE", "lastPrice": 1, "source": "NSE"},
                {"strikePrice": 23900, "OPTION_TYP": "PE", "lastPrice": 1, "source": "NSE"},
            ]
        )
        validation = validate_option_chain(option_chain)
        self.assertEqual(validation["provider_health"]["source"], "NSE")
        self.assertEqual(validation["provider_health"]["pairing_health"], "WEAK")
        self.assertEqual(validation["provider_health"]["summary_status"], "WEAK")

    def test_signal_and_execution_regime_classification(self):
        signal_regime = classify_signal_regime(
            direction="CALL",
            adjusted_trade_strength=81,
            final_flow_signal="BULLISH_FLOW",
            gamma_regime="SHORT_GAMMA_ZONE",
            confirmation_status="CONFIRMED",
            event_lockdown_flag=False,
            data_quality_status="GOOD",
        )
        execution_regime = classify_execution_regime(
            trade_status="TRADE",
            signal_regime=signal_regime,
            data_quality_score=88,
            macro_position_size_multiplier=0.7,
        )

        self.assertEqual(signal_regime, "EXPANSION_BIAS")
        self.assertEqual(execution_regime, "RISK_REDUCED")

    def test_global_risk_trade_modifiers_apply_requested_penalties(self):
        modifiers = derive_global_risk_trade_modifiers(
            {
                "global_risk_state": "VOL_SHOCK",
                "global_risk_adjustment_score": -2,
                "overnight_hold_allowed": False,
                "global_risk_features": {
                    "oil_shock_score": 0.8,
                    "commodity_risk_score": 0.6,
                    "volatility_shock_score": 1.0,
                    "volatility_explosion_probability": 0.82,
                },
            }
        )

        self.assertEqual(modifiers["base_adjustment_score"], -2)
        self.assertEqual(modifiers["feature_adjustment_score"], -10)
        self.assertEqual(modifiers["effective_adjustment_score"], -12)
        self.assertTrue(modifiers["overnight_trade_block"])
        self.assertIn("volatility_explosion_probability_high", modifiers["adjustment_reasons"])
        self.assertIn("oil_shock_score_high", modifiers["adjustment_reasons"])

    def test_global_risk_trade_modifiers_force_no_trade_on_event_lockdown(self):
        modifiers = derive_global_risk_trade_modifiers(
            {
                "global_risk_state": "EVENT_LOCKDOWN",
                "global_risk_adjustment_score": -4,
                "overnight_hold_allowed": False,
                "global_risk_features": {},
            }
        )

        self.assertTrue(modifiers["force_no_trade"])
        self.assertTrue(modifiers["overnight_trade_block"])

    def test_global_risk_trade_modifiers_force_no_trade_when_global_veto_is_active(self):
        modifiers = derive_global_risk_trade_modifiers(
            {
                "global_risk_state": "VOL_SHOCK",
                "global_risk_veto": True,
                "global_risk_adjustment_score": -8,
                "overnight_hold_allowed": False,
                "global_risk_features": {},
            }
        )

        self.assertTrue(modifiers["force_no_trade"])

    def test_global_risk_trade_modifiers_do_not_penalize_neutralized_market_features(self):
        modifiers = derive_global_risk_trade_modifiers(
            {
                "global_risk_state": "GLOBAL_NEUTRAL",
                "global_risk_adjustment_score": 0,
                "overnight_hold_allowed": True,
                "global_risk_features": {
                    "oil_shock_score": 0.0,
                    "commodity_risk_score": 0.0,
                    "volatility_shock_score": 0.0,
                    "volatility_explosion_probability": 0.0,
                    "market_features_neutralized": True,
                },
            }
        )

        self.assertEqual(modifiers["effective_adjustment_score"], 0)
        self.assertEqual(modifiers["adjustment_reasons"], [])

    def test_gamma_vol_trade_modifiers_reward_aligned_convexity(self):
        gamma_vol_state = build_gamma_vol_acceleration_state(
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="ABOVE_FLIP",
            gamma_flip_distance_pct=0.08,
            dealer_hedging_bias="UPSIDE_ACCELERATION",
            liquidity_vacuum_state="BREAKOUT_ZONE",
            intraday_range_pct=0.85,
            volatility_compression_score=0.7,
            volatility_shock_score=0.3,
            macro_event_risk_score=12,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.52,
        )

        modifiers = derive_gamma_vol_trade_modifiers(gamma_vol_state, direction="CALL")

        self.assertGreaterEqual(modifiers["effective_adjustment_score"], 3)
        self.assertEqual(modifiers["directional_convexity_state"], "UPSIDE_SQUEEZE_RISK")
        self.assertIn("upside_squeeze_alignment", modifiers["adjustment_reasons"])

    def test_gamma_vol_trade_modifiers_penalize_directional_conflict(self):
        gamma_vol_state = build_gamma_vol_acceleration_state(
            gamma_regime="SHORT_GAMMA_ZONE",
            spot_vs_flip="BELOW_FLIP",
            gamma_flip_distance_pct=0.10,
            dealer_hedging_bias="DOWNSIDE_ACCELERATION",
            liquidity_vacuum_state="NEAR_VACUUM",
            intraday_range_pct=0.95,
            volatility_compression_score=0.4,
            volatility_shock_score=0.7,
            macro_event_risk_score=25,
            global_risk_state={"global_risk_state": "RISK_OFF"},
            volatility_explosion_probability=0.62,
        )

        modifiers = derive_gamma_vol_trade_modifiers(gamma_vol_state, direction="CALL")

        self.assertLess(modifiers["effective_adjustment_score"], 0)
        self.assertEqual(modifiers["directional_convexity_state"], "DOWNSIDE_AIRPOCKET_RISK")
        self.assertIn("downside_convexity_conflict", modifiers["adjustment_reasons"])

    def test_dealer_pressure_trade_modifiers_reward_aligned_hedging_pressure(self):
        dealer_state = build_dealer_hedging_pressure_state(
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
            macro_event_risk_score=15,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.4,
            gamma_vol_acceleration_score=72,
        )

        modifiers = derive_dealer_pressure_trade_modifiers(dealer_state, direction="CALL")

        self.assertGreater(modifiers["effective_adjustment_score"], 0)
        self.assertEqual(modifiers["dealer_flow_state"], "UPSIDE_HEDGING_ACCELERATION")
        self.assertIn("upside_hedging_alignment", modifiers["adjustment_reasons"])

    def test_dealer_pressure_trade_modifiers_dampen_pinning_option_buying(self):
        dealer_state = build_dealer_hedging_pressure_state(
            spot=22000,
            gamma_regime="LONG_GAMMA_ZONE",
            spot_vs_flip="ABOVE_FLIP",
            gamma_flip_distance_pct=0.32,
            dealer_position="Long Gamma",
            dealer_hedging_bias="PINNING",
            dealer_hedging_flow="BUY_FUTURES",
            gamma_clusters=[21980, 22010, 22030],
            liquidity_levels=[21990, 22020],
            support_wall=21985,
            resistance_wall=22025,
            liquidity_vacuum_state="NO_VACUUM",
            intraday_gamma_state="VOL_SUPPRESSION",
            intraday_range_pct=0.22,
            flow_signal="NEUTRAL_FLOW",
            smart_money_flow="NEUTRAL_FLOW",
            macro_event_risk_score=6,
            global_risk_state={"global_risk_state": "GLOBAL_NEUTRAL"},
            volatility_explosion_probability=0.05,
            gamma_vol_acceleration_score=16,
        )

        modifiers = derive_dealer_pressure_trade_modifiers(dealer_state, direction="CALL")

        self.assertLess(modifiers["effective_adjustment_score"], 0)
        self.assertEqual(modifiers["dealer_flow_state"], "PINNING_DOMINANT")
        self.assertIn("pinning_dampens_option_buying", modifiers["adjustment_reasons"])

    def test_option_efficiency_trade_modifiers_reward_efficient_setup(self):
        state = build_option_efficiency_state(
            spot=22000,
            atm_iv=18.0,
            expiry_value="2026-03-21",
            valuation_time="2026-03-14T10:00:00+05:30",
            direction="CALL",
            strike=22000,
            option_type="CE",
            entry_price=110,
            target=145,
            stop_loss=92,
            gamma_vol_acceleration_score=72,
            dealer_hedging_pressure_score=66,
            liquidity_vacuum_state="BREAKOUT_ZONE",
        )

        modifiers = derive_option_efficiency_trade_modifiers(state)

        self.assertGreater(modifiers["option_efficiency_score"], 60)
        self.assertGreater(modifiers["effective_adjustment_score"], 0)
        self.assertTrue(modifiers["overnight_hold_allowed"])

    def test_option_efficiency_trade_modifiers_penalize_poor_overnight_setup(self):
        state = build_option_efficiency_state(
            spot=22000,
            atm_iv=13.0,
            expiry_value="2026-03-17",
            valuation_time="2026-03-14T15:10:00+05:30",
            direction="CALL",
            strike=22300,
            option_type="CE",
            entry_price=145,
            target=210,
            stop_loss=105,
            holding_profile="OVERNIGHT",
            global_risk_state={
                "global_risk_state": "GLOBAL_NEUTRAL",
                "holding_context": {
                    "holding_profile": "OVERNIGHT",
                    "overnight_relevant": True,
                },
            },
        )

        modifiers = derive_option_efficiency_trade_modifiers(state)

        self.assertLess(modifiers["effective_adjustment_score"], 0)
        self.assertFalse(modifiers["overnight_hold_allowed"])
        self.assertGreater(modifiers["overnight_option_efficiency_penalty"], 0)

    def test_execution_regime_treats_no_trade_as_blocked(self):
        execution_regime = classify_execution_regime(
            trade_status="NO_TRADE",
            signal_regime="LOCKDOWN",
            data_quality_score=92,
            macro_position_size_multiplier=1.0,
        )

        self.assertEqual(execution_regime, "BLOCKED")

    def test_global_risk_layer_blocks_event_lockdown(self):
        decision = evaluate_global_risk_layer(
            data_quality={"score": 92, "status": "STRONG", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=78,
            min_trade_strength=45,
            event_window_status="PRE_EVENT_LOCKDOWN",
            macro_event_risk_score=85,
            event_lockdown_flag=True,
            next_event_name="RBI Policy",
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 0.0, "event_lockdown_flag": False},
        )

        self.assertEqual(decision["risk_trade_status"], "EVENT_LOCKDOWN")
        self.assertEqual(decision["global_risk_action"], "BLOCK")
        self.assertEqual(decision["global_risk_level"], "BLOCKED")

    def test_global_risk_layer_reduces_size_without_blocking(self):
        decision = evaluate_global_risk_layer(
            data_quality={"score": 88, "status": "GOOD", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=72,
            min_trade_strength=45,
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=10,
            event_lockdown_flag=False,
            next_event_name=None,
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 0.8, "event_lockdown_flag": False},
        )

        self.assertIsNone(decision["risk_trade_status"])
        self.assertEqual(decision["global_risk_action"], "REDUCE")
        self.assertEqual(decision["global_risk_size_cap"], 0.8)

    def test_large_move_probability_recognizes_short_gamma_zone(self):
        short_gamma_prob = large_move_probability(
            "SHORT_GAMMA_ZONE",
            "BREAKOUT_ZONE",
            "UPSIDE_ACCELERATION",
            "BULLISH_FLOW",
        )
        long_gamma_prob = large_move_probability(
            "LONG_GAMMA_ZONE",
            "BREAKOUT_ZONE",
            "UPSIDE_ACCELERATION",
            "BULLISH_FLOW",
        )

        self.assertGreater(short_gamma_prob, long_gamma_prob)


if __name__ == "__main__":
    unittest.main()
