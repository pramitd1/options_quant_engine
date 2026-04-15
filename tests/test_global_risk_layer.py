from __future__ import annotations

import unittest

from risk import build_global_risk_state, evaluate_global_risk_layer
from risk.global_risk_layer import _fallback_global_risk_state


class GlobalRiskLayerStageOneTests(unittest.TestCase):
    def test_fallback_global_risk_preserves_overnight_relevance_for_auto_profile(self):
        state = _fallback_global_risk_state(
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=0,
            event_lockdown_flag=False,
            holding_profile="AUTO",
        )
        self.assertTrue(state["holding_context"]["overnight_relevant"])

    def test_build_global_risk_state_gracefully_falls_back_to_neutral(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 0,
                "event_window_status": "NO_EVENT_DATA",
                "event_lockdown_flag": False,
                "event_data_available": False,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": True,
                "news_confidence_score": 0,
            },
            holding_profile="AUTO",
            as_of="2026-03-14T10:00:00+05:30",
        )

        self.assertEqual(state["global_risk_state"], "GLOBAL_NEUTRAL")
        self.assertEqual(state["global_risk_score"], 0)
        self.assertEqual(state["global_risk_adjustment_score"], 0)
        self.assertTrue(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "overnight_risk_contained")
        self.assertEqual(state["overnight_risk_penalty"], 0)
        self.assertTrue(state["neutral_fallback"])

    def test_build_global_risk_state_blocks_extreme_overnight_context(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 75,
                "event_window_status": "PRE_EVENT_WATCH",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "RISK_OFF",
                "macro_sentiment_score": -28,
                "volatility_shock_score": 90,
                "news_confidence_score": 88,
                "headline_velocity": 1.0,
                "global_risk_bias": -0.8,
                "headline_count": 6,
                "classified_headline_count": 6,
                "neutral_fallback": False,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 8.5,
                    "gold_change_24h": 3.4,
                    "copper_change_24h": -5.2,
                    "vix_change_24h": 18.0,
                    "sp500_change_24h": -2.5,
                    "nasdaq_change_24h": -2.2,
                    "us10y_change_bp": 14.0,
                    "usdinr_change_24h": 0.9,
                    "realized_vol_5d": 0.11,
                    "realized_vol_30d": 0.28,
                },
            },
            holding_profile="OVERNIGHT",
            as_of="2026-03-14T15:10:00+05:30",
        )

        self.assertEqual(state["global_risk_state"], "VOL_SHOCK")
        self.assertFalse(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "vol_shock_block")
        self.assertEqual(state["overnight_risk_penalty"], 10)
        self.assertTrue(state["global_risk_veto"])
        self.assertGreaterEqual(state["overnight_gap_risk_score"], 79)

    def test_evaluate_global_risk_layer_blocks_when_global_risk_veto_is_active(self):
        global_risk_state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 80,
                "event_window_status": "PRE_EVENT_WATCH",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "RISK_OFF",
                "macro_sentiment_score": -32,
                "volatility_shock_score": 92,
                "news_confidence_score": 90,
                "headline_velocity": 1.0,
                "global_risk_bias": -0.85,
                "neutral_fallback": False,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 8.8,
                    "gold_change_24h": 3.5,
                    "copper_change_24h": -5.4,
                    "vix_change_24h": 19.0,
                    "sp500_change_24h": -2.7,
                    "nasdaq_change_24h": -2.4,
                    "us10y_change_bp": 16.0,
                    "usdinr_change_24h": 1.0,
                    "realized_vol_5d": 0.10,
                    "realized_vol_30d": 0.30,
                },
            },
            holding_profile="OVERNIGHT",
            as_of="2026-03-14T15:20:00+05:30",
        )

        decision = evaluate_global_risk_layer(
            data_quality={"score": 90, "status": "GOOD", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=82,
            min_trade_strength=45,
            event_window_status="PRE_EVENT_WATCH",
            macro_event_risk_score=80,
            event_lockdown_flag=False,
            next_event_name="US CPI",
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 1.0, "event_lockdown_flag": False},
            global_risk_state=global_risk_state,
            holding_profile="OVERNIGHT",
        )

        self.assertEqual(decision["risk_trade_status"], "GLOBAL_RISK_BLOCKED")
        self.assertEqual(decision["global_risk_action"], "BLOCK")
        self.assertEqual(decision["global_risk_state"], "VOL_SHOCK")

    def test_evaluate_global_risk_layer_combines_global_and_macro_size_caps(self):
        global_risk_state = {
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 40,
            "overnight_gap_risk_score": 20,
            "volatility_expansion_risk_score": 35,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_risk_contained",
            "overnight_risk_penalty": 0,
            "global_risk_adjustment_score": -2,
            "global_risk_veto": False,
            "global_risk_position_size_multiplier": 0.85,
            "global_risk_reasons": ["volatility_expansion_elevated"],
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "holding_context": {"overnight_relevant": False},
        }

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
            global_risk_state=global_risk_state,
            holding_profile="AUTO",
        )

        self.assertEqual(decision["global_risk_action"], "REDUCE")
        self.assertEqual(decision["global_risk_size_cap"], 0.8)
        self.assertEqual(decision["global_risk_adjustment_score"], -2)

    def test_evaluate_global_risk_layer_watchlists_large_macro_uncertainty_window(self):
        global_risk_state = {
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 28,
            "overnight_gap_risk_score": 10,
            "volatility_expansion_risk_score": 20,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_risk_contained",
            "overnight_risk_penalty": 0,
            "global_risk_adjustment_score": 0,
            "global_risk_veto": False,
            "global_risk_position_size_multiplier": 1.0,
            "global_risk_reasons": [],
            "global_risk_features": {
                "macro_uncertainty_score": 0.82,
                "headline_data_stale": True,
                "global_macro_data_stale": True,
                "event_uncertainty_score": 0.75,
            },
            "global_risk_diagnostics": {},
            "holding_context": {"overnight_relevant": False},
        }

        decision = evaluate_global_risk_layer(
            data_quality={"score": 91, "status": "GOOD", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=84,
            min_trade_strength=45,
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=10,
            event_lockdown_flag=False,
            next_event_name=None,
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 1.0, "event_lockdown_flag": False},
            global_risk_state=global_risk_state,
            holding_profile="AUTO",
        )

        self.assertEqual(decision["risk_trade_status"], "WATCHLIST")
        self.assertEqual(decision["global_risk_action"], "WATCHLIST")
        self.assertIn("macro_uncertainty_window", decision["global_risk_reasons"])

    def test_evaluate_global_risk_layer_watchlists_when_same_way_options_book_is_overcrowded(self):
        global_risk_state = {
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 24,
            "overnight_gap_risk_score": 8,
            "volatility_expansion_risk_score": 15,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_risk_contained",
            "overnight_risk_penalty": 0,
            "global_risk_adjustment_score": 0,
            "global_risk_veto": False,
            "global_risk_position_size_multiplier": 1.0,
            "global_risk_reasons": [],
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "holding_context": {"overnight_relevant": False},
        }

        decision = evaluate_global_risk_layer(
            data_quality={"score": 93, "status": "GOOD", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=86,
            min_trade_strength=45,
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=8,
            event_lockdown_flag=False,
            next_event_name=None,
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 1.0, "event_lockdown_flag": False},
            global_risk_state=global_risk_state,
            holding_profile="AUTO",
            portfolio_context={
                "direction": "CALL",
                "recent_signal_count": 5,
                "same_direction_count": 4,
                "same_direction_share": 0.8,
                "same_direction_avg_close_bps": -18.0,
                "same_direction_avg_tradeability_score": 49.0,
            },
        )

        self.assertEqual(decision["risk_trade_status"], "WATCHLIST")
        self.assertEqual(decision["global_risk_action"], "WATCHLIST")
        self.assertIn("portfolio_same_way_concentration", decision["global_risk_reasons"])

    def test_evaluate_global_risk_layer_reduces_size_when_same_way_book_is_elevated(self):
        global_risk_state = {
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 18,
            "overnight_gap_risk_score": 5,
            "volatility_expansion_risk_score": 10,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_risk_contained",
            "overnight_risk_penalty": 0,
            "global_risk_adjustment_score": 0,
            "global_risk_veto": False,
            "global_risk_position_size_multiplier": 1.0,
            "global_risk_reasons": [],
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "holding_context": {"overnight_relevant": False},
        }

        decision = evaluate_global_risk_layer(
            data_quality={"score": 92, "status": "GOOD", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=78,
            min_trade_strength=45,
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=5,
            event_lockdown_flag=False,
            next_event_name=None,
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 0.95, "event_lockdown_flag": False},
            global_risk_state=global_risk_state,
            holding_profile="AUTO",
            portfolio_context={
                "direction": "PUT",
                "recent_signal_count": 4,
                "same_direction_count": 3,
                "same_direction_share": 0.75,
                "same_direction_avg_close_bps": 6.0,
                "same_direction_avg_tradeability_score": 58.0,
            },
        )

        self.assertEqual(decision["global_risk_action"], "REDUCE")
        self.assertLess(decision["global_risk_size_cap"], 0.95)
        self.assertIn("portfolio_same_way_concentration", decision["global_risk_reasons"])

    def test_evaluate_global_risk_layer_escalates_book_heat_in_toxic_regime(self):
        global_risk_state = {
            "global_risk_state": "RISK_OFF",
            "global_risk_score": 44,
            "overnight_gap_risk_score": 18,
            "volatility_expansion_risk_score": 32,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_risk_contained",
            "overnight_risk_penalty": 0,
            "global_risk_adjustment_score": -2,
            "global_risk_veto": False,
            "global_risk_position_size_multiplier": 0.9,
            "global_risk_reasons": ["risk_off_pressure"],
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "holding_context": {"overnight_relevant": False},
        }

        decision = evaluate_global_risk_layer(
            data_quality={"score": 78, "status": "CAUTION", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=82,
            min_trade_strength=45,
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=16,
            event_lockdown_flag=False,
            next_event_name=None,
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 0.9, "event_lockdown_flag": False},
            global_risk_state=global_risk_state,
            holding_profile="AUTO",
            portfolio_context={
                "direction": "PUT",
                "recent_signal_count": 5,
                "same_direction_count": 3,
                "same_direction_share": 0.60,
                "same_direction_avg_close_bps": 4.0,
                "same_direction_avg_tradeability_score": 57.0,
                "gamma_regime": "NEGATIVE_GAMMA",
                "vol_regime": "VOL_EXPANSION",
                "macro_regime": "RISK_OFF",
                "provider_health_summary": "CAUTION",
                "data_quality_status": "CAUTION",
            },
        )

        self.assertEqual(decision["global_risk_action"], "REDUCE")
        self.assertIn(decision["portfolio_concentration_guard"]["heat_label"], {"HOT", "CRITICAL"})
        self.assertGreaterEqual(decision["portfolio_concentration_guard"]["heat_score"], 60)
        self.assertIn("portfolio_regime_heat", decision["global_risk_reasons"])

    def test_evaluate_global_risk_layer_watchlists_when_book_heat_turns_critical(self):
        global_risk_state = {
            "global_risk_state": "RISK_OFF",
            "global_risk_score": 46,
            "overnight_gap_risk_score": 20,
            "volatility_expansion_risk_score": 35,
            "overnight_hold_allowed": True,
            "overnight_hold_reason": "overnight_risk_contained",
            "overnight_risk_penalty": 0,
            "global_risk_adjustment_score": -3,
            "global_risk_veto": False,
            "global_risk_position_size_multiplier": 0.85,
            "global_risk_reasons": ["risk_off_pressure"],
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "holding_context": {"overnight_relevant": False},
        }

        decision = evaluate_global_risk_layer(
            data_quality={"score": 71, "status": "CAUTION", "fatal": False},
            confirmation={"status": "CONFIRMED", "veto": False},
            adjusted_trade_strength=85,
            min_trade_strength=45,
            event_window_status="NO_EVENT_DATA",
            macro_event_risk_score=20,
            event_lockdown_flag=False,
            next_event_name=None,
            active_event_name=None,
            macro_news_adjustments={"macro_position_size_multiplier": 0.85, "event_lockdown_flag": False},
            global_risk_state=global_risk_state,
            holding_profile="AUTO",
            portfolio_context={
                "direction": "PUT",
                "recent_signal_count": 5,
                "same_direction_count": 4,
                "same_direction_share": 0.8,
                "same_direction_avg_close_bps": -12.0,
                "same_direction_avg_tradeability_score": 48.0,
                "gamma_regime": "NEGATIVE_GAMMA",
                "vol_regime": "VOL_EXPANSION",
                "macro_regime": "RISK_OFF",
                "provider_health_summary": "WEAK",
                "data_quality_status": "CAUTION",
            },
        )

        self.assertEqual(decision["risk_trade_status"], "WATCHLIST")
        self.assertEqual(decision["portfolio_concentration_guard"]["heat_label"], "CRITICAL")
        self.assertIn("portfolio_regime_heat", decision["global_risk_reasons"])


if __name__ == "__main__":
    unittest.main()
