from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from data.global_market_snapshot import build_global_market_snapshot
from risk import build_global_risk_state
from risk.global_risk_features import build_global_risk_features


class GlobalRiskFeatureModelTests(unittest.TestCase):
    def test_build_global_market_snapshot_returns_neutral_when_disabled(self):
        with patch("data.global_market_snapshot.GLOBAL_MARKET_DATA_ENABLED", False):
            snapshot = build_global_market_snapshot("NIFTY", as_of="2026-03-14T10:00:00+05:30")

        self.assertFalse(snapshot["data_available"])
        self.assertTrue(snapshot["neutral_fallback"])
        self.assertIn("global_market_data_disabled", snapshot["warnings"])
        self.assertEqual(snapshot["market_inputs"], {})

    def test_build_global_market_snapshot_batches_downloads(self):
        download_calls = []

        def fake_download(tickers, **kwargs):
            download_calls.append(tickers)
            index = pd.to_datetime(["2026-03-13", "2026-03-14"], utc=True)
            tickers_list = tickers if isinstance(tickers, list) else [tickers]
            payload = {
                (ticker, "Close"): [100.0 + idx, 101.0 + idx]
                for idx, ticker in enumerate(tickers_list)
            }
            frame = pd.DataFrame(payload, index=index)
            frame.index.name = "Date"
            frame.columns = pd.MultiIndex.from_tuples(frame.columns)
            return frame

        with patch("data.global_market_snapshot.yf.download", side_effect=fake_download):
            snapshot = build_global_market_snapshot("NIFTY", as_of="2026-03-14T10:00:00+05:30")

        self.assertEqual(len(download_calls), 1)
        self.assertTrue(snapshot["market_inputs"]["oil_change_24h"] is not None)
        self.assertTrue(snapshot["market_inputs"]["sp500_change_24h"] is not None)

    def test_build_global_risk_features_computes_cross_asset_scores(self):
        features = build_global_risk_features(
            macro_event_state={
                "macro_event_risk_score": 65,
                "event_window_status": "PRE_EVENT_WATCH",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": False,
                "news_confidence_score": 75,
                "headline_velocity": 0.4,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 8.0,
                    "gold_change_24h": 2.4,
                    "copper_change_24h": -4.0,
                    "vix_change_24h": 12.0,
                    "sp500_change_24h": -1.5,
                    "nasdaq_change_24h": -0.8,
                    "us10y_change_bp": 12.0,
                    "usdinr_change_24h": 0.9,
                    "realized_vol_5d": 0.12,
                    "realized_vol_30d": 0.25,
                },
            },
            holding_profile="AUTO",
            as_of="2026-03-14T10:00:00+05:30",
        )

        self.assertEqual(features["oil_shock_score"], 1.0)
        self.assertEqual(features["gold_risk_score"], 0.5)
        self.assertEqual(features["copper_growth_signal"], -0.6)
        self.assertAlmostEqual(features["commodity_risk_score"], 0.53, places=4)
        self.assertEqual(features["volatility_shock_score"], 0.7)
        self.assertEqual(features["us_equity_risk_score"], 0.4)
        self.assertEqual(features["rates_shock_score"], 0.6)
        self.assertEqual(features["currency_shock_score"], 0.5)
        self.assertEqual(features["volatility_compression_score"], 0.7)
        self.assertAlmostEqual(features["risk_off_intensity"], 0.614, places=3)
        self.assertAlmostEqual(features["volatility_explosion_probability"], 0.945, places=3)
        self.assertTrue(features["market_data_available"])
        self.assertFalse(features["neutral_fallback"])

    def test_build_global_risk_features_neutralizes_stale_market_snapshot(self):
        features = build_global_risk_features(
            macro_event_state={
                "macro_event_risk_score": 0,
                "event_window_status": "NO_EVENT_DATA",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": False,
                "news_confidence_score": 70,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": False,
                "neutral_fallback": True,
                "stale": True,
                "issues": [],
                "warnings": ["global_market_snapshot_stale:6d"],
                "market_inputs": {
                    "oil_change_24h": 9.0,
                    "gold_change_24h": 3.5,
                    "copper_change_24h": -5.5,
                    "vix_change_24h": 19.0,
                    "sp500_change_24h": -2.8,
                    "nasdaq_change_24h": -2.4,
                    "us10y_change_bp": 15.0,
                    "usdinr_change_24h": 1.0,
                    "realized_vol_5d": 0.10,
                    "realized_vol_30d": 0.30,
                },
            },
            holding_profile="AUTO",
            as_of="2026-03-14T10:00:00+05:30",
        )

        self.assertFalse(features["market_data_available"])
        self.assertTrue(features["market_features_neutralized"])
        self.assertEqual(features["market_neutralization_reason"], "market_data_stale")
        self.assertEqual(features["market_feature_confidence"], 0.0)
        self.assertEqual(features["oil_shock_score"], 0.0)
        self.assertEqual(features["volatility_shock_score"], 0.0)
        self.assertEqual(features["volatility_explosion_probability"], 0.0)
        self.assertEqual(features["raw_market_inputs"]["oil_change_24h"], 9.0)
        self.assertIn("market_features_neutralized:market_data_stale", features["warnings"])

    def test_build_global_risk_state_tracks_partial_market_input_confidence(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 12,
                "event_window_status": "NO_EVENT_DATA",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": False,
                "news_confidence_score": 72,
                "headline_velocity": 0.2,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 8.0,
                    "gold_change_24h": None,
                    "copper_change_24h": None,
                    "vix_change_24h": 12.0,
                    "sp500_change_24h": None,
                    "nasdaq_change_24h": None,
                    "us10y_change_bp": None,
                    "usdinr_change_24h": None,
                    "realized_vol_5d": 0.12,
                    "realized_vol_30d": 0.24,
                },
            },
            holding_profile="AUTO",
            as_of="2026-03-14T10:00:00+05:30",
        )

        diagnostics = state["global_risk_diagnostics"]
        self.assertAlmostEqual(diagnostics["market_feature_confidence"], 0.4, places=4)
        self.assertFalse(diagnostics["market_features_neutralized"])
        self.assertEqual(diagnostics["market_input_availability"]["oil_change_24h"], True)
        self.assertEqual(diagnostics["market_input_availability"]["gold_change_24h"], False)
        self.assertIn("dominant_risk_driver", diagnostics)

    def test_build_global_risk_state_uses_market_shocks_even_when_news_is_neutral(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 55,
                "event_window_status": "PRE_EVENT_WATCH",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": True,
                "news_confidence_score": 0,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 8.2,
                    "gold_change_24h": 2.8,
                    "copper_change_24h": -5.1,
                    "vix_change_24h": 16.0,
                    "sp500_change_24h": -2.4,
                    "nasdaq_change_24h": -2.1,
                    "us10y_change_bp": 13.0,
                    "usdinr_change_24h": 0.8,
                    "realized_vol_5d": 0.10,
                    "realized_vol_30d": 0.25,
                },
            },
            holding_profile="OVERNIGHT",
            as_of="2026-03-14T15:15:00+05:30",
        )

        self.assertFalse(state["neutral_fallback"])
        self.assertEqual(state["global_risk_state"], "VOL_SHOCK")
        self.assertGreaterEqual(state["global_risk_score"], 35)
        self.assertGreaterEqual(state["volatility_expansion_risk_score"], 60)
        self.assertFalse(state["overnight_hold_allowed"])

    def test_build_global_risk_state_enters_event_lockdown_on_high_event_risk(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 78,
                "event_window_status": "PRE_EVENT_WATCH",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": False,
                "news_confidence_score": 65,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 0.0,
                    "gold_change_24h": 0.0,
                    "copper_change_24h": 0.0,
                    "vix_change_24h": 2.0,
                    "sp500_change_24h": -0.2,
                    "nasdaq_change_24h": -0.1,
                    "us10y_change_bp": 2.0,
                    "usdinr_change_24h": 0.1,
                    "realized_vol_5d": 0.20,
                    "realized_vol_30d": 0.22,
                },
            },
            holding_profile="OVERNIGHT",
            as_of="2026-03-14T15:15:00+05:30",
        )

        self.assertEqual(state["global_risk_state"], "EVENT_LOCKDOWN")
        self.assertFalse(state["overnight_hold_allowed"])

    def test_build_global_risk_state_can_mark_risk_on_when_cross_asset_risk_is_supportive(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 5,
                "event_window_status": "NO_EVENT_DATA",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "RISK_ON",
                "macro_sentiment_score": 26,
                "global_risk_bias": 0.7,
                "neutral_fallback": False,
                "news_confidence_score": 85,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": -6.0,
                    "gold_change_24h": -0.5,
                    "copper_change_24h": -1.0,
                    "vix_change_24h": -4.0,
                    "sp500_change_24h": 1.4,
                    "nasdaq_change_24h": 1.8,
                    "us10y_change_bp": -4.0,
                    "usdinr_change_24h": -0.2,
                    "realized_vol_5d": 0.18,
                    "realized_vol_30d": 0.19,
                },
            },
            holding_profile="AUTO",
            as_of="2026-03-14T11:00:00+05:30",
        )

        self.assertEqual(state["global_risk_state"], "RISK_ON")
        self.assertTrue(state["overnight_hold_allowed"])
        self.assertLess(state["global_risk_diagnostics"]["regime_score"], -0.3)

    def test_build_global_risk_state_assigns_overnight_penalty_for_oil_and_us_equity_stress(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 10,
                "event_window_status": "NO_EVENT_DATA",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": False,
                "news_confidence_score": 70,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 5.0,
                    "gold_change_24h": 0.0,
                    "copper_change_24h": -1.0,
                    "vix_change_24h": 3.0,
                    "sp500_change_24h": -2.2,
                    "nasdaq_change_24h": -1.1,
                    "us10y_change_bp": 4.0,
                    "usdinr_change_24h": 0.2,
                    "realized_vol_5d": 0.20,
                    "realized_vol_30d": 0.22,
                },
            },
            holding_profile="OVERNIGHT",
            as_of="2026-03-14T15:10:00+05:30",
        )

        self.assertTrue(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "oil_shock_elevated")
        self.assertEqual(state["overnight_risk_penalty"], 6)

    def test_build_global_risk_state_blocks_overnight_when_combined_penalty_is_extreme(self):
        state = build_global_risk_state(
            macro_event_state={
                "macro_event_risk_score": 50,
                "event_window_status": "PRE_EVENT_WATCH",
                "event_lockdown_flag": False,
                "event_data_available": True,
            },
            macro_news_state={
                "macro_regime": "MACRO_NEUTRAL",
                "neutral_fallback": False,
                "news_confidence_score": 80,
            },
            global_market_snapshot={
                "provider": "TEST",
                "data_available": True,
                "neutral_fallback": False,
                "stale": False,
                "issues": [],
                "warnings": [],
                "market_inputs": {
                    "oil_change_24h": 5.5,
                    "gold_change_24h": 0.0,
                    "copper_change_24h": -1.0,
                    "vix_change_24h": 4.0,
                    "sp500_change_24h": -2.3,
                    "nasdaq_change_24h": -1.3,
                    "us10y_change_bp": 5.0,
                    "usdinr_change_24h": 0.3,
                    "realized_vol_5d": 0.20,
                    "realized_vol_30d": 0.24,
                },
            },
            holding_profile="OVERNIGHT",
            as_of="2026-03-14T15:12:00+05:30",
        )

        self.assertFalse(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "macro_event_risk_elevated")
        self.assertEqual(state["overnight_risk_penalty"], 8)


if __name__ == "__main__":
    unittest.main()
