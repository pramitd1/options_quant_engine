from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.macro_news_scenario_runner import run_scenario
from config.settings import BASE_DIR
from macro.engine_adjustments import compute_macro_news_adjustments
from macro.scope_utils import headline_mentions_symbol
from news.classifier import classify_headline
from news.models import HeadlineRecord, coerce_headline_timestamp


SCENARIO_FILE = Path(BASE_DIR) / "config/macro_news_scenarios.json"


def _load_scenarios():
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


class MacroNewsLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = {item["name"]: item for item in _load_scenarios()}

    def test_headline_classification_policy(self):
        record = HeadlineRecord(
            timestamp=coerce_headline_timestamp("2026-03-13T09:05:00+05:30"),
            source="TEST",
            headline="RBI Governor says inflation trajectory remains watchful",
            url_or_identifier="test-rbi-001",
            category="MACRO",
        )
        result = classify_headline(record)
        self.assertEqual(result.primary_category, "policy")
        self.assertGreater(result.volatility_shock_score, 0)
        self.assertGreater(result.headline_impact_score, 0)

    def test_neutral_day_scenario(self):
        result = run_scenario(self.scenarios["neutral_day"])
        state = result["macro_news_state"]
        self.assertEqual(state["macro_regime"], "MACRO_NEUTRAL")
        self.assertTrue(state["neutral_fallback"])
        self.assertEqual(state["headline_count"], 0)

    def test_rbi_event_window_scenario(self):
        result = run_scenario(self.scenarios["rbi_event_window"])
        self.assertEqual(result["event_state"]["event_window_status"], "PRE_EVENT_LOCKDOWN")
        self.assertTrue(result["macro_news_state"]["event_lockdown_flag"])
        self.assertEqual(result["macro_news_state"]["macro_regime"], "EVENT_LOCKDOWN")

    def test_risk_off_geopolitical_burst_scenario(self):
        result = run_scenario(self.scenarios["risk_off_geopolitical_burst"])
        state = result["macro_news_state"]
        self.assertEqual(state["macro_regime"], "RISK_OFF")
        self.assertIn("volatility_shock_high", state["macro_regime_reasons"])
        self.assertGreater(state["volatility_shock_score"], 0)
        self.assertGreater(state["news_confidence_score"], 0)
        self.assertGreaterEqual(state["headline_count"], 3)

    def test_risk_on_soft_landing_scenario(self):
        result = run_scenario(self.scenarios["risk_on_soft_landing"])
        state = result["macro_news_state"]
        self.assertEqual(state["macro_regime"], "RISK_ON")
        self.assertIn("sentiment_risk_on", state["macro_regime_reasons"])

    def test_stale_news_feed_scenario(self):
        result = run_scenario(self.scenarios["stale_news_feed"])
        state = result["macro_news_state"]
        self.assertTrue(state["neutral_fallback"])
        self.assertEqual(state["macro_regime"], "MACRO_NEUTRAL")
        self.assertIn("neutral_fallback", state["macro_regime_reasons"])

    def test_engine_adjustment_risk_off_conflicting_call(self):
        adjustments = compute_macro_news_adjustments(
            direction="CALL",
            macro_news_state={
                "macro_regime": "RISK_OFF",
                "macro_sentiment_score": -30,
                "volatility_shock_score": 72,
                "news_confidence_score": 65,
                "event_lockdown_flag": False,
                "neutral_fallback": False,
            },
        )
        self.assertLess(adjustments["macro_adjustment_score"], 0)
        self.assertLess(adjustments["macro_confirmation_adjustment"], 0)
        self.assertLess(adjustments["macro_position_size_multiplier"], 1.0)

    def test_engine_adjustment_event_lockdown(self):
        adjustments = compute_macro_news_adjustments(
            direction="PUT",
            macro_news_state={
                "macro_regime": "EVENT_LOCKDOWN",
                "macro_sentiment_score": 0,
                "volatility_shock_score": 80,
                "news_confidence_score": 55,
                "event_lockdown_flag": True,
                "neutral_fallback": False,
            },
        )
        self.assertTrue(adjustments["event_lockdown_flag"])
        self.assertEqual(adjustments["macro_position_size_multiplier"], 0.0)

    def test_engine_graceful_degradation_when_news_missing(self):
        adjustments = compute_macro_news_adjustments(
            direction="CALL",
            macro_news_state=None,
        )
        self.assertEqual(adjustments["macro_regime"], "MACRO_NEUTRAL")
        self.assertEqual(adjustments["macro_adjustment_score"], 0)
        self.assertEqual(adjustments["macro_position_size_multiplier"], 1.0)

    def test_symbol_relevance_helper(self):
        self.assertTrue(headline_mentions_symbol("NIFTY", "Nifty gains as bond yields cool"))
        self.assertTrue(headline_mentions_symbol("BANKNIFTY", "Financials rally lifts Bank Nifty sentiment"))
        self.assertTrue(headline_mentions_symbol("RELIANCE", "Reliance board approves capex update"))
        self.assertFalse(headline_mentions_symbol("RELIANCE", "Infosys board discusses buyback plan"))


if __name__ == "__main__":
    unittest.main()
