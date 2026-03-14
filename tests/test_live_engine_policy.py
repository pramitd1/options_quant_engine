from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.provider_normalization import normalize_live_option_chain
from data.option_chain_validation import validate_option_chain
from engine.trading_engine import (
    classify_execution_regime,
    classify_signal_regime,
    classify_spot_vs_flip_for_symbol,
    decide_direction,
)


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


if __name__ == "__main__":
    unittest.main()
