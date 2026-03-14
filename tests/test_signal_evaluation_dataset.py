from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.signal_evaluation.dataset import ensure_signals_dataset_exists, load_signals_dataset
from research.signal_evaluation.evaluator import (
    build_signal_evaluation_row,
    build_regime_fingerprint,
    evaluate_signal_outcomes,
    save_signal_evaluation,
    update_signal_dataset_outcomes,
)
from research.signal_evaluation.policy import (
    CAPTURE_POLICY_ACTIONABLE,
    CAPTURE_POLICY_ALL,
    CAPTURE_POLICY_TRADE_ONLY,
    normalize_capture_policy,
    should_capture_signal,
)


class SignalEvaluationDatasetTests(unittest.TestCase):
    def _sample_result(self):
        return {
            "source": "NSE",
            "mode": "LIVE",
            "symbol": "NIFTY",
            "spot_summary": {
                "spot": 22000.0,
                "day_open": 21940.0,
                "day_high": 22050.0,
                "day_low": 21910.0,
                "prev_close": 21900.0,
                "timestamp": "2026-03-14T10:00:00+05:30",
                "lookback_avg_range_pct": 0.92,
                "ticker": "^NSEI",
            },
            "saved_paths": {
                "spot": "debug_samples/spot.json",
                "chain": "debug_samples/chain.csv",
            },
            "option_chain_validation": {
                "provider_health": {
                    "summary_status": "GOOD",
                    "row_health": "GOOD",
                    "pricing_health": "GOOD",
                    "pairing_health": "GOOD",
                    "iv_health": "CAUTION",
                    "duplicate_health": "GOOD",
                }
            },
            "trade": {
                "selected_expiry": "2026-03-26",
                "direction": "CALL",
                "option_type": "CE",
                "strike": 22000,
                "entry_price": 110.5,
                "target": 143.65,
                "stop_loss": 93.93,
                "trade_strength": 81,
                "signal_quality": "STRONG",
                "signal_regime": "EXPANSION_BIAS",
                "execution_regime": "ACTIVE",
                "trade_status": "TRADE",
                "direction_source": "FLOW+HEDGING_BIAS",
                "final_flow_signal": "BULLISH_FLOW",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "spot_vs_flip": "ABOVE_FLIP",
                "macro_regime": "MACRO_NEUTRAL",
                "dealer_position": "Short Gamma",
                "dealer_hedging_bias": "UPSIDE_ACCELERATION",
                "volatility_regime": "VOL_EXPANSION",
                "liquidity_vacuum_state": "BREAKOUT_ZONE",
                "confirmation_status": "CONFIRMED",
                "macro_event_risk_score": 12,
                "data_quality_score": 88,
                "data_quality_status": "STRONG",
                "rule_move_probability": 0.61,
                "hybrid_move_probability": 0.72,
                "ml_move_probability": 0.68,
                "large_move_probability": 0.72,
            },
        }

    def test_build_row_has_stable_primary_key_and_context(self):
        row_a = build_signal_evaluation_row(self._sample_result())
        row_b = build_signal_evaluation_row(self._sample_result())

        self.assertEqual(row_a["signal_id"], row_b["signal_id"])
        self.assertEqual(row_a["symbol"], "NIFTY")
        self.assertEqual(row_a["provider_health_status"], "GOOD")
        self.assertEqual(row_a["move_probability"], 0.72)
        self.assertEqual(row_a["rule_move_probability"], 0.61)
        self.assertTrue(str(row_a["regime_fingerprint"]).startswith("signal_regime="))
        self.assertEqual(len(str(row_a["regime_fingerprint_id"])), 16)
        self.assertEqual(row_a["signal_calibration_bucket"], "80_100")

    def test_regime_fingerprint_is_deterministic(self):
        trade = self._sample_result()["trade"]
        provider_health = self._sample_result()["option_chain_validation"]["provider_health"]
        fp_a, fp_id_a = build_regime_fingerprint(trade, provider_health)
        fp_b, fp_id_b = build_regime_fingerprint(trade, provider_health)

        self.assertEqual(fp_a, fp_b)
        self.assertEqual(fp_id_a, fp_id_b)

    def test_dataset_file_is_created_with_headers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            ensure_signals_dataset_exists(dataset_path)

            self.assertTrue(dataset_path.exists())
            frame = load_signals_dataset(dataset_path)
            self.assertIn("signal_id", frame.columns)
            self.assertIn("move_probability", frame.columns)
            self.assertEqual(len(frame), 0)

    def test_evaluate_signal_outcomes_enriches_row_without_duplication(self):
        row = build_signal_evaluation_row(self._sample_result())
        realized_path = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-14T10:05:00+05:30",
                    "2026-03-14T10:15:00+05:30",
                    "2026-03-14T10:30:00+05:30",
                    "2026-03-14T11:00:00+05:30",
                    "2026-03-14T15:25:00+05:30",
                    "2026-03-15T09:20:00+05:30",
                    "2026-03-15T15:25:00+05:30",
                ],
                "spot": [22020, 22035, 22050, 22010, 22080, 22110, 22140],
            }
        )

        enriched = evaluate_signal_outcomes(row, realized_path, as_of="2026-03-15T15:25:00+05:30")

        self.assertEqual(enriched["outcome_status"], "COMPLETE")
        self.assertEqual(enriched["spot_5m"], 22020)
        self.assertEqual(enriched["spot_close_same_day"], 22080)
        self.assertEqual(enriched["spot_next_open"], 22110)
        self.assertEqual(enriched["spot_next_close"], 22140)
        selfGreater = self.assertGreater
        selfGreater(enriched["realized_return_5m"], 0)
        self.assertGreater(enriched["signed_return_60m_bps"], 0)
        self.assertGreater(enriched["mfe_points"], 0)
        self.assertGreater(enriched["direction_score"], 0)
        self.assertGreater(enriched["magnitude_score"], 0)
        self.assertGreater(enriched["timing_score"], 0)
        self.assertGreater(enriched["tradeability_score"], 0)
        self.assertGreater(enriched["composite_signal_score"], 0)
        self.assertEqual(enriched["correct_session_close"], 1)

    def test_save_signal_evaluation_upserts_by_signal_id(self):
        result = self._sample_result()
        realized_path = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-14T10:15:00+05:30",
                    "2026-03-14T10:30:00+05:30",
                ],
                "spot": [22040, 22060],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            save_signal_evaluation(result, dataset_path=dataset_path)
            save_signal_evaluation(result, dataset_path=dataset_path, realized_spot_path=realized_path)

            frame = load_signals_dataset(dataset_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["signal_id"], build_signal_evaluation_row(result)["signal_id"])
            self.assertEqual(frame.iloc[0]["outcome_status"], "PARTIAL")

    def test_update_dataset_outcomes_merges_updated_rows(self):
        result = self._sample_result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            save_signal_evaluation(result, dataset_path=dataset_path)

            def fake_fetch(symbol, signal_timestamp, as_of=None):
                return pd.DataFrame(
                    {
                        "timestamp": [
                            "2026-03-14T10:05:00+05:30",
                            "2026-03-14T10:15:00+05:30",
                            "2026-03-14T10:30:00+05:30",
                            "2026-03-14T11:00:00+05:30",
                            "2026-03-14T15:25:00+05:30",
                            "2026-03-15T09:20:00+05:30",
                            "2026-03-15T15:25:00+05:30",
                        ],
                        "spot": [22005, 22030, 22020, 22070, 22090, 22110, 22130],
                    }
                )

            frame = update_signal_dataset_outcomes(
                dataset_path=dataset_path,
                as_of="2026-03-15T15:25:00+05:30",
                fetch_spot_path_fn=fake_fetch,
            )

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["outcome_status"], "COMPLETE")
            self.assertFalse(pd.isna(frame.iloc[0]["spot_next_open"]))
            self.assertFalse(pd.isna(frame.iloc[0]["spot_next_close"]))
            self.assertGreater(float(frame.iloc[0]["directional_consistency_score"]), 0)

    def test_capture_policy_switch(self):
        trade = self._sample_result()["trade"]

        self.assertTrue(should_capture_signal(trade, CAPTURE_POLICY_ALL))
        self.assertTrue(should_capture_signal(trade, CAPTURE_POLICY_ACTIONABLE))
        self.assertTrue(should_capture_signal(trade, CAPTURE_POLICY_TRADE_ONLY))

        watchlist_trade = dict(trade)
        watchlist_trade["trade_status"] = "WATCHLIST"
        self.assertTrue(should_capture_signal(watchlist_trade, CAPTURE_POLICY_ALL))
        self.assertTrue(should_capture_signal(watchlist_trade, CAPTURE_POLICY_ACTIONABLE))
        self.assertFalse(should_capture_signal(watchlist_trade, CAPTURE_POLICY_TRADE_ONLY))

        no_signal_trade = dict(trade)
        no_signal_trade["trade_status"] = "NO_SIGNAL"
        self.assertTrue(should_capture_signal(no_signal_trade, CAPTURE_POLICY_ALL))
        self.assertFalse(should_capture_signal(no_signal_trade, CAPTURE_POLICY_ACTIONABLE))
        self.assertFalse(should_capture_signal(no_signal_trade, CAPTURE_POLICY_TRADE_ONLY))

        self.assertEqual(normalize_capture_policy("trade_only"), CAPTURE_POLICY_TRADE_ONLY)
        self.assertEqual(normalize_capture_policy("unknown"), CAPTURE_POLICY_ALL)


if __name__ == "__main__":
    unittest.main()
