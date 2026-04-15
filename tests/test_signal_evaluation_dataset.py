from __future__ import annotations

import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from research.signal_evaluation.dataset import ensure_signals_dataset_exists, load_signals_dataset
from research.signal_evaluation.dataset import write_signals_dataset
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
                "selected_option_last_price": 110.5,
                "selected_option_volume": 138212815,
                "selected_option_open_interest": 8704150,
                "selected_option_iv": 55.79,
                "selected_option_iv_is_proxy": False,
                "selected_option_delta": 0.4735,
                "selected_option_delta_is_proxy": False,
                "selected_option_gamma": 0.0124,
                "selected_option_theta": -0.084,
                "selected_option_vega": 0.221,
                "selected_option_vanna": 0.013,
                "selected_option_charm": -0.009,
                "selected_option_capital_per_lot": 13685.75,
                "selected_option_ba_spread_ratio": 0.012,
                "selected_option_ba_spread_pct": 1.2,
                "selected_option_score": 27.81,
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
                "global_risk_state": "GLOBAL_NEUTRAL",
                "global_risk_score": 24,
                "gamma_vol_acceleration_score": 68,
                "squeeze_risk_state": "HIGH_ACCELERATION_RISK",
                "directional_convexity_state": "UPSIDE_SQUEEZE_RISK",
                "upside_squeeze_risk": 0.74,
                "downside_airpocket_risk": 0.41,
                "overnight_convexity_risk": 0.52,
                "gamma_vol_adjustment_score": 4,
                "dealer_hedging_pressure_score": 66,
                "dealer_flow_state": "UPSIDE_HEDGING_ACCELERATION",
                "upside_hedging_pressure": 0.81,
                "downside_hedging_pressure": 0.32,
                "pinning_pressure_score": 0.18,
                "dealer_pressure_adjustment_score": 3,
                "expected_move_points": 165.4,
                "expected_move_pct": 0.7518,
                "target_reachability_score": 78,
                "premium_efficiency_score": 74,
                "strike_efficiency_score": 78,
                "option_efficiency_score": 77,
                "option_efficiency_adjustment_score": 4,
                "consistency_check_status": "PASS",
                "consistency_check_issue_count": 0,
                "consistency_check_critical_issue_count": 0,
                "consistency_check_escalated": False,
                "consistency_check_findings": [],
                "oil_shock_score": 0.7,
                "commodity_risk_score": 0.53,
                "market_volatility_shock_score": 0.7,
                "volatility_explosion_probability": 0.45,
                "dealer_position": "Short Gamma",
                "dealer_hedging_bias": "UPSIDE_ACCELERATION",
                "dealer_hedging_flow": 0.63,
                "delta_exposure": 142500.0,
                "gamma_exposure_greeks": -8420.0,
                "theta_exposure": -315.0,
                "vega_exposure": 2240.0,
                "vanna_exposure": 190.0,
                "charm_exposure": -44.0,
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
        self.assertEqual(row_a["selected_option_last_price"], 110.5)
        self.assertEqual(row_a["selected_option_delta"], 0.4735)
        self.assertEqual(row_a["selected_option_gamma"], 0.0124)
        self.assertEqual(row_a["selected_option_charm"], -0.009)
        self.assertEqual(row_a["market_gamma_exposure"], -8420.0)
        self.assertEqual(row_a["market_charm_exposure"], -44.0)
        self.assertAlmostEqual(row_a["target_premium_return_pct"], 30.0, places=4)
        self.assertAlmostEqual(row_a["stop_loss_premium_return_pct"], -14.9955, places=4)
        self.assertEqual(row_a["provider_health_status"], "GOOD")
        self.assertEqual(row_a["hybrid_move_probability"], 0.72)
        self.assertEqual(row_a["rule_move_probability"], 0.61)
        self.assertEqual(row_a["global_risk_state"], "GLOBAL_NEUTRAL")
        self.assertEqual(row_a["global_risk_score"], 24)
        self.assertEqual(row_a["gamma_vol_acceleration_score"], 68)
        self.assertEqual(row_a["squeeze_risk_state"], "HIGH_ACCELERATION_RISK")
        self.assertEqual(row_a["directional_convexity_state"], "UPSIDE_SQUEEZE_RISK")
        self.assertEqual(row_a["upside_squeeze_risk"], 0.74)
        self.assertEqual(row_a["downside_airpocket_risk"], 0.41)
        self.assertEqual(row_a["overnight_convexity_risk"], 0.52)
        self.assertEqual(row_a["gamma_vol_adjustment_score"], 4)
        self.assertEqual(row_a["dealer_hedging_pressure_score"], 66)
        self.assertEqual(row_a["dealer_flow_state"], "UPSIDE_HEDGING_ACCELERATION")
        self.assertEqual(row_a["upside_hedging_pressure"], 0.81)
        self.assertEqual(row_a["downside_hedging_pressure"], 0.32)
        self.assertEqual(row_a["pinning_pressure_score"], 0.18)
        self.assertEqual(row_a["dealer_pressure_adjustment_score"], 3)
        self.assertEqual(row_a["expected_move_points"], 165.4)
        self.assertEqual(row_a["expected_move_pct"], 0.7518)
        self.assertEqual(row_a["target_reachability_score"], 78)
        self.assertEqual(row_a["premium_efficiency_score"], 74)
        self.assertEqual(row_a["strike_efficiency_score"], 78)
        self.assertEqual(row_a["option_efficiency_score"], 77)
        self.assertEqual(row_a["option_efficiency_adjustment_score"], 4)
        self.assertEqual(row_a["consistency_check_status"], "PASS")
        self.assertEqual(row_a["consistency_check_issue_count"], 0)
        self.assertEqual(row_a["consistency_check_critical_issue_count"], 0)
        self.assertEqual(row_a["consistency_check_escalated"], False)
        self.assertEqual(row_a["oil_shock_score"], 0.7)
        self.assertEqual(row_a["commodity_risk_score"], 0.53)
        self.assertEqual(row_a["volatility_shock_score"], 0.7)
        self.assertEqual(row_a["volatility_explosion_probability"], 0.45)
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
            self.assertIn("global_risk_state", frame.columns)
            self.assertIn("gamma_vol_acceleration_score", frame.columns)
            self.assertIn("squeeze_risk_state", frame.columns)
            self.assertIn("directional_convexity_state", frame.columns)
            self.assertIn("upside_squeeze_risk", frame.columns)
            self.assertIn("downside_airpocket_risk", frame.columns)
            self.assertIn("overnight_convexity_risk", frame.columns)
            self.assertIn("gamma_vol_adjustment_score", frame.columns)
            self.assertIn("dealer_hedging_pressure_score", frame.columns)
            self.assertIn("dealer_flow_state", frame.columns)
            self.assertIn("upside_hedging_pressure", frame.columns)
            self.assertIn("downside_hedging_pressure", frame.columns)
            self.assertIn("pinning_pressure_score", frame.columns)
            self.assertIn("dealer_pressure_adjustment_score", frame.columns)
            self.assertIn("expected_move_points", frame.columns)
            self.assertIn("expected_move_pct", frame.columns)
            self.assertIn("target_reachability_score", frame.columns)
            self.assertIn("premium_efficiency_score", frame.columns)
            self.assertIn("selected_option_delta", frame.columns)
            self.assertIn("selected_option_iv", frame.columns)
            self.assertIn("market_gamma_exposure", frame.columns)
            self.assertIn("market_charm_exposure", frame.columns)
            self.assertIn("strike_efficiency_score", frame.columns)
            self.assertIn("option_efficiency_score", frame.columns)
            self.assertIn("option_efficiency_adjustment_score", frame.columns)
            self.assertIn("consistency_check_status", frame.columns)
            self.assertIn("consistency_check_issue_count", frame.columns)
            self.assertIn("consistency_check_critical_issue_count", frame.columns)
            self.assertIn("consistency_check_escalated", frame.columns)
            self.assertIn("consistency_check_findings", frame.columns)
            self.assertIn("oil_shock_score", frame.columns)
            self.assertIn("commodity_risk_score", frame.columns)
            self.assertIn("volatility_shock_score", frame.columns)
            self.assertIn("volatility_explosion_probability", frame.columns)
            self.assertEqual(len(frame), 0)

    def test_dataset_writes_sqlite_sidecar_for_durable_storage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            save_signal_evaluation(self._sample_result(), dataset_path=dataset_path)

            sqlite_path = dataset_path.with_suffix(".sqlite")
            self.assertTrue(sqlite_path.exists())

            frame = load_signals_dataset(dataset_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["signal_id"], build_signal_evaluation_row(self._sample_result())["signal_id"])

    def test_append_sqlite_rows_auto_migrates_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            sqlite_path = dataset_path.with_suffix(".sqlite")

            # Simulate an older schema that predates recently added columns.
            with sqlite3.connect(sqlite_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE signals (
                        signal_id TEXT,
                        signal_timestamp TEXT,
                        source TEXT,
                        mode TEXT,
                        symbol TEXT
                    )
                    """
                )

            save_signal_evaluation(self._sample_result(), dataset_path=dataset_path)

            with sqlite3.connect(sqlite_path) as connection:
                columns = {
                    row[1]
                    for row in connection.execute('PRAGMA table_info("signals")').fetchall()
                }

            self.assertIn("event_intelligence_enabled", columns)

            frame = load_signals_dataset(dataset_path)
            self.assertEqual(len(frame), 1)

    def test_save_signal_evaluation_uses_signal_timestamp_as_default_capture_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            save_signal_evaluation(self._sample_result(), dataset_path=dataset_path)

            frame = load_signals_dataset(dataset_path)
            self.assertEqual(frame.iloc[0]["created_at"], "2026-03-14T10:00:00+05:30")
            self.assertEqual(frame.iloc[0]["updated_at"], "2026-03-14T10:00:00+05:30")

    def test_sparse_frame_normalization_does_not_emit_fragmentation_warning(self):
        sparse_frame = pd.DataFrame(
            [
                {
                    "signal_id": "sig-1",
                    "signal_timestamp": "2026-03-14T10:00:00+05:30",
                    "symbol": "NIFTY",
                    "trade_status": "TRADE",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                write_signals_dataset(sparse_frame, dataset_path)

            performance_warnings = [
                warning
                for warning in caught
                if issubclass(warning.category, pd.errors.PerformanceWarning)
            ]

            self.assertEqual(performance_warnings, [])

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

    def test_evaluate_signal_outcomes_labels_early_alpha_decay_and_exit_pressure(self):
        row = build_signal_evaluation_row(self._sample_result())
        realized_path = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-14T10:05:00+05:30",
                    "2026-03-14T10:15:00+05:30",
                    "2026-03-14T10:30:00+05:30",
                    "2026-03-14T11:00:00+05:30",
                    "2026-03-14T15:25:00+05:30",
                ],
                "spot": [22040, 22110, 22070, 22015, 21980],
            }
        )

        enriched = evaluate_signal_outcomes(row, realized_path, as_of="2026-03-14T15:25:00+05:30")

        self.assertEqual(enriched["best_outcome_horizon"], "15m")
        self.assertEqual(enriched["horizon_edge_label"], "EARLY_ALPHA_DECAY")
        self.assertEqual(enriched["exit_quality_label"], "EARLY_EXIT")
        self.assertLess(float(enriched["peak_to_close_decay_bps"]), 0)
        self.assertIn(enriched["tradeability_tier"], {"HIGH", "USABLE", "FRAGILE"})

    def test_evaluate_signal_outcomes_respects_as_of_without_future_leakage(self):
        row = build_signal_evaluation_row(self._sample_result())
        realized_path = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-14T10:05:00+05:30",
                    "2026-03-14T10:10:00+05:30",
                    "2026-03-14T11:05:00+05:30",
                ],
                "spot": [22010, 22020, 22100],
            }
        )

        enriched = evaluate_signal_outcomes(
            row,
            realized_path,
            as_of="2026-03-14T10:10:00+05:30",
        )

        self.assertTrue(pd.isna(enriched.get("spot_60m")))
        self.assertTrue(pd.isna(enriched.get("signed_return_60m_bps")))
        self.assertIn(enriched["outcome_status"], {"PENDING", "PARTIAL"})

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

    def test_save_signal_evaluation_supports_append_only_live_capture(self):
        result_a = self._sample_result()
        result_b = self._sample_result()
        result_b["spot_summary"] = dict(result_b["spot_summary"])
        result_b["spot_summary"]["timestamp"] = "2026-03-14T10:30:00+05:30"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "signals_dataset.csv"
            returned = save_signal_evaluation(result_a, dataset_path=dataset_path, return_frame=False)
            self.assertIsNone(returned)

            returned = save_signal_evaluation(result_b, dataset_path=dataset_path, return_frame=False)
            self.assertIsNone(returned)

            frame = load_signals_dataset(dataset_path)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame["signal_id"].nunique(), 2)

    def test_no_direction_signal_does_not_force_directional_scoring(self):
        result = self._sample_result()
        result["trade"] = dict(result["trade"])
        result["trade"]["direction"] = None
        result["trade"]["option_type"] = None
        result["trade"]["strike"] = None
        result["trade"]["entry_price"] = None
        result["trade"]["target"] = None
        result["trade"]["stop_loss"] = None
        result["trade"]["trade_status"] = "NO_SIGNAL"

        row = build_signal_evaluation_row(result)
        realized_path = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-14T10:05:00+05:30",
                    "2026-03-14T10:15:00+05:30",
                    "2026-03-14T15:25:00+05:30",
                ],
                "spot": [22020, 22035, 22080],
            }
        )

        enriched = evaluate_signal_outcomes(row, realized_path, as_of="2026-03-14T15:25:00+05:30")

        self.assertEqual(enriched["outcome_status"], "PARTIAL")
        self.assertEqual(enriched["spot_5m"], 22020)
        self.assertGreater(enriched["realized_return_5m"], 0)
        self.assertTrue(pd.isna(enriched["signed_return_5m_bps"]))
        self.assertTrue(pd.isna(enriched["correct_5m"]))
        self.assertTrue(pd.isna(enriched["signed_return_session_close_bps"]))
        self.assertTrue(pd.isna(enriched["directional_consistency_score"]))
        self.assertTrue(pd.isna(enriched["direction_score"]))
        self.assertTrue(pd.isna(enriched["magnitude_score"]))
        self.assertTrue(pd.isna(enriched["timing_score"]))
        self.assertTrue(pd.isna(enriched["tradeability_score"]))

    def test_row_builder_infers_missing_contract_keys_from_ranked_strikes(self):
        result = self._sample_result()
        result["trade"] = dict(result["trade"])
        result["trade"]["selected_expiry"] = None
        result["trade"]["option_type"] = None
        result["trade"]["strike"] = None
        result["option_chain_validation"] = dict(result["option_chain_validation"])
        result["option_chain_validation"]["selected_expiry"] = "2026-03-26"
        result["ranked_strikes"] = [
            {
                "strike": 22100,
                "option_type": "PE",
                "selected_expiry": "2026-03-26",
                "score": 25.0,
            },
            {
                "strike": 22000,
                "option_type": "CE",
                "selected_expiry": "2026-03-26",
                "score": 27.81,
            },
        ]

        row = build_signal_evaluation_row(result)

        self.assertEqual(row["selected_expiry"], "2026-03-26")
        self.assertEqual(row["option_type"], "CE")
        self.assertEqual(row["strike"], 22000)

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


class CumulativeDatasetArchivalTests(unittest.TestCase):
    """Tests for the cumulative dataset syncing and archival mechanism."""

    def _make_rows(self, signal_ids, date="2026-03-18"):
        return [
            {
                "signal_id": sid,
                "signal_date": date,
                "updated_at": f"{date}T10:00:00+05:30",
                "symbol": "NIFTY",
            }
            for sid in signal_ids
        ]

    def test_sync_to_cumulative_appends_new_rows(self):
        from research.signal_evaluation.dataset import (
            _sync_to_cumulative,
            CUMULATIVE_DATASET_PATH,
            _dataset_store_path,
        )
        import research.signal_evaluation.dataset as ds_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            cumul_csv = Path(tmp_dir) / "signals_dataset_cumul.csv"
            cumul_sqlite = Path(tmp_dir) / "signals_dataset_cumul.sqlite"

            # Temporarily override the module-level paths
            orig_cumul = ds_mod.CUMULATIVE_DATASET_PATH
            ds_mod.CUMULATIVE_DATASET_PATH = cumul_csv

            try:
                # First sync — creates cumulative from scratch
                df1 = pd.DataFrame(self._make_rows(["sig_a", "sig_b"]))
                _sync_to_cumulative(df1)
                cumul = pd.read_csv(cumul_csv)
                self.assertEqual(len(cumul), 2)

                # Second sync — only new rows appended
                df2 = pd.DataFrame(self._make_rows(["sig_b", "sig_c"]))
                _sync_to_cumulative(df2)
                cumul = pd.read_csv(cumul_csv)
                self.assertEqual(len(cumul), 3)
                self.assertSetEqual(set(cumul["signal_id"]), {"sig_a", "sig_b", "sig_c"})
            finally:
                ds_mod.CUMULATIVE_DATASET_PATH = orig_cumul

    def test_sync_to_cumulative_skips_empty_frame(self):
        from research.signal_evaluation.dataset import _sync_to_cumulative
        import research.signal_evaluation.dataset as ds_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            cumul_csv = Path(tmp_dir) / "signals_dataset_cumul.csv"
            orig_cumul = ds_mod.CUMULATIVE_DATASET_PATH
            ds_mod.CUMULATIVE_DATASET_PATH = cumul_csv

            try:
                _sync_to_cumulative(pd.DataFrame())
                self.assertFalse(cumul_csv.exists())
            finally:
                ds_mod.CUMULATIVE_DATASET_PATH = orig_cumul

    def test_sync_live_to_cumulative_returns_new_count(self):
        from research.signal_evaluation.dataset import (
            sync_live_to_cumulative,
            write_signals_dataset,
        )
        import research.signal_evaluation.dataset as ds_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            live_csv = Path(tmp_dir) / "signals_dataset.csv"
            cumul_csv = Path(tmp_dir) / "signals_dataset_cumul.csv"

            orig_live = ds_mod.SIGNAL_DATASET_PATH
            orig_cumul = ds_mod.CUMULATIVE_DATASET_PATH
            ds_mod.SIGNAL_DATASET_PATH = live_csv
            ds_mod.CUMULATIVE_DATASET_PATH = cumul_csv

            try:
                # Write live dataset with 3 rows
                live_df = pd.DataFrame(self._make_rows(["sig_1", "sig_2", "sig_3"]))
                write_signals_dataset(live_df, live_csv)

                # First sync — all 3 should be new
                synced = sync_live_to_cumulative()
                self.assertEqual(synced, 3)

                # Second sync — idempotent, no new rows
                synced = sync_live_to_cumulative()
                self.assertEqual(synced, 0)
            finally:
                ds_mod.SIGNAL_DATASET_PATH = orig_live
                ds_mod.CUMULATIVE_DATASET_PATH = orig_cumul

    def test_upsert_auto_syncs_to_cumulative(self):
        from research.signal_evaluation.dataset import upsert_signal_rows
        import research.signal_evaluation.dataset as ds_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            live_csv = Path(tmp_dir) / "signals_dataset.csv"
            cumul_csv = Path(tmp_dir) / "signals_dataset_cumul.csv"

            orig_live = ds_mod.SIGNAL_DATASET_PATH
            orig_cumul = ds_mod.CUMULATIVE_DATASET_PATH
            ds_mod.SIGNAL_DATASET_PATH = live_csv
            ds_mod.CUMULATIVE_DATASET_PATH = cumul_csv

            try:
                # Upsert to the live path — should auto-sync to cumulative
                upsert_signal_rows(
                    self._make_rows(["sig_x", "sig_y"]),
                    path=live_csv,
                    return_frame=False,
                )
                self.assertTrue(cumul_csv.exists())
                cumul = pd.read_csv(cumul_csv)
                self.assertEqual(len(cumul), 2)
                self.assertSetEqual(set(cumul["signal_id"]), {"sig_x", "sig_y"})
            finally:
                ds_mod.SIGNAL_DATASET_PATH = orig_live
                ds_mod.CUMULATIVE_DATASET_PATH = orig_cumul


if __name__ == "__main__":
    unittest.main()
