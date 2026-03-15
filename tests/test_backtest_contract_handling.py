from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from backtest.intraday_backtester import run_intraday_backtest
from backtest.pnl_engine import calculate_trade_pnl
from strategy.exit_model import calculate_exit


class BacktestContractHandlingTests(unittest.TestCase):
    def test_pnl_engine_respects_selected_expiry(self):
        trade = {
            "strike": 22000,
            "option_type": "CE",
            "selected_expiry": "2026-03-26",
            "entry_price": 100.0,
            "target": 200.0,
            "stop_loss": 10.0,
            "lot_size": 1,
            "number_of_lots": 1,
        }
        exit_snapshot = pd.DataFrame(
            [
                {"strikePrice": 22000, "OPTION_TYP": "CE", "EXPIRY_DT": "2026-04-02", "lastPrice": 80.0},
                {"strikePrice": 22000, "OPTION_TYP": "CE", "EXPIRY_DT": "2026-03-26", "lastPrice": 120.0},
            ]
        )

        pnl = calculate_trade_pnl(trade, exit_snapshot)

        self.assertEqual(pnl["exit_reason"], "TIME_EXIT")
        self.assertGreater(pnl["exit_price"], 100.0)

    def test_exit_model_supports_override_parameters(self):
        target, stop_loss = calculate_exit(100.0, target_profit_percent=40.0, stop_loss_percent=20.0)

        self.assertEqual(target, 140.0)
        self.assertEqual(stop_loss, 80.0)

    def test_backtest_filters_to_front_expiry_and_passes_exit_overrides(self):
        historical_df = pd.DataFrame(
            [
                {
                    "timestamp": "2026-03-14T09:15:00+05:30",
                    "spot": 22000.0,
                    "strikePrice": 22000,
                    "OPTION_TYP": "CE",
                    "lastPrice": 100.0,
                    "openInterest": 1000,
                    "EXPIRY_DT": "2026-03-26",
                },
                {
                    "timestamp": "2026-03-14T09:15:00+05:30",
                    "spot": 22000.0,
                    "strikePrice": 22000,
                    "OPTION_TYP": "CE",
                    "lastPrice": 90.0,
                    "openInterest": 1000,
                    "EXPIRY_DT": "2026-04-02",
                },
            ]
        )
        captured = {}

        def fake_generate_trade(**kwargs):
            captured["expiries"] = sorted(kwargs["option_chain"]["EXPIRY_DT"].astype(str).unique().tolist())
            captured["target_profit_percent"] = kwargs["target_profit_percent"]
            captured["stop_loss_percent"] = kwargs["stop_loss_percent"]
            return {"trade_status": "NO_SIGNAL", "direction": None}

        with patch("backtest.intraday_backtester.load_option_chain", return_value=historical_df):
            with patch("backtest.intraday_backtester.generate_trade", side_effect=fake_generate_trade):
                with patch("backtest.intraday_backtester.compute_performance_metrics", return_value={}):
                    result = run_intraday_backtest(
                        "NIFTY",
                        years=1,
                        target_profit_percent=40.0,
                        stop_loss_percent=20.0,
                    )

        self.assertEqual(captured["expiries"], ["2026-03-26"])
        self.assertEqual(captured["target_profit_percent"], 40.0)
        self.assertEqual(captured["stop_loss_percent"], 20.0)
        self.assertEqual(result["bar_interval"], "1d_default_synthetic")


if __name__ == "__main__":
    unittest.main()
