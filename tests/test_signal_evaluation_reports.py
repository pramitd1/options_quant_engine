from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from research.signal_evaluation.reports import (
    average_realized_return_by_horizon,
    average_score_by_signal_quality,
    build_research_report,
    hit_rate_by_macro_regime,
    hit_rate_by_trade_strength,
    move_probability_calibration,
    regime_fingerprint_performance,
    signal_count_by_regime,
)
from research.signal_evaluation.reporting import (
    build_signal_evaluation_summary,
    write_signal_evaluation_report,
)


class SignalEvaluationReportsTests(unittest.TestCase):
    def _sample_frame(self):
        return pd.DataFrame(
            [
                {
                    "signal_id": "a1",
                    "signal_calibration_bucket": "80_100",
                    "trade_strength": 82,
                    "macro_regime": "RISK_ON",
                    "signal_quality": "STRONG",
                    "signal_regime": "EXPANSION_BIAS",
                    "gamma_regime": "SHORT_GAMMA_ZONE",
                    "regime_fingerprint_id": "fp_01",
                    "regime_fingerprint": "signal_regime=EXPANSION_BIAS|macro_regime=RISK_ON|gamma_regime=SHORT_GAMMA_ZONE",
                    "move_probability": 0.78,
                    "probability_calibration_bucket": "0.65_0.79",
                    "composite_signal_score": 82,
                    "realized_return_5m": 0.0010,
                    "realized_return_15m": 0.0018,
                    "realized_return_30m": 0.0026,
                    "realized_return_60m": 0.0030,
                    "spot_at_signal": 22000,
                    "spot_next_close": 22120,
                    "spot_close_same_day": 22090,
                    "spot_60m": 22070,
                    "direction": "CALL",
                    "direction_score": 90,
                    "magnitude_score": 76,
                    "timing_score": 84,
                    "tradeability_score": 78,
                },
                {
                    "signal_id": "a2",
                    "signal_calibration_bucket": "50_64",
                    "trade_strength": 58,
                    "macro_regime": "RISK_OFF",
                    "signal_quality": "MEDIUM",
                    "signal_regime": "CONFLICTED",
                    "gamma_regime": "LONG_GAMMA_ZONE",
                    "regime_fingerprint_id": "fp_02",
                    "regime_fingerprint": "signal_regime=CONFLICTED|macro_regime=RISK_OFF|gamma_regime=LONG_GAMMA_ZONE",
                    "move_probability": 0.42,
                    "probability_calibration_bucket": "0.35_0.49",
                    "composite_signal_score": 48,
                    "realized_return_5m": -0.0004,
                    "realized_return_15m": -0.0008,
                    "realized_return_30m": -0.0010,
                    "realized_return_60m": -0.0016,
                    "spot_at_signal": 22000,
                    "spot_next_close": 22110,
                    "spot_close_same_day": 22040,
                    "spot_60m": 22020,
                    "direction": "PUT",
                    "direction_score": 35,
                    "magnitude_score": 42,
                    "timing_score": 38,
                    "tradeability_score": 46,
                },
            ]
        )

    def test_grouped_reports_are_generated(self):
        frame = self._sample_frame()

        self.assertEqual(len(hit_rate_by_trade_strength(frame)), 2)
        self.assertEqual(len(hit_rate_by_macro_regime(frame)), 2)
        self.assertEqual(len(average_score_by_signal_quality(frame)), 2)
        self.assertEqual(len(average_realized_return_by_horizon(frame)), 4)
        self.assertGreater(len(signal_count_by_regime(frame)), 0)
        self.assertEqual(len(move_probability_calibration(frame)), 2)
        self.assertEqual(len(regime_fingerprint_performance(frame)), 2)

        report = build_research_report(frame)
        self.assertIn("hit_rate_by_trade_strength", report)
        self.assertIn("move_probability_calibration", report)
        self.assertIn("regime_fingerprint_performance", report)
        self.assertFalse(report["move_probability_calibration"].empty)

    def test_structured_signal_evaluation_summary_and_artifacts_are_generated(self):
        frame = self._sample_frame()
        frame["signal_timestamp"] = [
            "2026-03-10T09:20:00+05:30",
            "2026-03-11T09:20:00+05:30",
        ]
        frame["correct_60m"] = [1, 0]
        frame["signed_return_60m_bps"] = [54.0, -23.0]
        frame["outcome_status"] = ["COMPLETE", "COMPLETE"]
        summary = build_signal_evaluation_summary(
            frame,
            production_pack_name="baseline_v1",
            dataset_path="research/signal_evaluation/signals_dataset.csv",
        )

        self.assertEqual(summary["production_pack_name"], "baseline_v1")
        self.assertEqual(summary["total_signal_count"], 2)
        self.assertIn("signals_by_symbol", summary)
        self.assertIn("horizon_performance", summary)
        self.assertIn("score_statistics", summary)
        self.assertIn("score_bucket_performance", summary)

        tmp_dir = Path(self.id().replace(".", "_"))
        try:
            artifact = write_signal_evaluation_report(
                frame,
                production_pack_name="baseline_v1",
                dataset_path="research/signal_evaluation/signals_dataset.csv",
                output_dir=tmp_dir,
                report_name="unit_test_signal_eval",
            )
            self.assertTrue(Path(artifact["json_path"]).exists())
            self.assertTrue(Path(artifact["markdown_path"]).exists())
            self.assertIn("signals_by_symbol", artifact["csv_paths"])
        finally:
            if tmp_dir.exists():
                for child in sorted(tmp_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()


if __name__ == "__main__":
    unittest.main()
