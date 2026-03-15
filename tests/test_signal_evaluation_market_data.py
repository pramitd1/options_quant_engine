from __future__ import annotations

import pandas as pd

from research.signal_evaluation.dataset import write_signals_dataset
from research.signal_evaluation.evaluator import update_signal_dataset_outcomes


def test_update_signal_dataset_outcomes_batches_history_fetches_by_symbol(tmp_path):
    dataset_path = tmp_path / "signals.csv"
    fetch_calls = []

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "signal_timestamp": "2026-01-02T09:20:00+05:30",
                "symbol": "NIFTY",
                "direction": "CALL",
                "spot_at_signal": 22000.0,
                "trade_strength": 60,
                "hybrid_move_probability": 0.60,
                "outcome_status": "PENDING",
            },
            {
                "signal_id": "sig-2",
                "signal_timestamp": "2026-01-02T10:20:00+05:30",
                "symbol": "NIFTY",
                "direction": "CALL",
                "spot_at_signal": 22020.0,
                "trade_strength": 55,
                "hybrid_move_probability": 0.58,
                "outcome_status": "PENDING",
            },
        ]
    )
    write_signals_dataset(frame, dataset_path)

    def fake_fetch_history(symbol, *, start_ts, end_ts, interval="5m"):
        fetch_calls.append((symbol, str(start_ts), str(end_ts), interval))
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-02T09:20:00+05:30",
                        "2026-01-02T09:25:00+05:30",
                        "2026-01-02T10:20:00+05:30",
                        "2026-01-02T10:25:00+05:30",
                        "2026-01-02T11:20:00+05:30",
                    ],
                    utc=True,
                ).tz_convert("Asia/Kolkata"),
                "spot": [22000.0, 22010.0, 22020.0, 22035.0, 22060.0],
            }
        )

    updated = update_signal_dataset_outcomes(
        dataset_path=dataset_path,
        as_of="2026-01-02T11:30:00+05:30",
        fetch_spot_history_fn=fake_fetch_history,
    )

    assert len(fetch_calls) == 1
    assert len(updated) == 2
    assert set(updated["outcome_status"].astype(str)) <= {"PARTIAL", "COMPLETE"}
