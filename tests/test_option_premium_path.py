from __future__ import annotations

import pandas as pd

from research.signal_evaluation.option_premium_path import enrich_option_premium_paths


def _write_chain(path, *, premium: float) -> None:
    pd.DataFrame(
        [
            {
                "strikePrice": 22000,
                "OPTION_TYP": "CE",
                "EXPIRY_DT": "2026-03-26",
                "lastPrice": premium,
                "bidPrice": premium - 0.5,
                "askPrice": premium + 0.5,
                "totalTradedVolume": 1000,
                "openInterest": 5000,
            },
            {
                "strikePrice": 22100,
                "OPTION_TYP": "CE",
                "EXPIRY_DT": "2026-03-26",
                "lastPrice": premium / 2,
                "totalTradedVolume": 100,
                "openInterest": 500,
            },
        ]
    ).to_csv(path, index=False)


def test_enrich_option_premium_paths_from_saved_chain_snapshots(tmp_path):
    snapshot_dir = tmp_path / "option_chain_snapshots"
    snapshot_dir.mkdir()
    for horizon, premium in [(5, 105), (15, 112), (30, 118), (60, 130), (120, 145)]:
        hour = 10 + ((horizon) // 60)
        minute = horizon % 60
        path = snapshot_dir / f"NIFTY_ICICI_option_chain_snapshot_2026-03-14T{hour:02d}-{minute:02d}-01+05-30.csv"
        _write_chain(path, premium=float(premium))

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "signal_timestamp": "2026-03-14T10:00:00+05:30",
                "symbol": "NIFTY",
                "source": "ICICI",
                "option_source": "ICICI",
                "selected_expiry": "2026-03-26",
                "option_type": "CE",
                "strike": 22000,
                "entry_price": 100.0,
                "option_entry_premium": 100.0,
            }
        ]
    )

    enriched, summary = enrich_option_premium_paths(
        frame,
        snapshot_dir=snapshot_dir,
        as_of="2026-03-14T12:10:00+05:30",
        max_lag_seconds=90,
        lot_size=50,
    )

    row = enriched.iloc[0]
    assert summary["rows_updated"] == 1
    assert summary["premium_points_filled"] == 5
    assert row["option_premium_path_status"] == "COMPLETE"
    assert float(row["option_premium_5m"]) == 105.0
    assert float(row["option_premium_return_5m_pct"]) == 5.0
    assert float(row["option_premium_return_5m_bps"]) == 500.0
    assert float(row["option_premium_60m"]) == 130.0
    assert float(row["option_premium_return_60m_bps"]) == 3000.0
    assert float(row["option_premium_pnl_per_lot_60m"]) == 1500.0
