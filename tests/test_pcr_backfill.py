from __future__ import annotations

import pandas as pd

from research.signal_evaluation.pcr_backfill import backfill_pcr_fields, compute_pcr_from_option_chain


def _chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strikePrice": [23300, 23300, 23350, 23350, 23400, 23400],
            "OPTION_TYP": ["CE", "PE", "CE", "PE", "CE", "PE"],
            "openInterest": [100.0, 130.0, 200.0, 300.0, 100.0, 170.0],
            "totalTradedVolume": [10.0, 12.0, 20.0, 16.0, 10.0, 8.0],
            "EXPIRY_DT": ["19-May-2026"] * 6,
        }
    )


def test_compute_pcr_from_option_chain_prefers_open_interest_basis():
    pcr = compute_pcr_from_option_chain(_chain(), spot=23360.0)

    assert pcr["open_interest_pcr"] == 1.5
    assert pcr["volume_pcr"] == 0.9
    assert pcr["volume_pcr_atm"] == 0.9
    assert pcr["pcr_value"] == 1.5
    assert pcr["pcr_basis"] == "OPEN_INTEREST"
    assert pcr["pcr_bucket"] == "HIGH_PCR"


def test_backfill_pcr_fields_uses_saved_chain_snapshot(tmp_path):
    chain_path = tmp_path / "chain.csv"
    _chain().to_csv(chain_path, index=False)
    frame = pd.DataFrame(
        [
            {
                "signal_id": "s1",
                "signal_timestamp": "2026-05-18T10:00:00+05:30",
                "saved_chain_snapshot_path": str(chain_path),
                "spot_at_signal": 23360.0,
            }
        ]
    )

    updated, summary = backfill_pcr_fields(frame)

    row = updated.iloc[0]
    assert summary["saved_path_backfilled"] == 1
    assert row["pcr_value"] == 1.5
    assert row["pcr_data_source"] == "BACKFILL_SAVED_CHAIN"
