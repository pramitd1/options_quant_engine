from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.heston.heston_pricer import HestonParams, heston_implied_vol_proxy, heston_price
from research.signal_evaluation.dataset import load_signals_dataset, write_signals_dataset
from research.signal_evaluation.heston_backfill import (
    OFFLINE_HESTON_BACKFILL_VERSION,
    backfill_heston_research_dataset,
    enrich_heston_research_features_from_snapshots,
)


VALUATION_TIME = "2026-05-19T09:30:00+05:30"


def _synthetic_chain(*, spot: float = 10000.0, expiry: str = "2026-05-26") -> pd.DataFrame:
    params = HestonParams(kappa=1.6, theta=0.050, vol_of_vol=0.50, rho=-0.38, v0=0.046)
    expiry_ts = pd.Timestamp(expiry).tz_localize("Asia/Kolkata").replace(hour=15, minute=30)
    tte_years = (expiry_ts - pd.Timestamp(VALUATION_TIME)).total_seconds() / (365.0 * 24.0 * 3600.0)
    rows = []
    for strike in [9700, 9800, 9900, 10000, 10100, 10200, 10300]:
        for option_type in ("CE", "PE"):
            price = heston_price(
                spot=spot,
                strike=strike,
                time_to_expiry_years=tte_years,
                option_type=option_type,
                params=params,
            )
            iv_proxy = heston_implied_vol_proxy(
                spot=spot,
                strike=strike,
                time_to_expiry_years=tte_years,
                params=params,
            )
            rows.append(
                {
                    "strikePrice": strike,
                    "OPTION_TYP": option_type,
                    "lastPrice": round(float(price or 0.0), 4),
                    "EXPIRY_DT": expiry,
                    "impliedVolatility": round(float(iv_proxy or 0.0) * 100.0, 4),
                    "totalTradedVolume": 50000,
                    "openInterest": 100000,
                    "bidPrice": round(float(price or 0.0) * 0.995, 4),
                    "askPrice": round(float(price or 0.0) * 1.005, 4),
                }
            )
    return pd.DataFrame(rows)


def _signal_row(chain_path: Path) -> dict:
    return {
        "signal_id": "sig_heston_001",
        "signal_timestamp": VALUATION_TIME,
        "symbol": "NIFTY",
        "source": "ICICI",
        "option_source": "ICICI",
        "saved_chain_snapshot_path": str(chain_path),
        "spot_at_signal": 10000.0,
        "selected_expiry": "2026-05-26",
        "direction": "CALL",
        "option_type": "CE",
        "strike": 10000,
        "selected_option_iv": 22.0,
        "selected_option_delta": 0.50,
        "selected_option_gamma": 0.001,
    }


def test_enrich_heston_research_features_from_saved_chain_snapshot(tmp_path: Path):
    chain_path = tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-05-19T09-30-00+05-30.csv"
    _synthetic_chain().to_csv(chain_path, index=False)

    frame = pd.DataFrame([_signal_row(chain_path)])
    updated, summary = enrich_heston_research_features_from_snapshots(
        frame,
        snapshot_dir=tmp_path,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
    )

    assert summary["rows_seen"] == 1
    assert summary["rows_updated"] == 1
    assert updated.iloc[0]["heston_research_enabled"] is True
    assert updated.iloc[0]["heston_calibration_status"] in {"CALIBRATED", "FAILED", "REJECTED"}
    diagnostics = json.loads(updated.iloc[0]["heston_diagnostics_json"])
    assert diagnostics["backfill_version"] == OFFLINE_HESTON_BACKFILL_VERSION
    assert diagnostics["live_trade_decision_unchanged"] is True


def test_heston_backfill_handles_blank_float_heston_columns(tmp_path: Path):
    chain_path = tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-05-19T09-30-00+05-30.csv"
    _synthetic_chain().to_csv(chain_path, index=False)

    row = _signal_row(chain_path)
    row.update(
        {
            "heston_research_enabled": float("nan"),
            "heston_calibration_status": float("nan"),
            "heston_diagnostics_json": float("nan"),
        }
    )
    frame = pd.DataFrame([row])

    updated, summary = enrich_heston_research_features_from_snapshots(
        frame,
        snapshot_dir=tmp_path,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
    )

    assert summary["rows_updated"] == 1
    assert updated.iloc[0]["heston_research_enabled"] is True
    assert updated.iloc[0]["heston_calibration_status"] in {"CALIBRATED", "FAILED", "REJECTED"}


def test_heston_backfill_dataset_writes_when_requested(tmp_path: Path):
    snapshot_dir = tmp_path / "option_chain_snapshots"
    snapshot_dir.mkdir()
    chain_path = snapshot_dir / "NIFTY_ICICI_option_chain_snapshot_2026-05-19T09-30-00+05-30.csv"
    _synthetic_chain().to_csv(chain_path, index=False)
    dataset_path = tmp_path / "signals_dataset.csv"
    write_signals_dataset(pd.DataFrame([_signal_row(chain_path)]), dataset_path)

    dry_summary = backfill_heston_research_dataset(
        dataset_path=dataset_path,
        snapshot_dir=snapshot_dir,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
        dry_run=True,
    )
    dry_frame = load_signals_dataset(dataset_path)
    assert dry_summary["rows_updated"] == 1
    dry_status = dry_frame.iloc[0]["heston_calibration_status"]
    assert dry_status in (None, "") or pd.isna(dry_status)

    write_summary = backfill_heston_research_dataset(
        dataset_path=dataset_path,
        snapshot_dir=snapshot_dir,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
        dry_run=False,
    )
    written = load_signals_dataset(dataset_path)

    assert write_summary["dry_run"] is False
    assert written.iloc[0]["heston_research_enabled"] in {True, "True", "true", 1}
    assert OFFLINE_HESTON_BACKFILL_VERSION in str(written.iloc[0]["heston_diagnostics_json"])


def test_heston_backfill_skips_existing_offline_rows_without_force(tmp_path: Path):
    chain_path = tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-05-19T09-30-00+05-30.csv"
    _synthetic_chain().to_csv(chain_path, index=False)
    row = _signal_row(chain_path)
    row["heston_calibration_status"] = "CALIBRATED"
    row["heston_diagnostics_json"] = json.dumps({"backfill_version": OFFLINE_HESTON_BACKFILL_VERSION})

    _, summary = enrich_heston_research_features_from_snapshots(
        pd.DataFrame([row]),
        snapshot_dir=tmp_path,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
    )

    assert summary["rows_skipped_existing"] == 1
    assert summary["rows_updated"] == 0
