from __future__ import annotations

import pandas as pd

from scripts.ops import run_segmented_calibration_governance as gov


def _base_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_id": "A",
                "signal_timestamp": "2026-04-01T09:15:00+05:30",
                "symbol": "NIFTY",
                "source": "ICICI",
                "direction": "CALL",
                "gamma_regime": "NEGATIVE_GAMMA",
                "vol_regime": "VOL_EXPANSION",
                "volatility_regime": "VOL_EXPANSION",
                "data_quality_status": "WEAK",
                "confirmation_status": "WEAK",
                "provider_health_status": "FRAGILE",
                "correct_60m": 1,
                "signed_return_60m_bps": 15.0,
                "saved_spot_snapshot_path": "/tmp/spot_A.json",
                "saved_chain_snapshot_path": "/tmp/chain_A.csv",
            },
            {
                "signal_id": "B",
                "signal_timestamp": "2026-04-01T09:20:00+05:30",
                "symbol": "NIFTY",
                "source": "ICICI",
                "direction": "PUT",
                "gamma_regime": "NEGATIVE_GAMMA",
                "vol_regime": "VOL_EXPANSION",
                "volatility_regime": "VOL_EXPANSION",
                "data_quality_status": "CAUTION",
                "confirmation_status": "CONFIRMED",
                "provider_health_status": "GOOD",
                "correct_60m": 0,
                "signed_return_60m_bps": -10.0,
                "saved_spot_snapshot_path": "/tmp/spot_B.json",
                "saved_chain_snapshot_path": "/tmp/chain_B.csv",
            },
            {
                "signal_id": "C",
                "signal_timestamp": "2026-04-01T09:25:00+05:30",
                "symbol": "NIFTY",
                "source": "ZERODHA",
                "direction": "CALL",
                "gamma_regime": "NEUTRAL_GAMMA",
                "vol_regime": "NORMAL_VOL",
                "volatility_regime": "NORMAL_VOL",
                "data_quality_status": "GOOD",
                "confirmation_status": "CONFIRMED",
                "provider_health_status": "GOOD",
                "correct_60m": 1,
                "signed_return_60m_bps": 8.0,
                "saved_spot_snapshot_path": "/tmp/spot_C.json",
                "saved_chain_snapshot_path": "/tmp/chain_C.csv",
            },
        ]
    )


def _proxy_compare(signal_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_id": signal_ids,
            "baseline_calibrated_runtime_composite_score": [60.0] * len(signal_ids),
            "segmented_calibrated_runtime_composite_score": [62.0] * len(signal_ids),
            "baseline_proxy_selected": [False] * len(signal_ids),
            "segmented_proxy_selected": [False] * len(signal_ids),
        }
    )


def test_replay_selector_marks_only_high_weak_data_rows_in_weak_data_mode(monkeypatch):
    frame = _base_candidate_frame()
    proxy = _proxy_compare(frame["signal_id"].tolist())

    monkeypatch.setattr(gov, "_eligible_replay_rows", lambda dataset: frame.copy())
    monkeypatch.setattr(gov, "_backfill_outcome_ready_replay_rows", lambda dataset, limit: pd.DataFrame())
    monkeypatch.setattr(gov, "_archived_replay_rows", lambda limit: pd.DataFrame())

    selected, replay_source = gov._replay_candidates_with_priority(
        dataset=frame,
        proxy_compare=proxy,
        replay_limit=3,
        replay_priority_mode="weak-data-heavy",
        weak_data_min_score=2,
    )

    assert replay_source == "prioritized_weak_data_heavy_replay_rows"
    priorities = dict(zip(selected["signal_id"].tolist(), selected["replay_priority"].tolist()))
    assert priorities["A"] == "weak_data_focus"
    assert priorities["B"] != "weak_data_focus"


def test_replay_selector_uses_balanced_source_by_default(monkeypatch):
    frame = _base_candidate_frame()
    proxy = _proxy_compare(frame["signal_id"].tolist())

    monkeypatch.setattr(gov, "_eligible_replay_rows", lambda dataset: frame.copy())
    monkeypatch.setattr(gov, "_backfill_outcome_ready_replay_rows", lambda dataset, limit: pd.DataFrame())
    monkeypatch.setattr(gov, "_archived_replay_rows", lambda limit: pd.DataFrame())

    _, replay_source = gov._replay_candidates_with_priority(
        dataset=frame,
        proxy_compare=proxy,
        replay_limit=2,
        replay_priority_mode="balanced",
    )

    assert replay_source == "prioritized_dataset_replay_rows"
