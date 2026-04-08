from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.legacy_backfill import (
    apply_repair_proposals_to_dataset,
    audit_unresolved_signal_contract_matches,
    backfill_signal_contract_fields,
    partition_repair_proposals,
    propose_repairs_for_unresolved_signal_contract_matches,
)


def test_legacy_backfill_reconstructs_selected_option_fields(tmp_path: Path) -> None:
    chain_path = tmp_path / "NIFTY_TEST_option_chain_snapshot_2026-03-14T10-00-00+05-30.csv"
    chain = pd.DataFrame(
        [
            {
                "strikePrice": 22000,
                "OPTION_TYP": "CE",
                "lastPrice": 111.2,
                "openInterest": 120000,
                "totalTradedVolume": 6500,
                "impliedVolatility": 18.5,
                "EXPIRY_DT": "2026-03-26",
            }
        ]
    )
    chain.to_csv(chain_path, index=False)

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "signal_timestamp": "2026-03-14T10:00:00+05:30",
                "saved_chain_snapshot_path": str(chain_path),
                "selected_expiry": "2026-03-26",
                "option_type": "CE",
                "strike": 22000,
                "spot_at_signal": 22010,
                "entry_price": 110.5,
                "target": 143.65,
                "stop_loss": 93.93,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
                "target_premium_return_pct": pd.NA,
                "stop_loss_premium_return_pct": pd.NA,
            }
        ]
    )

    updated, stats = backfill_signal_contract_fields(frame, project_root=tmp_path)

    assert stats["rows_backfilled"] == 1
    row = updated.iloc[0]
    assert float(row["selected_option_last_price"]) == 111.2
    assert float(row["selected_option_iv"]) == 18.5
    assert pd.notna(row["selected_option_delta"])
    assert float(row["target_premium_return_pct"]) > 0
    assert float(row["stop_loss_premium_return_pct"]) < 0


def test_legacy_backfill_tracks_missing_snapshot_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-1",
                "saved_chain_snapshot_path": "debug_samples/option_chain_snapshots/missing.csv",
                "selected_expiry": "2026-03-26",
                "option_type": "CE",
                "strike": 22000,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
            }
        ]
    )

    updated, stats = backfill_signal_contract_fields(frame, project_root=tmp_path)

    assert stats["rows_skipped_missing_snapshot"] == 1
    assert stats["rows_backfilled"] == 0
    assert pd.isna(updated.iloc[0]["selected_option_last_price"])


def test_legacy_backfill_uses_alias_expiry_and_nearest_strike_fallback(tmp_path: Path) -> None:
    chain_path = tmp_path / "NIFTY_TEST_option_chain_snapshot_2026-03-14T10-00-00+05-30.csv"
    chain = pd.DataFrame(
        [
            {
                "strikePrice": 21950,
                "OPTION_TYP": "CALL",
                "lastPrice": 128.4,
                "openInterest": 90000,
                "totalTradedVolume": 4000,
                "impliedVolatility": 17.0,
                "EXPIRY_DT": "26-Mar-2026",
            },
            {
                "strikePrice": 22050,
                "OPTION_TYP": "CE",
                "lastPrice": 96.1,
                "openInterest": 75000,
                "totalTradedVolume": 3500,
                "impliedVolatility": 16.8,
                "EXPIRY_DT": "2026/03/26",
            },
        ]
    )
    chain.to_csv(chain_path, index=False)

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-2",
                "signal_timestamp": "2026-03-14T10:00:00+05:30",
                "saved_chain_snapshot_path": str(chain_path),
                "selected_expiry": "2026-03-26",
                "option_type": "C",
                "strike": 22000,
                "spot_at_signal": 22020,
                "entry_price": 120.0,
                "target": 150.0,
                "stop_loss": 102.0,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
            }
        ]
    )

    updated, stats = backfill_signal_contract_fields(frame, project_root=tmp_path)

    assert stats["rows_backfilled"] == 1
    row = updated.iloc[0]
    assert float(row["selected_option_last_price"]) == 128.4
    assert float(row["selected_option_iv"]) == 17.0


def test_unresolved_contract_audit_includes_top_candidates(tmp_path: Path) -> None:
    chain_path = tmp_path / "NIFTY_TEST_option_chain_snapshot_2026-03-14T10-00-00+05-30.csv"
    pd.DataFrame(
        [
            {
                "strikePrice": 21950,
                "OPTION_TYP": "CALL",
                "lastPrice": 128.4,
                "openInterest": 90000,
                "totalTradedVolume": 4000,
                "impliedVolatility": 17.0,
                "EXPIRY_DT": "26-Mar-2026",
            },
            {
                "strikePrice": 22050,
                "OPTION_TYP": "PUT",
                "lastPrice": 96.1,
                "openInterest": 75000,
                "totalTradedVolume": 3500,
                "impliedVolatility": 16.8,
                "EXPIRY_DT": "2026/03/26",
            },
        ]
    ).to_csv(chain_path, index=False)

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-audit",
                "signal_timestamp": "2026-03-14T10:00:00+05:30",
                "symbol": "NIFTY",
                "saved_chain_snapshot_path": str(chain_path),
                "selected_expiry": "2026-04-30",
                "option_type": "XX",
                "strike": 22000,
                "spot_at_signal": 22020,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
            }
        ]
    )

    audit = audit_unresolved_signal_contract_matches(frame, project_root=tmp_path, top_n=2)

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["signal_id"] == "sig-audit"
    assert row["selected_option_type_normalized"] == "XX"
    assert row["selected_expiry_normalized"] == "2026-04-30"
    assert row["top_candidate_count"] == 2
    assert '"option_type_normalized": "PE"' in row["top_candidates_json"]


def test_repair_proposals_infer_option_type_and_strike_from_direction_and_candidates(tmp_path: Path) -> None:
    chain_path = tmp_path / "NIFTY_TEST_option_chain_snapshot_2026-03-14T10-00-00+05-30.csv"
    pd.DataFrame(
        [
            {
                "strikePrice": 23100,
                "OPTION_TYP": "PE",
                "lastPrice": 142.85,
                "openInterest": 8739185,
                "totalTradedVolume": 285808900,
                "impliedVolatility": 17.1,
                "EXPIRY_DT": "17-Mar-2026",
            },
            {
                "strikePrice": 23200,
                "OPTION_TYP": "CE",
                "lastPrice": 130.45,
                "openInterest": 9142575,
                "totalTradedVolume": 314846805,
                "impliedVolatility": 16.7,
                "EXPIRY_DT": "17-Mar-2026",
            },
        ]
    ).to_csv(chain_path, index=False)

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-repair",
                "signal_timestamp": "2026-03-16T14:05:00+05:30",
                "symbol": "NIFTY",
                "direction": "PUT",
                "saved_chain_snapshot_path": str(chain_path),
                "selected_expiry": "17-Mar-2026",
                "option_type": pd.NA,
                "strike": pd.NA,
                "spot_at_signal": 23131.5996,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
            }
        ]
    )

    audit = audit_unresolved_signal_contract_matches(frame, project_root=tmp_path, top_n=2)
    proposals = propose_repairs_for_unresolved_signal_contract_matches(audit)

    assert len(proposals) == 1
    row = proposals.iloc[0]
    assert row["direction_implied_option_type"] == "PE"
    assert row["proposed_option_type"] == "PE"
    assert float(row["proposed_strike"]) == 23100.0
    assert row["proposal_confidence"] in {"MEDIUM", "HIGH"}


def test_repair_proposals_use_exact_replay_evidence_to_raise_confidence(tmp_path: Path) -> None:
    chain_path = tmp_path / "debug_samples" / "NIFTY_ICICI_option_chain_snapshot_2026-03-16T12-51-26.783567+05-30.csv"
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "strikePrice": 23000,
                "OPTION_TYP": "PE",
                "lastPrice": 142.85,
                "openInterest": 8739185,
                "totalTradedVolume": 285808900,
                "impliedVolatility": 17.1,
                "EXPIRY_DT": "17-Mar-2026",
            }
        ]
    ).to_csv(chain_path, index=False)

    replay_dir = tmp_path / "research" / "ml_evaluation" / "confirmation_mode_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "chain_snapshot": "debug_samples/NIFTY_ICICI_option_chain_snapshot_2026-03-16T12-51-26.783567+05-30.csv",
                "spot_snapshot": "debug_samples/NIFTY_spot_snapshot_2026-03-16T12-50-00+05-30.json",
                "timestamp_x": "2026-03-16T12:50:00+05:30",
                "trade_status_disc": "WATCHLIST",
                "direction_disc": "PUT",
                "direction_source_x": "FLOW+CHARM",
                "confirmation_status_disc": "STRONG_CONFIRMATION",
                "no_trade_reason_code_disc": "PROVIDER_HEALTH_CAUTION_BLOCK",
            }
        ]
    ).to_csv(replay_dir / "confirmation_mode_replay_delta.csv", index=False)

    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-replay-high",
                "signal_timestamp": "2026-03-16T12:50:00+05:30",
                "symbol": "NIFTY",
                "direction": pd.NA,
                "saved_chain_snapshot_path": str(chain_path),
                "selected_expiry": "17-Mar-2026",
                "option_type": pd.NA,
                "strike": pd.NA,
                "spot_at_signal": 23097.3496,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
            }
        ]
    )

    audit = audit_unresolved_signal_contract_matches(frame, project_root=tmp_path, top_n=1)
    proposals = propose_repairs_for_unresolved_signal_contract_matches(audit, project_root=tmp_path)

    assert len(proposals) == 1
    row = proposals.iloc[0]
    assert row["archived_replay_match_type"] == "EXACT_CHAIN_SNAPSHOT"
    assert row["archived_direction"] == "PUT"
    assert row["proposed_direction"] == "PUT"
    assert row["proposal_confidence"] == "HIGH"


def test_apply_repair_proposals_updates_missing_fields_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-accepted",
                "direction": pd.NA,
                "direction_source": pd.NA,
                "option_type": pd.NA,
                "strike": pd.NA,
                "selected_expiry": "2026-03-17",
            },
            {
                "signal_id": "sig-rejected",
                "direction": pd.NA,
                "direction_source": pd.NA,
                "option_type": pd.NA,
                "strike": pd.NA,
                "selected_expiry": pd.NA,
            },
        ]
    )
    proposals = pd.DataFrame(
        [
            {
                "signal_id": "sig-accepted",
                "proposed_direction": "PUT",
                "archived_direction_source": "FLOW+CHARM",
                "proposed_option_type": "PE",
                "proposed_strike": 23000.0,
                "proposed_expiry": "2026-03-17",
                "proposal_confidence": "MEDIUM",
            },
            {
                "signal_id": "sig-rejected",
                "proposed_direction": "CALL",
                "archived_direction_source": "FLOW+GAMMA_FLIP",
                "proposed_option_type": "CE",
                "proposed_strike": 23500.0,
                "proposed_expiry": "2026-03-17",
                "proposal_confidence": "LOW",
            },
        ]
    )

    repaired, summary = apply_repair_proposals_to_dataset(frame, proposals, min_confidence="MEDIUM")

    accepted = repaired.loc[repaired["signal_id"] == "sig-accepted"].iloc[0]
    rejected = repaired.loc[repaired["signal_id"] == "sig-rejected"].iloc[0]

    assert accepted["direction"] == "PUT"
    assert accepted["direction_source"] == "FLOW+CHARM"
    assert accepted["option_type"] == "PE"
    assert float(accepted["strike"]) == 23000.0
    assert accepted["selected_expiry"] == "2026-03-17"
    assert pd.isna(rejected["direction"])
    assert pd.isna(rejected["option_type"])
    assert summary["rows_accepted"] == 1
    assert summary["rows_updated"] == 1


def test_partition_repair_proposals_routes_medium_to_review_queue_in_high_only_mode() -> None:
    proposals = pd.DataFrame(
        [
            {"signal_id": "sig-high", "proposal_confidence": "HIGH"},
            {"signal_id": "sig-medium", "proposal_confidence": "MEDIUM"},
            {"signal_id": "sig-low", "proposal_confidence": "LOW"},
        ]
    )

    accepted, review_queue = partition_repair_proposals(proposals, min_confidence="HIGH")

    assert accepted["signal_id"].tolist() == ["sig-high"]
    assert review_queue["signal_id"].tolist() == ["sig-medium", "sig-low"]


def test_apply_repair_proposals_high_only_leaves_medium_rows_unchanged() -> None:
    frame = pd.DataFrame(
        [
            {"signal_id": "sig-high", "direction": pd.NA, "option_type": pd.NA, "strike": pd.NA},
            {"signal_id": "sig-medium", "direction": pd.NA, "option_type": pd.NA, "strike": pd.NA},
        ]
    )
    proposals = pd.DataFrame(
        [
            {
                "signal_id": "sig-high",
                "proposed_direction": "PUT",
                "proposed_option_type": "PE",
                "proposed_strike": 23000.0,
                "proposal_confidence": "HIGH",
            },
            {
                "signal_id": "sig-medium",
                "proposed_direction": "CALL",
                "proposed_option_type": "CE",
                "proposed_strike": 23500.0,
                "proposal_confidence": "MEDIUM",
            },
        ]
    )

    repaired, summary = apply_repair_proposals_to_dataset(frame, proposals, min_confidence="HIGH")

    high_row = repaired.loc[repaired["signal_id"] == "sig-high"].iloc[0]
    medium_row = repaired.loc[repaired["signal_id"] == "sig-medium"].iloc[0]

    assert high_row["direction"] == "PUT"
    assert high_row["option_type"] == "PE"
    assert float(high_row["strike"]) == 23000.0
    assert pd.isna(medium_row["direction"])
    assert pd.isna(medium_row["option_type"])
    assert summary["rows_accepted"] == 1


def test_partition_repair_proposals_medium_mode_accepts_high_and_medium() -> None:
    proposals = pd.DataFrame(
        [
            {"signal_id": "sig-high", "proposal_confidence": "HIGH"},
            {"signal_id": "sig-medium", "proposal_confidence": "MEDIUM"},
            {"signal_id": "sig-low", "proposal_confidence": "LOW"},
        ]
    )

    accepted, review_queue = partition_repair_proposals(proposals, min_confidence="MEDIUM")

    assert accepted["signal_id"].tolist() == ["sig-high", "sig-medium"]
    assert review_queue["signal_id"].tolist() == ["sig-low"]
