from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.replay_loader import resolve_nearest_replay_snapshot_paths


ROWS_PATH = REPO_ROOT / "research" / "reviews" / "signal_dataset_stickiness_2026-04-02_cumul_replay" / "signal_dataset_cumul_replay_rows.csv"
DATASET_PATH = REPO_ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul_backfilled.csv"
SPOT_DIR = REPO_ROOT / "debug_samples" / "spot_snapshots"
CHAIN_DIR = REPO_ROOT / "debug_samples" / "option_chain_snapshots"
OUTPUT_DIR = REPO_ROOT / "research" / "reviews" / "snapshot_backfill_2026-04-02"

MAX_SPOT_DELTA_SECONDS = 8 * 3600.0
MAX_CHAIN_DELTA_SECONDS = 8 * 3600.0


def _timestamp_token(ts: pd.Timestamp) -> str:
    token = ts.isoformat()
    token = re.sub(r"T(\d{2}):(\d{2}):(\d{2})", r"T\1-\2-\3", token)
    token = re.sub(r"\+(\d{2}):(\d{2})$", r"+\1-\2", token)
    token = re.sub(r"-(\d{2}):(\d{2})$", r"-\1-\2", token)
    return token


def _ensure_tz(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata")


def _load_missing_timestamps() -> pd.DataFrame:
    rows = pd.read_csv(ROWS_PATH)
    rows["signal_timestamp"] = pd.to_datetime(rows["signal_timestamp"], errors="coerce")
    replay_ok = rows["replay_ok"].astype(str).str.strip().str.lower()
    missing = rows[
        (
            rows["replay_error"].astype(str).str.contains("missing_saved_snapshot", case=False, na=False)
            | replay_ok.isin(["false", "0"])
        )
        & rows["signal_timestamp"].notna()
    ].copy()
    missing = missing[["signal_timestamp"]].drop_duplicates().sort_values("signal_timestamp").reset_index(drop=True)
    return missing


def _lookup_symbol_source() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce")
    df = df[df["signal_timestamp"].notna()].copy()
    df = df[["signal_timestamp", "symbol", "source"]].drop_duplicates(subset=["signal_timestamp"], keep="last")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPOT_DIR.mkdir(parents=True, exist_ok=True)
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)

    missing = _load_missing_timestamps()
    symbol_source = _lookup_symbol_source()
    joined = missing.merge(symbol_source, on="signal_timestamp", how="left")
    joined["symbol"] = joined["symbol"].fillna("NIFTY").astype(str)
    joined["source"] = joined["source"].fillna("ICICI").astype(str)

    results = []
    created_spot = 0
    created_chain = 0
    unresolved = 0

    for row in joined.itertuples(index=False):
        ts = _ensure_tz(pd.to_datetime(row.signal_timestamp, errors="coerce"))
        symbol = str(row.symbol or "NIFTY").upper().strip()
        source = str(row.source or "ICICI").upper().strip()
        token = _timestamp_token(ts)

        target_spot = SPOT_DIR / f"{symbol}_spot_snapshot_{token}.json"
        target_chain = CHAIN_DIR / f"{symbol}_{source}_option_chain_snapshot_{token}.csv"

        nearest = resolve_nearest_replay_snapshot_paths(
            symbol,
            target_timestamp=ts,
            replay_dir=str(REPO_ROOT / "debug_samples"),
            source_label=source,
            max_spot_delta_seconds=MAX_SPOT_DELTA_SECONDS,
            max_chain_delta_seconds=MAX_CHAIN_DELTA_SECONDS,
        )

        source_spot = Path(nearest["spot_path"]) if nearest.get("spot_path") else None
        source_chain = Path(nearest["chain_path"]) if nearest.get("chain_path") else None

        spot_status = "existing"
        chain_status = "existing"

        if not target_spot.exists():
            if source_spot and source_spot.exists():
                payload = json.loads(source_spot.read_text(encoding="utf-8"))
                payload["timestamp"] = ts.isoformat()
                target_spot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                created_spot += 1
                spot_status = "created"
            else:
                spot_status = "missing_source"

        if not target_chain.exists():
            if source_chain and source_chain.exists():
                shutil.copy2(source_chain, target_chain)
                created_chain += 1
                chain_status = "created"
            else:
                chain_status = "missing_source"

        if spot_status == "missing_source" or chain_status == "missing_source":
            unresolved += 1

        results.append(
            {
                "signal_timestamp": ts.isoformat(),
                "symbol": symbol,
                "source": source,
                "target_spot_path": str(target_spot.relative_to(REPO_ROOT)),
                "target_chain_path": str(target_chain.relative_to(REPO_ROOT)),
                "spot_status": spot_status,
                "chain_status": chain_status,
                "source_spot_path": str(source_spot.relative_to(REPO_ROOT)) if source_spot else None,
                "source_chain_path": str(source_chain.relative_to(REPO_ROOT)) if source_chain else None,
                "spot_delta_seconds": nearest.get("spot_delta_seconds"),
                "chain_delta_seconds": nearest.get("chain_delta_seconds"),
                "selection_reason": nearest.get("selection_reason"),
            }
        )

    mapping_path = OUTPUT_DIR / "archive_gap_fill_mapping.csv"
    summary_path = OUTPUT_DIR / "archive_gap_fill_summary.json"
    md_path = OUTPUT_DIR / "archive_gap_fill_summary.md"

    mapping_df = pd.DataFrame(results)
    mapping_df.to_csv(mapping_path, index=False)

    summary = {
        "rows_targeted": int(len(joined)),
        "spot_created": int(created_spot),
        "chain_created": int(created_chain),
        "unresolved_rows": int(unresolved),
        "max_spot_delta_seconds": MAX_SPOT_DELTA_SECONDS,
        "max_chain_delta_seconds": MAX_CHAIN_DELTA_SECONDS,
        "mapping_path": str(mapping_path.relative_to(REPO_ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Archive Gap Fill Summary",
                "",
                f"- Rows targeted: `{summary['rows_targeted']}`",
                f"- Spot snapshots created: `{summary['spot_created']}`",
                f"- Chain snapshots created: `{summary['chain_created']}`",
                f"- Unresolved rows: `{summary['unresolved_rows']}`",
                f"- Mapping CSV: `{summary['mapping_path']}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()