from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.replay_loader import resolve_nearest_replay_snapshot_paths


DATASET_PATH = REPO_ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul.csv"
OUTPUT_DATASET_PATH = REPO_ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul_backfilled.csv"
OUTPUT_DIR = REPO_ROOT / "research" / "reviews" / "snapshot_backfill_2026-04-02"
MAX_SPOT_DELTA_SECONDS = 7200.0
MAX_CHAIN_DELTA_SECONDS = 14400.0


def _to_repo_relative(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce")

    records = []
    backfilled_rows = 0
    exact_rows = 0

    for idx, row in df.iterrows():
        spot_raw = row.get("saved_spot_snapshot_path")
        chain_raw = row.get("saved_chain_snapshot_path")
        spot_path = REPO_ROOT / str(spot_raw) if pd.notna(spot_raw) and str(spot_raw).strip() else None
        chain_path = REPO_ROOT / str(chain_raw) if pd.notna(chain_raw) and str(chain_raw).strip() else None

        spot_exists = bool(spot_path and spot_path.exists())
        chain_exists = bool(chain_path and chain_path.exists())
        exact_ok = spot_exists and chain_exists

        nearest = resolve_nearest_replay_snapshot_paths(
            str(row.get("symbol") or "NIFTY"),
            target_timestamp=row.get("signal_timestamp"),
            replay_dir=str(REPO_ROOT / "debug_samples"),
            source_label=str(row.get("source") or "ICICI"),
            max_spot_delta_seconds=MAX_SPOT_DELTA_SECONDS,
            max_chain_delta_seconds=MAX_CHAIN_DELTA_SECONDS,
        )

        chosen_spot = str(spot_path) if spot_exists else nearest.get("spot_path")
        chosen_chain = str(chain_path) if chain_exists else nearest.get("chain_path")
        if exact_ok:
            exact_rows += 1
        elif chosen_spot or chosen_chain:
            backfilled_rows += 1

        df.at[idx, "saved_spot_snapshot_path"] = _to_repo_relative(chosen_spot)
        df.at[idx, "saved_chain_snapshot_path"] = _to_repo_relative(chosen_chain)
        df.at[idx, "snapshot_backfill_status"] = (
            "exact"
            if exact_ok
            else "backfilled"
            if (chosen_spot or chosen_chain)
            else "missing"
        )
        df.at[idx, "snapshot_backfill_spot_delta_seconds"] = nearest.get("spot_delta_seconds")
        df.at[idx, "snapshot_backfill_chain_delta_seconds"] = nearest.get("chain_delta_seconds")

        if idx < 5000:
            records.append(
                {
                    "signal_timestamp": str(row.get("signal_timestamp")),
                    "symbol": row.get("symbol"),
                    "source": row.get("source"),
                    "original_spot_path": spot_raw,
                    "original_chain_path": chain_raw,
                    "resolved_spot_path": _to_repo_relative(chosen_spot),
                    "resolved_chain_path": _to_repo_relative(chosen_chain),
                    "spot_exists_exact": spot_exists,
                    "chain_exists_exact": chain_exists,
                    "spot_delta_seconds": nearest.get("spot_delta_seconds"),
                    "chain_delta_seconds": nearest.get("chain_delta_seconds"),
                    "selection_reason": nearest.get("selection_reason"),
                    "backfill_status": df.at[idx, "snapshot_backfill_status"],
                }
            )

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATASET_PATH, index=False)
    mapping_df = pd.DataFrame(records)
    mapping_df.to_csv(OUTPUT_DIR / "snapshot_backfill_mapping.csv", index=False)

    summary = {
        "dataset_path": str(DATASET_PATH),
        "output_dataset_path": str(OUTPUT_DATASET_PATH),
        "rows": int(len(df)),
        "exact_rows": int(exact_rows),
        "backfilled_rows": int(backfilled_rows),
        "missing_rows": int((df["snapshot_backfill_status"] == "missing").sum()),
        "coverage_after_backfill": round(float((df["snapshot_backfill_status"] != "missing").mean()), 4),
        "max_spot_delta_seconds": MAX_SPOT_DELTA_SECONDS,
        "max_chain_delta_seconds": MAX_CHAIN_DELTA_SECONDS,
        "spot_delta_seconds_median": pd.to_numeric(df["snapshot_backfill_spot_delta_seconds"], errors="coerce").median(),
        "chain_delta_seconds_median": pd.to_numeric(df["snapshot_backfill_chain_delta_seconds"], errors="coerce").median(),
    }
    (OUTPUT_DIR / "snapshot_backfill_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "snapshot_backfill_summary.md").write_text(
        "\n".join(
            [
                "# Snapshot Backfill Summary",
                "",
                f"- Input dataset: `{DATASET_PATH}`",
                f"- Output dataset: `{OUTPUT_DATASET_PATH}`",
                f"- Coverage after backfill: `{summary['coverage_after_backfill']}`",
                f"- Exact rows: `{summary['exact_rows']}`",
                f"- Backfilled rows: `{summary['backfilled_rows']}`",
                f"- Missing rows: `{summary['missing_rows']}`",
                f"- Median spot delta seconds: `{summary['spot_delta_seconds_median']}`",
                f"- Median chain delta seconds: `{summary['chain_delta_seconds_median']}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()