from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

from research.signal_evaluation import load_signals_dataset, write_signals_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_approve_repair_review_queue_end_to_end(tmp_path: Path) -> None:
    dataset_path = tmp_path / "signals_dataset_cumul.csv"
    source_frame = pd.DataFrame(
        [
            {
                "signal_id": "sig-medium",
                "signal_timestamp": "2026-03-16T14:05:00+05:30",
                "symbol": "NIFTY",
                "direction": pd.NA,
                "direction_source": pd.NA,
                "option_type": pd.NA,
                "strike": pd.NA,
                "selected_expiry": "17-Mar-2026",
                "saved_chain_snapshot_path": pd.NA,
                "selected_option_last_price": pd.NA,
                "selected_option_iv": pd.NA,
                "selected_option_delta": pd.NA,
            }
        ]
    )
    write_signals_dataset(source_frame, dataset_path)

    review_queue_path = tmp_path / "repair_review_queue.csv"
    review_queue = pd.DataFrame(
        [
            {
                "signal_id": "sig-medium",
                "signal_timestamp": "2026-03-16T14:05:00+05:30",
                "proposal_confidence": "MEDIUM",
                "proposed_direction": "PUT",
                "archived_direction_source": "FLOW+CHARM",
                "proposed_option_type": "PE",
                "proposed_strike": 23000.0,
                "proposed_expiry": "2026-03-17",
            }
        ]
    )
    review_queue.to_csv(review_queue_path, index=False)

    output_dir = tmp_path / "approval_output"
    script_path = PROJECT_ROOT / "scripts" / "approve_repair_review_queue.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--review-queue-csv",
            str(review_queue_path),
            "--dataset-path",
            str(dataset_path),
            "--signal-ids",
            "sig-medium",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout

    summary_path = output_dir / "review_queue_approval_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["approved_rows"] == 1
    assert summary["remaining_review_queue_rows"] == 0
    assert summary["promoted_to_source_dataset"] is False

    repaired_dataset_path = Path(summary["repaired_dataset_path"])
    assert repaired_dataset_path.exists()
    repaired = load_signals_dataset(repaired_dataset_path)
    row = repaired.loc[repaired["signal_id"] == "sig-medium"].iloc[0]
    assert str(row["direction"]).upper() == "PUT"
    assert str(row["option_type"]).upper() == "PE"
    assert float(row["strike"]) == 23000.0
