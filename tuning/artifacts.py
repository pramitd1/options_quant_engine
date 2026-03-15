"""
Common file-artifact helpers for tuning research outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def append_jsonl_record(record: dict[str, Any], path: str | Path) -> Path:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record or {}), sort_keys=True))
        handle.write("\n")
    return ledger_path


def load_jsonl_frame(path: str | Path) -> pd.DataFrame:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return pd.DataFrame()

    records = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)
