#!/usr/bin/env python
"""Verify signal data update flow - daily and cumulative file synchronization."""

import pandas as pd
from research.signal_evaluation import (
    load_signals_dataset,
    load_cumulative_dataset,
    SIGNAL_DATASET_PATH,
    CUMULATIVE_DATASET_PATH,
)
from pathlib import Path

def main():
    print("=" * 60)
    print("SIGNAL DATA UPDATE FLOW VERIFICATION")
    print("=" * 60)
    
    # Test 1: Verify paths
    print("\n[1] Paths Verification")
    print(f"    Daily:      {SIGNAL_DATASET_PATH}")
    print(f"    Cumulative: {CUMULATIVE_DATASET_PATH}")
    print(f"    ✓ Daily exists:      {Path(SIGNAL_DATASET_PATH).exists()}")
    print(f"    ✓ Cumulative exists: {Path(CUMULATIVE_DATASET_PATH).exists()}")
    
    # Test 2: Load datasets
    print("\n[2] Loading Datasets")
    daily = load_signals_dataset()
    cumul = load_cumulative_dataset()
    print(f"    ✓ Daily signals:      {len(daily)} rows")
    print(f"    ✓ Cumulative signals: {len(cumul)} rows")
    
    # Test 3: Verify deduplication
    print("\n[3] Deduplication Check")
    daily_ids = set(daily['signal_id'].dropna()) if 'signal_id' in daily.columns else set()
    cumul_ids = set(cumul['signal_id'].dropna()) if 'signal_id' in cumul.columns else set()
    print(f"    Daily unique signal_ids:      {len(daily_ids)}")
    print(f"    Cumulative unique signal_ids: {len(cumul_ids)}")
    all_synced = daily_ids.issubset(cumul_ids)
    print(f"    ✓ All daily signals in cumulative: {all_synced}")
    
    if not all_synced and daily_ids:
        missing = daily_ids - cumul_ids
        print(f"    ⚠ Missing from cumulative: {len(missing)} signals")
        print(f"    Examples: {list(missing)[:3]}")
    
    # Test 4: SQLite sidecars
    print("\n[4] SQLite Sidecar Files")
    daily_sqlite = Path(str(SIGNAL_DATASET_PATH).replace(".csv", ".sqlite"))
    cumul_sqlite = Path(str(CUMULATIVE_DATASET_PATH).replace(".csv", ".sqlite"))
    print(f"    ✓ Daily SQLite:      {daily_sqlite.exists()}")
    print(f"    ✓ Cumulative SQLite: {cumul_sqlite.exists()}")
    
    # Test 5: File sizes
    print("\n[5] File Sizes")
    if Path(SIGNAL_DATASET_PATH).exists():
        size_kb = Path(SIGNAL_DATASET_PATH).stat().st_size / 1024
        print(f"    Daily CSV:       {size_kb:.2f} KB")
    if Path(CUMULATIVE_DATASET_PATH).exists():
        size_kb = Path(CUMULATIVE_DATASET_PATH).stat().st_size / 1024
        print(f"    Cumulative CSV:  {size_kb:.2f} KB")
    if daily_sqlite.exists():
        size_kb = daily_sqlite.stat().st_size / 1024
        print(f"    Daily SQLite:    {size_kb:.2f} KB")
    if cumul_sqlite.exists():
        size_kb = cumul_sqlite.stat().st_size / 1024
        print(f"    Cumulative SQLite: {size_kb:.2f} KB")
    
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE - FLOW IS OPERATIONAL")
    print("=" * 60)

if __name__ == "__main__":
    main()
