#!/usr/bin/env python3
"""
Comprehensive threshold sweep for provider-health override optimization.

Tests multiple parameter combinations to identify binding constraints and
quantify impact of each parameter on trade pass-through.

Usage:
    python scripts/comprehensive_threshold_sweep.py --snapshot-limit 25 --out-dir research/artifacts/sweep_results
"""

import argparse
import json
import glob
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from app.engine_runner import run_engine_snapshot
from config.signal_policy import TRADE_RUNTIME_THRESHOLDS


def get_recent_snapshot_pairs(limit: int = 25) -> List[Tuple[str, str]]:
    """Get most recent chain/spot snapshot pairs by date/time extraction."""
    from datetime import datetime, timedelta
    
    # Get all chain snapshots, sorted by mtime descending
    chains = sorted(
        glob.glob("debug_samples/*option_chain_snapshot*.csv"),
        key=lambda f: os.path.getmtime(f),
        reverse=True
    )[:limit*2]  # Get more to account for potential mismatches
    
    # Get all spot snapshots sorted by timestamp
    spots = sorted(
        glob.glob("debug_samples/NIFTY_spot_snapshot*.json"),
        key=lambda f: os.path.getmtime(f),
        reverse=True
    )
    
    pairs = []
    
    for chain_file in chains:
        if len(pairs) >= limit:
            break
            
        # Extract timestamp from chain file: e.g. "...2026-03-30T11-23-32..."
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})', chain_file)
        if not m:
            continue
            
        chain_date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        
        # Try to find nearest prior spot snapshot on same day
        # Spot files have format: "...2026-03-30T11-20-00..."
        best_spot = None
        best_time_diff = float('inf')
        
        for spot_file in spots:
            spot_m = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})', spot_file)
            if not spot_m:
                continue
                
            spot_date_str = f"{spot_m.group(1)}-{spot_m.group(2)}-{spot_m.group(3)}"
            
            # Only match same date or prior day
            if spot_date_str != chain_date_str and spot_date_str > chain_date_str:
                continue
                
            # If same date, prefer most recent spot before chain time
            if spot_date_str == chain_date_str:
                chain_hm = f"{m.group(4)}{m.group(5)}"  # e.g., "1123"
                spot_hm = f"{spot_m.group(4)}{spot_m.group(5)}"  # e.g., "1120"
                
                try:
                    chain_mins = int(chain_hm)
                    spot_mins = int(spot_hm)
                    
                    if spot_mins <= chain_mins:
                        time_diff = chain_mins - spot_mins
                        if time_diff < best_time_diff:
                            best_time_diff = time_diff
                            best_spot = spot_file
                except:
                    pass
        
        if best_spot:
            pairs.append((chain_file, best_spot))
    
    return pairs


def run_threshold_config(
    param_overrides: Dict[str, Any],
    snapshot_pairs: List[Tuple[str, str]],
    config_name: str
) -> Dict[str, Any]:
    """
    Run engine with specified threshold overrides on snapshot pairs.
    
    Args:
        param_overrides: Dict of threshold parameter values to override
        snapshot_pairs: List of (chain_path, spot_path) tuples
        config_name: Name for this configuration
        
    Returns:
        Dict with metrics: trade_count, override_count, blocked_reasons_dist, per_snapshot rows
    """
    # Apply overrides to runtime thresholds
    original = TRADE_RUNTIME_THRESHOLDS.copy()
    for key, val in param_overrides.items():
        TRADE_RUNTIME_THRESHOLDS[key] = val
    
    results = {
        "config_name": config_name,
        "param_overrides": param_overrides,
        "snapshot_count": len(snapshot_pairs),
        "trade_count": 0,
        "override_active_count": 0,
        "override_eligible_count": 0,
        "blocked_count": 0,
        "caution_count": 0,
        "blocked_reasons_distribution": {},
        "override_fail_reasons_distribution": {},
        "rows": []
    }
    
    for chain_file, spot_file in snapshot_pairs:
        try:
            output = run_engine_snapshot(
                symbol="NIFTY",
                mode="REPLAY",
                source=f"SWEEP_{config_name}",
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=65,
                max_capital=20000,
                replay_spot=spot_file,
                replay_chain=chain_file,
                replay_dir="debug_samples",
                capture_signal_evaluation=False,
                signal_capture_policy="ALL_SIGNALS",
                previous_chain=None,
                holding_profile="AUTO",
                headline_service=None,
                data_router=None,
            )
            
            trade_status = output.get("trade_status", "UNKNOWN")
            trade_suggestion = output.get("trade_suggestion", {})
            
            row = {
                "chain": os.path.basename(chain_file),
                "spot": os.path.basename(spot_file),
                "trade_status": trade_status,
                "override_active": trade_suggestion.get("provider_health_override_active", False),
                "override_eligible": output.get("provider_health_override_eligible", False),
                "override_reason": trade_suggestion.get("provider_health_override_reason", None),
                "override_fail_reasons": output.get("provider_health_override_diagnostics", {}).get("fail_reasons", []),
                "composite_score": output.get("signal_composite_score", None),
                "trade_strength": output.get("adjusted_trade_strength", None),
                "provider_health_status": output.get("provider_health_status", None),
                "provider_health_message": output.get("provider_health_message", None),
            }
            results["rows"].append(row)
            
            if trade_status == "TRADE":
                results["trade_count"] += 1
                if row["override_active"]:
                    results["override_active_count"] += 1
            
            if row["override_eligible"]:
                results["override_eligible_count"] += 1
            
            if "provider_health_message" in output:
                msg = output["provider_health_message"]
                if "BLOCK" in msg:
                    results["blocked_count"] += 1
                    # Extract block reason
                    if "core_iv_weak" in msg:
                        results["blocked_reasons_distribution"]["core_iv_weak"] = \
                            results["blocked_reasons_distribution"].get("core_iv_weak", 0) + 1
                    if "composite" in msg.lower():
                        results["blocked_reasons_distribution"]["composite_too_low"] = \
                            results["blocked_reasons_distribution"].get("composite_too_low", 0) + 1
                elif "CAUTION" in msg:
                    results["caution_count"] += 1
            
            # Aggregate override failure reasons
            for fail_reason in row["override_fail_reasons"]:
                results["override_fail_reasons_distribution"][fail_reason] = \
                    results["override_fail_reasons_distribution"].get(fail_reason, 0) + 1
                    
        except Exception as e:
            print(f"ERROR processing {chain_file}: {e}")
            continue
    
    # Restore original thresholds
    TRADE_RUNTIME_THRESHOLDS.clear()
    TRADE_RUNTIME_THRESHOLDS.update(original)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive threshold sweep")
    parser.add_argument("--snapshot-limit", type=int, default=25, help="Max snapshot pairs to test")
    parser.add_argument("--out-dir", type=str, default="research/artifacts/sweep_results", help="Output directory")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Get snapshot pairs
    snapshot_pairs = get_recent_snapshot_pairs(args.snapshot_limit)
    if not snapshot_pairs:
        print("ERROR: No snapshot pairs found")
        return
    
    print(f"Found {len(snapshot_pairs)} snapshot pairs")
    
    # Define test configurations
    configurations = {
        "baseline_buffer8_strength12": {
            "provider_health_override_min_composite_buffer": 8,
            "provider_health_override_min_strength_buffer": 12,
        },
        "tuned_buffer4_strength12": {
            "provider_health_override_min_composite_buffer": 4,
            "provider_health_override_min_strength_buffer": 12,
        },
        "tuned_buffer4_strength10": {
            "provider_health_override_min_composite_buffer": 4,
            "provider_health_override_min_strength_buffer": 10,
        },
        "aggressive_buffer2_strength10": {
            "provider_health_override_min_composite_buffer": 2,
            "provider_health_override_min_strength_buffer": 10,
        },
        "aggressive_buffer2_strength8": {
            "provider_health_override_min_composite_buffer": 2,
            "provider_health_override_min_strength_buffer": 8,
        },
        "relaxed_size_cap50_buffer4": {
            "provider_health_override_min_composite_buffer": 4,
            "provider_health_override_min_strength_buffer": 12,
            "provider_health_override_size_cap": 0.50,
        },
    }
    
    all_results = {}
    
    for config_name, overrides in configurations.items():
        print(f"\nTesting config: {config_name}")
        print(f"  Overrides: {overrides}")
        
        result = run_threshold_config(overrides, snapshot_pairs, config_name)
        all_results[config_name] = result
        
        print(f"  Trade count: {result['trade_count']}")
        print(f"  Override eligible: {result['override_eligible_count']}")
        print(f"  Override active: {result['override_active_count']}")
        print(f"  Blocked: {result['blocked_count']}")
        print(f"  Caution: {result['caution_count']}")
        print(f"  Blocked reasons: {result['blocked_reasons_distribution']}")
        print(f"  Override fail reasons: {result['override_fail_reasons_distribution']}")
    
    # Write results
    out_file = os.path.join(args.out_dir, "comprehensive_sweep_results.json")
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results to {out_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<35} {'Trades':<8} {'Override':<10} {'Eligible':<10} {'Blocked':<8}")
    print("-"*80)
    for config_name in sorted(all_results.keys()):
        r = all_results[config_name]
        print(f"{config_name:<35} {r['trade_count']:<8} {r['override_active_count']:<10} {r['override_eligible_count']:<10} {r['blocked_count']:<8}")
    
    # Analyze which configuration performs best
    best_config = max(all_results.items(), key=lambda x: x[1]["trade_count"])
    print(f"\nBest performing config: {best_config[0]} with {best_config[1]['trade_count']} trades")
    
    # Analyze override failure reasons across all configs
    print("\n" + "="*80)
    print("OVERRIDE FAILURE REASONS (AGGREGATE)")
    print("="*80)
    reason_distribution = {}
    for config_name, result in all_results.items():
        for reason, count in result["override_fail_reasons_distribution"].items():
            reason_distribution[reason] = reason_distribution.get(reason, 0) + count
    
    for reason, count in sorted(reason_distribution.items(), key=lambda x: -x[1]):
        print(f"  {reason:<40} {count:>4} occurrences")


if __name__ == "__main__":
    main()
