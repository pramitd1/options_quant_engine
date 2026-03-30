#!/usr/bin/env python3
"""
Unified threshold tuning harness using working sweep mechanism.
Tests multiple parameter combinations systematically.
"""

import json
import os
import subprocess
from pathlib import Path

def run_sweep_config(buffer, strength, limit=20):
    """Run sweep with specific buffer and strength parameters."""
    config_name = f"buf{buffer}_str{strength}"
    out_file = f"research/artifacts/sweep_results/{config_name}.json"
    
    cmd = (
        f'cd /Users/pramitdutta/Desktop/Quant\\ Engines/options_quant_engine && '
        f'PYTHONPATH=. /Users/pramitdutta/Desktop/Quant\\ Engines/options_quant_engine/.venv/bin/python -c '
        f'"'
        f'import sys, json; '
        f'sys.path.insert(0, "."); '
        f'from config import signal_policy as sp; '
        f'from app.engine_runner import run_engine_snapshot; '
        f'from scripts.run_provider_override_sweep import build_pairs; '
        f'sp.TRADE_RUNTIME_THRESHOLDS["provider_health_override_min_composite_buffer"] = {buffer}; '
        f'sp.TRADE_RUNTIME_THRESHOLDS["provider_health_override_min_strength_buffer"] = {strength}; '
        f'rows = []; '
        f'for spot, chain in build_pairs(limit={limit}): '
        f'  r = run_engine_snapshot(symbol="NIFTY", mode="REPLAY", source="TUNE", apply_budget_constraint=False, requested_lots=1, lot_size=65, max_capital=20000, replay_spot=spot, replay_chain=chain); '
        f'  t = r.get("trade") or {{}}; '
        f'  rows.append({{"chain": chain.split("/")[-1], "trade_status": "TRADE" if t.get("signal_status") == "PASSED_ALL_GATES" else "BLOCKED", "strength": r.get("adjusted_trade_strength"), "composite": r.get("signal_composite_score"), "override": t.get("provider_health_override_active", False)}}); '
        f'result = {{"config": "{config_name}", "buffer": {buffer}, "strength": {strength}, "trade_count": sum(1 for x in rows if x["trade_status"] == "TRADE"), "override_count": sum(1 for x in rows if x["override"]), "rows": rows}}; '
        f'json.dump(result, open("{out_file}", "w"), indent=2); '
        f'print(f"Completed {{config_name}}: {{result[\"trade_count\"]}} trades")' 
        f'"'
    )
    
    print(f"Running: {config_name}...")
    ret = os.system(cmd + " 2>/dev/null")
    
    try:
        with open(out_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    Path("research/artifacts/sweep_results").mkdir(parents=True, exist_ok=True)
    
    # Define test configurations
    test_configs = [
        # (buffer, strength, description)
        (8, 12, "BASELINE"),
        (4, 12, "TUNED_BUFFER_ONLY"),
        (4, 10, "TUNED_BUFFER_AND_STRENGTH"),
        (3, 10, "AGGRESSIVE_BUFFER"),
        (2, 10, "VERY_AGGRESSIVE"),
        (6, 11, "MIDDLE_GROUND"),
    ]
    
    print("="*80)
    print("COMPREHENSIVE THRESHOLD TUNING SWEEP")
    print("="*80)
    
    all_results = {}
    
    for buffer, strength, desc in test_configs:
        print(f"\n[{desc}] Testing buffer={buffer}, strength={strength}")
        result = run_sweep_config(buffer, strength, limit=20)
        if result:
            all_results[f"buf{buffer}_str{strength}"] = result
            print(f"  ✓ Result: {result['trade_count']} trades | {result['override_count']} overrides")
        else:
            print(f"  ✗ Failed")
    
    # Summary
    print("\n" + "="*80)
    print("TUNING RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Config':<25} {'Description':<20} {'Trades':<10} {'Overrides':<10}")
    print("-"*80)
    
    config_map = {f"buf{b}_str{s}": d for b, s, d in test_configs}
    
    for cfg_name in sorted(all_results.keys()):
        result = all_results[cfg_name]
        desc = config_map.get(cfg_name, "")
        print(f"{cfg_name:<25} {desc:<20} {result['trade_count']:<10} {result['override_count']:<10}")
    
    # Analysis
    if all_results:
        best_config = max(all_results.items(), key=lambda x: x[1]['trade_count'])
        baseline = all_results.get("buf8_str12", {})
        tuned = all_results.get("buf4_str12", {})
        strength_tuned = all_results.get("buf4_str10", {})
        
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        print(f"\nBest performer: {best_config[0]}")
        print(f"  Trades: {best_config[1]['trade_count']}")
        print(f"  Override activations: {best_config[1]['override_count']}")
        
        if baseline:
            print(f"\nImpact of buffer tuning (8→4, strength=12):")
            lift = (tuned.get('trade_count', 0) - baseline.get('trade_count', 0)) if tuned else 0
            print(f"  Baseline trades: {baseline.get('trade_count', 0)}")
            print(f"  Tuned trades: {tuned.get('trade_count', 0) if tuned else 'N/A'}")
            print(f"  Additional trades: {lift}")
        
        if strength_tuned:
            print(f"\nImpact of strength tuning (12→10, buffer=4):")
            lift = strength_tuned.get('trade_count', 0) - (tuned.get('trade_count', 0) if tuned else 0)
            print(f"  Buffer-tuned trades: {tuned.get('trade_count', 0) if tuned else 'N/A'}")
            print(f"  Strength+buffer-tuned trades: {strength_tuned.get('trade_count', 0)}")
            print(f"  Additional trades from strength: {lift}")
    
    # Save comprehensive results
    summary_file = "research/artifacts/sweep_results/tuning_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Full results saved to: {summary_file}")


if __name__ == "__main__":
    main()
