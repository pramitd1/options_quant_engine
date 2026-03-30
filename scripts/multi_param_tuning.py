#!/usr/bin/env python3
"""
Multi-parameter threshold tuning analysis using existing sweep framework.
Tests various parameter combinations and analyzes results.
"""

import json
import subprocess
import os
from pathlib import Path

def run_sweep(buffer_val, strength_val):
    """Run a single sweep configuration."""
    config_name = f"buffer{buffer_val}_strength{strength_val}"
    out_file = f"research/artifacts/sweep_results/{config_name}.json"
    
    # Modify config in memory and run sweep
    cmd = f"""PYTHONPATH=. "/Users/pramitdutta/Desktop/Quant Engines/options_quant_engine/.venv/bin/python" -c "
import sys
sys.path.insert(0, '.')
from config import signal_policy as sp
from scripts.run_provider_override_sweep import build_pairs, run_engine_snapshot

# Override thresholds
sp.TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_composite_buffer'] = {buffer_val}
sp.TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_strength_buffer'] = {strength_val}

rows = []
for spot_path, chain_path in build_pairs(limit=20):
    result = run_engine_snapshot(
        symbol='NIFTY',
        mode='REPLAY',
        source='TUNING',
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=65,
        max_capital=20000,
        replay_spot=spot_path,
        replay_chain=chain_path,
        replay_dir='debug_samples',
        capture_signal_evaluation=False,
        signal_capture_policy='ALL_SIGNALS',
        previous_chain=None,
        holding_profile='AUTO',
        headline_service=None,
        data_router=None,
    )
    
    trade = result.get('trade') or {{}}
    trade_status = 'TRADE' if (trade.get('signal_status') == 'PASSED_ALL_GATES') else 'BLOCKED'
    
    row = {{
        'chain': chain_path.split('/')[-1],
        'spot': spot_path.split('/')[-1],
        'trade_status': trade_status,
        'signal_status': trade.get('signal_status'),
        'trade_strength': result.get('adjusted_trade_strength'),
        'composite_score': result.get('signal_composite_score'),
        'provider_status': result.get('provider_health_status'),
        'provider_message': result.get('provider_health_message'),
        'override_active': trade.get('provider_health_override_active', False),
        'override_reason': trade.get('provider_health_override_reason'),
    }}
    rows.append(row)

import json
result = {{
    'config_name': '{config_name}',
    'buffer': {buffer_val},
    'strength': {strength_val},
    'trade_count': sum(1 for r in rows if r['trade_status'] == 'TRADE'),
    'override_count': sum(1 for r in rows if r.get('override_active')),
    'rows': rows,
}}

with open('{out_file}', 'w') as f:
    json.dump(result, f, indent=2)
    
print(f'Config: {config_name}, Trades: {{result[\"trade_count\"]}}, Override: {{result[\"override_count\"]}}')
"
"""
    os.system(f"cd /Users/pramitdutta/Desktop/Quant\\ Engines/options_quant_engine && {cmd}")
    
    try:
        with open(out_file) as f:
            return json.load(f)
    except:
        return None


def main():
    Path("research/artifacts/sweep_results").mkdir(parents=True, exist_ok=True)
    
    # Test configurations
    configs = [
        (8, 12, "baseline_current"),
        (4, 12, "tuned_buffer_only"),
        (4, 10, "tuned_buffer_strength"),
        (2, 12, "aggressive_buffer"),
        (2, 10, "aggressive_both"),
        (6, 11, "middle_ground"),
    ]
    
    results_summary = {}
    
    print("Running comprehensive threshold tuning...")
    print("="*70)
    
    for buffer, strength, label in configs:
        print(f"\nTesting: {label} (buffer={buffer}, strength={strength})")
        result = run_sweep(buffer, strength)
        if result:
            results_summary[f"buffer{buffer}_strength{strength}"] = result
            print(f"  Result: {result['trade_count']} trades, {result['override_count']} overrides")
        else:
            print(f"  ERROR: Failed to run sweep")
    
    # Save summary
    summary_file = "research/artifacts/sweep_results/comprehensive_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Trades':<10} {'Overrides':<10}")
    print("-"*70)
    
    for config_name, result in sorted(results_summary.items()):
        print(f"{config_name:<30} {result['trade_count']:<10} {result['override_count']:<10}")
    
    # Find best configs
    best_config = max(results_summary.items(), key=lambda x: x[1]['trade_count'])
    print(f"\nBest performer: {best_config[0]} with {best_config[1]['trade_count']} trades")
    
    print(f"\nDetailed results saved to: {summary_file}")


if __name__ == "__main__":
    main()
