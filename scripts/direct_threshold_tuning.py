#!/usr/bin/env python3
"""
Direct threshold tuning using in-process configuration modification.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/Users/pramitdutta/Desktop/Quant Engines/options_quant_engine')

from config import signal_policy as sp
from app.engine_runner import run_engine_snapshot
from scripts.run_provider_override_sweep import build_pairs

# Create output directory
Path('research/artifacts/sweep_results').mkdir(parents=True, exist_ok=True)

# Define test configurations
configs = [
    ("baseline_buf8_str12", 8, 12),
    ("tuned_buf4_str12", 4, 12),
    ("tuned_buf4_str10", 4, 10),
    ("aggressive_buf3_str10", 3, 10),
    ("aggressive_buf2_str10", 2, 10),
    ("middle_buf6_str11", 6, 11),
]

results_all = {}

print("="*80)
print("COMPREHENSIVE THRESHOLD TUNING")
print("="*80)

for config_name, buffer, strength in configs:
    print(f"\nTesting: {config_name} (buffer={buffer}, strength={strength})")
    
    # Apply configuration
    sp.TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_composite_buffer'] = buffer
    sp.TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_strength_buffer'] = strength
    
    rows = []
    trade_count = 0
    override_count = 0
    
    try:
        for i, (spot_path, chain_path) in enumerate(build_pairs(limit=20)):
            result = run_engine_snapshot(
                symbol='NIFTY',
                mode='REPLAY',
                source=f'TUNE_{config_name}',
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=65,
                max_capital=20000,
                replay_spot=spot_path,
                replay_chain=chain_path,
                capture_signal_evaluation=False,
                signal_capture_policy='ALL_SIGNALS',
            )
            
            trade = result.get('trade') or {}
            is_trade = trade.get('signal_status') == 'PASSED_ALL_GATES'
            is_override = trade.get('provider_health_override_active', False)
            
            if is_trade:
                trade_count += 1
            if is_override:
                override_count += 1
            
            rows.append({
                'chain': chain_path.split('/')[-1],
                'spot': spot_path.split('/')[-1],
                'trade_status': 'TRADE' if is_trade else 'BLOCKED',
                'override_active': is_override,
                'trade_strength': result.get('adjusted_trade_strength'),
                'composite_score': result.get('signal_composite_score'),
                'provider_message': result.get('provider_health_message'),
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/20 snapshots...")
        
        result_obj = {
            'config': config_name,
            'buffer': buffer,
            'strength': strength,
            'trade_count': trade_count,
            'override_count': override_count,
            'total_snapshots': len(rows),
            'rows': rows,
        }
        
        results_all[config_name] = result_obj
        
        out_file = f'research/artifacts/sweep_results/{config_name}.json'
        with open(out_file, 'w') as f:
            json.dump(result_obj, f, indent=2)
        
        print(f"  ✓ Completed: {trade_count} trades, {override_count} overrides")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Print summary
print("\n" + "="*80)
print("TUNING RESULTS SUMMARY")
print("="*80)
print(f"\n{'Config':<30} {'Buffer':<8} {'Strength':<10} {'Trades':<10} {'Overrides':<10}")
print("-"*80)

for config_name, buffer, strength in configs:
    if config_name in results_all:
        r = results_all[config_name]
        print(f"{config_name:<30} {r['buffer']:<8} {r['strength']:<10} {r['trade_count']:<10} {r['override_count']:<10}")

# Analysis
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)

if results_all:
    best_config = max(results_all.items(), key=lambda x: x[1]['trade_count'])
    print(f"\n✓ Best performing configuration:")
    print(f"  Config: {best_config[0]}")
    print(f"  Trades: {best_config[1]['trade_count']}")
    print(f"  Overrides: {best_config[1]['override_count']}")
    
    baseline = results_all.get("baseline_buf8_str12")
    tuned_buf = results_all.get("tuned_buf4_str12")
    tuned_both = results_all.get("tuned_buf4_str10")
    
    if baseline and tuned_buf:
        lift = tuned_buf['trade_count'] - baseline['trade_count']
        print(f"\n📊 Buffer tuning impact (8→4, strength=12):")
        print(f"  Baseline: {baseline['trade_count']} trades")
        print(f"  Tuned: {tuned_buf['trade_count']} trades")
        print(f"  Delta: {lift:+d} trades")
    
    if tuned_buf and tuned_both:
        lift = tuned_both['trade_count'] - tuned_buf['trade_count']
        print(f"\n📊 Strength tuning impact (12→10, buffer=4):")
        print(f"  Buffer-only: {tuned_buf['trade_count']} trades")
        print(f"  Buffer+Strength: {tuned_both['trade_count']} trades")
        print(f"  Delta: {lift:+d} trades")
    
    # Check which parameters matter more
    print(f"\n📈 Parameter sensitivity:")
    variance_buffer = []
    variance_strength = []
    
    for cfg_name, res in results_all.items():
        if res['buffer'] in [4, 8] and res['strength'] == 12:  # Vary buffer, fix strength
            variance_buffer.append(res['trade_count'])
        if res['buffer'] == 4 and res['strength'] in [10, 12]:  # Fix buffer, vary strength
            variance_strength.append(res['trade_count'])
    
    if variance_buffer:
        print(f"  Buffer variance (fixed strength=12): {max(variance_buffer,default=0) - min(variance_buffer,default=0)} trade difference")
    if variance_strength:
        print(f"  Strength variance (fixed buffer=4): {max(variance_strength,default=0) - min(variance_strength,default=0)} trade difference")

# Save comprehensive results
summary_file = 'research/artifacts/sweep_results/COMPREHENSIVE_TUNING_RESULTS.json'
with open(summary_file, 'w') as f:
    json.dump({
        'configurations_tested': configs,
        'detailed_results': results_all,
        'best_config': best_config[0] if results_all else None,
        'summary_stats': {
            'total_configs': len(configs),
            'configs_successful': len(results_all),
            'max_trades': max((r['trade_count'] for r in results_all.values()), default=0),
        }
    }, f, indent=2)

print(f"\n✓ All results saved to:")
print(f"  - Individual configs: research/artifacts/sweep_results/*.json")
print(f"  - Summary: {summary_file}")

if results_all:
    print(f"\n✓ Recommendation:")
    best = max(results_all.items(), key=lambda x: x[1]['trade_count'])
    print(f"  Deploy with: buffer={best[1]['buffer']}, strength={best[1]['strength']}")
    print(f"  Expected trades per 20-snapshot window: {best[1]['trade_count']}")
