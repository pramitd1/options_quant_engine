#!/usr/bin/env python3
"""
Simple threshold tuning by running existing sweep framework multiple times.
"""

import subprocess
import json
import os

configs = [
    ("baseline_buf8_str12", 8, 12),
    ("tuned_buf4_str12", 4, 12),
    ("aggressive_buf4_str10", 4, 10),
    ("aggressive_buf3_str10", 3, 10),
    ("aggressive_buf2_str10", 2, 10),
    ("middle_buf6_str11", 6, 11),
]

os.makedirs("research/artifacts/sweep_results", exist_ok=True)

results = {}

for config_name, buffer, strength in configs:
    print(f"\nTesting: {config_name} (buffer={buffer}, strength={strength})")
    
    # Create inline Python that modifies config and runs the sweep
    py_code = f"""
import sys
sys.path.insert(0, '.')
from config.signal_policy import TRADE_RUNTIME_THRESHOLDS
TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_composite_buffer'] = {buffer}
TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_strength_buffer'] = {strength}

# Now run the sweep
from scripts.run_provider_override_sweep import build_pairs
from app.engine_runner import run_engine_snapshot
import json

rows = []
trade_count = 0
override_count = 0

for spot_path, chain_path in build_pairs(limit=20):
    result = run_engine_snapshot(
        symbol='NIFTY',
        mode='REPLAY',
        source='TUNE',
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=65,
        max_capital=20000,
        replay_spot=spot_path,
        replay_chain=chain_path,
        capture_signal_evaluation=False,
        signal_capture_policy='ALL_SIGNALS',
    )
    
    trade = result.get('trade') or {{}}
    is_trade = trade.get('signal_status') == 'PASSED_ALL_GATES'
    is_override = trade.get('provider_health_override_active', False)
    
    if is_trade:
        trade_count += 1
    if is_override:
        override_count += 1
    
    rows.append({{
        'chain': chain_path.split('/')[-1],
        'spot': spot_path.split('/')[-1],
        'trade': is_trade,
        'override': is_override,
        'strength': result.get('adjusted_trade_strength'),
        'composite': result.get('signal_composite_score'),
    }})

result_obj = {{
    'config': '{config_name}',
    'buffer': {buffer},
    'strength': {strength},
    'trade_count': trade_count,
    'override_count': override_count,
    'total_snapshots': len(rows),
    'rows': rows,
}}

out_file = 'research/artifacts/sweep_results/{config_name}.json'
with open(out_file, 'w') as f:
    json.dump(result_obj, f, indent=2)

print(f"{{config_name}}: {{trade_count}} trades, {{override_count}} overrides")
"""
    
    cmd = (
        f'cd /Users/pramitdutta/Desktop/Quant\\ Engines/options_quant_engine && '
        f'PYTHONPATH=. /Users/pramitdutta/Desktop/Quant\\ Engines/options_quant_engine/.venv/bin/python -c '
        f'"{py_code}"'
    )
    
    ret = os.system(f"{cmd} 2>/dev/null")
    
    # Load result
    try:
        with open(f"research/artifacts/sweep_results/{config_name}.json") as f:
            result = json.load(f)
            results[config_name] = result
            print(f"  ✓ Trades: {result['trade_count']}, Overrides: {result['override_count']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Print summary
print("\n" + "="*80)
print("COMPREHENSIVE TUNING SUMMARY")
print("="*80)
print(f"\n{'Config':<30} {'Buffer':<8} {'Strength':<10} {'Trades':<10} {'Overrides':<10}")
print("-"*80)

for cfg_name, buffer, strength in configs:
    if cfg_name in results:
        r = results[cfg_name]
        print(f"{cfg_name:<30} {r['buffer']:<8} {r['strength']:<10} {r['trade_count']:<10} {r['override_count']:<10}")

# Analysis
if results:
    best_config = max(results.items(), key=lambda x: x[1]['trade_count'])
    print(f"\nBest config: {best_config[0]} with {best_config[1]['trade_count']} trades")
    
    baseline = results.get("baseline_buf8_str12", {})
    tuned = results.get("tuned_buf4_str12", {})
    aggressive = results.get("aggressive_buf4_str10", {})
    
    if baseline and tuned:
        print(f"\nBuffer tuning impact (8→4):")
        print(f"  Baseline: {baseline['trade_count']} trades")
        print(f"  Tuned: {tuned['trade_count']} trades")
        print(f"  Lift: {tuned['trade_count'] - baseline['trade_count']}")
    
    if tuned and aggressive:
        print(f"\nStrength tuning impact (12→10):")
        print(f"  Buffer-only: {tuned['trade_count']} trades")
        print(f"  Buffer+strength: {aggressive['trade_count']} trades")
        print(f"  Lift: {aggressive['trade_count'] - tuned['trade_count']}")

# Save summary
summary_file = "research/artifacts/sweep_results/TUNING_RESULTS.json"
with open(summary_file, 'w') as f:
    json.dump({
        'configs_tested': configs,
        'results': results,
        'best_config': max(results.items(), key=lambda x: x[1]['trade_count'])[0] if results else None,
    }, f, indent=2)

print(f"\n✓ Summary saved to: {summary_file}")
