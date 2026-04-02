#!/usr/bin/env python3
"""Deployment validation test."""

import sys
sys.path.insert(0, '/Users/pramitdutta/Desktop/Quant Engines/options_quant_engine')

from config.signal_policy import TRADE_RUNTIME_THRESHOLDS
from app.engine_runner import run_engine_snapshot

print("="*80)
print("DEPLOYMENT VALIDATION TEST")
print("="*80)

# 1. Verify configuration
print("\n✓ CONFIGURATION VERIFICATION")
print("-"*80)
print(f"provider_health_override_min_composite_buffer: {TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_composite_buffer']}")
print(f"provider_health_override_min_strength_buffer: {TRADE_RUNTIME_THRESHOLDS['provider_health_override_min_strength_buffer']}")
print(f"provider_health_override_size_cap: {TRADE_RUNTIME_THRESHOLDS['provider_health_override_size_cap']}")
print(f"provider_health_override_hold_cap_minutes: {TRADE_RUNTIME_THRESHOLDS['provider_health_override_hold_cap_minutes']}")
print(f"min_trade_strength: {TRADE_RUNTIME_THRESHOLDS['min_trade_strength']}")
print(f"min_composite_score: {TRADE_RUNTIME_THRESHOLDS['min_composite_score']}")

# 2. Run a quick engine smoke test
print("\n✓ ENGINE SMOKE TEST")
print("-"*80)

try:
    result = run_engine_snapshot(
        symbol='NIFTY',
        mode='REPLAY',
        source='DEPLOYMENT_TEST',
        apply_budget_constraint=False,
        requested_lots=1,
        lot_size=65,
        max_capital=20000,
        replay_spot='debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-30T10-35-00+05-30.json',
        replay_chain='debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-30T10-35-44.046236+05-30.csv',
        capture_signal_evaluation=False,
        signal_capture_policy='ALL_SIGNALS',
    )
    
    print(f"✓ Engine executed successfully")
    print(f"  Trade status: {result.get('trade', {}).get('signal_status', 'N/A')}")
    trade_suggestion = result.get('trade_suggestion', {})
    if trade_suggestion:
        print(f"  Override active: {trade_suggestion.get('provider_health_override_active', False)}")
        if trade_suggestion.get('provider_health_override_active'):
            print(f"  Override reason: {trade_suggestion.get('provider_health_override_reason', 'N/A')}")
    print(f"  Composite score: {result.get('signal_composite_score')}")
    print(f"  Trade strength: {result.get('adjusted_trade_strength')}")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ VALIDATION COMPLETE - READY FOR DEPLOYMENT")
print("="*80)
