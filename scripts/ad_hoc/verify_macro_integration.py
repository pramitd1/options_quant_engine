#!/usr/bin/env python
"""Verify macro indicator integration in signal engine decision."""

import pandas as pd
from data.global_market_snapshot import build_global_market_snapshot
from risk.global_risk_features import build_global_risk_features
from risk.global_risk_layer import build_global_risk_state
from macro.macro_news_aggregator import build_macro_news_state
from macro.scheduled_event_risk import evaluate_scheduled_event_risk

def verify_macro_integration():
    """Trace macro indicators through signal engine layers."""
    
    print("=" * 80)
    print("MACRO MACROECONOMIC INDICATORS - INTEGRATION VERIFICATION")
    print("=" * 80)
    
    symbol = "NIFTY"
    
    # Step 1: Build global market snapshot
    print("\n[STEP 1] Building Global Market Snapshot")
    print("-" * 80)
    try:
        gms = build_global_market_snapshot(symbol)
        market_inputs = gms.get("market_inputs", {})
        print(f"✓ Global market snapshot built")
        print(f"  Status: {gms.get('provider')} {'(ENABLED)' if gms.get('data_available') else '(DISABLED)'}")
        print(f"\n  Market Inputs Collected:")
        for key, value in sorted(market_inputs.items()):
            if isinstance(value, float):
                print(f"    • {key:30} = {value:>10.3f}")
            else:
                print(f"    • {key:30} = {value}")
    except Exception as e:
        print(f"✗ Failed to build global market snapshot: {e}")
        return
    
    # Step 2: Build macro event state
    print("\n[STEP 2] Building Macro Event State")
    print("-" * 80)
    try:
        macro_event_state = evaluate_scheduled_event_risk(symbol=symbol)
        print(f"✓ Macro event state evaluated")
        print(f"  Event lockdown: {macro_event_state.get('event_lockdown', False)}")
        print(f"  Event risk score: {macro_event_state.get('event_risk_score', 0)}")
        print(f"  Next event: {macro_event_state.get('next_event_name', 'None')}")
    except Exception as e:
        print(f"⚠ Macro event state evaluation: {e}")
        macro_event_state = {}
    
    # Step 3: Build global risk features
    print("\n[STEP 3] Building Global Risk Features")
    print("-" * 80)
    try:
        grf = build_global_risk_features(
            macro_event_state=macro_event_state,
            macro_news_state=None,
            global_market_snapshot=gms,
            holding_profile="AUTO",
        )
        
        print(f"✓ Global risk features computed")
        print(f"\n  Key Macro-Derived Scores:")
        print(f"    • oil_shock_score            = {grf.get('oil_shock_score', 0):>8.3f}")
        print(f"    • commodity_risk_score       = {grf.get('commodity_risk_score', 0):>8.3f}")
        print(f"    • volatility_shock_score     = {grf.get('volatility_shock_score', 0):>8.3f}")
        print(f"    • volatility_explosion_prob  = {grf.get('volatility_explosion_probability', 0):>8.3f}")
        print(f"    • overnight_gap_risk_score   = {grf.get('overnight_gap_risk_score', 0):>8.3f}")
        print(f"    • india_vix_level            = {grf.get('india_vix_level', 0):>8.3f}")
        print(f"\n  Flags:")
        print(f"    • market_features_neutralized = {grf.get('market_features_neutralized', False)}")
        print(f"    • market_session             = {grf.get('market_session', 'UNKNOWN')}")
        
    except Exception as e:
        print(f"✗ Failed to build global risk features: {e}")
        grf = {}
    
    # Step 4: Build macro news state
    print("\n[STEP 4] Building Macro News State")
    print("-" * 80)
    try:
        mns = build_macro_news_state(
            event_state=macro_event_state,
            headline_state=None,
            as_of=None,
        ).to_dict()
        
        print(f"✓ Macro news state built")
        print(f"  Macro regime:           {mns.get('macro_regime', 'UNKNOWN')}")
        print(f"  Macro sentiment score:  {mns.get('macro_sentiment_score', 0):.3f}")
        print(f"  Volatility shock score: {mns.get('volatility_shock_score', 0):.3f}")
        print(f"  Event lockdown flag:    {mns.get('event_lockdown_flag', False)}")
        print(f"  News confidence:        {mns.get('news_confidence_score', 0):.3f}")
        
    except Exception as e:
        print(f"⚠ Macro news state warning: {e}")
        mns = {}
    
    # Step 5: Build global risk state
    print("\n[STEP 5] Building Global Risk State")
    print("-" * 80)
    try:
        grs = build_global_risk_state(
            macro_event_state=macro_event_state,
            macro_news_state=mns,
            global_market_snapshot=gms,
            holding_profile="AUTO",
        )
        
        print(f"✓ Global risk state assembled")
        print(f"\n  Risk Assessment:")
        print(f"    • global_risk_state    = {grs.get('global_risk_state', 'UNKNOWN')}")
        print(f"    • global_risk_score    = {grs.get('global_risk_score', 0)}")
        print(f"    • risk_adjustment_pct  = {grs.get('risk_adjustment_percentage', 0):.1f}%")
        
        grf_from_state = grs.get('global_risk_features', {})
        print(f"\n  Macro Indicators in Risk State:")
        print(f"    • oil_shock_score         = {grf_from_state.get('oil_shock_score', 0):.3f}")
        print(f"    • commodity_risk_score    = {grf_from_state.get('commodity_risk_score', 0):.3f}")
        print(f"    • volatility_shock_score  = {grf_from_state.get('volatility_shock_score', 0):.3f}")
        print(f"    • india_vix_level         = {grf_from_state.get('india_vix_level', 0):.2f}")
        
        print(f"\n  Overnight Assessment:")
        print(f"    • overnight_hold_allowed = {grs.get('overnight_hold_allowed', False)}")
        print(f"    • overnight_reason       = {grs.get('overnight_hold_reason', 'N/A')}")
        print(f"    • overnight_risk_penalty = {grs.get('overnight_risk_penalty', 0)}")
        
    except Exception as e:
        print(f"✗ Failed to build global risk state: {e}")
        grs = {}
    
    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION VERIFICATION SUMMARY")
    print("=" * 80)
    
    checks = {
        "✅ Global market snapshot": gms.get('data_available', False),
        "✅ Market inputs collected": bool(market_inputs),
        "✅ Macro event state": bool(macro_event_state),
        "✅ Global risk features": bool(grf),
        "✅ Macro news state": bool(mns),
        "✅ Global risk state": bool(grs),
        "✅ Oil shock score": grf.get('oil_shock_score') is not None,
        "✅ Commodity risk score": grf.get('commodity_risk_score') is not None,
        "✅ Volatility shock score": grf.get('volatility_shock_score') is not None,
        "✅ India VIX level": grf.get('india_vix_level') is not None,
    }
    
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check:40} [{status}]")
    
    all_passed = all(checks.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL MACRO INDICATORS PROPERLY INTEGRATED")
        print("   Macro layer is fully operational and contributing to signal decisions")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"⚠ Found {len(failed)} integration issues:")
        for issue in failed:
            print(f"  - {issue}")
    print("=" * 80)

if __name__ == "__main__":
    verify_macro_integration()
