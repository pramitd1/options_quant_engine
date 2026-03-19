"""
Script: signal_bias_analysis.py

Purpose:
    Deep analysis of today's generated signals to identify systematic biases,
    decision errors, and areas for improvement.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_signals():
    df = pd.read_csv('research/signal_evaluation/signals_dataset.csv')
    
    print("\n" + "="*90)
    print("SIGNAL BIAS & ERROR ANALYSIS - TODAY'S ENGINE DECISIONS (March 19, 2026)")
    print("="*90)
    
    # === SECTION 1: DIRECTIONAL BIAS ===
    print("\n\n1️⃣  DIRECTIONAL BIAS ANALYSIS")
    print("-" * 90)
    
    direction_counts = df['direction'].value_counts()
    total = len(df)
    
    calls = direction_counts.get('CALL', 0)
    puts = direction_counts.get('PUT', 0)
    
    print(f"Calls: {calls:>3} ({calls*100/total:.1f}%)")
    print(f"Puts:  {puts:>3} ({puts*100/total:.1f}%)")
    
    if calls > 0 and puts > 0:
        ratio = calls / puts
        print(f"Call/Put Ratio: {ratio:.2f}x")
        
        if ratio > 1.3 or ratio < 0.77:
            print(f"\n⚠️  BIAS DETECTED: Significant directional skew detected")
            print(f"   → System favoring {'CALLS' if ratio > 1 else 'PUTS'} by {abs(ratio-1)*100:.1f}%")
            
            # Analyze accuracy by direction
            calls_df = df[df['direction'] == 'CALL']
            puts_df = df[df['direction'] == 'PUT']
            
            calls_completed = calls_df[calls_df['observed_minutes'].notna() & (calls_df['observed_minutes'] > 0)]
            puts_completed = puts_df[puts_df['observed_minutes'].notna() & (puts_df['observed_minutes'] > 0)]
            
            if len(calls_completed) > 5:
                calls_acc_5m = calls_completed['correct_5m'].sum() / len(calls_completed) * 100
                print(f"   → CALLS accuracy @ 5m: {calls_acc_5m:.1f}%")
            
            if len(puts_completed) > 5:
                puts_acc_5m = puts_completed['correct_5m'].sum() / len(puts_completed) * 100
                print(f"   → PUTS accuracy @ 5m:  {puts_acc_5m:.1f}%")
            
            if len(calls_completed) > 5 and len(puts_completed) > 5:
                calls_acc_5m = calls_completed['correct_5m'].sum() / len(calls_completed) * 100
                puts_acc_5m = puts_completed['correct_5m'].sum() / len(puts_completed) * 100
                if abs(calls_acc_5m - puts_acc_5m) > 15:
                    better = 'CALLS' if calls_acc_5m > puts_acc_5m else 'PUTS'
                    worse_acc = min(calls_acc_5m, puts_acc_5m)
                    print(f"   → ⚡ {better} performing {abs(calls_acc_5m - puts_acc_5m):.1f}% better - consider rebalancing")
    
    # === SECTION 2: REGIME JUDGMENT ===
    print("\n\n2️⃣  REGIME CLASSIFICATION JUDGMENT")
    print("-" * 90)
    
    macro_regimes = df['macro_regime'].value_counts()
    print("Macro Regimes:")
    for regime, count in macro_regimes.items():
        pct = count * 100 / total
        print(f"  {regime}: {count:>3} ({pct:>5.1f}%)")
    
    global_risk_states = df['global_risk_state'].value_counts()
    print("\nGlobal Risk States:")
    for state, count in global_risk_states.items():
        pct = count * 100 / total
        print(f"  {state}: {count:>3} ({pct:>5.1f}%)")
    
    # Check if regime matches market reality
    avg_vix_shock = df['volatility_shock_score'].mean()
    avg_vol = df['volatility_shock_score'].std()
    
    print(f"\nVolatility Assessment:")
    print(f"  Avg volatility shock score: {avg_vix_shock:.3f}")
    print(f"  Volatility perception (1-5): {'LOW' if avg_vix_shock < 0.3 else 'MEDIUM' if avg_vix_shock < 0.6 else 'HIGH'}")
    print(f"  → Market today: India VIX +13%, S&P 500 -1.36%, Nasdaq -1.46%")
    
    if avg_vix_shock < 0.5:
        print(f"\n⚠️  JUDGMENT ERROR: Volatility shock score too LOW ({avg_vix_shock:.3f})")
        print(f"   → Given India VIX up 13%, expected shock score > 0.7")
        print(f"   → Currently {(0.7-avg_vix_shock)*100:.1f}% UNDERESTIMATING volatility stress")
    
    # === SECTION 3: STRIKE SELECTION ===
    print("\n\n3️⃣  STRIKE SELECTION BIAS")
    print("-" * 90)
    
    df['strike_pct'] = (df['strike'] - df['spot_at_signal']) / df['spot_at_signal'] * 100
    calls_df = df[df['direction'] == 'CALL']
    puts_df = df[df['direction'] == 'PUT']
    
    if len(calls_df) > 0:
        calls_itm = (calls_df['strike_pct'] < 0).sum()
        calls_otm = (calls_df['strike_pct'] > 0).sum()
        calls_atm = ((calls_df['strike_pct'].abs() < 0.5)).sum()
        
        print(f"CALLS Strike Selection:")
        print(f"  ITM:  {calls_itm} ({calls_itm*100/len(calls_df):.1f}%)")
        print(f"  ATM:  {calls_atm} ({calls_atm*100/len(calls_df):.1f}%)")
        print(f"  OTM:  {calls_otm} ({calls_otm*100/len(calls_df):.1f}%)")
        
        avg_strike_dist_call = abs(calls_df['strike_pct']).mean()
        print(f"  Avg distance from spot: {avg_strike_dist_call:.2f}%")
    
    if len(puts_df) > 0:
        puts_itm = (puts_df['strike_pct'] > 0).sum()
        puts_otm = (puts_df['strike_pct'] < 0).sum()
        puts_atm = ((puts_df['strike_pct'].abs() < 0.5)).sum()
        
        print(f"\nPUTS Strike Selection:")
        print(f"  ITM:  {puts_itm} ({puts_itm*100/len(puts_df):.1f}%)")
        print(f"  ATM:  {puts_atm} ({puts_atm*100/len(puts_df):.1f}%)")
        print(f"  OTM:  {puts_otm} ({puts_otm*100/len(puts_df):.1f}%)")
        
        avg_strike_dist_put = abs(puts_df['strike_pct']).mean()
        print(f"  Avg distance from spot: {avg_strike_dist_put:.2f}%")
    
    # Check if strikes are too conservative/aggressive
    overall_strike_dist = abs(df['strike_pct']).mean()
    if overall_strike_dist > 2.0:
        print(f"\n⚠️  STRIKE SELECTION BIAS: Too aggressive (OTM heavy)")
        print(f"   → Avg strike distance: {overall_strike_dist:.2f}% from spot")
        print(f"   → Lower probability of reaching targets")
    elif overall_strike_dist < 0.3:
        print(f"\n⚠️  STRIKE SELECTION BIAS: Too conservative (ITM heavy)")
        print(f"   → Avg strike distance: {overall_strike_dist:.2f}% from spot")
        print(f"   → Missing better risk-reward opportunities")
    
    # === SECTION 4: SIGNAL QUALITY ===
    print("\n\n4️⃣  SIGNAL QUALITY & CONFIDENCE")
    print("-" * 90)
    
    print(f"Trade Strength Distribution:")
    print(f"  Mean:     {df['trade_strength'].mean():.1f} / 100")
    print(f"  Median:   {df['trade_strength'].median():.1f} / 100")
    print(f"  Std Dev:  {df['trade_strength'].std():.1f}")
    print(f"  Min/Max:  {df['trade_strength'].min():.1f} / {df['trade_strength'].max():.1f}")
    
    weak_signals_count = (df['trade_strength'] < 50).sum()
    print(f"\nWeak signals (< 50): {weak_signals_count} ({weak_signals_count*100/total:.1f}%)")
    
    if weak_signals_count > total * 0.2:
        print(f"⚠️  ERROR: Too many weak signals ({weak_signals_count*100/total:.1f}%)")
        print(f"   → Consider raising entry bar or improving filtering")
    
    print(f"\nSignal Confidence:")
    print(f"  Mean:     {df['signal_confidence_score'].mean():.3f}")
    print(f"  Median:   {df['signal_confidence_score'].median():.3f}")
    
    # === SECTION 5: PERFORMANCE ANALYSIS ===
    print("\n\n5️⃣  PREDICTION ACCURACY & PERFORMANCE")
    print("-" * 90)
    
    completed = df[df['observed_minutes'].notna() & (df['observed_minutes'] > 0)]
    
    if len(completed) > 0:
        print(f"Completed signals: {len(completed)} / {total} ({len(completed)*100/total:.1f}%)")
        
        acc_5m = completed['correct_5m'].sum() if 'correct_5m' in completed.columns else 0
        acc_15m = completed['correct_15m'].sum() if 'correct_15m' in completed.columns else 0
        acc_60m = completed['correct_60m'].sum() if 'correct_60m' in completed.columns else 0
        
        print(f"\nAccuracy by Time Horizon:")
        print(f"  @ 5m:  {acc_5m}/{len(completed)} ({acc_5m*100/len(completed):.1f}%)")
        print(f"  @ 15m: {acc_15m}/{len(completed)} ({acc_15m*100/len(completed):.1f}%)")
        print(f"  @ 60m: {acc_60m}/{len(completed)} ({acc_60m*100/len(completed):.1f}%)")
        
        # Correlation with signal strength
        if len(completed) > 10:
            strong_signals = completed[completed['trade_strength'] >= df['trade_strength'].median()]
            weak_signals = completed[completed['trade_strength'] < df['trade_strength'].median()]
            
            if len(strong_signals) > 3:
                strong_acc = strong_signals['correct_5m'].sum() / len(strong_signals) * 100
                print(f"\n  Strong signals (≥median) accuracy @ 5m: {strong_acc:.1f}%")
            
            if len(weak_signals) > 3:
                weak_acc = weak_signals['correct_5m'].sum() / len(weak_signals) * 100
                print(f"  Weak signals (<median) accuracy @ 5m:  {weak_acc:.1f}%")
                
                if abs(strong_acc - weak_acc) > 20:
                    print(f"\n✅ GOOD SIGNAL FILTERING: {abs(strong_acc - weak_acc):.1f}% acc gap between strong/weak")
    else:
        print("No completed signals yet - market still active")
    
    # === SECTION 6: RISK ASSESSMENT ERRORS ===
    print("\n\n6️⃣  RISK ASSESSMENT JUDGMENT")
    print("-" * 90)
    
    print(f"Average Risk Scores:")
    print(f"  Oil shock:            {df['oil_shock_score'].mean():.3f}")
    print(f"  Commodity risk:       {df['commodity_risk_score'].mean():.3f}")
    print(f"  Volatility shock:     {df['volatility_shock_score'].mean():.3f}")
    print(f"  Dealer hedging press: {df['dealer_hedging_pressure_score'].mean():.3f}")
    print(f"  Global risk score:    {df['global_risk_score'].mean():.1f} / 100")
    
    # Risk-reward check
    print(f"\nRisk-Reward Assessment:")
    print(f"  Expected move (avg %): {df['expected_move_pct'].mean():.2f}%")
    print(f"  Target reachability:   {df['target_reachability_score'].mean():.3f}")
    
    avg_tradeability = df['tradeability_score'].mean()
    if avg_tradeability < 0.5:
        print(f"\n⚠️  ERROR: Low tradeability score ({avg_tradeability:.3f})")
        print(f"   → Signals may have poor geometric setup")
    
    # === SECTION 7: OVERNIGHT HOLDING ===
    print("\n\n7️⃣  OVERNIGHT HOLDING JUDGMENT")
    print("-" * 90)
    
    overnight_allowed = (df['overnight_hold_allowed'].astype(str) == 'True').sum()
    print(f"Overnight allowed: {overnight_allowed}/{total} ({overnight_allowed*100/total:.1f}%)")
    
    if overnight_allowed > total * 0.5:
        print(f"⚠️  JUDGMENT: High overnight allowance ({overnight_allowed*100/total:.1f}%)")
        print(f"   → Consider if gap risk penalties are calibrated correctly")
    elif overnight_allowed < total * 0.1:
        print(f"⚠️  JUDGMENT: Very low overnight allowance ({overnight_allowed*100/total:.1f}%)")
        print(f"   → May be overly conservative, restricting position holding")
    
    # === SECTION 8: RECOMMENDATIONS ===
    print("\n\n" + "="*90)
    print("OVERALL ASSESSMENT & RECOMMENDATIONS")
    print("="*90)
    
    recommendations = []
    
    # Check for bias
    if calls > 0 and puts > 0:
        ratio = calls / puts
        if ratio > 1.3 or ratio < 0.77:
            recommendations.append(
                f"1. REBALANCE DIRECTIONAL BIAS: Call/Put ratio is {ratio:.2f}x - consider adjusting direction selection logic"
            )
    
    # Check volatility assessment
    if avg_vix_shock < 0.5:
        recommendations.append(
            f"2. INCREASE VOLATILITY SENSITIVITY: Current avg shock={avg_vix_shock:.3f}, but market VIX elevated +13% - thresholds may be too high"
        )
    
    # Check strike selection
    if overall_strike_dist > 2.0:
        recommendations.append(
            f"3. REDUCE STRIKE AGGRESSIVENESS: Avg {overall_strike_dist:.2f}% OTM - targets less reachable, consider ITM/ATM bias"
        )
    
    # Check strike selection
    if overall_strike_dist < 0.3:
        recommendations.append(
            f"3. INCREASE STRIKE AGGRESSIVENESS: Avg {overall_strike_dist:.2f}% delta - too conservative, missing opportunities"
        )
    
    # Check signal quality
    if weak_signals_count > total * 0.2:
        recommendations.append(
            f"4. TIGHTEN SIGNAL FILTERING: {weak_signals_count*100/total:.1f}% of signals have trade_strength < 50 - raise minimum threshold"
        )
    
    # Check tradeability
    if avg_tradeability < 0.5:
        recommendations.append(
            f"5. IMPROVE GEOMETRIC SETUP: Tradeability score only {avg_tradeability:.3f} - validate strike/target/SL logic"
        )
    
    if not recommendations:
        print("\n✅ NO CRITICAL BIASES DETECTED")
        print("   System appears well-calibrated for current market regime")
    else:
        print(f"\n❌ IDENTIFIED ISSUES:")
        for rec in recommendations:
            print(f"   {rec}")
    
    print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    analyze_signals()
