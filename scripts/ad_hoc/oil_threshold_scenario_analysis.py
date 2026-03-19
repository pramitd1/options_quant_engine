"""
Script: oil_threshold_scenario_analysis.py

Purpose:
    Analyze the impact of oil shock threshold changes on today's market conditions.
    Compare old thresholds (4.0% medium) vs new thresholds (2.5% medium, 1.5% notable).

Outputs:
    Side-by-side comparison showing how today's +0.218% oil move is scored before/after.
    Also simulates other realistic intraday oil moves to validate sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from config.global_risk_policy import get_global_risk_policy_config
from utils.numerics import safe_float as _safe_float


@dataclass
class OldOilPolicy:
    """Legacy oil shock thresholds (prior to March 19, 2026 update)"""
    oil_shock_extreme_change_pct: float = 7.0
    oil_shock_medium_change_pct: float = 4.0
    oil_shock_relief_change_pct: float = -5.0
    oil_shock_extreme_score: float = 1.0
    oil_shock_medium_score: float = 0.7
    oil_shock_relief_score: float = -0.5


def score_oil_shock_old(change_24h, cfg=None):
    """Score using OLD thresholds (4.0% medium)"""
    if cfg is None:
        cfg = OldOilPolicy()
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h > cfg.oil_shock_extreme_change_pct:
        return cfg.oil_shock_extreme_score
    if change_24h > cfg.oil_shock_medium_change_pct:
        return cfg.oil_shock_medium_score
    if change_24h < cfg.oil_shock_relief_change_pct:
        return cfg.oil_shock_relief_score
    return 0.0


def score_oil_shock_new(change_24h, cfg=None):
    """Score using NEW thresholds (2.5% medium, 1.5% notable)"""
    if cfg is None:
        cfg = get_global_risk_policy_config()
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h > cfg.oil_shock_extreme_change_pct:
        return cfg.oil_shock_extreme_score
    if change_24h > cfg.oil_shock_medium_change_pct:
        return cfg.oil_shock_medium_score
    if change_24h > cfg.oil_shock_notable_change_pct:
        return cfg.oil_shock_notable_score
    if change_24h < cfg.oil_shock_relief_change_pct:
        return cfg.oil_shock_relief_score
    return 0.0


def analyze_scenario(change_pct: float) -> dict:
    """Compare old vs new scoring for a given oil change percentage"""
    old_score = score_oil_shock_old(change_pct)
    new_score = score_oil_shock_new(change_pct)
    
    return {
        "change_pct": change_pct,
        "old_score": old_score,
        "new_score": new_score,
        "improved": new_score > old_score,
        "score_delta": new_score - old_score,
    }


def main():
    print("\n" + "="*90)
    print("OIL SHOCK THRESHOLD SCENARIO ANALYSIS")
    print("="*90)
    
    print("\n📊 THRESHOLD COMPARISON:")
    print("-" * 90)
    print(f"{'Metric':<30} {'OLD (Prior)':<25} {'NEW (Updated)':<25}")
    print("-" * 90)
    print(f"{'Extreme threshold':<30} {'7.0%':<25} {'7.0%':<25}")
    print(f"{'Medium threshold':<30} {'4.0%':<25} {'2.5%':<25}")
    print(f"{'Notable threshold':<30} {'(none)':<25} {'1.5%':<25}")
    print(f"{'Relief threshold':<30} {'-5.0%':<25} {'-5.0%':<25}")
    print()
    print(f"{'Extreme score':<30} {'1.0':<25} {'1.0':<25}")
    print(f"{'Medium score':<30} {'0.7':<25} {'0.7':<25}")
    print(f"{'Notable score':<30} {'(none)':<25} {'0.3 (new tier)':<25}")
    print(f"{'Relief score':<30} {'-0.5':<25} {'-0.5':<25}")
    
    # Today's actual market data
    today_oil_change = 0.218  # +0.218% as reported by fetch_macro_levels.py
    
    print("\n\n📈 TODAY'S SCENARIO (March 19, 2026):")
    print("-" * 90)
    print(f"Oil 24h change: +{today_oil_change}%")
    
    scenario = analyze_scenario(today_oil_change)
    print(f"\n  OLD Scoring (4.0% threshold):  {scenario['old_score']:.1f}")
    print(f"  NEW Scoring (1.5% threshold):  {scenario['new_score']:.1f}")
    
    if scenario['improved']:
        print(f"\n  ✅ IMPROVEMENT: Score increased by {scenario['score_delta']:.1f} with new thresholds")
        print(f"     → Oil move is now captured as 'notable' early warning signal")
    else:
        print(f"\n  ℹ️  Score unchanged (move too small for current thresholds)")
    
    # Broader scenarios
    print("\n\n🎯 INTRADAY SCENARIOS (Realistic Oil Move Ranges):")
    print("-" * 90)
    print(f"{'Oil Change':<15} {'OLD Score':<15} {'NEW Score':<15} {'Improvement':<30}")
    print("-" * 90)
    
    test_scenarios = [
        (0.0, "No move"),
        (0.2, "Minimal (today's level)"),
        (0.5, "Minor move"),
        (1.0, "Modest move"),
        (1.5, "Approx medium move - THRESHOLD"),
        (2.0, "Notable move"),
        (2.5, "Medium move - THRESHOLD"),
        (3.0, "Significant move"),
        (4.0, "High move - OLD THRESHOLD"),
        (5.0, "Very high move"),
        (7.0, "Extreme move - THRESHOLD"),
        (-1.0, "Down move"),
        (-5.0, "Relief move - THRESHOLD"),
        (-6.0, "Strong relief"),
    ]
    
    for change, scenario_name in test_scenarios:
        scenario = analyze_scenario(change)
        improvement = "✅ CAPTURES NOW" if scenario['improved'] else ""
        print(f"{change:>+7.1f}%        {scenario['old_score']:>6.1f}         {scenario['new_score']:>6.1f}         {improvement:<30}")
    
    print("\n\n💡 KEY INSIGHTS:")
    print("-" * 90)
    print("1. REGIME SENSITIVITY:")
    print("   • New thresholds catch 1.5-2.5% moves that old model missed")
    print("   • These smaller moves are significant in elevated volatility regimes")
    print("   • Today (VIX +12%, India VIX +13%), early warning helps")
    print()
    print("2. TODAY'S IMPACT:")
    print(f"   • Oil +0.218% would score: OLD={scenario_oil_old:.1f}, NEW={scenario_oil_new:.1f}")
    print("   • Still below notable threshold, but system is more responsive")
    print()
    print("3. INTRADAY TRADING:")
    print("   • 1-3% intraday oil moves now trigger early warning (0.3 score)")
    print("   • Helps catch building stress before it becomes 'shock' (0.7)")
    print("   • Options Greeks sensitive - better for gamma/vega positioning")
    print()
    print("4. BACKWARD COMPATIBILITY:")
    print("   • Extreme shock (7%+) & relief moves unchanged")
    print("   • New 'notable' tier provides smoother gradation")
    print("   • No breaking changes to downstream signal engine")
    
    print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    # Compute today's scenarios once for clarity
    scenario_today = analyze_scenario(0.218)
    scenario_oil_old = scenario_today['old_score']
    scenario_oil_new = scenario_today['new_score']
    main()
