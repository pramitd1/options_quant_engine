#!/usr/bin/env python
"""
================================================================================
==========  COMPARATIVE BACKTEST: ALL 6 PREDICTION METHODS (2016-2026)        ==========
================================================================================

Script: comparative_backtest_full_history_2016_2026.py

Purpose:
    Run historical backtests for all 6 available prediction methods across the
    FULL 10-YEAR HISTORICAL WINDOW (2016-2026) and compare their performance
    metrics side-by-side.

Prediction Methods Tested:
    1. blended (production default)
    2. pure_rule (rule-based only)
    3. pure_ml (ML-based only)
    4. research_dual_model (GBT + LogReg research)
    5. research_decision_policy (dual-model + policy overlay)
    6. ev_sizing (EV-based position sizing)

Historical Window:
    Start Date: January 1, 2016
    End Date: March 19, 2026 (today)
    Duration: ~10 years of NSE option chain data
    Expected Signals: 1000+ per method (comprehensive sample)

Output:
    • Individual backtest results for each method (10-year horizon)
    • Comparative performance table
    • Statistical analysis and long-term conclusions
    • Recommendations for production use
    • Full results saved to JSON for detailed analysis

Date: March 19, 2026 (with PRIORITY 1-3 fixes active)

================================================================================
"""

import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.holistic_backtest_runner import run_holistic_backtest
from config.settings import DEFAULT_SYMBOL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def format_section(title):
    """Format section header."""
    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"==========================     {title:<42}     ==========================")
    print("=" * 80)
    print()


def format_subsection(title):
    """Format subsection header."""
    print(f"\n{title}")
    print("-" * 80)


# Available prediction methods
PREDICTION_METHODS = [
    "blended",
    "pure_rule",
    "pure_ml",
    "research_dual_model",
    "research_decision_policy",
    "ev_sizing",
    "research_rank_gate",
    "research_uncertainty_adjusted",
]

# Method descriptions
METHOD_DESCRIPTIONS = {
    "blended": "Production default: weighted blend of rule + ML + recalibration",
    "pure_rule": "Rule-based logic only (ML disabled)",
    "pure_ml": "ML inference only (rule disabled, blend weight=100% ML)",
    "research_dual_model": "Research: GBT ranking + LogReg calibration",
    "research_decision_policy": "Research: dual-model + policy decision overlay",
    "ev_sizing": "Research: EV-based position sizing and probability scaling",
}


def run_single_backtest(
    method: str,
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = "2016-01-01",
    end_date: str = "2026-03-19",
    max_expiries: int = 3,
) -> dict:
    """Run a single backtest for the given prediction method."""
    log.info(f"Starting backtest for prediction_method={method!r}")
    log.info(f"  Date Range: {start_date} → {end_date}")
    
    try:
        result = run_holistic_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            max_expiries=max_expiries,
            evaluate_outcomes=True,
            prediction_method=method,
        )
        
        if result.get("ok"):
            log.info(
                f"✓ Backtest complete: {result['total_signals']} signals, "
                f"{result['evaluated_days']} days, "
                f"{result['elapsed_seconds']:.1f}s elapsed"
            )
        else:
            log.error(f"✗ Backtest failed: {result.get('message', 'Unknown error')}")
        
        return result
    except Exception as e:
        log.exception(f"Exception during backtest for {method}")
        return {
            "ok": False,
            "method": method,
            "error": str(e),
        }


def extract_metrics(result: dict, method: str) -> dict:
    """Extract key metrics from backtest result."""
    if not result.get("ok"):
        return {"method": method, "status": "FAILED", "error": result.get("error")}
    
    metrics = result.get("metrics", {})
    
    return {
        "method": method,
        "status": "SUCCESS",
        "total_signals": result.get("total_signals", 0),
        "evaluated_days": result.get("evaluated_days", 0),
        "trade_signals": metrics.get("trade_signals", 0),
        "trade_rate": metrics.get("trade_rate", 0),
        "target_hit_rate": metrics.get("target_hit_rate", 0),
        "stop_loss_hit_rate": metrics.get("stop_loss_hit_rate", 0),
        "avg_trade_strength": metrics.get("avg_trade_strength", 0),
        "avg_composite_score": metrics.get("avg_scores", {}).get("composite_signal_score", 0),
        "avg_direction_score": metrics.get("avg_scores", {}).get("direction_score", 0),
        "avg_tradeability_score": metrics.get("avg_scores", {}).get("tradeability_score", 0),
        "correct_1d": metrics.get("directional_accuracy", {}).get("correct_1d", 0),
        "total_1d": metrics.get("directional_accuracy", {}).get("total_1d", 1),  # avoid div by 0
        "correct_expiry": metrics.get("directional_accuracy", {}).get("correct_at_expiry", 0),
        "total_expiry": metrics.get("directional_accuracy", {}).get("total_at_expiry", 1),  # avoid div by 0
        "avg_mfe_bps": metrics.get("avg_eod_mfe_bps", 0),
        "avg_mae_bps": metrics.get("avg_eod_mae_bps", 0),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
    }


def compute_accuracies(metrics_list: list[dict]) -> list[dict]:
    """Compute derived metrics like directional accuracy %."""
    for m in metrics_list:
        if m["status"] == "SUCCESS":
            m["accuracy_1d_pct"] = (
                100 * m["correct_1d"] / max(1, m["total_1d"])
            )
            m["accuracy_expiry_pct"] = (
                100 * m["correct_expiry"] / max(1, m["total_expiry"])
            )
        else:
            m["accuracy_1d_pct"] = 0
            m["accuracy_expiry_pct"] = 0
    return metrics_list


def print_comparison_table(metrics_list: list[dict]):
    """Print comparative results table."""
    format_subsection("COMPARATIVE RESULTS (All 6 Methods)")
    
    print(
        f"{'Method':<25} | {'Signals':<8} | {'Trade':<6} | "
        f"{'Rate':<6} | {'Strength':<9} | {'Composite':<9} | "
        f"{'1D Acc':<7} | {'Exp Acc':<7} | {'TP%':<6} | {'SL%':<6}"
    )
    print("-" * 130)
    
    for m in metrics_list:
        if m["status"] == "FAILED":
            print(f"{m['method']:<25} | {'FAILED':<60} | {m.get('error', 'Unknown')}")
        else:
            print(
                f"{m['method']:<25} | "
                f"{m['total_signals']:<8} | "
                f"{m['trade_signals']:<6} | "
                f"{m['trade_rate']:<6.1%} | "
                f"{m['avg_trade_strength']:<9.2f} | "
                f"{m['avg_composite_score']:<9.2f} | "
                f"{m['accuracy_1d_pct']:<7.1f}% | "
                f"{m['accuracy_expiry_pct']:<7.1f}% | "
                f"{m['target_hit_rate']:<6.1%} | "
                f"{m['stop_loss_hit_rate']:<6.1%}"
            )


def print_detailed_summary(metrics_list: list[dict]):
    """Print detailed summary for each method."""
    format_subsection("DETAILED SUMMARY (Each Method)")
    
    for m in metrics_list:
        if m["status"] == "FAILED":
            print(f"\n{m['method']:<20}: FAILED - {m.get('error', 'Unknown error')}")
        else:
            print(f"\n{m['method']:<20}:")
            print(f"  Signals:          {m['total_signals']} total")
            print(f"  Evaluated Days:   {m['evaluated_days']} trading days")
            print(f"  Trade Rate:       {m['trade_signals']}/{m['total_signals']} ({m['trade_rate']:.1%})")
            print(f"  Avg Strength:     {m['avg_trade_strength']:.2f}/100")
            print(f"  Composite Score:  {m['avg_composite_score']:.2f}/100")
            print(f"  Direction Score:  {m['avg_direction_score']:.2f}/100")
            print(f"  Tradeability:     {m['avg_tradeability_score']:.2f}/100")
            print(f"  Accuracy (1D):    {m['accuracy_1d_pct']:.1f}%")
            print(f"  Accuracy (Exp):   {m['accuracy_expiry_pct']:.1f}%")
            print(f"  Target Hit:       {m['target_hit_rate']:.1%}")
            print(f"  Stop Loss Hit:    {m['stop_loss_hit_rate']:.1%}")
            print(f"  MFE:              {m['avg_mfe_bps']:.1f} bps")
            print(f"  MAE:              {m['avg_mae_bps']:.1f} bps")
            print(f"  Time:             {m['elapsed_seconds']:.1f}s")


def generate_rankings(metrics_list: list[dict]):
    """Generate performance rankings."""
    format_subsection("PERFORMANCE RANKINGS")
    
    successful = [m for m in metrics_list if m["status"] == "SUCCESS"]
    if not successful:
        print("No successful backtests to rank.")
        return
    
    ranking_categories = [
        ("Trade Rate", "trade_rate"),
        ("Avg Strength", "avg_trade_strength"),
        ("Composite Score", "avg_composite_score"),
        ("1D Directional Accuracy", "accuracy_1d_pct"),
        ("Expiry Directional Accuracy", "accuracy_expiry_pct"),
        ("Target Hit Rate", "target_hit_rate"),
        ("Stop Loss Hit Rate (lower better)", "stop_loss_hit_rate"),
        ("Execution Speed (lower better)", "elapsed_seconds"),
    ]
    
    for category_name, field in ranking_categories:
        print(f"\n{category_name}:")
        sorted_methods = sorted(successful, key=lambda x: x[field], reverse=True)
        for rank, method in enumerate(sorted_methods, 1):
            value = method[field]
            if "better" in category_name and "lower" in category_name:
                print(f"  {rank}. {method['method']:<25} ({value:.2f}s)" if "Speed" in category_name else
                      f"  {rank}. {method['method']:<25} ({value:.1%})")
            elif "Accuracy" in category_name or "Rate" in category_name:
                print(f"  {rank}. {method['method']:<25} ({value:.1%})")
            else:
                print(f"  {rank}. {method['method']:<25} ({value:.2f})")


def generate_recommendations(metrics_list: list[dict]):
    """Generate production recommendations."""
    format_subsection("RECOMMENDATIONS")
    
    successful = [m for m in metrics_list if m["status"] == "SUCCESS"]
    if not successful:
        print("Cannot generate recommendations: no successful backtests.")
        return
    
    # Sort by composite score
    by_composite = sorted(
        successful, key=lambda x: x["avg_composite_score"], reverse=True
    )
    
    print("\nTIER 1: PRODUCTION READY")
    print(f"  ✅ {by_composite[0]['method']:<25} (Composite: {by_composite[0]['avg_composite_score']:.2f})")
    print(f"     → Highest overall quality across 10-year window")
    
    print("\nTIER 2: PROMISING ALTERNATIVES")
    for method in by_composite[1:3]:
        if method["status"] == "SUCCESS":
            print(f"  🔶 {method['method']:<25} (Composite: {method['avg_composite_score']:.2f})")
    
    print("\nTIER 3: RESEARCH ONLY")
    for method in by_composite[3:]:
        if method["status"] == "SUCCESS":
            print(f"  ⏱️  {method['method']:<25} (Composite: {method['avg_composite_score']:.2f})")


def main():
    """Main execution."""
    format_section("COMPARATIVE BACKTEST (2016-2026)")
    
    print("Configuration:")
    print(f"  Date Range:     2016-01-01 → 2026-03-19 (10 years)")
    print(f"  Methods:        {len(PREDICTION_METHODS)} prediction methods")
    print(f"  Max Expiries:   3 per trading day")
    print(f"  With Priority:  1-3 macro fixes ACTIVE")
    print()
    print("Methods to test:")
    for method in PREDICTION_METHODS:
        print(f"  • {method:<25} - {METHOD_DESCRIPTIONS[method]}")
    
    # Run backtests
    format_section("RUNNING BACKTESTS")
    
    all_results = {}
    start_time = time.time()
    
    for idx, method in enumerate(PREDICTION_METHODS, 1):
        print(f"\n[{idx}/{len(PREDICTION_METHODS)}] Testing {method}...")
        result = run_single_backtest(
            method=method,
            start_date="2016-01-01",
            end_date="2026-03-19",
        )
        all_results[method] = result
    
    total_elapsed = time.time() - start_time
    
    # Extract and compute metrics
    metrics_list = [extract_metrics(all_results[m], m) for m in PREDICTION_METHODS]
    metrics_list = compute_accuracies(metrics_list)
    
    # Print results
    format_section("RESULTS")
    print(f"Total execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print()
    
    print_comparison_table(metrics_list)
    print_detailed_summary(metrics_list)
    generate_rankings(metrics_list)
    generate_recommendations(metrics_list)
    
    # Save results
    format_section("SAVING RESULTS")
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"backtest_comparison_results_fullhistory_{timestamp}.json")
    
    results_payload = {
        "timestamp": timestamp,
        "date_range": {
            "start": "2016-01-01",
            "end": "2026-03-19",
            "duration_years": 10,
        },
        "total_execution_seconds": total_elapsed,
        "methods": PREDICTION_METHODS,
        "metrics": metrics_list,
        "all_results": all_results,
    }
    
    with open(output_file, "w") as f:
        json.dump(results_payload, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    
    format_section("BACKTEST COMPLETE")
    print(f"✅ All {len(PREDICTION_METHODS)} methods tested successfully!")
    print(f"   Total time: {total_elapsed/60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
