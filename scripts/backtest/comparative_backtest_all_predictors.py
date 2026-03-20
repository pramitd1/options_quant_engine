#!/usr/bin/env python
from __future__ import annotations

"""
================================================================================
==========     COMPARATIVE BACKTEST: ALL 6 PREDICTION METHODS                  ==========
================================================================================

Script: comparative_backtest_all_predictors.py

Purpose:
    Run historical backtests for all 6 available prediction methods and
    compare their performance metrics side-by-side.

Prediction Methods Tested:
    1. blended (production default)
    2. pure_rule (rule-based only)
    3. pure_ml (ML-based only)
    4. research_dual_model (GBT + LogReg research)
    5. research_decision_policy (dual-model + policy overlay)
    6. ev_sizing (EV-based position sizing)
    7. research_rank_gate (dual-model + rank-threshold gate)
    8. research_uncertainty_adjusted (dual-model + uncertainty discount)

Output:
    • Individual backtest results for each method
    • Comparative performance table
    • Statistical analysis and conclusions
    • Recommendations for production use

Date: March 19, 2026 (with PRIORITY 1-3 fixes active)

================================================================================
"""

import json
import logging
import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtest.holistic_backtest_runner import run_holistic_backtest
from config.settings import DEFAULT_SYMBOL
from data.historical_snapshot import get_available_dates

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
    "research_rank_gate": "Research: dual-model + rank-threshold gating",
    "research_uncertainty_adjusted": "Research: dual-model + uncertainty discounting",
}


def run_single_backtest(
    method: str,
    symbol: str = DEFAULT_SYMBOL,
    start_date: str | None = None,
    end_date: str | None = None,
    max_expiries: int = 3,
) -> dict:
    """Run a single backtest for the given prediction method."""
    log.info(f"Starting backtest for prediction_method={method!r}")
    
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
                f"{result['elapsed_seconds']}s elapsed"
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
        "correct_expiry": metrics.get("directional_accuracy", {}).get("correct_at_expiry", 0),
        "avg_mfe_bps": metrics.get("avg_eod_mfe_bps", 0),
        "avg_mae_bps": metrics.get("avg_eod_mae_bps", 0),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
    }


def print_comparison_table(all_metrics: list[dict]):
    """Print comparison table of all methods."""
    format_section("COMPARATIVE RESULTS - ALL PREDICTION METHODS")
    
    if not all_metrics or all([m.get("status") == "FAILED" for m in all_metrics]):
        print("✗ No successful backtests to compare")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    successful = df[df["status"] == "SUCCESS"].copy()
    
    if successful.empty:
        print("✗ No successful results")
        return
    
    print(f"Backtests: {len(successful)}/{len(df)} successful\n")
    
    # Key metrics table
    display_cols = [
        "method",
        "total_signals",
        "trade_signals",
        "trade_rate",
        "avg_trade_strength",
        "avg_composite_score",
        "correct_1d",
        "correct_expiry",
        "target_hit_rate",
        "stop_loss_hit_rate",
    ]
    
    print("Method                     | Signals | Trade | Rate  | Strength | Composite | 1D Acc | Expiry | TP%   | SL%")
    print("-" * 110)
    
    for _, row in successful.iterrows():
        method = row["method"]
        signals = int(row["total_signals"])
        trades = int(row["trade_signals"])
        rate = f"{row['trade_rate']*100:.1f}%"
        strength = f"{row['avg_trade_strength']:.2f}"
        composite = f"{row['avg_composite_score']:.2f}"
        acc_1d = f"{row['correct_1d']*100:.1f}%" if row['correct_1d'] else "N/A"
        acc_exp = f"{row['correct_expiry']*100:.1f}%" if row['correct_expiry'] else "N/A"
        tp_hit = f"{row['target_hit_rate']*100:.1f}%"
        sl_hit = f"{row['stop_loss_hit_rate']*100:.1f}%"
        
        print(f"{method:<26} | {signals:>7} | {trades:>5} | {rate:>5} | {strength:>8} | {composite:>9} | {acc_1d:>6} | {acc_exp:>6} | {tp_hit:>5} | {sl_hit:>5}")
    
    print("\n" + "-" * 110)
    
    # Ranking
    print("\n📊 RANKINGS:\n")
    
    rankings = {
        "Most Signals": ("total_signals", False),
        "Highest Trade Rate": ("trade_rate", False),
        "Highest Avg Strength": ("avg_trade_strength", False),
        "Highest Composite Score": ("avg_composite_score", False),
        "Best 1D Accuracy": ("correct_1d", False),
        "Best Expiry Accuracy": ("correct_expiry", False),
        "Highest TP Rate": ("target_hit_rate", False),
        "Lowest SL Rate": ("stop_loss_hit_rate", True),
        "Fastest Execution": ("elapsed_seconds", True),
    }
    
    for metric_name, (col_name, lower_is_better) in rankings.items():
        valid = successful[successful[col_name].notna()].copy()
        if valid.empty:
            continue
        
        if lower_is_better:
            best_idx = valid[col_name].idxmin()
        else:
            best_idx = valid[col_name].idxmax()
        
        best_row = valid.loc[best_idx]
        value = best_row[col_name]
        method = best_row["method"]
        
        if isinstance(value, float):
            if col_name in ["target_hit_rate", "stop_loss_hit_rate", "trade_rate", "correct_1d", "correct_expiry"]:
                display_value = f"{value*100:.1f}%"
            elif col_name == "elapsed_seconds":
                display_value = f"{value:.1f}s"
            else:
                display_value = f"{value:.2f}"
        else:
            display_value = str(value)
        
        print(f"  🏆 {metric_name:<25}: {method:<25} ({display_value})")


def print_detailed_summary(all_metrics: list[dict]):
    """Print detailed summary for each method."""
    format_section("DETAILED SUMMARY - EACH PREDICTION METHOD")
    
    for metrics in all_metrics:
        if metrics.get("status") == "FAILED":
            format_subsection(f"{metrics['method'].upper()} - FAILED")
            print(f"Error: {metrics.get('error', 'Unknown error')}")
        else:
            format_subsection(metrics["method"].upper())
            print(f"Status: {metrics['status']}")
            print(f"Total Signals: {metrics.get('total_signals', 0)}")
            print(f"Trade Signals: {metrics.get('trade_signals', 0)}")
            print(f"Trade Rate: {metrics.get('trade_rate', 0)*100:.2f}%")
            print(f"Avg Trade Strength: {metrics.get('avg_trade_strength', 0):.2f}")
            print(f"Avg Composite Score: {metrics.get('avg_composite_score', 0):.2f}")
            print(f"Avg Direction Score: {metrics.get('avg_direction_score', 0):.2f}")
            print(f"Avg Tradeability Score: {metrics.get('avg_tradeability_score', 0):.2f}")
            print(f"1D Directional Accuracy: {metrics.get('correct_1d', 0)*100:.2f}%")
            print(f"Expiry Directional Accuracy: {metrics.get('correct_expiry', 0)*100:.2f}%")
            print(f"Target Hit Rate: {metrics.get('target_hit_rate', 0)*100:.2f}%")
            print(f"Stop-Loss Hit Rate: {metrics.get('stop_loss_hit_rate', 0)*100:.2f}%")
            print(f"Avg MFE: {metrics.get('avg_mfe_bps', 0):.2f} bps")
            print(f"Avg MAE: {metrics.get('avg_mae_bps', 0):.2f} bps")
            print(f"Execution Time: {metrics.get('elapsed_seconds', 0):.2f}s")


def print_recommendations(all_metrics: list[dict]):
    """Print recommendations based on results."""
    format_section("RECOMMENDATIONS & INSIGHTS")
    
    successful = [m for m in all_metrics if m.get("status") == "SUCCESS"]
    
    if not successful:
        print("✗ No successful backtests available for analysis")
        return
    
    df_success = pd.DataFrame(successful)
    
    # Find best performer overall
    strength_scores = df_success["avg_trade_strength"].copy()
    composite_scores = df_success["avg_composite_score"].copy()
    
    # Normalized scoring (higher is better)
    if strength_scores.max() > 0:
        strength_normalized = strength_scores / strength_scores.max()
    else:
        strength_normalized = strength_scores
    
    if composite_scores.max() > 0:
        composite_normalized = composite_scores / composite_scores.max()
    else:
        composite_normalized = composite_scores
    
    overall_score = (strength_normalized * 0.4 + composite_normalized * 0.6)
    best_idx = overall_score.idxmax()
    best_method = df_success.loc[best_idx, "method"]
    
    print(f"🎯 Best Overall Performance: {best_method.upper()}")
    print(f"   Composite Score: {df_success.loc[best_idx, 'avg_composite_score']:.2f}")
    print(f"   Trade Strength: {df_success.loc[best_idx, 'avg_trade_strength']:.2f}")
    
    print(f"\n📊 Method Comparison Summary:\n")
    
    for col in ["total_signals", "trade_rate", "avg_composite_score", "correct_expiry"]:
        if col not in df_success.columns:
            continue
        
        vals = df_success[col].dropna()
        if vals.empty:
            continue
        
        mean_val = vals.mean()
        std_val = vals.std()
        
        print(f"   {col}:")
        print(f"     • Mean: {mean_val:.2f} (±{std_val:.2f})")
        print(f"     • Range: {vals.min():.2f} - {vals.max():.2f}")
    
    print(f"\n💡 Key Insights:\n")
    
    # Insight 1: Signal volume
    print(f"   1. Signal Volume:")
    max_signals = df_success["total_signals"].idxmax()
    min_signals = df_success["total_signals"].idxmin()
    max_method = df_success.loc[max_signals, "method"]
    min_method = df_success.loc[min_signals, "method"]
    print(f"      • {max_method} generates most signals ({int(df_success.loc[max_signals, 'total_signals'])})")
    print(f"      • {min_method} generates least signals ({int(df_success.loc[min_signals, 'total_signals'])})")
    
    # Insight 2: Quality vs quantity
    print(f"\n   2. Quality vs Quantity Trade-off:")
    high_strength = df_success.loc[df_success["avg_trade_strength"].idxmax()]
    print(f"      • Highest quality: {high_strength['method']} (strength={high_strength['avg_trade_strength']:.2f})")
    
    # Insight 3: Accuracy
    print(f"\n   3. Directional Accuracy:")
    high_accuracy = df_success.loc[df_success["correct_expiry"].idxmax()]
    print(f"      • Best accuracy: {high_accuracy['method']} ({high_accuracy['correct_expiry']*100:.1f}% expiry accuracy)")
    
    # Insight 4: Risk management
    print(f"\n   4. Risk Management (Target/SL Rates):")
    for idx, row in df_success.iterrows():
        print(f"      • {row['method']}: TP={row['target_hit_rate']*100:.1f}%, SL={row['stop_loss_hit_rate']*100:.1f}%")
    
    print(f"\n⚠️  PRODUCTION RECOMMENDATION:\n")
    print(f"   Current Production: blended")
    blended_metrics = next((m for m in all_metrics if m.get("method") == "blended"), None)
    if blended_metrics is not None:
        print(f"   Status: {blended_metrics.get('status', 'unknown')}")
        print(f"   Recommendation: STICK with 'blended' - stable, production-proven baseline")
        print(f"                  Consider research_dual_model for future evaluation")
    else:
        print("   Status: not evaluated in this run (subset mode)")
        print("   Recommendation: this subset run excludes 'blended'; compare subset metrics")
        print("                  against the most recent blended benchmark before switching")


def save_results(results: dict, filename: str = None):
    """Save detailed results to JSON."""
    if filename is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_comparison_results_{timestamp}.json"
    
    filepath = Path(__file__).resolve().parent / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"Results saved to {filepath}")
    return filepath


def main():
    """Main entry point for comparative backtest."""
    parser = argparse.ArgumentParser(description="Comparative backtest across predictor methods")
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of methods, e.g. --methods pure_rule pure_ml",
    )
    args = parser.parse_args()

    methods_to_run = list(PREDICTION_METHODS)
    if args.methods:
        requested = [m.strip() for m in args.methods if m and m.strip()]
        unknown = [m for m in requested if m not in PREDICTION_METHODS]
        if unknown:
            print(f"Unknown methods: {unknown}")
            print(f"Available methods: {PREDICTION_METHODS}")
            return 1
        methods_to_run = requested

    format_section("STARTING COMPARATIVE BACKTEST - ALL 6 PREDICTION METHODS")
    
    print("Prediction Methods to Test:")
    for i, method in enumerate(methods_to_run, 1):
        print(f"  {i}. {method:<25} - {METHOD_DESCRIPTIONS.get(method, 'Unknown')}")
    
    print(f"\nSymbol: {DEFAULT_SYMBOL}")
    available_dates = get_available_dates(DEFAULT_SYMBOL)
    if not available_dates:
        print("Date Range: unavailable (no historical data found)")
        return 1

    requested_start = date(2016, 1, 1)
    latest_available = max(available_dates)
    effective_start = max(requested_start, min(available_dates))
    effective_end = latest_available

    print(f"Date Range: {effective_start} to {effective_end} (full historical window)")
    print(f"Max Expiries/Day: 3")
    
    # Run backtests
    format_section("RUNNING BACKTESTS")
    
    # Use full requested historical window capped by available data.
    start_date = effective_start
    end_date = effective_end
    
    all_results = {}
    all_metrics = []
    
    for i, method in enumerate(methods_to_run, 1):
        print(f"\n[{i}/{len(methods_to_run)}] Testing {method.upper()}...")
        
        t0 = time.time()
        result = run_single_backtest(
            method,
            symbol=DEFAULT_SYMBOL,
            start_date=str(start_date),
            end_date=str(end_date),
            max_expiries=3,
        )
        elapsed = time.time() - t0
        
        all_results[method] = result
        
        metrics = extract_metrics(result, method)
        all_metrics.append(metrics)
        
        if result.get("ok"):
            print(f"✓ Complete ({elapsed:.1f}s): {result['total_signals']} signals")
        else:
            print(f"✗ Failed: {result.get('message', 'Unknown error')}")
    
    # Print comparisons
    print_comparison_table(all_metrics)
    print_detailed_summary(all_metrics)
    print_recommendations(all_metrics)
    
    # Save results
    format_section("SAVING RESULTS")
    full_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "symbol": DEFAULT_SYMBOL,
        "date_range": {"start": str(start_date), "end": str(end_date)},
        "methods": methods_to_run,
        "individual_results": all_results,
        "comparison_metrics": all_metrics,
    }
    
    filepath = save_results(full_results)
    print(f"✓ Results saved to {filepath}")
    
    format_section("BACKTEST COMPLETE")
    print("✅ Comparative backtest finished successfully")
    print("\nNext Steps:")
    print("  • Review results in comparison table above")
    print("  • Check detailed results file for full analysis")
    print("  • Consider recommendations for production method selection")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
