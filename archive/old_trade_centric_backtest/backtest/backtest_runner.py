"""
Module: backtest_runner.py

Purpose:
    Implement backtest runner logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed

from config.settings import BACKTEST_YEARS, BACKTEST_DATA_SOURCE, MC_SIMULATIONS, MAX_WORKERS
from backtest.intraday_backtester import run_intraday_backtest
from backtest.monte_carlo import monte_carlo_reshuffle
from backtest.parameter_sweep import build_parameter_grid, summarize_sweep_results
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def _run_one_sweep(args):
    """
    Purpose:
        Process run one sweep for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        args (Any): Input associated with args.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    symbol, years, signal_persistence, max_hold_bars, tp, sl, data_source = args
    result = run_intraday_backtest(
        symbol=symbol,
        years=years,
        signal_persistence=signal_persistence,
        max_hold_bars=max_hold_bars,
        target_profit_percent=tp,
        stop_loss_percent=sl,
        data_source=data_source,
    )
    result.update({
        "signal_persistence": signal_persistence,
        "max_hold_bars": max_hold_bars,
        "tp_percent": tp,
        "sl_percent": sl
    })
    return result


def run_backtest():
    """
    Purpose:
        Process run backtest for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    symbol = input("Symbol to backtest: ").strip().upper()
    years_input = input(f"Years of data [{BACKTEST_YEARS}]: ").strip()
    years = int(years_input) if years_input else BACKTEST_YEARS

    print(f"\nData source (current default: {BACKTEST_DATA_SOURCE}):")
    print("1. historical  — real NSE bhav-copy data")
    print("2. live         — synthetic Black-Scholes chain")
    print("3. combined     — historical + live for missing dates")
    ds_input = input("Enter choice (1/2/3) or press Enter for default: ").strip()
    ds_map = {"1": "historical", "2": "live", "3": "combined"}
    data_source = ds_map.get(ds_input, BACKTEST_DATA_SOURCE)
    print(f"Using data source: {data_source}")

    print("\nChoose mode:")
    print("1. Single backtest")
    print("2. Parameter sweep")

    mode = input("Enter choice (1/2): ").strip()

    if mode == "2":
        grid = build_parameter_grid()
        jobs = [(symbol, years, sp, mh, tp, sl, data_source) for sp, mh, tp, sl in grid]

        results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_run_one_sweep, job) for job in jobs]
            for fut in as_completed(futures):
                results.append(fut.result())

        ranked = summarize_sweep_results(results)

        print("\nTOP 10 PARAMETER SWEEP RESULTS")
        print("--------------------------------")
        for row in ranked[:10]:
            print(row)

        return

    print("\nRunning backtest...")
    print("If historical option-chain data is not cached, it will be built automatically.")

    results = run_intraday_backtest(symbol, years, data_source=data_source)
    mc = monte_carlo_reshuffle(results.get("trades", []), simulations=MC_SIMULATIONS)

    print("\nBACKTEST RESULTS")
    print("-------------------------")
    for key, value in results.items():
        if key != "trades":
            print(f"{key}: {value}")

    print("\nMONTE CARLO")
    print("-------------------------")
    for key, value in mc.items():
        print(f"{key}: {value}")

    if results.get("trades"):
        print("\nLAST 5 TRADES")
        print("-------------------------")
        for trade in results["trades"][-5:]:
            print(trade)


if __name__ == "__main__":
    run_backtest()
