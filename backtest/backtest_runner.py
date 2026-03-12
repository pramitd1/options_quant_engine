from concurrent.futures import ProcessPoolExecutor, as_completed

from config.settings import BACKTEST_YEARS, MC_SIMULATIONS, MAX_WORKERS
from backtest.intraday_backtester import run_intraday_backtest
from backtest.monte_carlo import monte_carlo_reshuffle
from backtest.parameter_sweep import build_parameter_grid, summarize_sweep_results
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def _run_one_sweep(args):
    symbol, years, signal_persistence, max_hold_bars, tp, sl = args
    result = run_intraday_backtest(
        symbol=symbol,
        years=years,
        signal_persistence=signal_persistence,
        max_hold_bars=max_hold_bars,
    )
    result.update({
        "signal_persistence": signal_persistence,
        "max_hold_bars": max_hold_bars,
        "tp_percent": tp,
        "sl_percent": sl
    })
    return result


def run_backtest():
    symbol = input("Symbol to backtest: ").strip().upper()
    years_input = input(f"Years of data [{BACKTEST_YEARS}]: ").strip()
    years = int(years_input) if years_input else BACKTEST_YEARS

    print("\nChoose mode:")
    print("1. Single backtest")
    print("2. Parameter sweep")

    mode = input("Enter choice (1/2): ").strip()

    if mode == "2":
        grid = build_parameter_grid()
        jobs = [(symbol, years, sp, mh, tp, sl) for sp, mh, tp, sl in grid]

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

    results = run_intraday_backtest(symbol, years)
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