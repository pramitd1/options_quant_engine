# Archived: Old Trade-Centric Backtest System

**Archived on:** 2026-03-17  
**Reason:** Replaced by the signal-centric holistic backtest runner (`backtest/holistic_backtest_runner.py`)

## What Was This?

A trade-centric backtesting pipeline that simulated opening/closing option positions
based on signal engine outputs, tracking PnL, equity curves, drawdowns, and win rates.

## Key Difference from Current System

| Aspect | Old (Trade-Centric) | New (Signal-Centric) |
|--------|---------------------|----------------------|
| Focus | Simulated trades & portfolio PnL | Signal quality evaluation |
| Output | Equity curve, Sharpe, max drawdown | 136-col signal evaluation row per signal |
| Trade management | Open/close, hold bars, stop-loss | None — evaluates direction, magnitude, timing |
| Data source | yfinance live/recent chains | Historical NSE parquet database (2012-2026) |
| Per-day | One trade decision | Multiple signals (one per upcoming expiry) |

## Files

```
backtest/
  backtest_runner.py          — Orchestrator with parameter sweep
  intraday_backtester.py      — Core trade simulation (bar-by-bar replay)
  pnl_engine.py               — Slippage, spread, commission calculation
  performance_metrics.py      — Equity curve, Sharpe, drawdown
  monte_carlo.py              — PnL resampling for path robustness
  parameter_sweep.py          — Cartesian grid of (persistence, hold_bars, TP%, SL%)
data/
  historical_option_chain.py  — yfinance-based option chain loader
tests/
  test_backtest_contract_handling.py — Tests for PnL/contract handling
```

## Dependency Chain

```
backtest_runner.py
  ├── intraday_backtester.py
  │     ├── pnl_engine.py
  │     ├── performance_metrics.py
  │     └── data/historical_option_chain.py
  ├── monte_carlo.py
  └── parameter_sweep.py
```

## Restoring

To restore, move files back to their original locations and re-add imports.
All files are self-contained within this dependency cluster.
