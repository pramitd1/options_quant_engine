"""Profile the engine's critical path to identify actual bottlenecks."""
import time
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- utilities ----------
_timings = {}

def _time(label, func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    _timings[label] = elapsed
    return result


# ---------- build a realistic synthetic option chain ----------
spot = 23000.0
strikes = np.arange(22000, 24050, 50)  # 41 strikes
rows = []
for strike in strikes:
    for ot in ["CE", "PE"]:
        dist = abs(strike - spot)
        if ot == "CE":
            intrinsic = max(0, spot - strike)
            lp = max(5.0, intrinsic + np.random.uniform(20, 150) * np.exp(-dist / 1000))
        else:
            intrinsic = max(0, strike - spot)
            lp = max(5.0, intrinsic + np.random.uniform(20, 150) * np.exp(-dist / 1000))
        rows.append({
            "strikePrice": float(strike),
            "OPTION_TYP": ot,
            "lastPrice": round(lp, 2),
            "openInterest": int(np.random.uniform(5000, 500000)),
            "changeinOpenInterest": int(np.random.uniform(-50000, 50000)),
            "totalTradedVolume": int(np.random.uniform(1000, 100000)),
            "impliedVolatility": round(np.random.uniform(10, 25), 2),
            "EXPIRY_DT": "2026-03-17T06:00:00.000Z",
            "STRIKE_PR": float(strike),
            "OPEN_INT": int(np.random.uniform(5000, 500000)),
            "IV": round(np.random.uniform(10, 25), 2),
            "source": "ICICI",
        })

option_chain = pd.DataFrame(rows)
spot_snapshot = {"spot": spot, "timestamp": "2026-03-17T10:30:00", "day_high": spot + 50, "day_low": spot - 80, "day_open": spot - 20, "prev_close": spot - 30}
print(f"Spot: {spot}")
print(f"Chain rows: {len(option_chain)}")
print()

# ---------- profile individual analytics ----------
from analytics.greeks_engine import enrich_chain_with_greeks
from analytics.gamma_exposure import calculate_gamma_exposure
from analytics.market_gamma_map import largest_gamma_strikes
from analytics.volatility_surface import build_vol_surface, atm_vol
from analytics.volatility_regime import detect_volatility_regime
from analytics.gamma_walls import detect_gamma_walls
from analytics.gamma_flip import gamma_flip_level
from analytics.dealer_inventory import dealer_inventory_position
from analytics.options_flow_imbalance import flow_signal
from analytics.smart_money_flow import smart_money_signal
from analytics.liquidity_heatmap import strongest_liquidity_levels
from analytics.dealer_hedging_flow import dealer_hedging_flow
from analytics.intraday_gamma_shift import compute_gamma_profile

# ---------- profile full engine ----------
from app.engine_runner import run_preloaded_engine_snapshot
from news.service import build_default_headline_service

symbol = "NIFTY"
headline_svc = build_default_headline_service()
engine_kwargs = dict(
    symbol=symbol,
    mode="REPLAY",
    source="ICICI",
    spot_snapshot=spot_snapshot,
    option_chain=option_chain,
    apply_budget_constraint=False,
    requested_lots=1,
    lot_size=50,
    max_capital=100000,
    capture_signal_evaluation=False,
    headline_service=headline_svc,
    holding_profile="INTRADAY",
)

# Warm-up run (first run pays import + JIT costs)
run_preloaded_engine_snapshot(**engine_kwargs)

# Timed runs
runs = []
for _ in range(3):
    t0 = time.perf_counter()
    result = run_preloaded_engine_snapshot(**engine_kwargs)
    runs.append((time.perf_counter() - t0) * 1000)

full_ms = min(runs)  # best of 3
_timings["FULL_ENGINE"] = full_ms
print(f"\nFull engine runs (ms): {[f'{r:.1f}' for r in runs]}")
print(f"Best: {full_ms:.1f}ms")

# ---------- profile individual analytics (WARM — after engine import/cache) ----------
from analytics.greeks_engine import enrich_chain_with_greeks
from analytics.gamma_exposure import calculate_gamma_exposure
from analytics.market_gamma_map import largest_gamma_strikes
from analytics.volatility_surface import build_vol_surface, atm_vol
from analytics.volatility_regime import detect_volatility_regime
from analytics.gamma_walls import detect_gamma_walls
from analytics.gamma_flip import gamma_flip_level
from analytics.dealer_inventory import dealer_inventory_position
from analytics.options_flow_imbalance import flow_signal
from analytics.smart_money_flow import smart_money_signal
from analytics.liquidity_heatmap import strongest_liquidity_levels
from analytics.dealer_hedging_flow import dealer_hedging_flow
from analytics.intraday_gamma_shift import compute_gamma_profile
from data.option_chain_validation import validate_option_chain
from strategy.strike_selector import rank_strike_candidates
from risk.global_risk_layer import build_global_risk_state
from risk.gamma_vol_acceleration_layer import build_gamma_vol_acceleration_state
from risk.dealer_hedging_pressure_layer import build_dealer_hedging_pressure_state

macro_event_state = {"macro_event_risk_score": 0, "event_window_status": "OUTSIDE_WINDOW"}
macro_news_state = {"macro_adjustment_score": 0, "macro_confirmation_adjustment": 0, "macro_position_size_multiplier": 1.0}
global_market_snapshot = {}

# Best of 3 for each component
def _time_best(label, func, *args, **kwargs):
    best = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        best = min(best, elapsed)
    _timings[label] = best
    return res

df = option_chain.copy()
df = _time_best("enrich_greeks", enrich_chain_with_greeks, df, spot=spot)
_time_best("gamma_exposure", calculate_gamma_exposure, df, spot)
_time_best("gamma_map", largest_gamma_strikes, df, spot=spot)
_time_best("vol_surface", build_vol_surface, df)
_time_best("atm_vol", atm_vol, df, spot)
_time_best("vol_regime", detect_volatility_regime, df)
_time_best("gamma_walls", detect_gamma_walls, df)
_time_best("gamma_flip", gamma_flip_level, df, spot)
_time_best("dealer_position", dealer_inventory_position, df)
_time_best("flow_signal", flow_signal, df)
try:
    _time_best("smart_money", smart_money_signal, df)
except Exception:
    _timings["smart_money"] = 0.0
try:
    _time_best("liquidity_levels", strongest_liquidity_levels, df)
except Exception:
    _timings["liquidity_levels"] = 0.0
try:
    _time_best("hedging_flow", dealer_hedging_flow, df)
except Exception:
    _timings["hedging_flow"] = 0.0
try:
    _time_best("intraday_gamma", compute_gamma_profile, df, spot)
except Exception:
    _timings["intraday_gamma"] = 0.0

_time_best("validation", validate_option_chain, df)
_time_best("strike_ranking", rank_strike_candidates,
      option_chain=df, spot=spot, direction="CALL",
      support_wall=spot * 0.99, resistance_wall=spot * 1.01,
      gamma_clusters=[])
_time_best("global_risk", build_global_risk_state,
      macro_event_state=macro_event_state,
      macro_news_state=macro_news_state,
      global_market_snapshot=global_market_snapshot,
      holding_profile="INTRADAY")
try:
    _time_best("gamma_vol_accel", build_gamma_vol_acceleration_state,
          gamma_regime="NEUTRAL", holding_profile="INTRADAY")
except Exception:
    _timings["gamma_vol_accel"] = 0.0
try:
    _time_best("dealer_pressure", build_dealer_hedging_pressure_state,
          dealer_position="LONG_GAMMA", holding_profile="INTRADAY")
except Exception:
    _timings["dealer_pressure"] = 0.0

# ---------- sort and display ----------
print("=" * 60)
print(f"{'COMPONENT':<30} {'TIME (ms)':>10}")
print("=" * 60)
for label, ms in sorted(_timings.items(), key=lambda x: -x[1]):
    bar = "█" * int(ms / (max(_timings.values()) / 40))
    print(f"{label:<30} {ms:>10.1f}  {bar}")
print("=" * 60)
analytics_total = sum(v for k, v in _timings.items() if k != "FULL_ENGINE")
print(f"{'Analytics total':<30} {analytics_total:>10.1f}")
print(f"{'Full engine':<30} {full_ms:>10.1f}")
print(f"{'Engine overhead':<30} {full_ms - analytics_total:>10.1f}")
print(f"{'Analytics % of engine':<30} {analytics_total / full_ms * 100:>9.1f}%")
