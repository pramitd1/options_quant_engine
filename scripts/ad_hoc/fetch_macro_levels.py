#!/usr/bin/env python
"""Fetch and report current global macroeconomic indicators."""

import pandas as pd
import yfinance as yf
from datetime import datetime
from config.market_data_policy import IST_TIMEZONE, GLOBAL_MARKET_TICKERS

def fetch_macro_levels():
    """Fetch current levels for all tracked global macro indicators."""
    
    print("=" * 80)
    print("GLOBAL MACROECONOMIC INDICATORS - REAL-TIME SNAPSHOT")
    print(f"Report Time: {datetime.now(pd.Timestamp.now(tz=IST_TIMEZONE).tz).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 80)
    
    indicators = {}
    errors = []
    
    for name, ticker in GLOBAL_MARKET_TICKERS.items():
        try:
            # Fetch latest data
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            
            if data is None or len(data) == 0:
                errors.append(f"{name.upper():12} ({ticker:12}) - No data")
                continue
            
            # Get latest close and previous close
            latest = float(data['Close'].iloc[-1])
            previous = float(data['Close'].iloc[-2]) if len(data) > 1 else latest
            change = latest - previous
            change_pct = (change / previous * 100) if previous != 0 else 0
            
            indicators[name] = {
                "ticker": ticker,
                "latest": latest,
                "previous": previous,
                "change": change,
                "change_pct": change_pct,
                "status": "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
            }
            
        except Exception as e:
            errors.append(f"{name.upper():12} ({ticker:12}) - Error: {str(e)[:40]}")
    
    # Print formatted macro report
    print("\n📊 COMMODITY PRICES")
    print("-" * 80)
    commodities = ["oil", "gold", "copper"]
    for comm in commodities:
        if comm in indicators:
            ind = indicators[comm]
            arrow = "🔺" if ind["status"] == "UP" else "🔻" if ind["status"] == "DOWN" else "→"
            print(f"  {comm.upper():10} {arrow} {ind['latest']:>10.2f}  |  "
                  f"Change: {ind['change']:>7.2f} ({ind['change_pct']:>6.2f}%)")
    
    print("\n📈 VOLATILITY INDICES")
    print("-" * 80)
    volatility = ["vix", "india_vix"]
    for vol in volatility:
        if vol in indicators:
            ind = indicators[vol]
            arrow = "🔺" if ind["status"] == "UP" else "🔻" if ind["status"] == "DOWN" else "→"
            print(f"  {vol.upper():15} {arrow} {ind['latest']:>10.2f}  |  "
                  f"Change: {ind['change']:>7.2f} ({ind['change_pct']:>6.2f}%)")
    
    print("\n📊 EQUITY INDICES")
    print("-" * 80)
    equities = ["sp500", "nasdaq"]
    for eq in equities:
        if eq in indicators:
            ind = indicators[eq]
            arrow = "🔺" if ind["status"] == "UP" else "🔻" if ind["status"] == "DOWN" else "→"
            print(f"  {eq.upper().ljust(15):15} {arrow} {ind['latest']:>10.2f}  |  "
                  f"Change: {ind['change']:>7.2f} ({ind['change_pct']:>6.2f}%)")
    
    print("\n💵 RATES & CURRENCY")
    print("-" * 80)
    rates = ["us10y", "usdinr"]
    for rate in rates:
        if rate in indicators:
            ind = indicators[rate]
            arrow = "🔺" if ind["status"] == "UP" else "🔻" if ind["status"] == "DOWN" else "→"
            if rate == "us10y":
                print(f"  US 10Y Yield {arrow} {ind['latest']:>10.2f}%  |  "
                      f"Change: {ind['change']:>7.2f} bps ({ind['change_pct']:>6.2f}%)")
            else:
                print(f"  USD/INR       {arrow} {ind['latest']:>10.2f}  |  "
                      f"Change: {ind['change']:>7.2f} ({ind['change_pct']:>6.2f}%)")
    
    print("\n" + "=" * 80)
    print("MACRO REGIME ASSESSMENT")
    print("=" * 80)
    
    # Simple macro regime logic
    vix_level = indicators.get("vix", {}).get("latest", 0)
    oil_change = indicators.get("oil", {}).get("change_pct", 0)
    sp500_change = indicators.get("sp500", {}).get("change_pct", 0)
    india_vix = indicators.get("india_vix", {}).get("latest", 0)
    usdinr = indicators.get("usdinr", {}).get("latest", 0)
    
    print(f"\nVIX Level:        {vix_level:.2f} {'⚠️ ELEVATED' if vix_level > 25 else '😌 Normal'}")
    print(f"India VIX:        {india_vix:.2f} {'⚠️ ELEVATED' if india_vix > 25 else '😌 Normal'}")
    print(f"Oil Change 24h:   {oil_change:+.2f}% {'🔴 Major Shock' if abs(oil_change) > 5 else '🟡 Elevated' if abs(oil_change) > 2 else '🟢 Normal'}")
    print(f"S&P 500 Change:   {sp500_change:+.2f}% {'🔴 Risk Off' if sp500_change < -2 else '🟡 Weak' if sp500_change < 0 else '🟢 Risk On'}")
    print(f"USD/INR Level:    {usdinr:.2f} {'📈 Rupee Weak' if usdinr > 83 else '📉 Rupee Strong'}")
    
    # Determine overall regime
    if vix_level > 30 or india_vix > 30:
        regime = "🔴 HIGH VOLATILITY / RISK OFF"
    elif vix_level > 20 or india_vix > 20:
        regime = "🟡 ELEVATED VOLATILITY / CAUTION"
    elif oil_change > 5 or abs(sp500_change) > 2:
        regime = "🟡 MARKET STRESS / MACRO EVENT"
    else:
        regime = "🟢 NORMAL / RISK ON"
    
    print(f"\nOVERALL REGIME:   {regime}")
    
    print("\n" + "=" * 80)
    if errors:
        print("⚠️  DATA FETCH ISSUES:")
        for error in errors:
            print(f"   {error}")
    print("=" * 80)

if __name__ == "__main__":
    fetch_macro_levels()
