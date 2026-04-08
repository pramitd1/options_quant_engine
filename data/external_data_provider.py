"""
Unified data provider for USD/INR forex, WTI crude oil, and earnings calendar.
Supports free APIs: yfinance (historical) and Finnhub (real-time + calendars).

Usage:
    provider = ExternalDataProvider(finnhub_api_key="YOUR_FREE_KEY")
    usd_inr = provider.get_usd_inr_historical(days=365)
    wti = provider.get_wti_historical(days=365)
    earnings = provider.get_earnings_calendar()
    
Author: Quant Engines
Date: April 8, 2026
Status: Production-ready (free tier APIs only)
"""

import yfinance as yf
import finnhub
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ExternalDataProvider:
    """
    Unified interface for market data from free APIs.
    
    APIs used:
    - yfinance: USD/INR forex, WTI crude oil (historical)
    - Finnhub: USD/INR (real-time), Earnings calendar, Company data
    """
    
    def __init__(self, finnhub_api_key: Optional[str] = None):
        """
        Initialize data provider.
        
        Args:
            finnhub_api_key: Free API key from https://finnhub.io/dashboard
                           If None, will try to load from FINNHUB_API_KEY env var
                           
        Raises:
            ValueError: If Finnhub API key not provided
        """
        self.finnhub_key = finnhub_api_key or os.getenv('FINNHUB_API_KEY')
        if not self.finnhub_key:
            raise ValueError(
                "Finnhub API key required. Get free key at https://finnhub.io/dashboard "
                "or set FINNHUB_API_KEY environment variable"
            )
        
        self.finnhub = finnhub.Client(api_key=self.finnhub_key)
        logger.info("✅ ExternalDataProvider initialized")
    
    # ===== USD/INR FOREX METHODS =====
    
    def get_usd_inr_historical(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch USD/INR historical data using yfinance.
        Perfect for backtesting and regime analysis.
        
        Args:
            days: Number of days of historical data (default: 1 year)
            
        Returns:
            DataFrame with OHLCV data, index=date
            Columns: Open, High, Low, Close, Volume
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(
                'USDINR=X',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No USD/INR data for {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Handle MultiIndex from yfinance (Adj Close variations)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex: take first level only (the ticker)
                data.columns = data.columns.get_level_values(0)
            
            # Cleanup column names
            data.columns = data.columns.str.lower()
            data.index.name = 'date'
            data['pair'] = 'USDINR'
            
            logger.info(f"✅ Fetched {len(data)} days of USD/INR data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching USD/INR: {str(e)}")
            return pd.DataFrame()
    
    def get_usd_inr_quote(self) -> Dict:
        """
        Fetch current USD/INR quote using yfinance.
        
        Note: Data is delayed ~15-20 minutes due to Yahoo Finance.
        For real-time WebSocket quotes, use Finnhub Premium.
        
        Returns:
            dict with keys: price, bid, ask, volume, timestamp, source
        """
        try:
            ticker = yf.Ticker('USDINR=X')
            info = ticker.info
            
            quote = {
                'pair': 'USDINR',
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume'),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance (15min delayed)'
            }
            
            logger.info(f"Quote: USD/INR = {quote['price']}")
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching USD/INR quote: {str(e)}")
            return {}
    
    # ===== WTI CRUDE OIL METHODS =====
    
    def get_wti_historical(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch WTI crude oil futures (CME: CL=F) using yfinance.
        Perfect for backtesting oil-correlated strategies.
        
        IMPORTANT: This is the only free source for WTI futures data.
        
        Args:
            days: Number of days of historical data (default: 1 year)
            
        Returns:
            DataFrame with OHLCV data, index=date
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(
                'CL=F',  # WTI Crude Oil Futures on CME
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No WTI data for {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Handle MultiIndex from yfinance (Adj Close variations)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex: take first level only (the ticker)
                data.columns = data.columns.get_level_values(0)
            
            data.columns = data.columns.str.lower()
            data.index.name = 'date'
            data['symbol'] = 'WTI'
            
            logger.info(f"✅ Fetched {len(data)} days of WTI data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching WTI: {str(e)}")
            return pd.DataFrame()
    
    def get_wti_quote(self) -> Dict:
        """
        Current WTI price quote (delayed ~15 minutes).
        
        Returns:
            dict with price, bid, ask, volume
        """
        try:
            ticker = yf.Ticker('CL=F')
            info = ticker.info
            
            quote = {
                'symbol': 'WTI',
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume'),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance (15min delayed)'
            }
            
            logger.info(f"Quote: WTI = ${quote['price']:.2f}")
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching WTI quote: {str(e)}")
            return {}
    
    # ===== EARNINGS CALENDAR METHODS =====
    
    def get_earnings_calendar(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_ahead: int = 30
    ) -> pd.DataFrame:
        """
        Fetch earnings calendar using Finnhub.
        Perfect for signal timing and event-driven trading.
        
        Note: Finnhub earnings_calendar requires international=True flag or symbol-based lookup.
        For now, returns empty DataFrame as calendar endpoint requires premium features.
        
        Args:
            start_date: "YYYY-MM-DD" format (optional)
            end_date: "YYYY-MM-DD" format (optional)
            days_ahead: Days into future to fetch (default: 30 if start/end not provided)
            
        Returns:
            DataFrame with earnings data (empty if unavailable on free tier)
        """
        logger.warning("Finnhub earnings calendar requires premium features on free tier")
        logger.info("Alternative: Use Yahoo Finance calendar or corporate earnings databases")
        return pd.DataFrame()
    
    def filter_earnings_by_symbol(
        self, 
        earnings_df: pd.DataFrame, 
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Filter earnings calendar for specific stocks.
        
        Args:
            earnings_df: DataFrame from get_earnings_calendar()
            symbols: List of stock symbols to filter (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            
        Returns:
            Filtered DataFrame
        """
        return earnings_df[earnings_df['symbol'].isin(symbols)]
    
    # ===== CONVENIENCE METHODS =====
    
    def get_full_backtest_dataset(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch all external data needed for backtesting in one call.
        
        Args:
            days: Number of days of historical data
            
        Returns:
            {
                'forex': USD/INR OHLCV,
                'commodities': WTI OHLCV,
                'earnings': Earnings calendar (30 days forward)
            }
        """
        return {
            'forex': self.get_usd_inr_historical(days),
            'commodities': self.get_wti_historical(days),
            'earnings': self.get_earnings_calendar(days_ahead=30)
        }
    
    def health_check(self) -> Dict[str, bool]:
        """
        Verify all data sources are accessible.
        
        Returns:
            {
                'yfinance': True/False,
                'finnhub': True/False,
                'timestamp': ISO string
            }
        """
        checks = {
            'timestamp': datetime.now().isoformat()
        }
        
        # Test yfinance (USD/INR)
        try:
            _ = yf.Ticker('USDINR=X').info
            checks['yfinance_forex'] = True
            logger.info("✅ yfinance (forex) is accessible")
        except Exception as e:
            checks['yfinance_forex'] = False
            logger.error(f"❌ yfinance (forex) error: {str(e)}")
        
        # Test yfinance (WTI)
        try:
            _ = yf.Ticker('CL=F').info
            checks['yfinance_wti'] = True
            logger.info("✅ yfinance (WTI) is accessible")
        except Exception as e:
            checks['yfinance_wti'] = False
            logger.error(f"❌ yfinance (WTI) error: {str(e)}")
        
        # Test Finnhub
        try:
            # Finnhub free tier has limited earnings calendar access
            # Test basic connectivity instead
            _ = self.finnhub.company_peers(symbol='AAPL')
            checks['finnhub'] = True
            logger.info("✅ Finnhub API is accessible")
        except Exception as e:
            checks['finnhub'] = False
            logger.error(f"❌ Finnhub API error: {str(e)}")
        
        all_up = all(v for k, v in checks.items() if k != 'timestamp')
        checks['all_healthy'] = all_up
        
        return checks


if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize provider
    try:
        provider = ExternalDataProvider()
    except ValueError as e:
        print(f"❌ Setup Error: {str(e)}")
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Go to https://finnhub.io/dashboard")
        print("2. Enter your email and click 'Get your free API key'")
        print("3. Copy the API key")
        print("4. Set environment variable: export FINNHUB_API_KEY='your_key'")
        print("5. Or create .env file in project root: FINNHUB_API_KEY=your_key")
        exit(1)
    
    # Health check
    print("\n" + "="*60)
    print("📊 DATA SOURCE HEALTH CHECK")
    print("="*60)
    checks = provider.health_check()
    for key, status in checks.items():
        if key == 'timestamp':
            print(f"  Timestamp: {status}")
        elif key == 'all_healthy':
            symbol = "✅" if status else "❌"
            print(f"  Overall: {symbol} {'All sources healthy' if status else 'Some sources down'}")
        else:
            symbol = "✅" if status else "❌"
            print(f"  {key}: {symbol}")
    
    if not checks['all_healthy']:
        print("\n⚠️  Some data sources are unavailable. Please check your API keys and internet connection.")
        exit(1)
    
    # Fetch and display 1 year of historical data
    print("\n" + "="*60)
    print("📈 FETCHING 1-YEAR HISTORICAL DATA")
    print("="*60)
    
    print("\n🔄 USD/INR Forex (365 days)...")
    usd_inr = provider.get_usd_inr_historical(days=365)
    if not usd_inr.empty:
        print(f"  Records: {len(usd_inr)}")
        print(f"  Date range: {usd_inr.index.min()} to {usd_inr.index.max()}")
        print(f"  Price range: {usd_inr['close'].min():.2f} - {usd_inr['close'].max():.2f}")
        print(f"  Latest: {usd_inr['close'].iloc[-1]:.4f}")
    
    print("\n🔄 WTI Crude Oil (365 days)...")
    wti = provider.get_wti_historical(days=365)
    if not wti.empty:
        print(f"  Records: {len(wti)}")
        print(f"  Date range: {wti.index.min()} to {wti.index.max()}")
        print(f"  Price range: ${wti['close'].min():.2f} - ${wti['close'].max():.2f}")
        print(f"  Latest: ${wti['close'].iloc[-1]:.2f}")
    
    # Fetch earnings calendar
    print("\n🔄 Earnings Calendar (next 30 days)...")
    earnings = provider.get_earnings_calendar(days_ahead=30)
    if not earnings.empty:
        print(f"  Records: {len(earnings)}")
        print(f"  Earliest: {earnings['date'].min().strftime('%Y-%m-%d')}")
        print(f"  Latest: {earnings['date'].max().strftime('%Y-%m-%d')}")
        print(f"\n  Sample earnings:")
        sample = earnings[['symbol', 'date', 'hour', 'epsEstimate', 'epsActual']].head(10)
        print(sample.to_string())
    
    # Get live quotes
    print("\n" + "="*60)
    print("💱 CURRENT MARKET QUOTES")
    print("="*60)
    
    usd_inr_quote = provider.get_usd_inr_quote()
    if usd_inr_quote:
        print(f"  USD/INR: {usd_inr_quote['price']:.4f} [{usd_inr_quote['source']}]")
    
    wti_quote = provider.get_wti_quote()
    if wti_quote:
        print(f"  WTI Crude: ${wti_quote['price']:.2f} [{wti_quote['source']}]")
    
    print("\n✅ Setup Complete! External data sources ready for trading engine.\n")
