import yfinance as yf


def get_spot_price(symbol):

    """
    Fetch latest spot price for an index or stock
    """

    ticker_map = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^NSEFIN"
    }

    ticker = ticker_map.get(symbol, symbol)

    data = yf.Ticker(ticker)

    hist = data.history(period="1d")

    if hist.empty:
        raise ValueError("Unable to fetch spot price")

    return float(hist["Close"].iloc[-1])