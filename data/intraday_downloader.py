import yfinance as yf


def download_intraday(symbol, days=30):

    ticker = symbol + ".NS"

    data = yf.download(
        ticker,
        period=str(days) + "d",
        interval="5m"
    )

    return data