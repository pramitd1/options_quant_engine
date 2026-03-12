import requests
import pandas as pd


def load_live_option_chain(symbol):

    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)

    data = r.json()

    rows = []

    for row in data["records"]["data"]:

        strike = row["strikePrice"]

        if "CE" in row:

            ce = row["CE"]

            rows.append({

                "STRIKE_PR": strike,
                "OPTION_TYP": "CE",
                "OPEN_INT": ce["openInterest"],
                "IV": ce["impliedVolatility"],
                "DELTA": 0.5,
                "GAMMA": 0.005,
                "UNDERLYING_VALUE": ce["underlyingValue"]

            })

        if "PE" in row:

            pe = row["PE"]

            rows.append({

                "STRIKE_PR": strike,
                "OPTION_TYP": "PE",
                "OPEN_INT": pe["openInterest"],
                "IV": pe["impliedVolatility"],
                "DELTA": -0.5,
                "GAMMA": 0.005,
                "UNDERLYING_VALUE": pe["underlyingValue"]

            })

    return pd.DataFrame(rows)