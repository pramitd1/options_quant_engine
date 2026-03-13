import json
import os

from breeze_connect import BreezeConnect


API_KEY = os.getenv("ICICI_BREEZE_API_KEY", "")
SECRET_KEY = os.getenv("ICICI_BREEZE_SECRET_KEY", "")
SESSION_TOKEN = os.getenv("ICICI_BREEZE_SESSION_TOKEN", "")
EXPIRY_DATE = os.getenv("ICICI_DEFAULT_EXPIRY_DATE", "")
SYMBOL = os.getenv("ICICI_TEST_SYMBOL", "NIFTY")
RIGHT = os.getenv("ICICI_TEST_RIGHT", "call")
STRIKE_PRICE = os.getenv("ICICI_TEST_STRIKE", "")


def _require(name: str, value: str):
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")


_require("ICICI_BREEZE_API_KEY", API_KEY)
_require("ICICI_BREEZE_SECRET_KEY", SECRET_KEY)
_require("ICICI_BREEZE_SESSION_TOKEN", SESSION_TOKEN)
_require("ICICI_DEFAULT_EXPIRY_DATE", EXPIRY_DATE)

breeze = BreezeConnect(api_key=API_KEY)
breeze.generate_session(api_secret=SECRET_KEY, session_token=SESSION_TOKEN)

print("\nSession initialized\n")

try:
    resp = breeze.get_option_chain_quotes(
        stock_code=SYMBOL,
        exchange_code="NFO",
        product_type="options",
        right=RIGHT,
        strike_price=STRIKE_PRICE,
        expiry_date=EXPIRY_DATE,
    )

    print("Response:\n")
    print(json.dumps(resp, indent=2)[:2000])

except Exception as e:
    print("ERROR:", e)
