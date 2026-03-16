"""
Module: generate_token.py

Purpose:
    Define configuration values used by generate token.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""
from kiteconnect import KiteConnect

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
request_token = "PASTE_REQUEST_TOKEN"

kite = KiteConnect(api_key=api_key)

data = kite.generate_session(request_token, api_secret=api_secret)

print("ACCESS TOKEN:")
print(data["access_token"])