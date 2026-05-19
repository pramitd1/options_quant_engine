from __future__ import annotations

import unittest
from unittest.mock import patch

from data.icici_gift_nifty import (
    fetch_icici_gift_nifty_snapshot,
    invalidate_icici_gift_nifty_cache,
)


class FakeBreezeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get_quotes(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"Error": "exhausted"}


class ICICIGiftNiftyTests(unittest.TestCase):
    def tearDown(self):
        invalidate_icici_gift_nifty_cache()

    def test_fetch_tries_candidates_until_quote_has_change(self):
        invalidate_icici_gift_nifty_cache()
        client = FakeBreezeClient(
            [
                {"Error": "stock not found"},
                {
                    "Success": [
                        {
                            "ltp": "23750.50",
                            "previous_close": "23650.00",
                            "ltt": "19-May-2026 09:20:00",
                        }
                    ]
                },
            ]
        )

        with patch("data.icici_gift_nifty.GIFT_NIFTY_SOURCE", "ICICI"), patch(
            "data.icici_gift_nifty.GIFT_NIFTY_ICICI_CANDIDATES",
            (
                {"exchange_code": "NDX", "stock_code": "BAD", "product_type": ""},
                {"exchange_code": "NDX", "stock_code": "NIFTY", "product_type": ""},
            ),
        ), patch("data.icici_gift_nifty._build_breeze_client", return_value=client):
            quote = fetch_icici_gift_nifty_snapshot(force=True)

        self.assertTrue(quote["available"])
        self.assertEqual(quote["gift_nifty_source"], "ICICI")
        self.assertEqual(quote["gift_nifty_exchange_code"], "NDX")
        self.assertEqual(quote["gift_nifty_stock_code"], "NIFTY")
        self.assertAlmostEqual(quote["gift_nifty_change_24h"], 0.425, places=3)
        self.assertEqual(len(client.calls), 2)

    def test_fetch_accepts_provider_percent_change_field(self):
        invalidate_icici_gift_nifty_cache()
        client = FakeBreezeClient(
            [
                {
                    "Success": {
                        "last_price": "23801",
                        "percent_change": "0.72%",
                    }
                }
            ]
        )

        with patch("data.icici_gift_nifty.GIFT_NIFTY_SOURCE", "ICICI"), patch(
            "data.icici_gift_nifty.GIFT_NIFTY_ICICI_CANDIDATES",
            ({"exchange_code": "NDX", "stock_code": "NIFTY", "product_type": "futures"},),
        ), patch("data.icici_gift_nifty._build_breeze_client", return_value=client):
            quote = fetch_icici_gift_nifty_snapshot(force=True)

        self.assertTrue(quote["available"])
        self.assertAlmostEqual(quote["gift_nifty_change_24h"], 0.72)
        self.assertEqual(quote["gift_nifty_product_type"], "futures")

    def test_fetch_degrades_when_client_is_unavailable(self):
        invalidate_icici_gift_nifty_cache()

        with patch("data.icici_gift_nifty.GIFT_NIFTY_SOURCE", "ICICI"), patch(
            "data.icici_gift_nifty._build_breeze_client",
            side_effect=ValueError("missing session"),
        ):
            quote = fetch_icici_gift_nifty_snapshot(force=True)

        self.assertFalse(quote["available"])
        self.assertIn("gift_nifty_icici_client_unavailable:missing session", quote["issues"])


if __name__ == "__main__":
    unittest.main()
