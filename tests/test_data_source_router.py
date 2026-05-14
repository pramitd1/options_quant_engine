from __future__ import annotations

import logging

import pandas as pd

import data.data_source_router as data_source_router


class _WeakLoader:
    def __init__(self, source: str) -> None:
        self.source = source

    def fetch_option_chain(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strikePrice": 20000,
                    "OPTION_TYP": "CE",
                    "lastPrice": 100.0,
                    "EXPIRY_DT": "2026-05-23",
                },
                {
                    "strikePrice": 20100,
                    "OPTION_TYP": "PE",
                    "lastPrice": 95.0,
                    "EXPIRY_DT": "2026-05-23",
                },
            ]
        )

    def build_option_chain(self, symbol: str) -> pd.DataFrame:
        return self.fetch_option_chain(symbol)


class _UnexpectedFallbackLoader:
    def __init__(self, calls: list[str], source: str) -> None:
        calls.append(source)

    def fetch_option_chain(self, symbol: str) -> pd.DataFrame:
        raise AssertionError("fallback loader should not be used")

    def build_option_chain(self, symbol: str) -> pd.DataFrame:
        raise AssertionError("fallback loader should not be used")


def test_data_source_router_keeps_selected_source_on_weak_data(monkeypatch, caplog):
    """Weak selected-source data should warn, not switch to another provider."""
    fallback_calls: list[str] = []

    def _fake_loader_factories():
        return {
            "ICICI": lambda: _WeakLoader("ICICI"),
            "NSE": lambda: _UnexpectedFallbackLoader(fallback_calls, "NSE"),
            "ZERODHA": lambda: _UnexpectedFallbackLoader(fallback_calls, "ZERODHA"),
        }

    monkeypatch.setattr(data_source_router, "_build_loader_factories", _fake_loader_factories)

    router = data_source_router.DataSourceRouter("ICICI")
    with caplog.at_level(logging.WARNING, logger=data_source_router.__name__):
        result = router.get_option_chain("NIFTY")

    assert fallback_calls == []
    assert result["source"].unique().tolist() == ["ICICI"]
    assert result["underlying_symbol"].iloc[0] == "NIFTY"
    assert router.last_validation["is_valid"] is False
    assert "Keeping the user-selected source" in caplog.text
