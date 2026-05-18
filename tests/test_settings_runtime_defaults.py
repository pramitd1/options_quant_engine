from __future__ import annotations

import importlib


def test_core_runtime_defaults_can_be_overridden_by_env(monkeypatch):
    import config.settings as settings

    overrides = {
        "OQE_DEFAULT_SYMBOL": "banknifty",
        "OQE_DEFAULT_DATA_SOURCE": "zerodha",
        "OQE_REFRESH_INTERVAL": "7",
        "OQE_ICICI_REFRESH_INTERVAL": "6",
        "OQE_TARGET_PROFIT_PERCENT": "22.5",
        "OQE_STOP_LOSS_PERCENT": "11.5",
        "OQE_LOT_SIZE": "35",
        "OQE_RISK_FREE_RATE": "0.065",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    try:
        reloaded = importlib.reload(settings)

        assert reloaded.DEFAULT_SYMBOL == "BANKNIFTY"
        assert reloaded.DEFAULT_DATA_SOURCE == "ZERODHA"
        assert reloaded.REFRESH_INTERVAL == 7
        assert reloaded.ICICI_REFRESH_INTERVAL == 6
        assert reloaded.TARGET_PROFIT_PERCENT == 22.5
        assert reloaded.STOP_LOSS_PERCENT == 11.5
        assert reloaded.LOT_SIZE == 35
        assert reloaded.RISK_FREE_RATE == 0.065
    finally:
        for key in overrides:
            monkeypatch.delenv(key, raising=False)
        importlib.reload(settings)


def test_invalid_default_data_source_fails_fast(monkeypatch):
    import config.settings as settings

    monkeypatch.setenv("OQE_DEFAULT_DATA_SOURCE", "bad_source")
    try:
        try:
            importlib.reload(settings)
            raise AssertionError("settings reload should reject invalid default source")
        except ValueError as exc:
            assert "Invalid DEFAULT_DATA_SOURCE" in str(exc)
    finally:
        monkeypatch.delenv("OQE_DEFAULT_DATA_SOURCE", raising=False)
        importlib.reload(settings)
