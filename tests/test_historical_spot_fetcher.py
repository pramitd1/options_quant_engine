from __future__ import annotations

import pandas as pd

from data import historical_spot_fetcher as hs


def test_fetch_historical_spot_ohlc_cache_only_returns_empty_when_no_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "historical_spot"
    monkeypatch.setattr(hs, "HISTORICAL_SPOT_DIR", cache_dir)

    result = hs.fetch_historical_spot_ohlc(
        symbol="NIFTY",
        start_date="2026-04-12",
        end_date="2026-05-12",
        cache_only=True,
    )

    assert result.empty


def test_fetch_historical_spot_ohlc_cache_only_loads_existing_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "historical_spot"
    cache_dir.mkdir(parents=True)

    symbol = "NIFTY"
    start_date = "2026-04-12"
    end_date = "2026-05-12"
    interval = "1D"

    cache_df = pd.DataFrame({
        "timestamp": pd.date_range(start=start_date, periods=5, freq="D"),
        "open": [23500, 23550, 23600, 23650, 23700],
        "high": [23550, 23600, 23650, 23700, 23750],
        "low": [23450, 23500, 23550, 23600, 23650],
        "close": [23525, 23575, 23625, 23675, 23725],
        "volume": [1000, 1100, 1200, 1150, 1300],
    })

    cache_file = cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}.parquet"
    cache_df.to_parquet(cache_file, index=False)
    monkeypatch.setattr(hs, "HISTORICAL_SPOT_DIR", cache_dir)

    result = hs.fetch_historical_spot_ohlc(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        cache_only=True,
    )

    assert not result.empty
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(result) == 5
    assert result.iloc[0]["open"] == 23500
