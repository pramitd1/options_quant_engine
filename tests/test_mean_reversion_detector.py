from __future__ import annotations

import pandas as pd

from config.policy_resolver import temporary_parameter_pack
from analytics.mean_reversion_detector import compute_mean_reversion_features, detect_mean_reversion_opportunity


def test_compute_mean_reversion_features_detects_trend_continuation_for_geometric_price_series():
    prices = [100.0 * (1.01 ** i) for i in range(20)]
    features = compute_mean_reversion_features(prices)

    assert features["mean_reversion_signal"] == "TREND_CONTINUATION"
    assert features["mean_reversion_strength"] <= 10.0
    assert isinstance(features["mean_reversion_zscore"], float)


def test_detect_mean_reversion_opportunity_returns_true_for_large_deviation():
    prices = [100.0] * 18 + [110.0, 150.0]
    assert detect_mean_reversion_opportunity(prices, threshold=1.0)


def test_get_mean_reversion_features_for_trade_uses_historical_close_price_axis(monkeypatch):
    import analytics.mean_reversion_detector as mrd

    monkeypatch.setattr(
        mrd,
        "get_recent_spot_history",
        lambda symbol, days_history: pd.DataFrame(
            {"close": [100.0, 101.5, 102.0, 101.0, 99.0, 98.5, 100.0, 99.5, 100.5]}
        ),
    )

    features = mrd.get_mean_reversion_features_for_trade("NIFTY", 100.5, days_history=9)

    assert isinstance(features, dict)
    assert "mean_reversion_signal" in features
    assert "mean_reversion_strength" in features
    assert "mean_reversion_zscore" in features
    assert 0.0 <= features["mean_reversion_strength"] <= 100.0


def test_mean_reversion_defaults_are_runtime_policy_driven(monkeypatch):
    import analytics.mean_reversion_detector as mrd

    captured = {}

    def _fake_history(symbol, days_history):
        captured["symbol"] = symbol
        captured["days_history"] = days_history
        return pd.DataFrame({"close": [100.0 + idx for idx in range(12)]})

    monkeypatch.setattr(mrd, "get_recent_spot_history", _fake_history)

    with temporary_parameter_pack(
        "mean_reversion_policy_test",
        overrides={
            "analytics.mean_reversion.default_history_days": 12,
            "analytics.mean_reversion.lookback": 6,
            "analytics.mean_reversion.zscore_threshold": 0.25,
            "analytics.mean_reversion.strength_scale": 20.0,
        },
    ):
        features = mrd.get_mean_reversion_features_for_trade("NIFTY", 111.0)

    assert captured == {"symbol": "NIFTY", "days_history": 12}
    assert "mean_reversion_signal" in features
    assert 0.0 <= features["mean_reversion_strength"] <= 100.0


def test_get_mean_reversion_features_for_trade_does_not_fetch_live_history_when_disallowed(monkeypatch):
    import analytics.mean_reversion_detector as mrd

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("live history fetch should not be called")

    monkeypatch.setattr(mrd, "get_recent_spot_history", _raise_if_called)

    features = mrd.get_mean_reversion_features_for_trade(
        "NIFTY",
        100.5,
        allow_live_history=False,
    )

    assert features["mean_reversion_signal"] == "INSUFFICIENT_DATA"
    assert features["mean_reversion_reason"] == "mean_reversion_history_not_supplied_for_historical_mode"


def test_compute_mean_reversion_features_handles_insufficient_history():
    features = compute_mean_reversion_features([])
    assert features["mean_reversion_signal"] == "INSUFFICIENT_DATA"
    assert features["mean_reversion_strength"] == 0.0
