from __future__ import annotations

import pandas as pd

from features.ta_indicators import get_ta_features_for_trade


def test_get_ta_features_for_trade_does_not_fetch_live_history_when_disallowed(monkeypatch):
    import features.ta_indicators as ta

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("live history fetch should not be called")

    monkeypatch.setattr(ta, "get_recent_spot_history", _raise_if_called)

    features = get_ta_features_for_trade(
        "NIFTY",
        100.5,
        allow_live_history=False,
    )

    assert features["ta_direction"] == "NO_SIGNAL"
    assert features["ta_confidence"] == 0.0
    assert features["ta_regime"] == "point_in_time_unavailable"
    assert features["ta_warning"] == "ta_history_not_supplied_for_historical_mode"


def test_get_ta_features_for_trade_filters_supplied_history_as_of(monkeypatch):
    import features.ta_indicators as ta

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("live history fetch should not be called")

    monkeypatch.setattr(ta, "get_recent_spot_history", _raise_if_called)
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01 09:15", periods=60, freq="D", tz="Asia/Kolkata"),
            "close": [100.0 + idx for idx in range(60)],
        }
    )

    features = get_ta_features_for_trade(
        "NIFTY",
        120.0,
        history_df=history,
        as_of="2026-03-31T15:30:00+05:30",
        allow_live_history=False,
    )

    assert features["ta_regime"] != "point_in_time_unavailable"
    assert features["indicators"]
