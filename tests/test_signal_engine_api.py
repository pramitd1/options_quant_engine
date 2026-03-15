from __future__ import annotations

import engine.signal_engine as signal_engine
import engine.trading_engine as trading_engine


def test_trading_engine_facade_matches_signal_engine():
    assert trading_engine.generate_trade is signal_engine.generate_trade
