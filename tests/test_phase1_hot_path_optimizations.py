import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from analytics.greeks_engine import enrich_chain_with_greeks
from analytics.smart_money_flow import classify_flow, detect_unusual_volume


def test_front_expiry_atm_slice_reuses_cached_slice(monkeypatch):
    option_chain = pd.DataFrame(
        {
            "strikePrice": [95, 100, 105, 110, 115],
            "EXPIRY_DT": ["2099-12-31"] * 5,
            "OPTION_TYP": ["CE", "CE", "PE", "PE", "CE"],
            "openInterest": [100, 120, 140, 160, 180],
        }
    )

    call_counts = {"resolve": 0, "filter": 0}

    def _fake_resolve(df):
        call_counts["resolve"] += 1
        return "2099-12-31"

    def _fake_filter(df, expiry):
        call_counts["filter"] += 1
        return df.copy()

    monkeypatch.setattr("analytics.flow_utils.resolve_selected_expiry", _fake_resolve)
    monkeypatch.setattr("analytics.flow_utils.filter_option_chain_by_expiry", _fake_filter)

    first = front_expiry_atm_slice(option_chain, spot=105.0, strike_window_steps=1)
    second = front_expiry_atm_slice(option_chain, spot=105.0, strike_window_steps=1)

    assert not first.empty
    assert first.equals(second)
    assert call_counts == {"resolve": 1, "filter": 1}


def test_enrich_chain_with_greeks_keeps_iv_columns_in_sync():
    option_chain = pd.DataFrame(
        {
            "strikePrice": [100, 100],
            "EXPIRY_DT": ["2099-12-31", "2099-12-31"],
            "OPTION_TYP": ["CE", "PE"],
            "lastPrice": [8.5, 7.8],
            "IV": [0.0, 0.0],
            "openInterest": [1000, 1000],
        }
    )

    enriched = enrich_chain_with_greeks(
        option_chain,
        spot=100.0,
        valuation_time="2099-01-01T09:15:00Z",
    )

    assert len(enriched) == len(option_chain)
    assert "DELTA" in enriched.columns
    assert "GAMMA" in enriched.columns
    assert "VEGA" in enriched.columns
    assert enriched["IV"].notna().all()
    assert enriched["IV"].equals(enriched["impliedVolatility"])


def test_detect_unusual_volume_emits_bounded_opening_activity_weight():
    option_chain = pd.DataFrame(
        {
            "strikePrice": [100, 100],
            "EXPIRY_DT": ["2099-12-31", "2099-12-31"],
            "OPTION_TYP": ["CE", "PE"],
            "VOLUME": [10, 500],
            "OPEN_INT": [100, 200],
            "CHG_IN_OI": [100000, 5],
            "LAST_PRICE": [10.0, 20.0],
            "DELTA": [0.5, -0.5],
        }
    )

    spikes = detect_unusual_volume(option_chain, spot=100.0)

    assert "OPENING_ACTIVITY_WEIGHT" in spikes.columns
    assert spikes["OPENING_ACTIVITY_WEIGHT"].max() <= 4.0
    assert spikes["OPENING_ACTIVITY_RATIO"].max() > 0


def test_classify_flow_does_not_let_raw_oi_change_dominate_notional():
    spikes = pd.DataFrame(
        {
            "OPTION_TYP": ["CE", "PE"],
            "FLOW_NOTIONAL": [100.0, 5000.0],
            "OPENING_ACTIVITY": [100000.0, 0.0],
            "OPENING_ACTIVITY_WEIGHT": [2.0, 1.0],
        }
    )

    assert classify_flow(spikes) == "BEARISH_FLOW"
