from __future__ import annotations

from data.feature_reliability import compute_feature_reliability_weights


def test_feature_reliability_weights_degrade_with_weaker_tradable_data() -> None:
    strong = {
        "tradable_data": {
            "crossed_locked": {"crossed_or_locked_ratio": 0.01},
            "outlier_rejection": {"outlier_ratio": 0.01},
            "per_strike_confidence": {"mean": 0.9},
        }
    }
    weak = {
        "tradable_data": {
            "crossed_locked": {"crossed_or_locked_ratio": 0.25},
            "outlier_rejection": {"outlier_ratio": 0.25},
            "per_strike_confidence": {"mean": 0.3},
        }
    }

    strong_w = compute_feature_reliability_weights(strong)
    weak_w = compute_feature_reliability_weights(weak)

    assert weak_w["flow"] < strong_w["flow"]
    assert weak_w["vol_surface"] < strong_w["vol_surface"]
    assert weak_w["greeks"] < strong_w["greeks"]
    assert weak_w["liquidity"] < strong_w["liquidity"]
