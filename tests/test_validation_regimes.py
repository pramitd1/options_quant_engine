from __future__ import annotations

import pandas as pd

from tuning.regimes import label_validation_regimes


def test_label_validation_regimes_uses_configured_event_thresholds():
    frame = pd.DataFrame(
        [
            {
                "volatility_regime": "VOL_EXPANSION",
                "global_risk_state": "VOL_SHOCK",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "macro_regime": "RISK_OFF",
                "overnight_hold_allowed": True,
                "dealer_flow_state": "UPSIDE_HEDGING_ACCELERATION",
                "squeeze_risk_state": "HIGH_ACCELERATION_RISK",
                "macro_event_risk_score": 72.0,
            },
            {
                "volatility_regime": "NORMAL_VOL",
                "global_risk_state": "GLOBAL_NEUTRAL",
                "gamma_regime": "LONG_GAMMA_ZONE",
                "macro_regime": "MACRO_NEUTRAL",
                "overnight_hold_allowed": False,
                "dealer_flow_state": "PINNING_DOMINANT",
                "squeeze_risk_state": None,
                "macro_event_risk_score": 50.0,
            },
        ]
    )

    labeled = label_validation_regimes(frame)

    assert labeled["event_risk_bucket"].tolist() == ["HIGH_EVENT_RISK", "ELEVATED_EVENT_RISK"]
    assert labeled["squeeze_risk_bucket"].tolist() == ["HIGH_ACCELERATION_RISK", "PINNING_REGIME"]
