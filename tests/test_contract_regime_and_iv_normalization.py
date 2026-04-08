import numpy as np
import pandas as pd

from analytics.volatility_regime import detect_volatility_regime
from analytics.volatility_surface import compute_risk_reversal
from analytics.volatility_surface import iv_hv_spread
from analytics.volatility_surface import vol_regime
from config.signal_evaluation_scoring import get_signal_evaluation_selection_policy
from config.signal_policy import get_activation_score_policy_config, get_trade_runtime_thresholds
from engine.signal_engine import _resolve_regime_thresholds
from engine.trading_support.probability import _blend_move_probability, _compute_atm_iv_percentile
from strategy.enhanced_strike_scoring import compute_premium_efficiency
from tuning.registry import get_parameter_registry
from tuning.runtime import temporary_parameter_pack
from utils.regime_normalization import canonical_gamma_regime, normalize_iv_decimal


def test_vol_surface_regime_is_unit_invariant():
    # Same volatility level expressed in decimal or percent should map identically.
    assert vol_regime(0.30) == "HIGH_VOL"
    assert vol_regime(30.0) == "HIGH_VOL"

    assert vol_regime(0.18) == "NORMAL_VOL"
    assert vol_regime(18.0) == "NORMAL_VOL"

    assert vol_regime(0.10) == "LOW_VOL"
    assert vol_regime(10.0) == "LOW_VOL"


def test_enhanced_strike_expected_move_is_unit_invariant():
    rows = pd.DataFrame(
        {
            "lastPrice": [100.0, 200.0, 300.0],
            "strikePrice": [22000.0, 22100.0, 22200.0],
        }
    )

    decimal_iv = compute_premium_efficiency(
        rows,
        spot=22000.0,
        atm_iv=0.20,
        days_to_expiry=7.0,
    )
    percent_iv = compute_premium_efficiency(
        rows,
        spot=22000.0,
        atm_iv=20.0,
        days_to_expiry=7.0,
    )

    assert np.allclose(
        decimal_iv.to_numpy(dtype=float),
        percent_iv.to_numpy(dtype=float),
        atol=1e-9,
    )


def test_gamma_regime_synonyms_have_identical_threshold_behavior():
    runtime_thresholds = get_trade_runtime_thresholds()

    short_zone = _resolve_regime_thresholds(
        runtime_thresholds=runtime_thresholds,
        base_min_trade_strength=60,
        base_min_composite_score=55,
        market_state={
            "spot_vs_flip": "ABOVE_FLIP",
            "gamma_regime": "SHORT_GAMMA_ZONE",
            "dealer_pos": "NEUTRAL",
        },
    )
    negative = _resolve_regime_thresholds(
        runtime_thresholds=runtime_thresholds,
        base_min_trade_strength=60,
        base_min_composite_score=55,
        market_state={
            "spot_vs_flip": "ABOVE_FLIP",
            "gamma_regime": "NEGATIVE_GAMMA",
            "dealer_pos": "NEUTRAL",
        },
    )

    assert short_zone == negative

    long_zone = _resolve_regime_thresholds(
        runtime_thresholds=runtime_thresholds,
        base_min_trade_strength=60,
        base_min_composite_score=55,
        market_state={
            "spot_vs_flip": "ABOVE_FLIP",
            "gamma_regime": "LONG_GAMMA_ZONE",
            "dealer_pos": "NEUTRAL",
        },
    )
    positive = _resolve_regime_thresholds(
        runtime_thresholds=runtime_thresholds,
        base_min_trade_strength=60,
        base_min_composite_score=55,
        market_state={
            "spot_vs_flip": "ABOVE_FLIP",
            "gamma_regime": "POSITIVE_GAMMA",
            "dealer_pos": "NEUTRAL",
        },
    )

    assert long_zone == positive


def test_move_probability_floor_contract_is_shared_between_live_and_research():
    selection_policy = get_signal_evaluation_selection_policy()
    activation_cfg = get_activation_score_policy_config()
    assert selection_policy["move_probability_floor"] == activation_cfg.move_probability_floor


def test_normalization_helpers_contract():
    assert normalize_iv_decimal(20.0) == 0.2
    assert normalize_iv_decimal(0.2) == 0.2
    assert canonical_gamma_regime("short_gamma_zone") == "NEGATIVE_GAMMA"
    assert canonical_gamma_regime("long_gamma_zone") == "POSITIVE_GAMMA"


def test_atm_iv_percentile_is_unit_invariant():
    assert _compute_atm_iv_percentile(18.0) == _compute_atm_iv_percentile(0.18)


def test_probability_blend_rule_fallback_contract():
    # With no ML leg, hybrid must equal bounded rule probability (no extra calibration drift).
    assert _blend_move_probability(0.62, None) == 0.62


def test_tuning_registry_move_probability_floor_bounds_are_probability_scale():
    registry = get_parameter_registry()
    definition = registry.get("evaluation_thresholds.selection.move_probability_floor")
    assert definition.min_value == 0.0
    assert definition.max_value == 1.0


def test_detect_volatility_regime_is_iv_unit_invariant():
    chain_pct = pd.DataFrame({"IV": [12.0, 16.0, 22.0, 35.0]})
    chain_dec = pd.DataFrame({"IV": [0.12, 0.16, 0.22, 0.35]})
    assert detect_volatility_regime(chain_pct) == detect_volatility_regime(chain_dec)


def test_risk_reversal_is_iv_unit_invariant():
    chain_pct = pd.DataFrame(
        {
            "STRIKE_PR": [985.0, 1015.0],
            "OPTION_TYP": ["PE", "CE"],
            "IV": [22.0, 18.0],
            "EXPIRY_DT": ["2026-03-27", "2026-03-27"],
        }
    )
    chain_dec = chain_pct.copy()
    chain_dec["IV"] = chain_dec["IV"] / 100.0

    rr_pct = compute_risk_reversal(chain_pct, spot=1000.0)
    rr_dec = compute_risk_reversal(chain_dec, spot=1000.0)

    assert rr_pct["rr_value"] == rr_dec["rr_value"]
    assert rr_pct["rr_value_points"] == rr_dec["rr_value_points"]
    assert rr_pct["rr_value_decimal"] == rr_dec["rr_value_decimal"]
    assert rr_pct["rr_unit"] == "VOL_POINTS"
    assert rr_dec["rr_unit"] == "VOL_POINTS"
    assert rr_pct["rr_regime"] == rr_dec["rr_regime"]


def test_iv_hv_regime_relative_threshold_low_vol():
    # In low vol, +1 vol point can be materially rich.
    out = iv_hv_spread(0.09, 0.08)
    assert out["iv_hv_spread"] == 0.01
    assert out["iv_hv_spread_relative"] == 0.125
    assert out["iv_hv_regime"] == "IV_FAIR"

    out_rich = iv_hv_spread(0.095, 0.08)
    assert out_rich["iv_hv_spread_relative"] == 0.1875
    assert out_rich["iv_hv_regime"] == "IV_RICH"


def test_iv_hv_regime_relative_threshold_high_vol():
    # In high vol, a larger absolute spread can still be fair on a relative basis.
    out = iv_hv_spread(0.42, 0.40)
    assert out["iv_hv_spread"] == 0.02
    assert out["iv_hv_spread_relative"] == 0.05
    assert out["iv_hv_regime"] == "IV_FAIR"

    out_cheap = iv_hv_spread(0.32, 0.40)
    assert out_cheap["iv_hv_spread_relative"] == -0.2
    assert out_cheap["iv_hv_regime"] == "IV_CHEAP"


def test_iv_hv_spread_respects_runtime_policy_override():
    with temporary_parameter_pack(
        "iv_hv_override_test",
        overrides={
            "analytics.iv_hv_spread.rich_threshold_relative": 0.10,
            "analytics.iv_hv_spread.cheap_threshold_relative": -0.10,
        },
    ):
        out = iv_hv_spread(0.09, 0.08)
    assert out["iv_hv_spread_relative"] == 0.125
    assert out["iv_hv_regime"] == "IV_RICH"
