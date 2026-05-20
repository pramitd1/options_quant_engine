from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from models.heston.heston_calibration import (
    HestonCalibrationResult,
    _bound_hit_fields,
    calibrate_heston_to_chain,
    prepare_calibration_points,
)
from models.heston.heston_features import (
    HESTON_FEATURE_COLUMNS,
    build_heston_research_features,
    default_heston_research_features,
)
from models.heston.heston_pricer import (
    HestonParams,
    black_scholes_price,
    heston_implied_vol_proxy,
    heston_price,
)
from research.signal_evaluation.heston_research_report import (
    build_heston_research_report,
    render_heston_research_markdown,
    write_heston_research_report,
)


VALUATION_TIME = "2026-05-19T09:30:00+05:30"


def _synthetic_chain(*, spot: float = 10000.0, expiry: str = "2026-05-26") -> pd.DataFrame:
    params = HestonParams(kappa=1.7, theta=0.052, vol_of_vol=0.55, rho=-0.42, v0=0.048)
    rows = []
    for strike in [9700, 9800, 9900, 10000, 10100, 10200, 10300]:
        tte = (pd.Timestamp(expiry).tz_localize("Asia/Kolkata").replace(hour=15, minute=30) - pd.Timestamp(VALUATION_TIME)).total_seconds()
        tte_years = max(tte / (365.0 * 24.0 * 3600.0), 1e-6)
        for option_type in ("CE", "PE"):
            price = heston_price(
                spot=spot,
                strike=strike,
                time_to_expiry_years=tte_years,
                option_type=option_type,
                params=params,
            )
            iv_proxy = heston_implied_vol_proxy(
                spot=spot,
                strike=strike,
                time_to_expiry_years=tte_years,
                params=params,
            )
            rows.append(
                {
                    "strikePrice": strike,
                    "OPTION_TYP": option_type,
                    "lastPrice": round(float(price or 0.0), 4),
                    "EXPIRY_DT": expiry,
                    "impliedVolatility": round(float(iv_proxy or 0.0) * 100.0, 4),
                    "totalTradedVolume": 50000 + abs(strike - spot),
                    "openInterest": 100000 + abs(strike - spot) * 10,
                    "bidPrice": round(float(price or 0.0) * 0.995, 4),
                    "askPrice": round(float(price or 0.0) * 1.005, 4),
                }
            )
    return pd.DataFrame(rows)


def test_heston_price_and_bs_price_are_finite_without_replacing_bs():
    params = HestonParams(kappa=1.5, theta=0.04, vol_of_vol=0.6, rho=-0.45, v0=0.05)

    bs_price = black_scholes_price(
        spot=10000,
        strike=10000,
        time_to_expiry_years=7 / 365,
        volatility=0.22,
        option_type="CE",
    )
    heston_proxy_price = heston_price(
        spot=10000,
        strike=10000,
        time_to_expiry_years=7 / 365,
        option_type="CE",
        params=params,
    )

    assert bs_price is not None and bs_price > 0
    assert heston_proxy_price is not None and heston_proxy_price > 0
    assert math.isfinite(bs_price)
    assert math.isfinite(heston_proxy_price)


def test_prepare_calibration_points_handles_same_day_date_only_expiry():
    chain = _synthetic_chain(expiry="2026-05-19")

    points = prepare_calibration_points(
        chain,
        spot=10000,
        valuation_time=VALUATION_TIME,
        max_rows=12,
    )

    assert len(points) == 12
    assert all(point.time_to_expiry_years > 0 for point in points)


def test_prepare_calibration_points_keeps_one_sided_quotes_with_valid_ltp():
    chain = _synthetic_chain()
    chain["askPrice"] = 0.0

    points = prepare_calibration_points(
        chain,
        spot=10000,
        valuation_time=VALUATION_TIME,
        max_rows=12,
    )

    assert len(points) == 12
    assert all(point.market_price > 0 for point in points)


def test_heston_calibration_succeeds_or_degrades_with_guarded_result():
    chain = _synthetic_chain()

    result = calibrate_heston_to_chain(
        chain,
        spot=10000,
        valuation_time=VALUATION_TIME,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
    )

    assert result.sample_size >= 8
    assert result.surface_quality in {"GOOD", "CAUTION", "WEAK", "REJECTED", "FAILED"}
    assert result.reason
    if result.success:
        assert result.params is not None
        assert result.calibration_error is not None
        assert 0.10 <= result.params.kappa <= 8.0
        assert -0.95 <= result.params.rho <= 0.25


def test_heston_feature_builder_is_optional_and_contract_complete():
    disabled = build_heston_research_features(_synthetic_chain(), spot=10000, enabled=False)

    assert set(HESTON_FEATURE_COLUMNS).issubset(disabled)
    assert disabled["heston_research_enabled"] is False
    assert disabled["heston_calibration_status"] == "DISABLED"

    default_enabled = default_heston_research_features(enabled=True)
    assert default_enabled["heston_calibration_status"] == "PENDING_SELECTION"


def test_heston_feature_builder_returns_research_only_comparison_fields():
    features = build_heston_research_features(
        _synthetic_chain(),
        spot=10000,
        selected_strike=10000,
        selected_option_type="CE",
        selected_expiry="2026-05-26",
        selected_iv=22.0,
        bs_delta=0.50,
        bs_gamma="N/A",
        valuation_time=VALUATION_TIME,
        enabled=True,
        min_rows=8,
        max_rows=12,
        reject_error=0.50,
        timeout_seconds=1.2,
    )

    assert set(HESTON_FEATURE_COLUMNS).issubset(features)
    assert features["heston_research_enabled"] is True
    assert features["heston_calibration_sample_size"] >= 8
    assert features["heston_calibration_status"] in {"CALIBRATED", "FAILED", "REJECTED"}
    diagnostics = json.loads(features["heston_diagnostics_json"])
    assert diagnostics["sample_size"] >= 8
    if features["heston_calibration_status"] == "CALIBRATED":
        assert features["heston_model_price"] is not None
        assert features["bs_model_price_for_heston"] is not None
        assert "heston_price_gap_rel_pct" in features
        assert features["greek_model_divergence_score"] is not None


def test_heston_bound_guard_flags_pinned_parameters():
    params = HestonParams(kappa=8.0, theta=1.0, vol_of_vol=0.55, rho=-0.95, v0=0.048)

    hits = _bound_hit_fields(params)

    assert "kappa_upper" in hits
    assert "theta_upper" in hits
    assert "rho_lower" in hits


def test_heston_feature_builder_rejects_extreme_bs_heston_price_gap(monkeypatch):
    from models.heston import heston_features

    def _fake_calibration(*args, **kwargs):
        return HestonCalibrationResult(
            True,
            HestonParams(kappa=1.5, theta=0.04, vol_of_vol=0.55, rho=-0.35, v0=0.04),
            0.04,
            "GOOD",
            "ok",
            12,
            1.0,
        )

    monkeypatch.setattr(heston_features, "calibrate_heston_to_chain", _fake_calibration)

    features = heston_features.build_heston_research_features(
        _synthetic_chain(),
        spot=10000,
        selected_strike=10000,
        selected_option_type="CE",
        selected_expiry="2026-05-26",
        selected_iv=0.03,
        bs_delta=0.50,
        bs_gamma=0.001,
        valuation_time=VALUATION_TIME,
        enabled=True,
    )

    assert features["heston_calibration_status"] == "REJECTED"
    assert features["heston_surface_quality"] == "REJECTED"
    assert "PRICE_GAP_REJECT" in features["heston_quality_flags"]
    assert features["heston_price_gap_rel_pct"] >= 100.0


def test_heston_feature_builder_suppresses_price_gap_when_selected_iv_is_proxy(monkeypatch):
    from models.heston import heston_features

    def _fake_calibration(*args, **kwargs):
        return HestonCalibrationResult(
            True,
            HestonParams(kappa=1.5, theta=0.04, vol_of_vol=0.55, rho=-0.35, v0=0.04),
            0.04,
            "GOOD",
            "ok",
            12,
            1.0,
        )

    monkeypatch.setattr(heston_features, "calibrate_heston_to_chain", _fake_calibration)

    features = heston_features.build_heston_research_features(
        _synthetic_chain(),
        spot=10000,
        selected_strike=10000,
        selected_option_type="CE",
        selected_expiry="2026-05-26",
        selected_iv=0.03,
        selected_iv_is_proxy=True,
        selected_iv_proxy_source="ATM_PROXY",
        bs_delta=0.50,
        bs_gamma=0.001,
        valuation_time=VALUATION_TIME,
        enabled=True,
    )

    assert features["heston_calibration_status"] == "CALIBRATED"
    assert features["heston_selected_iv_quality"] == "PROXY"
    assert "SELECTED_IV_PROXY" in features["heston_quality_flags"]
    assert "PRICE_GAP_SUPPRESSED_IV_QUALITY" in features["heston_quality_flags"]
    assert "PRICE_GAP_REJECT" not in features["heston_quality_flags"]


def test_heston_feature_builder_rejects_ultra_short_tte(monkeypatch):
    from models.heston import heston_features

    def _fake_calibration(*args, **kwargs):
        return HestonCalibrationResult(
            True,
            HestonParams(kappa=1.5, theta=0.04, vol_of_vol=0.55, rho=-0.35, v0=0.04),
            0.04,
            "GOOD",
            "ok",
            12,
            1.0,
        )

    monkeypatch.setattr(heston_features, "calibrate_heston_to_chain", _fake_calibration)

    features = heston_features.build_heston_research_features(
        _synthetic_chain(expiry="2026-05-19"),
        spot=10000,
        selected_strike=10000,
        selected_option_type="CE",
        selected_expiry="2026-05-19",
        selected_iv=22.0,
        bs_delta=0.50,
        bs_gamma=0.001,
        valuation_time="2026-05-19T15:15:00+05:30",
        enabled=True,
    )

    assert features["heston_calibration_status"] == "REJECTED"
    assert features["heston_tte_bucket"] == "ULTRA_SHORT_TTE"
    assert features["heston_short_tte_guard"] == "SHORT_TTE_REJECT"
    assert "SHORT_TTE_REJECT" in features["heston_quality_flags"]


def test_heston_research_report_summarizes_quality_and_importance(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "signal_timestamp": f"2026-05-19T09:{30 + idx:02d}:00+05:30",
                "heston_research_enabled": True,
                "heston_calibration_status": "CALIBRATED",
                "heston_surface_quality": "GOOD" if idx % 2 == 0 else "CAUTION",
                "heston_tte_days": 0.2 if idx < 2 else 5.0,
                "heston_tte_bucket": "EXPIRY_DAY" if idx < 2 else "FRONT_WEEK",
                "heston_expiry_context": "EXPIRY_DAY" if idx < 2 else "NON_EXPIRY",
                "heston_short_tte_guard": "SHORT_TTE_WEAK" if idx < 2 else "NONE",
                "heston_selected_iv_quality": "OK" if idx % 3 else "PROXY",
                "heston_kappa": 1.2 + idx * 0.01,
                "heston_theta": 0.04 + idx * 0.001,
                "heston_vol_of_vol": 0.5 + idx * 0.01,
                "heston_rho": -0.4 + idx * 0.005,
                "heston_v0": 0.04 + idx * 0.001,
                "heston_calibration_error": 0.04 + idx * 0.002,
                "heston_bound_hit_count": 0,
                "heston_forward_variance_proxy": 0.04 + idx * 0.001,
                "bs_vs_heston_price_gap": idx * 0.3,
                "heston_price_gap_rel_pct": idx * 1.5,
                "bs_vs_heston_greek_gap": idx * 0.01,
                "greek_model_divergence_score": 10 + idx,
                "heston_quality_flags": "PRICE_GAP_WEAK" if idx == 0 else "",
                "correct_60m": 1 if idx % 2 == 0 else 0,
                "signed_return_60m_bps": 8 - idx,
                "tradeability_score": 70 - idx,
                "option_efficiency_score": 65 + idx,
                "strike_efficiency_score": 60 + idx,
                "volatility_explosion_probability": 0.2 + idx * 0.01,
            }
            for idx in range(8)
        ]
    )

    report = build_heston_research_report(frame, dataset_path="signals.csv", min_sample=4)
    markdown = render_heston_research_markdown(report)
    artifact = write_heston_research_report(
        frame,
        dataset_path="signals.csv",
        output_dir=tmp_path,
        report_name="heston_test",
        min_sample=4,
    )

    assert report["heston_calibrated_row_count"] == 8
    assert report["quality_by_day"][0]["calibrated_count"] == 8
    assert report["quality_flag_counts"]["PRICE_GAP_WEAK"] == 1
    assert report["quality_by_tte_bucket"][0]["bucket"] in {"FRONT_WEEK", "EXPIRY_DAY"}
    assert report["quality_by_selected_iv_quality"]
    assert report["feature_importance"]
    assert "ml_feature_importance" in report
    assert "Black-Scholes remains the live Greek engine" in markdown
    assert Path(artifact["json_path"]).exists()
    assert Path(artifact["md_path"]).exists()
