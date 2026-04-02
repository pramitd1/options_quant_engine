from engine import signal_engine as se
from data.option_chain_validation import validate_option_chain
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot
from engine.signal_engine import generate_trade
from strategy.path_aware_filtering import PathAwareFilter
from strategy.time_decay_model import TimeDecayModel


def test_signal_elapsed_minutes_progression_and_reset():
    se._DECAY_SIGNAL_STATE.clear()

    t0 = "2026-03-25T10:00:00+05:30"
    t1 = "2026-03-25T10:05:00+05:30"

    m0 = se._compute_signal_elapsed_minutes(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t0,
        direction="CALL",
    )
    m1 = se._compute_signal_elapsed_minutes(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t1,
        direction="CALL",
    )

    assert m0 == 0.0
    assert m1 > 0.0

    # Direction flip should reset age.
    m_flip = se._compute_signal_elapsed_minutes(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t1,
        direction="PUT",
    )
    assert m_flip == 0.0


def test_path_observation_proxy_directional_signs():
    se._PATH_SIGNAL_STATE.clear()

    t0 = "2026-03-25T10:00:00+05:30"
    t1 = "2026-03-25T10:05:00+05:30"

    # First point has no previous state.
    mfe0, mae0 = se._compute_path_observation_bps(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t0,
        spot=22000.0,
        direction="CALL",
    )
    assert mfe0 is None and mae0 is None

    # Spot up move favors CALL.
    mfe1, mae1 = se._compute_path_observation_bps(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t1,
        spot=22022.0,
        direction="CALL",
    )
    assert mfe1 is not None and mae1 is not None
    assert mfe1 >= 0.0
    assert mae1 <= 0.0

    # Same up move is adverse for PUT.
    mfe2, mae2 = se._compute_path_observation_bps(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time="2026-03-25T10:10:00+05:30",
        spot=22044.0,
        direction="PUT",
    )
    assert mfe2 >= 0.0
    assert mae2 <= 0.0


def test_path_filter_flags_more_adverse_mae_as_hostile():
    pf = PathAwareFilter()
    mild = pf.check_path_geometry(
        gamma_regime="NEGATIVE_GAMMA",
        direction="CALL",
        mfe_observed_bps=0.0,
        mae_observed_bps=-20.0,
        window="5m",
        mae_zscore_threshold=1.0,
    )
    severe = pf.check_path_geometry(
        gamma_regime="NEGATIVE_GAMMA",
        direction="CALL",
        mfe_observed_bps=0.0,
        mae_observed_bps=-80.0,
        window="5m",
        mae_zscore_threshold=1.0,
    )

    assert severe["mae_zscore"] > mild["mae_zscore"]
    assert severe["path_status"] == "HOSTILE"


def test_time_decay_half_life_returns_half_factor():
    model = TimeDecayModel(
        positive_gamma_half_life_m=240,
        negative_gamma_half_life_m=240,
        neutral_gamma_half_life_m=230,
        steepness=1.5,
    )
    factor = model.compute_decay_factor(240.0, "POSITIVE_GAMMA", "NORMAL_VOL")
    assert 0.49 <= factor <= 0.51


def test_signal_elapsed_minutes_isolated_by_expiry_key():
    se._DECAY_SIGNAL_STATE.clear()

    t0 = "2026-03-25T10:00:00+05:30"
    t1 = "2026-03-25T10:05:00+05:30"

    # Build state for one expiry.
    _ = se._compute_signal_elapsed_minutes(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t0,
        direction="CALL",
    )
    elapsed_primary = se._compute_signal_elapsed_minutes(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t1,
        direction="CALL",
    )

    # A different expiry should start from a fresh state (no cross-talk).
    elapsed_secondary = se._compute_signal_elapsed_minutes(
        symbol="NIFTY",
        selected_expiry="2026-04-02",
        valuation_time=t1,
        direction="CALL",
    )

    assert elapsed_primary > 0.0
    assert elapsed_secondary == 0.0


def test_path_observation_isolated_by_symbol_key():
    se._PATH_SIGNAL_STATE.clear()

    t0 = "2026-03-25T10:00:00+05:30"
    t1 = "2026-03-25T10:05:00+05:30"

    # First symbol gets a baseline then a move.
    _ = se._compute_path_observation_bps(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t0,
        spot=22000.0,
        direction="CALL",
    )
    mfe_primary, mae_primary = se._compute_path_observation_bps(
        symbol="NIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t1,
        spot=22022.0,
        direction="CALL",
    )

    # Different symbol should not inherit prior state.
    mfe_secondary, mae_secondary = se._compute_path_observation_bps(
        symbol="BANKNIFTY",
        selected_expiry="2026-03-26",
        valuation_time=t1,
        spot=48000.0,
        direction="CALL",
    )

    assert mfe_primary is not None and mae_primary is not None
    assert mfe_secondary is None and mae_secondary is None


def test_generate_trade_exposes_score_calibration_metadata():
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot.get("spot"))
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    trade = generate_trade(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        option_chain_validation=option_chain_validation,
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
    )

    assert isinstance(trade, dict)
    assert "score_calibration_enabled" in trade
    assert "score_calibration_applied" in trade
    assert trade.get("score_calibration_backend") == "isotonic"
    assert trade.get("score_calibration_artifact_path")
    assert "time_decay_enabled" in trade
    assert "time_decay_applied" in trade
    assert "time_decay_fallback_used" in trade
    assert "time_decay_elapsed_source" in trade


def test_generate_trade_marks_reversal_age_time_decay_fallback(monkeypatch):
    spot_snapshot = load_spot_snapshot(
        "debug_samples/replay_fixtures/spot_snapshots/NIFTY_spot_snapshot_2026-03-25T09-50-00+05-30.json"
    )
    option_chain = load_option_chain_snapshot(
        "debug_samples/replay_fixtures/option_chain_snapshots/NIFTY_ICICI_option_chain_snapshot_2026-03-25T09-50-05.194161+05-30.csv"
    )
    spot = float(spot_snapshot.get("spot"))
    option_chain_validation = validate_option_chain(option_chain, spot=spot)

    monkeypatch.setattr(se, "_compute_signal_elapsed_minutes", lambda **kwargs: 0.0)
    monkeypatch.setattr(
        se,
        "_compute_signal_state",
        lambda **kwargs: {
            "direction": "CALL",
            "direction_source": "test_fixture",
            "trade_strength": 80,
            "scoring_breakdown": {},
            "confirmation": {
                "status": "CONFIRMED",
                "veto": False,
                "reasons": [],
                "breakdown": {},
                "score_adjustment": 0,
            },
        },
    )
    monkeypatch.setattr(
        se,
        "evaluate_global_risk_layer",
        lambda **kwargs: {
            "risk_trade_status": "TRADE_OK",
            "risk_message": "",
            "global_risk_state": "GLOBAL_NEUTRAL",
            "global_risk_score": 0,
            "global_risk_reasons": [],
            "overnight_gap_risk_score": 0,
            "volatility_expansion_risk_score": 0,
            "global_risk_features": {},
            "global_risk_diagnostics": {},
            "global_risk_level": "LOW",
            "global_risk_action": "ALLOW",
            "global_risk_size_cap": 1.0,
        },
    )

    trade = generate_trade(
        symbol="NIFTY",
        spot=spot,
        option_chain=option_chain,
        previous_chain=None,
        day_high=spot_snapshot.get("day_high"),
        day_low=spot_snapshot.get("day_low"),
        day_open=spot_snapshot.get("day_open"),
        prev_close=spot_snapshot.get("prev_close"),
        lookback_avg_range_pct=spot_snapshot.get("lookback_avg_range_pct"),
        spot_validation=spot_snapshot.get("validation") or {},
        option_chain_validation=option_chain_validation,
        backtest_mode=True,
        valuation_time=spot_snapshot.get("timestamp"),
        reversal_age=3,
    )

    assert isinstance(trade, dict)
    assert trade.get("time_decay_fallback_used") is True
    assert trade.get("time_decay_elapsed_source") == "reversal_age_fallback"
    assert trade.get("time_decay_elapsed_minutes") == 15.0
