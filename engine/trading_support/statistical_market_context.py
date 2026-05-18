"""Artifact-backed statistical market context for live signal enrichment."""

from __future__ import annotations

from functools import lru_cache
from math import isfinite
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATISTICAL_MARKET_CONTEXT_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "research"
    / "ml_research"
    / "historical_priors"
    / "latest_statistical_market_context_artifact.json"
)

STATISTICAL_MARKET_CONTEXT_VERSION = "statistical_market_context_v1"
STATISTICAL_MARKET_CONTEXT_DECISION_MODE = "LIVE_APPLIED"

DEFAULT_STATISTICAL_MARKET_CONTEXT_ARTIFACT = {
    "artifact_version": "statistical_market_context_artifact_v1",
    "source": "code_fallback",
    "source_run_id": "fallback_20260518",
    "baseline": {
        "daily_range_median_bps": 109.0477,
        "daily_range_p95_bps": 349.3738,
        "fwd_abs_move_1d_mean_bps": 86.0521,
        "daily_return_excess_kurtosis": 16.0173,
    },
    "numeric_bucket_priors": {
        "india_vix_level": [
            {
                "bucket": "(9.149, 13.612]",
                "lower": 9.15,
                "upper": 13.61,
                "n": 881,
                "mean_return_bps": 3.3651,
                "hit_positive": 0.5460,
                "expected_abs_move_bps": 48.6661,
                "abs_move_delta_vs_base_bps": -37.3860,
                "expected_range_bps": 79.1121,
            },
            {
                "bucket": "(13.612, 15.84]",
                "lower": 13.62,
                "upper": 15.84,
                "n": 881,
                "mean_return_bps": 4.2699,
                "hit_positive": 0.5403,
                "expected_abs_move_bps": 59.1590,
                "abs_move_delta_vs_base_bps": -26.8931,
                "expected_range_bps": 97.8795,
            },
            {
                "bucket": "(15.84, 18.69]",
                "lower": 15.85,
                "upper": 18.69,
                "n": 880,
                "mean_return_bps": -0.9486,
                "hit_positive": 0.5034,
                "expected_abs_move_bps": 71.6314,
                "abs_move_delta_vs_base_bps": -14.4207,
                "expected_range_bps": 118.0737,
            },
            {
                "bucket": "(18.69, 24.058]",
                "lower": 18.70,
                "upper": 24.05,
                "n": 879,
                "mean_return_bps": 5.4390,
                "hit_positive": 0.5330,
                "expected_abs_move_bps": 83.0702,
                "abs_move_delta_vs_base_bps": -2.9819,
                "expected_range_bps": 136.7081,
            },
            {
                "bucket": "(24.058, 85.13]",
                "lower": 24.06,
                "upper": 85.13,
                "n": 881,
                "mean_return_bps": 9.3218,
                "hit_positive": 0.5289,
                "expected_abs_move_bps": 156.7199,
                "abs_move_delta_vs_base_bps": 70.6678,
                "expected_range_bps": 262.2542,
            },
        ],
        "realized_vol_20d": [
            {
                "bucket": "(0.214, 0.888]",
                "lower": 0.214,
                "upper": 0.888,
                "n": 908,
                "expected_abs_move_bps": 156.8619,
                "abs_move_delta_vs_base_bps": 70.8098,
                "expected_range_bps": 259.0,
            },
        ],
    },
    "categorical_bucket_priors": {
        "trend_20d_bucket": {
            "selloff": {
                "n": 290,
                "mean_return_bps": 7.8,
                "hit_positive": 0.52,
                "expected_abs_move_bps": 166.85,
                "abs_move_delta_vs_base_bps": 80.8,
                "expected_range_bps": 275.0,
            }
        }
    },
    "macro_context": {
        "shock_priors": {
            "sp500_change_24h": {
                "bottom_decile": {
                    "n": 425,
                    "threshold": -1.35,
                    "mean_return_bps": -8.0,
                    "hit_positive": 0.46,
                    "expected_abs_move_bps": 135.71,
                    "abs_move_delta_vs_base_bps": 49.66,
                    "expected_range_bps": 220.0,
                }
            },
            "india_vix_change_24h": {
                "top_decile": {
                    "n": 425,
                    "threshold": 8.0,
                    "mean_return_bps": -6.0,
                    "hit_positive": 0.47,
                    "expected_abs_move_bps": 140.0,
                    "abs_move_delta_vs_base_bps": 53.95,
                    "expected_range_bps": 230.0,
                }
            },
        },
        "interaction_priors": {
            "macro_commodity_bucket=commodity_down|trend_20d_bucket=selloff": {
                "interaction": "macro_commodity_bucket=commodity_down|trend_20d_bucket=selloff",
                "left_feature": "macro_commodity_bucket",
                "left_bucket": "commodity_down",
                "right_feature": "trend_20d_bucket",
                "right_bucket": "selloff",
                "n": 80,
                "mean_return_bps": 5.0,
                "hit_positive": 0.52,
                "expected_abs_move_bps": 180.33,
                "abs_move_delta_vs_base_bps": 94.28,
                "expected_range_bps": 290.0,
            }
        },
        "pca": {"pc1": {"explained_variance_ratio": 0.3125}},
    },
    "application_rules": {
        "min_bucket_n": 50,
        "expanded_abs_move_delta_bps": 20.0,
        "high_abs_move_delta_bps": 50.0,
        "compressed_abs_move_delta_bps": -20.0,
        "directional_mean_edge_bps": 8.0,
        "directional_hit_edge": 0.56,
        "conflict_hit_edge": 0.46,
        "tail_risk_size_cap": 0.80,
        "elevated_risk_size_cap": 0.90,
        "macro_expanded_abs_move_delta_bps": 35.0,
        "macro_high_abs_move_delta_bps": 70.0,
        "macro_directional_mean_edge_bps": 10.0,
        "macro_directional_hit_edge": 0.57,
        "macro_conflict_hit_edge": 0.46,
        "macro_tail_risk_size_cap": 0.85,
        "macro_elevated_risk_size_cap": 0.90,
        "macro_conflict_size_cap": 0.85,
        "macro_max_score_adjustment": 3,
        "macro_max_probability_adjustment": 0.02,
    },
}

MACRO_SHOCK_FEATURES = [
    "oil_change_24h",
    "gold_change_24h",
    "copper_change_24h",
    "vix_change_24h",
    "india_vix_change_24h",
    "sp500_change_24h",
    "nasdaq_change_24h",
    "us10y_change_bp",
    "usdinr_change_24h",
]


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not isfinite(number):
        return default
    return number


def _round(value, digits=4):
    value = _safe_float(value, None)
    return round(value, digits) if value is not None else None


def _coerce_artifact(payload):
    if not isinstance(payload, dict):
        return DEFAULT_STATISTICAL_MARKET_CONTEXT_ARTIFACT
    if not isinstance(payload.get("numeric_bucket_priors"), dict):
        return DEFAULT_STATISTICAL_MARKET_CONTEXT_ARTIFACT
    return payload


@lru_cache(maxsize=4)
def load_statistical_market_context_artifact(path: str | Path | None = None) -> dict:
    """Load the compact statistical-market-context artifact with a safe fallback."""

    artifact_path = Path(path) if path is not None else DEFAULT_STATISTICAL_MARKET_CONTEXT_ARTIFACT_PATH
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_STATISTICAL_MARKET_CONTEXT_ARTIFACT
    return _coerce_artifact(payload)


def clear_statistical_market_context_artifact_cache() -> None:
    load_statistical_market_context_artifact.cache_clear()


def _lookup_numeric_prior(artifact, feature, value):
    value = _safe_float(value, None)
    priors = (artifact.get("numeric_bucket_priors") or {}).get(feature)
    if value is None or not isinstance(priors, list) or not priors:
        return None
    sorted_priors = sorted(
        [row for row in priors if isinstance(row, dict)],
        key=lambda row: (
            _safe_float(row.get("lower"), float("-inf")),
            _safe_float(row.get("upper"), float("inf")),
        ),
    )
    if not sorted_priors:
        return None
    rank_labels = ["low", "q2", "q3", "q4", "high"]
    for idx, row in enumerate(sorted_priors):
        lower = _safe_float(row.get("lower"), None)
        upper = _safe_float(row.get("upper"), None)
        lower_ok = lower is None or value >= lower
        upper_ok = upper is None or value <= upper
        if lower_ok and upper_ok:
            out = dict(row)
            out["feature"] = feature
            out["value"] = round(value, 4)
            out["rank_label"] = rank_labels[min(idx, len(rank_labels) - 1)]
            return out
    if value < _safe_float(sorted_priors[0].get("lower"), value):
        out = dict(sorted_priors[0])
        out["feature"] = feature
        out["value"] = round(value, 4)
        out["rank_label"] = "low"
        out["outside_range"] = "below_artifact_min"
        return out
    out = dict(sorted_priors[-1])
    out["feature"] = feature
    out["value"] = round(value, 4)
    out["rank_label"] = "high"
    out["outside_range"] = "above_artifact_max"
    return out


def _bucket_from_numeric_prior(artifact, feature, value):
    row = _lookup_numeric_prior(artifact, feature, value)
    return row.get("rank_label") if isinstance(row, dict) else None


def _expiry_bucket(days_to_expiry):
    dte = _safe_float(days_to_expiry, None)
    if dte is None:
        return None
    if dte <= 1:
        return "0-1d"
    if dte <= 3:
        return "2-3d"
    if dte <= 7:
        return "4-7d"
    if dte <= 14:
        return "8-14d"
    return "15d+"


def _trend_20d_bucket(ret_20d_bps):
    ret = _safe_float(ret_20d_bps, None)
    if ret is None:
        return None
    if ret <= -500:
        return "selloff"
    if ret <= -150:
        return "weak"
    if ret <= 150:
        return "flat"
    if ret <= 500:
        return "strong"
    return "surge"


def _weekday_label(weekday=None, valuation_time=None):
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    value = _safe_float(weekday, None)
    if value is None and valuation_time is not None:
        try:
            import pandas as _pd

            value = _pd.Timestamp(valuation_time).weekday()
        except Exception:
            value = None
    if value is None:
        return None
    index = int(value)
    if index < 0 or index >= len(names):
        return None
    return names[index]


def _extract_ret_20d_bps(market_state):
    market_state = market_state if isinstance(market_state, dict) else {}
    ta_features = market_state.get("ta_features") if isinstance(market_state.get("ta_features"), dict) else {}
    indicators = ta_features.get("indicators") if isinstance(ta_features.get("indicators"), dict) else {}
    for value in (
        indicators.get("ret_20d_bps"),
        ta_features.get("ret_20d_bps"),
        market_state.get("ret_20d_bps"),
        market_state.get("nifty_ret_20d_bps"),
    ):
        parsed = _safe_float(value, None)
        if parsed is not None:
            return parsed
    return None


def _normalize_realized_vol(value):
    parsed = _safe_float(value, None)
    if parsed is None:
        return None
    return parsed / 100.0 if parsed > 2.0 else parsed


def _distance_pct(level, spot):
    level = _safe_float(level, None)
    spot = _safe_float(spot, None)
    if level is None or spot in (None, 0.0):
        return None
    return abs((level - spot) / spot) * 100.0


def _extract_feature_values(*, market_state, global_risk_features, spot):
    market_state = market_state if isinstance(market_state, dict) else {}
    global_risk_features = global_risk_features if isinstance(global_risk_features, dict) else {}
    support = _safe_float(market_state.get("support_wall"), None)
    resistance = _safe_float(market_state.get("resistance_wall"), None)
    spot_value = _safe_float(spot, None)
    max_pain_dist = _safe_float(market_state.get("max_pain_dist"), None)
    wall_width = None
    if support is not None and resistance is not None and spot_value not in (None, 0.0):
        wall_width = abs(resistance - support) / spot_value * 100.0
    realized_vol = _normalize_realized_vol(
        market_state.get("realized_vol_20d", market_state.get("realized_hv_pct"))
    )
    atm_straddle_pct = _safe_float(market_state.get("atm_straddle_pct"), None)
    if atm_straddle_pct is None:
        atm_straddle_pct = _safe_float(market_state.get("expected_move_pct"), None)
    return {
        "india_vix_level": _safe_float(global_risk_features.get("india_vix_level"), None)
        or _safe_float(market_state.get("atm_iv"), None),
        "india_vix_change_24h": _safe_float(global_risk_features.get("india_vix_change_24h"), None),
        "realized_vol_20d": realized_vol,
        "pcr_oi": _safe_float(market_state.get("open_interest_pcr", market_state.get("oi_pcr")), None),
        "pcr_volume": _safe_float(market_state.get("volume_pcr"), None),
        "near_atm_pcr_oi": _safe_float(market_state.get("near_atm_pcr_oi"), None),
        "near_atm_pcr_volume": _safe_float(
            market_state.get("near_atm_pcr_volume", market_state.get("volume_pcr_atm")),
            None,
        ),
        "atm_straddle_pct": atm_straddle_pct,
        "max_pain_abs_dist_pct": abs(max_pain_dist / spot_value) * 100.0 if max_pain_dist is not None and spot_value not in (None, 0.0) else None,
        "wall_width_pct": wall_width,
        "ret_20d_bps": _extract_ret_20d_bps(market_state),
        "oil_change_24h": _safe_float(global_risk_features.get("oil_change_24h"), None),
        "gold_change_24h": _safe_float(global_risk_features.get("gold_change_24h"), None),
        "copper_change_24h": _safe_float(global_risk_features.get("copper_change_24h"), None),
        "vix_change_24h": _safe_float(
            global_risk_features.get("vix_change_24h", global_risk_features.get("us_vix_change_24h")),
            None,
        ),
        "usdinr_change_24h": _safe_float(global_risk_features.get("usdinr_change_24h"), None),
        "sp500_change_24h": _safe_float(global_risk_features.get("sp500_change_24h"), None),
        "nasdaq_change_24h": _safe_float(global_risk_features.get("nasdaq_change_24h"), None),
        "us10y_change_bp": _safe_float(global_risk_features.get("us10y_change_bp"), None),
    }


def _categorical_prior(artifact, feature, bucket):
    if bucket in (None, ""):
        return None
    priors = (artifact.get("categorical_bucket_priors") or {}).get(feature)
    if not isinstance(priors, dict):
        return None
    row = priors.get(str(bucket))
    if not isinstance(row, dict):
        return None
    out = dict(row)
    out["feature"] = feature
    out["bucket"] = str(bucket)
    return out


def _bias_from_prior(row, rules):
    mean_bps = _safe_float(row.get("mean_return_bps"), 0.0)
    hit = _safe_float(row.get("hit_positive"), 0.5)
    mean_edge = _safe_float(rules.get("directional_mean_edge_bps"), 8.0)
    hit_edge = _safe_float(rules.get("directional_hit_edge"), 0.56)
    conflict_hit_edge = _safe_float(rules.get("conflict_hit_edge"), 0.46)
    if mean_bps >= mean_edge or hit >= hit_edge:
        return "CALL"
    if mean_bps <= -mean_edge or hit <= conflict_hit_edge:
        return "PUT"
    return "NEUTRAL"


def _range_state(delta_bps, rules):
    delta = _safe_float(delta_bps, None)
    if delta is None:
        return "UNAVAILABLE"
    high = _safe_float(rules.get("high_abs_move_delta_bps"), 50.0)
    expanded = _safe_float(rules.get("expanded_abs_move_delta_bps"), 20.0)
    compressed = _safe_float(rules.get("compressed_abs_move_delta_bps"), -20.0)
    if delta >= high:
        return "EXPANDED_TAIL_RISK"
    if delta >= expanded:
        return "EXPANDED"
    if delta <= compressed:
        return "COMPRESSED"
    return "NORMAL"


def _range_severity(state):
    return {
        "UNAVAILABLE": 0,
        "COMPRESSED": 1,
        "NORMAL": 2,
        "EXPANDED": 3,
        "EXPANDED_TAIL_RISK": 4,
    }.get(str(state or "").upper(), 0)


def _vol_stress_score(delta_bps):
    delta = _safe_float(delta_bps, 0.0)
    return int(max(0, min(100, round(50 + delta * 0.5))))


def _select_primary_range_prior(matches):
    for feature in ["india_vix_level", "realized_vol_20d", "atm_straddle_pct", "pcr_oi"]:
        row = matches.get(feature)
        if isinstance(row, dict) and row.get("expected_range_bps") is not None:
            return row
    rows = [row for row in matches.values() if isinstance(row, dict) and row.get("expected_range_bps") is not None]
    if not rows:
        return None
    return max(rows, key=lambda row: _safe_float(row.get("n"), 0.0))


def _macro_application_rules(rules):
    return {
        "high_abs_move_delta_bps": _safe_float(
            rules.get("macro_high_abs_move_delta_bps", rules.get("high_abs_move_delta_bps")),
            70.0,
        ),
        "expanded_abs_move_delta_bps": _safe_float(
            rules.get("macro_expanded_abs_move_delta_bps", rules.get("expanded_abs_move_delta_bps")),
            35.0,
        ),
        "compressed_abs_move_delta_bps": _safe_float(
            rules.get("macro_compressed_abs_move_delta_bps", rules.get("compressed_abs_move_delta_bps")),
            -20.0,
        ),
        "directional_mean_edge_bps": _safe_float(
            rules.get("macro_directional_mean_edge_bps", rules.get("directional_mean_edge_bps")),
            10.0,
        ),
        "directional_hit_edge": _safe_float(
            rules.get("macro_directional_hit_edge", rules.get("directional_hit_edge")),
            0.57,
        ),
        "conflict_hit_edge": _safe_float(
            rules.get("macro_conflict_hit_edge", rules.get("conflict_hit_edge")),
            0.46,
        ),
    }


def _lookup_macro_shock_prior(macro_context, feature, value):
    value = _safe_float(value, None)
    priors = (macro_context.get("shock_priors") or {}).get(feature)
    if value is None or not isinstance(priors, dict):
        return None
    bottom = priors.get("bottom_decile") if isinstance(priors.get("bottom_decile"), dict) else None
    top = priors.get("top_decile") if isinstance(priors.get("top_decile"), dict) else None
    middle = priors.get("middle_80") if isinstance(priors.get("middle_80"), dict) else None

    row = None
    bucket = None
    bottom_threshold = _safe_float((bottom or {}).get("threshold"), None)
    top_threshold = _safe_float((top or {}).get("threshold"), None)
    if bottom and bottom_threshold is not None and value <= bottom_threshold:
        row = bottom
        bucket = "bottom_decile"
    elif top and top_threshold is not None and value >= top_threshold:
        row = top
        bucket = "top_decile"
    elif middle:
        row = middle
        bucket = "middle_80"
    if not row:
        return None
    out = dict(row)
    out["feature"] = feature
    out["bucket"] = bucket
    out["value"] = round(value, 4)
    return out


def _macro_bucket_state(shock_state):
    def _bucket(feature):
        row = shock_state.get(feature)
        return row.get("bucket") if isinstance(row, dict) else None

    risk_off = (
        _bucket("sp500_change_24h") == "bottom_decile"
        or _bucket("nasdaq_change_24h") == "bottom_decile"
        or _bucket("vix_change_24h") == "top_decile"
        or _bucket("india_vix_change_24h") == "top_decile"
    )
    risk_on = (
        _bucket("sp500_change_24h") == "top_decile"
        or _bucket("nasdaq_change_24h") == "top_decile"
        or _bucket("vix_change_24h") == "bottom_decile"
        or _bucket("india_vix_change_24h") == "bottom_decile"
    )
    if risk_off:
        macro_risk_bucket = "risk_off"
    elif risk_on:
        macro_risk_bucket = "risk_on"
    else:
        macro_risk_bucket = "neutral"

    commodity_down = _bucket("oil_change_24h") == "bottom_decile" or _bucket("copper_change_24h") == "bottom_decile"
    commodity_up = _bucket("oil_change_24h") == "top_decile" or _bucket("copper_change_24h") == "top_decile"
    if commodity_down:
        macro_commodity_bucket = "commodity_down"
    elif commodity_up:
        macro_commodity_bucket = "commodity_up"
    else:
        macro_commodity_bucket = "neutral"

    fx_rates_stress = _bucket("usdinr_change_24h") == "top_decile" or _bucket("us10y_change_bp") == "top_decile"
    fx_rates_easing = _bucket("usdinr_change_24h") == "bottom_decile" or _bucket("us10y_change_bp") == "bottom_decile"
    if fx_rates_stress:
        macro_fx_rates_bucket = "stress"
    elif fx_rates_easing:
        macro_fx_rates_bucket = "easing"
    else:
        macro_fx_rates_bucket = "neutral"

    return {
        "macro_risk_bucket": macro_risk_bucket,
        "macro_commodity_bucket": macro_commodity_bucket,
        "macro_fx_rates_bucket": macro_fx_rates_bucket,
    }


def _select_macro_range_row(rows):
    eligible = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("expected_range_bps") is not None
        and row.get("abs_move_delta_vs_base_bps") is not None
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            _safe_float(row.get("abs_move_delta_vs_base_bps"), -9999.0),
            _safe_float(row.get("n"), 0.0),
        ),
    )


def _select_macro_directional_row(rows, macro_rules):
    directional = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bias = _bias_from_prior(row, macro_rules)
        if bias == "NEUTRAL":
            continue
        item = dict(row)
        item["directional_prior"] = bias
        directional.append(item)
    if not directional:
        return None
    return max(
        directional,
        key=lambda row: (
            abs(_safe_float(row.get("mean_return_bps"), 0.0)),
            abs(_safe_float(row.get("hit_positive"), 0.5) - 0.5),
            _safe_float(row.get("n"), 0.0),
        ),
    )


def _build_macro_context(*, artifact, rules, features, categorical_state, direction_upper, min_bucket_n):
    macro_source = artifact.get("macro_context") if isinstance(artifact.get("macro_context"), dict) else {}
    macro_rules = _macro_application_rules(rules)
    shock_matches = {}
    for feature in MACRO_SHOCK_FEATURES:
        row = _lookup_macro_shock_prior(macro_source, feature, features.get(feature))
        if row and int(_safe_float(row.get("n"), 0.0)) >= min_bucket_n:
            shock_matches[feature] = row
    active_shock_matches = {
        feature: row
        for feature, row in shock_matches.items()
        if isinstance(row, dict) and row.get("bucket") != "middle_80"
    }

    macro_buckets = _macro_bucket_state(shock_matches)
    state_map = {}
    state_map.update(categorical_state)
    state_map.update(macro_buckets)

    interaction_matches = {}
    interaction_priors = macro_source.get("interaction_priors") if isinstance(macro_source.get("interaction_priors"), dict) else {}
    for key, row in interaction_priors.items():
        if not isinstance(row, dict) or int(_safe_float(row.get("n"), 0.0)) < min_bucket_n:
            continue
        left_feature = str(row.get("left_feature") or "")
        right_feature = str(row.get("right_feature") or "")
        left_bucket = str(row.get("left_bucket") or "")
        right_bucket = str(row.get("right_bucket") or "")
        if str(state_map.get(left_feature)) == left_bucket and str(state_map.get(right_feature)) == right_bucket:
            out = dict(row)
            out["interaction"] = out.get("interaction") or key
            interaction_matches[str(key)] = out

    all_rows = list(active_shock_matches.values()) + list(interaction_matches.values())
    range_row = _select_macro_range_row(all_rows)
    directional_row = _select_macro_directional_row(all_rows, macro_rules)

    macro_range_prior = _range_state(
        range_row.get("abs_move_delta_vs_base_bps") if isinstance(range_row, dict) else None,
        macro_rules,
    )
    macro_directional_prior = (
        directional_row.get("directional_prior") if isinstance(directional_row, dict) else "NEUTRAL"
    )
    directional_basis = None
    if isinstance(directional_row, dict):
        directional_basis = directional_row.get("feature") or directional_row.get("interaction")

    score_adjustment = 0
    probability_adjustment = 0.0
    threshold_adjustment = 0
    composite_threshold_adjustment = 0
    size_multiplier = 1.0
    reasons = []

    if macro_range_prior == "EXPANDED_TAIL_RISK":
        threshold_adjustment -= 1
        size_multiplier = min(size_multiplier, _safe_float(rules.get("macro_tail_risk_size_cap"), 0.85))
        reasons.append("macro_statistical_range_expanded_tail_risk")
    elif macro_range_prior == "EXPANDED":
        size_multiplier = min(size_multiplier, _safe_float(rules.get("macro_elevated_risk_size_cap"), 0.90))
        reasons.append("macro_statistical_range_expanded")
    elif macro_range_prior == "COMPRESSED":
        score_adjustment -= 1
        probability_adjustment -= 0.005
        threshold_adjustment += 1
        reasons.append("macro_statistical_range_compressed")

    if directional_row and direction_upper:
        edge_strength = 1
        mean_delta = abs(_safe_float(directional_row.get("mean_return_bps"), 0.0))
        hit_delta = abs(_safe_float(directional_row.get("hit_positive"), 0.5) - 0.5)
        if mean_delta >= 25 or hit_delta >= 0.12:
            edge_strength = 2
        if direction_upper == macro_directional_prior:
            score_adjustment += edge_strength
            probability_adjustment += min(0.015, 0.006 * edge_strength)
            reasons.append(f"macro_statistical_directional_prior_aligned_{directional_basis}")
        else:
            score_adjustment -= edge_strength + 1
            probability_adjustment -= min(0.020, 0.008 * edge_strength)
            threshold_adjustment += 1
            composite_threshold_adjustment += 1 if edge_strength >= 2 else 0
            size_multiplier = min(size_multiplier, _safe_float(rules.get("macro_conflict_size_cap"), 0.85))
            reasons.append(f"macro_statistical_directional_prior_conflicts_{directional_basis}")
    elif macro_directional_prior != "NEUTRAL":
        reasons.append(f"macro_statistical_directional_prior_{macro_directional_prior.lower()}_{directional_basis}")

    max_score = int(_safe_float(rules.get("macro_max_score_adjustment"), 3.0))
    max_probability = _safe_float(rules.get("macro_max_probability_adjustment"), 0.02)
    score_adjustment = int(round(max(-max_score, min(max_score, score_adjustment))))
    probability_adjustment = round(max(-max_probability, min(max_probability, probability_adjustment)), 4)
    threshold_adjustment = int(round(max(-1, min(2, threshold_adjustment))))
    composite_threshold_adjustment = int(round(max(0, min(1, composite_threshold_adjustment))))
    size_multiplier = round(max(0.75, min(1.0, size_multiplier)), 4)

    matched_count = len(active_shock_matches) + len(interaction_matches)
    return {
        "applied": bool(matched_count),
        "feature_values": {
            feature: _round(features.get(feature), 4)
            for feature in MACRO_SHOCK_FEATURES
            if features.get(feature) is not None
        },
        "shock_bucket_state": {
            feature: row.get("bucket")
            for feature, row in shock_matches.items()
            if isinstance(row, dict)
        },
        "macro_factor_buckets": macro_buckets,
        "matched_shocks": shock_matches,
        "matched_interactions": interaction_matches,
        "matched_count": matched_count,
        "macro_range_prior": macro_range_prior,
        "macro_range_basis": (range_row.get("feature") or range_row.get("interaction")) if isinstance(range_row, dict) else None,
        "macro_directional_prior": macro_directional_prior,
        "macro_directional_basis": directional_basis,
        "expected_range_bps": _round(range_row.get("expected_range_bps"), 4) if isinstance(range_row, dict) else None,
        "expected_abs_move_bps": _round(range_row.get("expected_abs_move_bps"), 4) if isinstance(range_row, dict) else None,
        "abs_move_delta_vs_base_bps": _round(range_row.get("abs_move_delta_vs_base_bps"), 4) if isinstance(range_row, dict) else None,
        "score_adjustment": score_adjustment,
        "probability_adjustment": probability_adjustment,
        "trade_strength_threshold_adjustment": threshold_adjustment,
        "composite_threshold_adjustment": composite_threshold_adjustment,
        "size_multiplier": size_multiplier,
        "reasons": reasons,
    }


def build_statistical_market_context(
    *,
    spot=None,
    market_state=None,
    global_risk_state=None,
    direction=None,
    valuation_time=None,
    weekday=None,
    artifact=None,
):
    """Return statistical market-context modifiers from the compact artifact."""

    market_state = market_state if isinstance(market_state, dict) else {}
    global_risk_state = global_risk_state if isinstance(global_risk_state, dict) else {}
    global_risk_features = global_risk_state.get("global_risk_features")
    global_risk_features = global_risk_features if isinstance(global_risk_features, dict) else {}
    artifact = artifact if isinstance(artifact, dict) else load_statistical_market_context_artifact()
    rules = artifact.get("application_rules") if isinstance(artifact.get("application_rules"), dict) else {}
    min_bucket_n = int(_safe_float(rules.get("min_bucket_n"), 50.0))
    features = _extract_feature_values(market_state=market_state, global_risk_features=global_risk_features, spot=spot)

    numeric_matches = {}
    for feature, value in features.items():
        row = _lookup_numeric_prior(artifact, feature, value)
        if row and int(_safe_float(row.get("n"), 0.0)) >= min_bucket_n:
            numeric_matches[feature] = row

    india_vix_bucket = _bucket_from_numeric_prior(artifact, "india_vix_level", features.get("india_vix_level"))
    pcr_oi_bucket = _bucket_from_numeric_prior(artifact, "pcr_oi", features.get("pcr_oi"))
    trend_bucket = _trend_20d_bucket(features.get("ret_20d_bps"))
    expiry_bucket = _expiry_bucket(market_state.get("days_to_expiry"))
    weekday_name = _weekday_label(weekday=weekday, valuation_time=valuation_time)
    categorical_state = {
        "trend_20d_bucket": trend_bucket,
        "expiry_bucket": expiry_bucket,
        "india_vix_bucket": india_vix_bucket,
        "pcr_oi_bucket": pcr_oi_bucket,
        "weekday": weekday_name,
        "macro_major_event": str(int(_safe_float(market_state.get("macro_major_event"), 0.0))),
        "near_call_wall": str(_distance_pct(market_state.get("resistance_wall"), spot) is not None and _distance_pct(market_state.get("resistance_wall"), spot) <= 0.35),
        "near_put_wall": str(_distance_pct(market_state.get("support_wall"), spot) is not None and _distance_pct(market_state.get("support_wall"), spot) <= 0.35),
    }
    categorical_matches = {}
    for feature, bucket in categorical_state.items():
        row = _categorical_prior(artifact, feature, bucket)
        if row and int(_safe_float(row.get("n"), 0.0)) >= min_bucket_n:
            categorical_matches[feature] = row

    primary_range_prior = _select_primary_range_prior(numeric_matches)
    expected_range_bps = _round(primary_range_prior.get("expected_range_bps"), 4) if primary_range_prior else None
    expected_abs_move_bps = _round(primary_range_prior.get("expected_abs_move_bps"), 4) if primary_range_prior else None
    abs_move_delta_bps = _round(primary_range_prior.get("abs_move_delta_vs_base_bps"), 4) if primary_range_prior else None
    range_expansion_prior = _range_state(abs_move_delta_bps, rules)
    expected_range_basis = primary_range_prior.get("feature") if isinstance(primary_range_prior, dict) else None

    score_adjustment = 0
    probability_adjustment = 0.0
    threshold_adjustment = 0
    composite_threshold_adjustment = 0
    size_multiplier = 1.0
    reasons = []
    direction_upper = str(direction or "").upper().strip()
    if direction_upper not in {"CALL", "PUT"}:
        direction_upper = None

    if range_expansion_prior == "EXPANDED_TAIL_RISK":
        threshold_adjustment -= 1
        size_multiplier = min(size_multiplier, _safe_float(rules.get("tail_risk_size_cap"), 0.80))
        reasons.append("statistical_range_expanded_tail_risk")
    elif range_expansion_prior == "EXPANDED":
        threshold_adjustment -= 1
        size_multiplier = min(size_multiplier, _safe_float(rules.get("elevated_risk_size_cap"), 0.90))
        reasons.append("statistical_range_expanded")
    elif range_expansion_prior == "COMPRESSED":
        score_adjustment -= 1
        probability_adjustment -= 0.005
        threshold_adjustment += 1
        reasons.append("statistical_range_compressed")

    directional_prior = "NEUTRAL"
    directional_basis = None
    directional_row = None
    for feature in ["trend_20d_bucket", "pcr_oi_bucket", "expiry_bucket"]:
        row = categorical_matches.get(feature)
        if not row:
            continue
        bias = _bias_from_prior(row, rules)
        if bias != "NEUTRAL":
            directional_prior = bias
            directional_basis = feature
            directional_row = row
            break
    if directional_row and direction_upper:
        edge_strength = 1
        delta = abs(_safe_float(directional_row.get("mean_return_bps"), 0.0))
        if delta >= 15 or abs(_safe_float(directional_row.get("hit_positive"), 0.5) - 0.5) >= 0.10:
            edge_strength = 2
        if direction_upper == directional_prior:
            score_adjustment += edge_strength
            probability_adjustment += min(0.020, 0.0075 * edge_strength)
            if edge_strength >= 2:
                threshold_adjustment -= 1
            reasons.append(f"statistical_directional_prior_aligned_{directional_basis}")
        else:
            score_adjustment -= edge_strength + 1
            probability_adjustment -= min(0.025, 0.010 * edge_strength)
            threshold_adjustment += 1 + (1 if edge_strength >= 2 else 0)
            composite_threshold_adjustment += 1 if edge_strength >= 2 else 0
            size_multiplier = min(size_multiplier, 0.85)
            reasons.append(f"statistical_directional_prior_conflicts_{directional_basis}")
    elif directional_prior != "NEUTRAL":
        reasons.append(f"statistical_directional_prior_{directional_prior.lower()}_{directional_basis}")

    macro_context = _build_macro_context(
        artifact=artifact,
        rules=rules,
        features=features,
        categorical_state=categorical_state,
        direction_upper=direction_upper,
        min_bucket_n=min_bucket_n,
    )
    if macro_context.get("applied"):
        score_adjustment += int(_safe_float(macro_context.get("score_adjustment"), 0.0))
        probability_adjustment += _safe_float(macro_context.get("probability_adjustment"), 0.0)
        threshold_adjustment += int(_safe_float(macro_context.get("trade_strength_threshold_adjustment"), 0.0))
        composite_threshold_adjustment += int(_safe_float(macro_context.get("composite_threshold_adjustment"), 0.0))
        size_multiplier = min(size_multiplier, _safe_float(macro_context.get("size_multiplier"), 1.0))
        reasons.extend(macro_context.get("reasons") or [])
        if _range_severity(macro_context.get("macro_range_prior")) > _range_severity(range_expansion_prior):
            range_expansion_prior = macro_context.get("macro_range_prior")
            expected_range_bps = macro_context.get("expected_range_bps")
            expected_abs_move_bps = macro_context.get("expected_abs_move_bps")
            abs_move_delta_bps = macro_context.get("abs_move_delta_vs_base_bps")
            expected_range_basis = macro_context.get("macro_range_basis") or "macro_context"

    macro_rows = list((macro_context.get("matched_shocks") or {}).values()) + list(
        (macro_context.get("matched_interactions") or {}).values()
    )
    matched_count = len(numeric_matches) + len(categorical_matches) + int(_safe_float(macro_context.get("matched_count"), 0.0))
    total_n = sum(
        int(_safe_float(row.get("n"), 0.0))
        for row in list(numeric_matches.values()) + list(categorical_matches.values()) + macro_rows
    )
    if matched_count >= 3 and total_n >= 1000:
        regime_confidence = "HIGH"
    elif matched_count >= 2 and total_n >= 300:
        regime_confidence = "MEDIUM"
    elif matched_count >= 1:
        regime_confidence = "LOW"
    else:
        regime_confidence = "UNAVAILABLE"

    if range_expansion_prior == "EXPANDED_TAIL_RISK":
        hold_time_hint = "ALLOW_LONGER_BUT_SIZE_DOWN"
    elif range_expansion_prior == "EXPANDED":
        hold_time_hint = "ALLOW_STANDARD_TO_LONGER_HOLD"
    elif range_expansion_prior == "COMPRESSED":
        hold_time_hint = "SHORTEN_OR_REQUIRE_FAST_CONFIRMATION"
    else:
        hold_time_hint = "STANDARD"

    score_adjustment = int(round(max(-4, min(4, score_adjustment))))
    probability_adjustment = round(max(-0.030, min(0.030, probability_adjustment)), 4)
    threshold_adjustment = int(round(max(-2, min(3, threshold_adjustment))))
    composite_threshold_adjustment = int(round(max(-1, min(2, composite_threshold_adjustment))))
    size_multiplier = round(max(0.65, min(1.0, size_multiplier)), 4)

    applied = bool(matched_count)
    return {
        "version": STATISTICAL_MARKET_CONTEXT_VERSION,
        "decision_mode": STATISTICAL_MARKET_CONTEXT_DECISION_MODE,
        "artifact_version": artifact.get("artifact_version"),
        "source_run_id": artifact.get("source_run_id"),
        "source_report_pdf": artifact.get("source_report_pdf"),
        "applied": applied,
        "feature_values": {key: _round(value, 4) for key, value in features.items()},
        "bucket_state": categorical_state,
        "numeric_matches": numeric_matches,
        "categorical_matches": categorical_matches,
        "matched_count": matched_count,
        "regime_confidence": regime_confidence,
        "vol_stress_score": _vol_stress_score(abs_move_delta_bps),
        "expected_range_prior": range_expansion_prior,
        "expected_range_basis": expected_range_basis,
        "expected_range_bps": expected_range_bps,
        "expected_abs_move_bps": expected_abs_move_bps,
        "abs_move_delta_vs_base_bps": abs_move_delta_bps,
        "directional_followthrough_prior": directional_prior,
        "directional_basis": directional_basis,
        "hold_time_hint": hold_time_hint,
        "score_adjustment": score_adjustment,
        "probability_adjustment": probability_adjustment,
        "trade_strength_threshold_adjustment": threshold_adjustment,
        "composite_threshold_adjustment": composite_threshold_adjustment,
        "size_multiplier": size_multiplier,
        "macro_context": macro_context,
        "reasons": reasons,
    }
