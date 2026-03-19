import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.gamma_exposure import calculate_gamma_exposure
from engine.trading_support.probability import _blend_move_probability
from risk.global_risk_features import build_global_risk_features
from strategy.strike_selector import rank_strike_candidates


def run_checks() -> dict:
    results = {}

    rows = []
    for s in range(21000, 24001, 100):
        rows.append(
            {
                "OPTION_TYP": "CE",
                "strikePrice": s,
                "lastPrice": 100,
                "openInterest": 1000,
                "totalTradedVolume": 1000,
                "impliedVolatility": 15,
            }
        )
    df = pd.DataFrame(rows)

    low = rank_strike_candidates(df, "CALL", spot=22500, volatility_shock_score=0.1, top_n=500)
    mid = rank_strike_candidates(df, "CALL", spot=22500, volatility_shock_score=0.5, top_n=500)
    high = rank_strike_candidates(df, "CALL", spot=22500, volatility_shock_score=1.0, top_n=500)
    override = rank_strike_candidates(
        df,
        "CALL",
        spot=22500,
        volatility_shock_score=0.1,
        strike_window_steps=9,
        top_n=500,
    )

    results["strike_selector"] = {
        "low_count": len(low),
        "mid_count": len(mid),
        "high_count": len(high),
        "override_count": len(override),
        "assert_monotonic_expansion": len(low) <= len(mid) <= len(high),
        "assert_override_not_tightened": len(override) >= len(low),
    }

    raw_schema = pd.DataFrame(
        {
            "strikePrice": [100, 100, 110, 110],
            "openInterest": [10, 8, 5, 7],
            "OPTION_TYP": ["CE", "PE", "CE", "PE"],
        }
    )
    alt_schema = pd.DataFrame(
        {
            "STRIKE_PR": [100, 100, 110, 110],
            "OPEN_INT": [10, 8, 5, 7],
            "OPTION_TYP": ["CE", "PE", "CE", "PE"],
        }
    )
    gex_raw = calculate_gamma_exposure(raw_schema, spot=105)
    gex_alt = calculate_gamma_exposure(alt_schema, spot=105)

    results["gamma_exposure"] = {
        "gex_raw": gex_raw,
        "gex_alt": gex_alt,
        "assert_schema_parity": abs(gex_raw - gex_alt) < 1e-12,
    }

    snap_missing = {
        "market_inputs": {},
        "data_available": False,
        "stale": False,
        "neutral_fallback": True,
    }
    risk_missing = build_global_risk_features(
        global_market_snapshot=snap_missing,
        macro_event_state={},
        macro_news_state={},
        holding_profile="AUTO",
    )
    results["global_risk_fallback"] = {
        "market_features_neutralized": risk_missing.get("market_features_neutralized"),
        "market_neutralization_reason": risk_missing.get("market_neutralization_reason"),
        "assert_neutralized_when_unusable": bool(risk_missing.get("market_features_neutralized")),
    }

    probs = [
        _blend_move_probability(0.0, 0.0),
        _blend_move_probability(1.0, 1.0),
        _blend_move_probability(0.5, 0.5),
        _blend_move_probability(0.2, None),
    ]
    results["blend_probability"] = {
        "samples": probs,
        "assert_all_bounded": all(0.0 <= p <= 1.0 for p in probs),
    }

    all_asserts = []
    for section in results.values():
        for k, v in section.items():
            if k.startswith("assert_"):
                all_asserts.append(bool(v))
    results["all_asserts_passed"] = all(all_asserts)

    return results


def main() -> None:
    out_dir = Path("research/runtime_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "runtime_regression_checks_20260319.json"

    results = run_checks()
    out_file.write_text(json.dumps(results, indent=2))

    print(f"artifact={out_file}")
    print(f"all_asserts_passed={results['all_asserts_passed']}")


if __name__ == "__main__":
    main()
