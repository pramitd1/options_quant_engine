"""
Scenario runner for the macro/news layer using local fixtures only.

Usage:
    python -m backtest.macro_news_scenario_runner
    python -m backtest.macro_news_scenario_runner --scenario risk_off_geopolitical_burst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import BASE_DIR, HEADLINE_STALE_MINUTES
from macro.engine_adjustments import compute_macro_news_adjustments
from macro.macro_news_aggregator import build_macro_news_state
from macro.scheduled_event_risk import evaluate_scheduled_event_risk
from news.models import HeadlineIngestionState, HeadlineRecord, coerce_headline_timestamp
from news.classifier import classify_headlines


def _load_scenarios(path: str | None = None):
    path = Path(path or (Path(BASE_DIR) / "config/macro_news_scenarios.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _build_headline_state(headlines, as_of, provider_name="SCENARIO"):
    records = []
    for raw in headlines or []:
        ts = coerce_headline_timestamp(raw.get("timestamp"))
        if ts is None:
            continue
        records.append(
            HeadlineRecord(
                timestamp=ts,
                source=str(raw.get("source", provider_name)),
                headline=str(raw.get("headline", "")).strip(),
                url_or_identifier=str(raw.get("url_or_identifier", raw.get("headline", ""))).strip(),
                category=raw.get("category"),
                raw_payload=dict(raw),
            )
        )

    as_of_ts = coerce_headline_timestamp(as_of)
    latest_ts = max((record.timestamp for record in records), default=None)
    is_stale = True
    if latest_ts is not None and as_of_ts is not None:
        age_minutes = max((as_of_ts - latest_ts).total_seconds() / 60.0, 0.0)
        is_stale = age_minutes > HEADLINE_STALE_MINUTES

    return HeadlineIngestionState(
        records=records,
        provider_name=provider_name,
        fetched_at=as_of_ts,
        latest_headline_at=latest_ts,
        is_stale=is_stale,
        data_available=bool(records) and not is_stale,
        neutral_fallback=(not records) or is_stale,
        stale_after_minutes=HEADLINE_STALE_MINUTES,
        provider_metadata={"scenario_record_count": len(records)},
        warnings=[f"headline_data_stale:{round(age_minutes, 2)}m"] if records and is_stale else [],
        issues=[],
    )


def run_scenario(scenario: dict):
    symbol = scenario["symbol"]
    as_of = scenario["as_of"]
    event_state = evaluate_scheduled_event_risk(
        symbol=symbol,
        as_of=as_of,
        events=scenario.get("events", []),
        enabled=True,
    )
    headline_state = _build_headline_state(scenario.get("headlines", []), as_of)
    macro_news_state = build_macro_news_state(
        event_state=event_state,
        headline_state=headline_state,
        as_of=as_of,
    ).to_dict()

    classifications = [item.to_dict() for item in classify_headlines(headline_state.records)]
    sample_adjustments = {
        "call": compute_macro_news_adjustments(direction="CALL", macro_news_state=macro_news_state),
        "put": compute_macro_news_adjustments(direction="PUT", macro_news_state=macro_news_state),
    }

    return {
        "name": scenario["name"],
        "symbol": symbol,
        "as_of": as_of,
        "event_state": event_state,
        "headline_state": headline_state.to_dict(),
        "classifications": classifications,
        "macro_news_state": macro_news_state,
        "sample_adjustments": sample_adjustments,
        "expected": scenario.get("expected", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, help="Optional scenario name from config/macro_news_scenarios.json")
    parser.add_argument("--scenario-file", default=None, help="Optional custom scenario file path")
    args = parser.parse_args()

    scenarios = _load_scenarios(args.scenario_file)
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario.get("name") == args.scenario]
        if not scenarios:
            raise ValueError(f"Scenario not found: {args.scenario}")

    for scenario in scenarios:
        result = run_scenario(scenario)
        print("\nSCENARIO")
        print("---------------------------")
        print(f"name                     : {result['name']}")
        print(f"symbol                   : {result['symbol']}")
        print(f"as_of                    : {result['as_of']}")
        print(f"macro_regime             : {result['macro_news_state']['macro_regime']}")
        print(f"event_window_status      : {result['event_state']['event_window_status']}")
        print(f"event_lockdown_flag      : {result['macro_news_state']['event_lockdown_flag']}")
        print(f"macro_sentiment_score    : {result['macro_news_state']['macro_sentiment_score']}")
        print(f"volatility_shock_score   : {result['macro_news_state']['volatility_shock_score']}")
        print(f"news_confidence_score    : {result['macro_news_state']['news_confidence_score']}")
        print(f"headline_velocity        : {result['macro_news_state']['headline_velocity']}")
        print(f"neutral_fallback         : {result['macro_news_state']['neutral_fallback']}")
        print(f"classified_count         : {len(result['classifications'])}")
        print(f"call_adjustment          : {result['sample_adjustments']['call']['macro_adjustment_score']}")
        print(f"put_adjustment           : {result['sample_adjustments']['put']['macro_adjustment_score']}")
        if result["expected"]:
            print(f"expected                 : {result['expected']}")


if __name__ == "__main__":
    main()
