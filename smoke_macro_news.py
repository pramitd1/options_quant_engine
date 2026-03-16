"""
Module: smoke_macro_news.py

Purpose:
    Provide a smoke-test entry point for macro-event and news-state integrations.

Role in the System:
    Part of the repository entry layer that wires top-level workflows into the trading system.

Key Outputs:
    CLI/runtime side effects, saved snapshots, and entry-point orchestration.

Downstream Usage:
    Consumed by operators and indirectly by runtime, replay, and capture workflows.
"""

from __future__ import annotations

import argparse

from macro.macro_news_aggregator import build_macro_news_state
from macro.scheduled_event_risk import evaluate_scheduled_event_risk
from news.service import build_default_headline_service


def parse_args():
    """
    Purpose:
        Parse command-line arguments for the current entry point.

    Context:
        Function inside the `smoke macro news` module. The module sits in the repository entry layer that wires top-level scripts into the trading workflow.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Parsed argument namespace.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--as-of", default=None)
    return parser.parse_args()


def main():
    """
    Purpose:
        Run the module entry point for command-line or operational execution.

    Context:
        Function inside the `smoke macro news` module. The module sits in the repository entry layer that wires top-level scripts into the trading workflow.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        Any: Exit status or workflow result returned by the implementation.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    args = parse_args()
    headline_service = build_default_headline_service()

    event_state = evaluate_scheduled_event_risk(
        symbol=args.symbol,
        as_of=args.as_of,
    )
    headline_state = headline_service.fetch(
        symbol=args.symbol,
        as_of=args.as_of,
    )
    macro_news_state = build_macro_news_state(
        event_state=event_state,
        headline_state=headline_state,
        as_of=args.as_of,
    )

    print("MACRO EVENT STATE")
    for key, value in event_state.items():
        print(f"{key:26}: {value}")

    print("\nHEADLINE INGESTION STATE")
    headline_dict = headline_state.to_dict()
    headline_summary = {
        "provider_name": headline_dict.get("provider_name"),
        "fetched_at": headline_dict.get("fetched_at"),
        "latest_headline_at": headline_dict.get("latest_headline_at"),
        "is_stale": headline_dict.get("is_stale"),
        "data_available": headline_dict.get("data_available"),
        "neutral_fallback": headline_dict.get("neutral_fallback"),
        "warnings": headline_dict.get("warnings"),
        "issues": headline_dict.get("issues"),
        "provider_metadata": headline_dict.get("provider_metadata"),
        "record_count": len(headline_state.records),
    }
    for key, value in headline_summary.items():
        print(f"{key:26}: {value}")

    print("\nMACRO / NEWS AGGREGATE")
    for key, value in macro_news_state.to_dict().items():
        print(f"{key:26}: {value}")


if __name__ == "__main__":
    main()
