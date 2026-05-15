#!/usr/bin/env python3
"""Run the read-only Monday readiness preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DEFAULT_DATA_SOURCE, DEFAULT_SYMBOL  # noqa: E402
from research.signal_evaluation.monday_readiness_preflight import (  # noqa: E402
    DEFAULT_GUARDED_STALENESS_PATH,
    DEFAULT_SHADOW_SOAK_PATH,
    PREFLIGHT_BLOCKED,
    build_monday_readiness_preflight_report,
)
from research.signal_evaluation.signal_quality_model_audit import (  # noqa: E402
    default_signal_quality_dataset_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a read-only Monday readiness checklist for the options signal engine. "
            "This command does not fetch providers, refresh outcomes, change data sources, "
            "alter parameter packs, or execute trades."
        )
    )
    parser.add_argument("--source", default=DEFAULT_DATA_SOURCE, help="User-selected option data source.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Underlying symbol for operator instructions.")
    parser.add_argument("--dataset", type=Path, default=default_signal_quality_dataset_path())
    parser.add_argument("--guarded-staleness-report", type=Path, default=DEFAULT_GUARDED_STALENESS_PATH)
    parser.add_argument("--shadow-soak-report", type=Path, default=DEFAULT_SHADOW_SOAK_PATH)
    parser.add_argument(
        "--option-chain",
        type=Path,
        default=None,
        help="Optional saved option-chain CSV/JSON/parquet snapshot to validate without fetching a provider.",
    )
    parser.add_argument("--spot", type=float, default=None, help="Optional spot used for ATM/core option checks.")
    parser.add_argument("--as-of", default=None, help="Optional timestamp for quote freshness/DTE checks.")
    parser.add_argument("--max-quote-age-seconds", type=float, default=None)
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit 2 when preflight_status is PREFLIGHT_BLOCKED.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_monday_readiness_preflight_report(
        source=args.source,
        symbol=args.symbol,
        dataset_path=args.dataset,
        guarded_staleness_path=args.guarded_staleness_report,
        shadow_soak_path=args.shadow_soak_report,
        option_chain_path=args.option_chain,
        spot=args.spot,
        as_of=args.as_of,
        max_quote_age_seconds=args.max_quote_age_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.fail_on_blocked and report.get("preflight_status") == PREFLIGHT_BLOCKED:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
