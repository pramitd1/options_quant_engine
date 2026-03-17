#!/usr/bin/env python3
"""
Daily Signal Research Report — CLI
====================================

Generate professional signal research reports evaluating
the predictive quality of engine-generated signals.

Supports two modes:

* ``daily``      — analyse a single trading day (default)
* ``cumulative`` — analyse the full signals dataset (overwritten each run)

Use ``--both`` to generate both reports in one invocation.

Usage:
    python scripts/daily_research_report.py
    python scripts/daily_research_report.py --date 2026-03-16
    python scripts/daily_research_report.py --mode cumulative
    python scripts/daily_research_report.py --both
    python scripts/daily_research_report.py --both --narrative

Output defaults to documentation/daily_reports/ (excluded from git via .gitignore).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.daily_research_report import (
    generate_daily_report,
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal research reports (daily, cumulative, or both).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Report date in YYYY-MM-DD format. Defaults to the latest date in the dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to signals_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--narrative",
        action="store_true",
        default=False,
        help="Enrich sections with AI-generated commentary (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "cumulative"],
        default="daily",
        help="Report mode: 'daily' analyses a single day, 'cumulative' analyses the full dataset.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        default=False,
        help="Generate both daily and cumulative reports in one run.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        default=True,
        help="Also render PDF alongside the Markdown report (default: on).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        default=False,
        help="Skip PDF rendering.",
    )

    args = parser.parse_args()
    if args.no_pdf:
        args.pdf = False

    report_date: date | None = None
    if args.date:
        report_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    modes = ["daily", "cumulative"] if args.both else [args.mode]

    for m in modes:
        report_path = generate_daily_report(
            report_date=report_date,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            narrative=args.narrative,
            mode=m,
        )

        label = "Daily" if m == "daily" else "Cumulative"
        print(f"\n{label} Signal Research Report generated successfully.")
        print(f"  Path: {report_path}")
        print(f"  Date: {report_date or 'latest'}")
        print(f"  Mode: {m}")
        if args.narrative:
            from research.signal_evaluation.narrative_provider import get_provider_name
            provider = get_provider_name()
            print(f"  AI Narrative: {provider or 'no provider available (skipped)'}")

        if args.pdf:
            try:
                from scripts.render_pdf import render_markdown_to_pdf
                pdf_path = render_markdown_to_pdf(report_path)
                print(f"  PDF : {pdf_path}")
            except Exception as exc:
                print(f"  PDF rendering failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
