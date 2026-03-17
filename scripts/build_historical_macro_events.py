#!/usr/bin/env python3
"""
Historical India Macro Event Schedule Builder
==============================================

Builds a comprehensive historical schedule of major Indian macro events
for backtesting. Events are generated from known release patterns and
official calendar archives.

Event categories (with severity matching the engine's policy):

  CRITICAL  — RBI MPC Policy Decision, Union Budget, India GDP
  MAJOR     — India CPI (Consumer Price Index), India IIP (Industrial Production)
  MEDIUM    — India WPI (Wholesale Price Index), India PMI (Manufacturing & Services)
  MINOR     — India Trade Balance (Merchandise Exports/Imports)

Data sources:
  - RBI MPC dates: https://rbi.org.in/Scripts/MPC_Dates.aspx (historical archive)
  - CPI/IIP/WPI/GDP: MoSPI release calendar + known ~12th-of-month pattern
  - Budget: Fixed annual date (Feb 1, or Jul pre-2017)
  - PMI: S&P Global India PMI, typically 1st-3rd business day of month

Output:
  data_store/historical/macro_events/india_macro_events_historical.json

Usage:
    python scripts/build_historical_macro_events.py
    python scripts/build_historical_macro_events.py --from-year 2015
    python scripts/build_historical_macro_events.py --to-year 2025
    python scripts/build_historical_macro_events.py --validate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("macro_hist")

OUTPUT_DIR = _PROJECT_ROOT / "data_store" / "historical" / "macro_events"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "india_macro_events_historical.json"

IST = "+05:30"

# ===================================================================
#  Known RBI MPC meeting dates (announcement day)
#  Source: RBI official MPC schedule archives
#  https://rbi.org.in/Scripts/MPC_Dates.aspx
# ===================================================================
# RBI typically announces at 10:00 IST on the final day of the meeting.
# Format: (year, month, day) of the announcement day.
RBI_MPC_DATES: list[tuple[int, int, int]] = [
    # 2016 (MPC framework introduced Oct 2016)
    (2016, 10, 4),
    (2016, 12, 7),
    # 2017
    (2017, 2, 8),
    (2017, 4, 6),
    (2017, 6, 7),
    (2017, 8, 2),
    (2017, 10, 4),
    (2017, 12, 6),
    # 2018
    (2018, 2, 7),
    (2018, 4, 5),
    (2018, 6, 6),
    (2018, 8, 1),
    (2018, 10, 5),
    (2018, 12, 5),
    # 2019
    (2019, 2, 7),
    (2019, 4, 4),
    (2019, 6, 6),
    (2019, 8, 7),
    (2019, 10, 4),
    (2019, 12, 5),
    # 2020
    (2020, 2, 6),
    (2020, 3, 27),   # Emergency COVID cut
    (2020, 5, 22),   # Off-cycle COVID
    (2020, 8, 6),
    (2020, 10, 9),
    (2020, 12, 4),
    # 2021
    (2021, 2, 5),
    (2021, 4, 7),
    (2021, 5, 5),    # Off-cycle
    (2021, 6, 4),
    (2021, 8, 6),
    (2021, 10, 8),
    (2021, 12, 8),
    # 2022
    (2022, 2, 10),
    (2022, 4, 8),
    (2022, 5, 4),    # Off-cycle rate hike
    (2022, 6, 8),
    (2022, 8, 5),
    (2022, 9, 30),
    (2022, 12, 7),
    # 2023
    (2023, 2, 8),
    (2023, 4, 6),
    (2023, 6, 8),
    (2023, 8, 10),
    (2023, 10, 6),
    (2023, 12, 8),
    # 2024
    (2024, 2, 8),
    (2024, 4, 5),
    (2024, 6, 7),
    (2024, 8, 8),
    (2024, 10, 9),
    (2024, 12, 6),
    # 2025
    (2025, 2, 7),
    (2025, 4, 9),
    (2025, 6, 6),
    (2025, 8, 8),
    (2025, 10, 8),
    (2025, 12, 5),
    # 2026
    (2026, 2, 6),
]

# ===================================================================
#  Union Budget dates
#  Pre-2017: typically last working day of February
#  Post-2017: February 1 (fixed)
# ===================================================================
BUDGET_DATES: list[tuple[int, int, int]] = [
    (2012, 3, 16),   # 2012-13 budget
    (2013, 2, 28),
    (2014, 2, 17),   # Interim budget before election
    (2014, 7, 10),   # Full budget after new govt
    (2015, 2, 28),
    (2016, 2, 29),
    (2017, 2, 1),    # First Feb 1 budget
    (2018, 2, 1),
    (2019, 2, 1),    # Interim budget (election year)
    (2019, 7, 5),    # Full budget after election
    (2020, 2, 1),
    (2021, 2, 1),
    (2022, 2, 1),
    (2023, 2, 1),
    (2024, 2, 1),    # Interim budget (election year)
    (2024, 7, 23),   # Full budget after election
    (2025, 2, 1),
    (2026, 2, 1),
]

# ===================================================================
#  India GDP release dates (quarterly, ~2 months after quarter end)
#  Q1 (Apr-Jun) → Aug/Sep, Q2 (Jul-Sep) → Nov/Dec,
#  Q3 (Oct-Dec) → Feb/Mar, Q4 (Jan-Mar) → May/Jun
#  MoSPI typically releases around the last day of the 2nd month.
# ===================================================================
GDP_DATES: list[tuple[int, int, int, str]] = [
    # (year, month, day, quarter_label)
    # 2016
    (2016, 5, 31, "Q4 FY2015-16"), (2016, 8, 31, "Q1 FY2016-17"),
    (2016, 11, 30, "Q2 FY2016-17"),
    # 2017
    (2017, 2, 28, "Q3 FY2016-17"), (2017, 5, 31, "Q4 FY2016-17"),
    (2017, 8, 31, "Q1 FY2017-18"), (2017, 11, 30, "Q2 FY2017-18"),
    # 2018
    (2018, 2, 28, "Q3 FY2017-18"), (2018, 5, 31, "Q4 FY2017-18"),
    (2018, 8, 31, "Q1 FY2018-19"), (2018, 11, 30, "Q2 FY2018-19"),
    # 2019
    (2019, 2, 28, "Q3 FY2018-19"), (2019, 5, 31, "Q4 FY2018-19"),
    (2019, 8, 30, "Q1 FY2019-20"), (2019, 11, 29, "Q2 FY2019-20"),
    # 2020
    (2020, 2, 28, "Q3 FY2019-20"), (2020, 5, 29, "Q4 FY2019-20"),
    (2020, 8, 31, "Q1 FY2020-21"), (2020, 11, 27, "Q2 FY2020-21"),
    # 2021
    (2021, 2, 26, "Q3 FY2020-21"), (2021, 5, 31, "Q4 FY2020-21"),
    (2021, 8, 31, "Q1 FY2021-22"), (2021, 11, 30, "Q2 FY2021-22"),
    # 2022
    (2022, 2, 28, "Q3 FY2021-22"), (2022, 5, 31, "Q4 FY2021-22"),
    (2022, 8, 31, "Q1 FY2022-23"), (2022, 11, 30, "Q2 FY2022-23"),
    # 2023
    (2023, 2, 28, "Q3 FY2022-23"), (2023, 5, 31, "Q4 FY2022-23"),
    (2023, 8, 31, "Q1 FY2023-24"), (2023, 11, 30, "Q2 FY2023-24"),
    # 2024
    (2024, 2, 29, "Q3 FY2023-24"), (2024, 5, 31, "Q4 FY2023-24"),
    (2024, 8, 30, "Q1 FY2024-25"), (2024, 11, 29, "Q2 FY2024-25"),
    # 2025
    (2025, 2, 28, "Q3 FY2024-25"), (2025, 5, 30, "Q4 FY2024-25"),
    (2025, 8, 29, "Q1 FY2025-26"), (2025, 11, 28, "Q2 FY2025-26"),
    # 2026
    (2026, 2, 27, "Q3 FY2025-26"), (2026, 5, 29, "Q4 FY2025-26"),
]


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Saturday=5, Sunday=6


# Known NSE market holidays (major ones for adjustment; not exhaustive)
_KNOWN_HOLIDAYS: set[date] = {
    date(2020, 3, 10),   # Holi
    date(2021, 3, 29),   # Holi
    date(2022, 3, 18),   # Holi
    date(2023, 3, 7),    # Holi
    date(2024, 3, 25),   # Holi
    date(2025, 3, 14),   # Holi
    date(2026, 3, 3),    # Holi
    date(2020, 8, 15),   # Independence Day
    date(2021, 8, 19),   # Muharram
    date(2022, 8, 15),   # Independence Day
    date(2023, 8, 15),   # Independence Day
    date(2024, 8, 15),   # Independence Day
    date(2025, 8, 15),   # Independence Day
    date(2026, 8, 17),   # Independence Day (observed)
    date(2020, 10, 2),   # Gandhi Jayanti
    date(2021, 10, 15),  # Dussehra
    date(2022, 10, 5),   # Dussehra
    date(2023, 10, 2),   # Gandhi Jayanti
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2022, 1, 26),   # Republic Day
    date(2023, 1, 26),   # Republic Day
    date(2024, 1, 26),   # Republic Day
    date(2025, 1, 26),   # Republic Day (Sunday, observed Mon)
    date(2026, 1, 26),   # Republic Day
}


def _next_trading_day(d: date) -> date:
    """Advance to the next trading day (skip weekends and known holidays)."""
    while _is_weekend(d) or d in _KNOWN_HOLIDAYS:
        d += timedelta(days=1)
    return d


def _adjust_to_trading_day(d: date) -> tuple[date, str | None]:
    """Adjust a date to the nearest trading day. Returns (adjusted_date, note_if_moved)."""
    original = d
    adjusted = _next_trading_day(d)
    if adjusted != original:
        note = f"Originally {original.isoformat()} ({original.strftime('%A')}), shifted to next trading day"
        return adjusted, note
    return adjusted, None


def _make_event(
    name: str,
    d: date,
    severity: str,
    time_str: str = "17:30",
    scope: list[str] | None = None,
    source: str = "HISTORICAL_ARCHIVE",
    notes: str | None = None,
    **window_overrides,
) -> dict:
    """Build one event dict in the engine's expected format."""
    adj_date, adj_note = _adjust_to_trading_day(d)
    all_notes = []
    if notes:
        all_notes.append(notes)
    if adj_note:
        all_notes.append(adj_note)

    # Default windows by severity
    windows = {
        "CRITICAL": {"warning_minutes": 240, "lockdown_minutes": 45, "event_duration_minutes": 15, "cooldown_minutes": 45},
        "MAJOR": {"warning_minutes": 180, "lockdown_minutes": 30, "event_duration_minutes": 10, "cooldown_minutes": 30},
        "MEDIUM": {"warning_minutes": 120, "lockdown_minutes": 20, "event_duration_minutes": 10, "cooldown_minutes": 20},
        "MINOR": {"warning_minutes": 60, "lockdown_minutes": 15, "event_duration_minutes": 5, "cooldown_minutes": 15},
    }
    win = {**windows.get(severity, windows["MEDIUM"]), **window_overrides}

    event = {
        "name": name,
        "timestamp": f"{adj_date.isoformat()}T{time_str}:00{IST}",
        "severity": severity,
        "scope": scope or ["ALL"],
        **win,
        "source": source,
    }
    if all_notes:
        event["notes"] = "; ".join(all_notes)
    return event


# ===================================================================
#  Monthly data release generators
# ===================================================================

def _generate_cpi_events(from_year: int, to_year: int) -> list[dict]:
    """
    India CPI (Consumer Price Index).
    Released by MoSPI around the 12th of every month for the prior month.
    Release time: ~17:30 IST (after market close, though sometimes intraday).
    """
    events = []
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            if year == to_year and month > 6:
                break
            d = date(year, month, 12)
            ref_month = month - 1 if month > 1 else 12
            ref_year = year if month > 1 else year - 1
            ref_label = date(ref_year, ref_month, 1).strftime("%b %Y")
            events.append(_make_event(
                name=f"India CPI ({ref_label})",
                d=d,
                severity="MAJOR",
                time_str="17:30",
                source="MOSPI_HISTORICAL",
            ))
    return events


def _generate_iip_events(from_year: int, to_year: int) -> list[dict]:
    """
    India IIP (Index of Industrial Production).
    Released by MoSPI around the 12th of every month (often with CPI).
    """
    events = []
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            if year == to_year and month > 6:
                break
            d = date(year, month, 12)
            # IIP lags by ~2 months
            ref_month = ((month - 3) % 12) + 1
            ref_year = year if month > 2 else year - 1
            ref_label = date(ref_year, ref_month, 1).strftime("%b %Y")
            events.append(_make_event(
                name=f"India IIP ({ref_label})",
                d=d,
                severity="MAJOR",
                time_str="17:30",
                source="MOSPI_HISTORICAL",
            ))
    return events


def _generate_wpi_events(from_year: int, to_year: int) -> list[dict]:
    """
    India WPI (Wholesale Price Index).
    Released by DPIIT around the 14th of every month.
    """
    events = []
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            if year == to_year and month > 6:
                break
            d = date(year, month, 14)
            ref_month = month - 1 if month > 1 else 12
            ref_year = year if month > 1 else year - 1
            ref_label = date(ref_year, ref_month, 1).strftime("%b %Y")
            events.append(_make_event(
                name=f"India WPI ({ref_label})",
                d=d,
                severity="MEDIUM",
                time_str="17:30",
                source="DPIIT_HISTORICAL",
            ))
    return events


def _generate_pmi_events(from_year: int, to_year: int) -> list[dict]:
    """
    India PMI (Manufacturing & Services Composite).
    S&P Global releases Manufacturing PMI on 1st business day,
    Services PMI on 3rd business day of each month.
    We track Manufacturing PMI as the primary event.
    """
    events = []
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            if year == to_year and month > 6:
                break
            # PMI is usually released 1st-3rd business day of month
            d = date(year, month, 1)
            d = _next_trading_day(d)
            ref_month = month - 1 if month > 1 else 12
            ref_year = year if month > 1 else year - 1
            ref_label = date(ref_year, ref_month, 1).strftime("%b %Y")
            events.append(_make_event(
                name=f"India Manufacturing PMI ({ref_label})",
                d=d,
                severity="MEDIUM",
                time_str="10:30",
                source="SPGLOBAL_HISTORICAL",
            ))
    return events


def _generate_trade_balance_events(from_year: int, to_year: int) -> list[dict]:
    """
    India Trade Balance (Merchandise Exports/Imports).
    Released by Ministry of Commerce around 15th of each month.
    """
    events = []
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            if year == to_year and month > 6:
                break
            d = date(year, month, 15)
            ref_month = month - 1 if month > 1 else 12
            ref_year = year if month > 1 else year - 1
            ref_label = date(ref_year, ref_month, 1).strftime("%b %Y")
            events.append(_make_event(
                name=f"India Trade Balance ({ref_label})",
                d=d,
                severity="MINOR",
                time_str="16:00",
                source="MOC_HISTORICAL",
            ))
    return events


# ===================================================================
#  Discrete event generators  (RBI, Budget, GDP)
# ===================================================================

def _generate_rbi_events(from_year: int, to_year: int) -> list[dict]:
    """RBI MPC Policy Decisions from known dates."""
    events = []
    for y, m, d in RBI_MPC_DATES:
        if y < from_year or y > to_year:
            continue
        events.append(_make_event(
            name="RBI MPC Policy Decision",
            d=date(y, m, d),
            severity="CRITICAL",
            time_str="10:00",
            source="RBI_MPC_ARCHIVE",
        ))
    return events


def _generate_budget_events(from_year: int, to_year: int) -> list[dict]:
    """Union Budget from known dates."""
    events = []
    for y, m, d in BUDGET_DATES:
        if y < from_year or y > to_year:
            continue
        events.append(_make_event(
            name="India Union Budget",
            d=date(y, m, d),
            severity="CRITICAL",
            time_str="11:00",
            source="BUDGET_ARCHIVE",
        ))
    return events


def _generate_gdp_events(from_year: int, to_year: int) -> list[dict]:
    """India GDP quarterly releases from known dates."""
    events = []
    for y, m, d, label in GDP_DATES:
        if y < from_year or y > to_year:
            continue
        events.append(_make_event(
            name=f"India GDP {label}",
            d=date(y, m, d),
            severity="CRITICAL",
            time_str="17:30",
            source="MOSPI_GDP_ARCHIVE",
        ))
    return events


# ===================================================================
#  Main builder
# ===================================================================

def build_historical_schedule(from_year: int = 2016, to_year: int = 2026) -> list[dict]:
    """Build the complete historical macro event schedule."""

    log.info("Building historical macro event schedule (%d → %d) …", from_year, to_year)

    all_events: list[dict] = []

    # CRITICAL events
    rbi = _generate_rbi_events(from_year, to_year)
    budget = _generate_budget_events(from_year, to_year)
    gdp = _generate_gdp_events(from_year, to_year)
    log.info("  CRITICAL: %d RBI MPC + %d Budget + %d GDP = %d",
             len(rbi), len(budget), len(gdp), len(rbi) + len(budget) + len(gdp))
    all_events.extend(rbi)
    all_events.extend(budget)
    all_events.extend(gdp)

    # MAJOR events
    cpi = _generate_cpi_events(from_year, to_year)
    iip = _generate_iip_events(from_year, to_year)
    log.info("  MAJOR: %d CPI + %d IIP = %d", len(cpi), len(iip), len(cpi) + len(iip))
    all_events.extend(cpi)
    all_events.extend(iip)

    # MEDIUM events
    wpi = _generate_wpi_events(from_year, to_year)
    pmi = _generate_pmi_events(from_year, to_year)
    log.info("  MEDIUM: %d WPI + %d PMI = %d", len(wpi), len(pmi), len(wpi) + len(pmi))
    all_events.extend(wpi)
    all_events.extend(pmi)

    # MINOR events
    trade = _generate_trade_balance_events(from_year, to_year)
    log.info("  MINOR: %d Trade Balance", len(trade))
    all_events.extend(trade)

    # Sort chronologically
    all_events.sort(key=lambda e: e["timestamp"])

    log.info("  TOTAL: %d events", len(all_events))
    return all_events


def validate_schedule(events: list[dict]) -> bool:
    """Validate event format and consistency."""
    errors = 0
    seen_timestamps: dict[str, str] = {}
    valid_severities = {"CRITICAL", "MAJOR", "MEDIUM", "MINOR"}

    for i, e in enumerate(events):
        # Required fields
        for field in ("name", "timestamp", "severity", "scope", "source"):
            if field not in e:
                log.error("  Event %d missing field '%s': %s", i, field, e.get("name", "?"))
                errors += 1

        # Severity check
        if e.get("severity") not in valid_severities:
            log.error("  Event %d invalid severity '%s': %s", i, e.get("severity"), e["name"])
            errors += 1

        # Timestamp format check
        ts = e.get("timestamp", "")
        if not ts or len(ts) < 19:
            log.error("  Event %d invalid timestamp '%s': %s", i, ts, e["name"])
            errors += 1

        # Check for exact same timestamp + same name (likely duplicate)
        key = f"{ts}|{e.get('name', '')}"
        if key in seen_timestamps:
            log.warning("  Duplicate: '%s' at %s", e["name"], ts)
        seen_timestamps[key] = e.get("name", "")

    if errors:
        log.error("Validation found %d errors", errors)
        return False

    log.info("Validation passed: %d events, 0 errors", len(events))
    return True


def print_summary(events: list[dict]):
    """Print a concise summary of the schedule."""
    from collections import Counter
    severity_counts = Counter(e["severity"] for e in events)
    name_counts = Counter(e["name"].split("(")[0].strip() for e in events)

    years = set()
    for e in events:
        years.add(e["timestamp"][:4])

    print(f"\n{'=' * 60}")
    print("Historical India Macro Event Schedule")
    print(f"{'=' * 60}")
    print(f"Total events  : {len(events)}")
    print(f"Year range    : {min(years)} → {max(years)}")
    print(f"\nBy severity:")
    for sev in ("CRITICAL", "MAJOR", "MEDIUM", "MINOR"):
        print(f"  {sev:<12} {severity_counts.get(sev, 0):>4}")
    print(f"\nBy type:")
    for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<40} {count:>4}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build historical India macro event schedule for backtesting",
    )
    parser.add_argument("--from-year", type=int, default=2016,
                        help="Start year (default: 2016, when MPC started)")
    parser.add_argument("--to-year", type=int, default=2026,
                        help="End year (default: 2026)")
    parser.add_argument("--validate", action="store_true",
                        help="Only validate existing schedule, don't rebuild")
    args = parser.parse_args()

    if args.validate:
        if not OUTPUT_FILE.exists():
            log.error("No schedule file found at %s", OUTPUT_FILE)
            sys.exit(1)
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
        events = data.get("events", [])
        ok = validate_schedule(events)
        if ok:
            print_summary(events)
        sys.exit(0 if ok else 1)

    events = build_historical_schedule(args.from_year, args.to_year)

    ok = validate_schedule(events)
    if not ok:
        log.error("Schedule has validation errors. Saving anyway for review.")

    output = {
        "description": "Historical India macro event schedule for backtesting",
        "generated_by": "scripts/build_historical_macro_events.py",
        "year_range": f"{args.from_year}-{args.to_year}",
        "event_count": len(events),
        "notes": [
            "RBI MPC dates sourced from official RBI archive",
            "CPI/IIP dates approximate (~12th of month, adjusted to trading days)",
            "WPI dates approximate (~14th of month, adjusted to trading days)",
            "PMI dates approximate (~1st trading day of month)",
            "Trade balance dates approximate (~15th of month, adjusted to trading days)",
            "GDP dates from MoSPI known release pattern (~end of 2nd month after quarter)",
            "Budget dates from official government records",
            "Weekend/holiday dates adjusted to next trading day",
        ],
        "events": events,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log.info("Schedule saved to %s", OUTPUT_FILE)
    print_summary(events)


if __name__ == "__main__":
    main()
