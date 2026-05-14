"""
Timestamp coercion helpers.

Uses pandas for parsing but carries no project-specific logic.
"""

from __future__ import annotations

import pandas as pd


def coerce_timestamp_series(values, *, utc: bool | None = None) -> pd.Series:
    """Parse timestamp-like values with pandas' mixed-format parser.

    Signal datasets have historically mixed ISO strings, timezone offsets, and
    space-separated timestamps in the same column.  Plain ``pd.to_datetime`` can
    infer one format from the first row and silently coerce later valid rows to
    ``NaT``; this helper keeps report/evaluation timestamp handling consistent.
    """
    index = getattr(values, "index", None)
    kwargs = {"errors": "coerce", "format": "mixed"}
    if utc is not None:
        kwargs["utc"] = utc

    try:
        parsed = pd.to_datetime(values, **kwargs)
    except (TypeError, ValueError):
        fallback_kwargs = {"errors": "coerce", "format": "mixed", "utc": True if utc is None else utc}
        try:
            parsed = pd.to_datetime(values, **fallback_kwargs)
        except (TypeError, ValueError):
            fallback_kwargs.pop("format", None)
            try:
                parsed = pd.to_datetime(values, **fallback_kwargs)
            except (TypeError, ValueError):
                parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)

    if isinstance(parsed, pd.Series):
        result = parsed
    else:
        result = pd.Series(parsed, index=index)

    if not hasattr(result, "dt"):
        result = pd.Series(pd.to_datetime(values, errors="coerce", format="mixed", utc=True), index=index)
    return result


def coerce_timestamp(value, *, tz: str | None = None, fallback=None):
    """Parse *value* into a timezone-aware ``pd.Timestamp``.

    Parameters
    ----------
    value : str | datetime | pd.Timestamp | None
        Raw timestamp to parse.
    tz : str, optional
        Target timezone (e.g. ``"Asia/Kolkata"``).  When *value* already
        carries timezone info it is converted; otherwise it is localized.
    fallback : optional
        Returned when *value* cannot be parsed.

    Returns
    -------
    pd.Timestamp | fallback
    """
    if value is None:
        return fallback
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if ts is pd.NaT:
            return fallback
        if tz:
            ts = ts.tz_convert(tz)
        return ts
    except Exception:
        return fallback
