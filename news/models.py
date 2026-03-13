"""
Normalized headline data structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


IST_TIMEZONE = "Asia/Kolkata"


def coerce_headline_timestamp(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(parsed):
        return None

    try:
        return parsed.tz_convert(IST_TIMEZONE)
    except Exception:
        return None


@dataclass(frozen=True)
class HeadlineRecord:
    timestamp: pd.Timestamp
    source: str
    headline: str
    url_or_identifier: str
    category: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(frozen=True)
class HeadlineIngestionState:
    records: list[HeadlineRecord]
    provider_name: str
    fetched_at: pd.Timestamp
    latest_headline_at: pd.Timestamp | None
    is_stale: bool
    data_available: bool
    neutral_fallback: bool
    stale_after_minutes: int
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "provider_name": self.provider_name,
            "fetched_at": self.fetched_at.isoformat(),
            "latest_headline_at": self.latest_headline_at.isoformat() if self.latest_headline_at is not None else None,
            "is_stale": self.is_stale,
            "data_available": self.data_available,
            "neutral_fallback": self.neutral_fallback,
            "stale_after_minutes": self.stale_after_minutes,
            "provider_metadata": self.provider_metadata,
            "issues": self.issues,
            "warnings": self.warnings,
        }
