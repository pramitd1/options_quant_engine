"""
Module: models.py

Purpose:
    Implement models logic used to classify headlines and derive news context.

Role in the System:
    Part of the news context layer that scores headline risk and directional news pressure.

Key Outputs:
    Headline state, news sentiment features, and risk flags.

Downstream Usage:
    Consumed by macro/news overlays, the signal engine, and research logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


IST_TIMEZONE = "Asia/Kolkata"


def coerce_headline_timestamp(value) -> pd.Timestamp | None:
    """
    Purpose:
        Coerce headline timestamp into a consistent representation.
    
    Context:
        Public function within the news layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        value (Any): Input associated with value.
    
    Returns:
        pd.Timestamp | None: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Dataclass representing HeadlineRecord within the repository.
    
    Context:
        Used within the news layer that classifies headlines and derives risk context. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        timestamp (pd.Timestamp): Value supplied for timestamp.
        source (str): Value supplied for source.
        headline (str): Value supplied for headline.
        url_or_identifier (str): Value supplied for url or identifier.
        category (str | None): Value supplied for category.
        raw_payload (dict[str, Any]): Structured mapping for raw payload.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    timestamp: pd.Timestamp
    source: str
    headline: str
    url_or_identifier: str
    category: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `HeadlineRecord` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(frozen=True)
class HeadlineIngestionState:
    """
    Purpose:
        Dataclass representing HeadlineIngestionState within the repository.
    
    Context:
        Used within the news layer that classifies headlines and derives risk context. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        records (list[HeadlineRecord]): Collection of records.
        provider_name (str): Value supplied for provider name.
        fetched_at (pd.Timestamp): Value supplied for fetched at.
        latest_headline_at (pd.Timestamp | None): Value supplied for latest headline at.
        is_stale (bool): Boolean flag controlling whether stale is active.
        data_available (bool): Boolean flag controlling whether data available is active.
        neutral_fallback (bool): Boolean flag controlling whether neutral fallback is active.
        stale_after_minutes (int): Number of minutes used for stale after.
        provider_metadata (dict[str, Any]): Structured mapping for provider metadata.
        issues (list[str]): Collection of issues.
        warnings (list[str]): Collection of warnings.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
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
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `HeadlineIngestionState` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
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
