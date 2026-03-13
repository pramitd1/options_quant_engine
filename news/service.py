"""
Headline ingestion service with stale-data handling and neutral fallback.
"""

from __future__ import annotations

import pandas as pd

from config.settings import (
    HEADLINE_INGESTION_ENABLED,
    HEADLINE_MAX_RECORDS,
    HEADLINE_MOCK_FILE,
    HEADLINE_PROVIDER,
    HEADLINE_RSS_TIMEOUT_SECONDS,
    HEADLINE_RSS_URLS,
    HEADLINE_RSS_USER_AGENT,
    HEADLINE_STALE_MINUTES,
)
from news.models import HeadlineIngestionState, HeadlineRecord, coerce_headline_timestamp
from news.providers import HeadlineProvider, build_headline_provider


IST_TIMEZONE = "Asia/Kolkata"


def _now_ts(as_of=None):
    if as_of is None:
        return pd.Timestamp.now(tz=IST_TIMEZONE)

    ts = coerce_headline_timestamp(as_of)
    if ts is not None:
        return ts

    return pd.Timestamp.now(tz=IST_TIMEZONE)


def _neutral_headline_state(
    *,
    provider_name: str,
    stale_after_minutes: int,
    fetched_at=None,
    provider_metadata=None,
    issues=None,
    warnings=None,
    data_available: bool = False,
):
    return HeadlineIngestionState(
        records=[],
        provider_name=provider_name,
        fetched_at=_now_ts(fetched_at),
        latest_headline_at=None,
        is_stale=True,
        data_available=data_available,
        neutral_fallback=True,
        stale_after_minutes=stale_after_minutes,
        provider_metadata=provider_metadata or {},
        issues=issues or [],
        warnings=warnings or [],
    )


class HeadlineIngestionService:
    def __init__(
        self,
        provider: HeadlineProvider,
        *,
        stale_after_minutes: int = HEADLINE_STALE_MINUTES,
        max_records: int = HEADLINE_MAX_RECORDS,
        enabled: bool = HEADLINE_INGESTION_ENABLED,
    ):
        self.provider = provider
        self.stale_after_minutes = stale_after_minutes
        self.max_records = max_records
        self.enabled = enabled

    def fetch(self, *, symbol: str | None = None, as_of=None, replay_mode: bool = False) -> HeadlineIngestionState:
        if not self.enabled:
            return _neutral_headline_state(
                provider_name=self.provider.provider_name,
                stale_after_minutes=self.stale_after_minutes,
                fetched_at=as_of,
                warnings=["headline_ingestion_disabled"],
            )

        # Live RSS feeds are not replay-deterministic, so keep replay neutral
        # unless the provider is backed by a snapshot/local source.
        if replay_mode and getattr(self.provider, "provider_name", "").upper() == "RSS":
            return _neutral_headline_state(
                provider_name=self.provider.provider_name,
                stale_after_minutes=self.stale_after_minutes,
                fetched_at=as_of,
                warnings=["headline_replay_live_provider_disabled"],
            )

        try:
            records = self.provider.fetch_headlines(symbol=symbol, limit=self.max_records)
        except Exception as exc:
            return _neutral_headline_state(
                provider_name=self.provider.provider_name,
                stale_after_minutes=self.stale_after_minutes,
                fetched_at=as_of,
                issues=[f"headline_provider_error:{type(exc).__name__}"],
            )

        provider_meta = {}
        try:
            provider_meta = self.provider.last_fetch_metadata()
        except Exception:
            provider_meta = {}

        fetched_at = _now_ts(as_of)
        if not records:
            warnings = ["headline_provider_returned_no_records"]
            status = str(provider_meta.get("status", "")).upper()
            if status == "MISSING_FILE":
                warnings.append("headline_mock_file_missing")
            elif status == "PARSE_ERROR":
                warnings.append("headline_mock_file_parse_error")
            elif status == "NO_URLS":
                warnings.append("headline_rss_urls_not_configured")
            return _neutral_headline_state(
                provider_name=self.provider.provider_name,
                stale_after_minutes=self.stale_after_minutes,
                fetched_at=fetched_at,
                provider_metadata=provider_meta,
                warnings=warnings,
            )

        latest_ts = max(record.timestamp for record in records)
        age_minutes = max((fetched_at - latest_ts).total_seconds() / 60.0, 0.0)
        is_stale = age_minutes > self.stale_after_minutes

        warnings = []
        if is_stale:
            warnings.append(f"headline_data_stale:{round(age_minutes, 2)}m")

        return HeadlineIngestionState(
            records=records,
            provider_name=self.provider.provider_name,
            fetched_at=fetched_at,
            latest_headline_at=latest_ts,
            is_stale=is_stale,
            data_available=not is_stale,
            neutral_fallback=is_stale,
            stale_after_minutes=self.stale_after_minutes,
            provider_metadata=provider_meta,
            warnings=warnings,
        )


def build_default_headline_service() -> HeadlineIngestionService:
    provider = build_headline_provider(
        HEADLINE_PROVIDER,
        mock_file=HEADLINE_MOCK_FILE,
        rss_urls=HEADLINE_RSS_URLS,
        rss_timeout_seconds=HEADLINE_RSS_TIMEOUT_SECONDS,
        rss_user_agent=HEADLINE_RSS_USER_AGENT,
    )
    return HeadlineIngestionService(
        provider,
        stale_after_minutes=HEADLINE_STALE_MINUTES,
        max_records=HEADLINE_MAX_RECORDS,
        enabled=HEADLINE_INGESTION_ENABLED,
    )


def headline_records_to_frame(records: list[HeadlineRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])
