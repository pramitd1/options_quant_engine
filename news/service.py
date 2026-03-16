"""
Module: service.py

Purpose:
    Implement service logic used to classify headlines and derive news context.

Role in the System:
    Part of the news context layer that scores headline risk and directional news pressure.

Key Outputs:
    Headline state, news sentiment features, and risk flags.

Downstream Usage:
    Consumed by macro/news overlays, the signal engine, and research logging.
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
    """
    Purpose:
        Process now timestamp for downstream use.
    
    Context:
        Internal helper within the news layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        as_of (Any): Input associated with as of.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Process neutral headline state for downstream use.
    
    Context:
        Internal helper within the news layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        provider_name (str): Human-readable name for provider.
        stale_after_minutes (int): Input associated with stale after minutes.
        fetched_at (Any): Input associated with fetched at.
        provider_metadata (Any): Input associated with provider metadata.
        issues (Any): Input associated with issues.
        warnings (Any): Input associated with warnings.
        data_available (bool): Boolean flag associated with data_available.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Service object that coordinates the `HeadlineIngestionService` workflow.

    Context:
        Used within the `service` module. The module sits in the news layer that converts provider headlines into structured macro state.

    Attributes:
        None: The class primarily groups behavior or protocol methods rather than declared fields.

    Notes:
        Documents the current role of the class without changing runtime behavior.
    """
    def __init__(
        self,
        provider: HeadlineProvider,
        *,
        stale_after_minutes: int = HEADLINE_STALE_MINUTES,
        max_records: int = HEADLINE_MAX_RECORDS,
        enabled: bool = HEADLINE_INGESTION_ENABLED,
    ):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `HeadlineIngestionService` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            provider (HeadlineProvider): Input associated with provider.
            stale_after_minutes (int): Input associated with stale after minutes.
            max_records (int): Input associated with max records.
            enabled (bool): Boolean flag associated with enabled.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        self.provider = provider
        self.stale_after_minutes = stale_after_minutes
        self.max_records = max_records
        self.enabled = enabled

    def fetch(self, *, symbol: str | None = None, as_of=None, replay_mode: bool = False) -> HeadlineIngestionState:
        """
        Purpose:
            Process fetch for downstream use.
        
        Context:
            Method on `HeadlineIngestionService` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            symbol (str | None): Underlying symbol or index identifier.
            as_of (Any): Input associated with as of.
            replay_mode (bool): Boolean flag associated with replay_mode.
        
        Returns:
            HeadlineIngestionState: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
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
    """
    Purpose:
        Build the default headline service used by downstream components.
    
    Context:
        Public function within the news layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        HeadlineIngestionService: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    """
    Purpose:
        Process headline records to frame for downstream use.
    
    Context:
        Public function within the news layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        records (list[HeadlineRecord]): Input associated with records.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    return pd.DataFrame([record.to_dict() for record in records])
