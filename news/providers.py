"""
Module: providers.py

Purpose:
    Implement providers logic used to classify headlines and derive news context.

Role in the System:
    Part of the news context layer that scores headline risk and directional news pressure.

Key Outputs:
    Headline state, news sentiment features, and risk flags.

Downstream Usage:
    Consumed by macro/news overlays, the signal engine, and research logging.
"""

from __future__ import annotations

import json
from email.utils import parsedate_to_datetime
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

from config.settings import (
    BASE_DIR,
    HEADLINE_RSS_TIMEOUT_SECONDS,
    HEADLINE_RSS_URLS,
    HEADLINE_RSS_USER_AGENT,
)
from macro.scope_utils import headline_mentions_symbol, normalize_scope, symbol_scope_matches
from news.models import HeadlineRecord, coerce_headline_timestamp


class HeadlineProvider(ABC):
    """
    Purpose:
        Provider adapter responsible for `HeadlineProvider` interactions.

    Context:
        Used within the `providers` module. The module sits in the news layer that converts provider headlines into structured macro state.

    Attributes:
        None: The class primarily groups behavior or protocol methods rather than declared fields.

    Notes:
        Documents the current role of the class without changing runtime behavior.
    """
    provider_name = "UNKNOWN"

    @abstractmethod
    def fetch_headlines(self, *, symbol: str | None = None, limit: int | None = None) -> list[HeadlineRecord]:
        """
        Purpose:
            Fetch headlines for the current evaluation step.
        
        Context:
            Method on `HeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            symbol (str | None): Underlying symbol or index identifier.
            limit (int | None): Input associated with limit.
        
        Returns:
            list[HeadlineRecord]: Computed value returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        raise NotImplementedError

    def last_fetch_metadata(self) -> dict:
        """
        Purpose:
            Process last fetch metadata for downstream use.
        
        Context:
            Method on `HeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        return {}


class MockHeadlineProvider(HeadlineProvider):
    """
    Purpose:
        Provider adapter responsible for `MockHeadlineProvider` interactions.

    Context:
        Used within the `providers` module. The module sits in the news layer that converts provider headlines into structured macro state.

    Attributes:
        None: The class primarily groups behavior or protocol methods rather than declared fields.

    Notes:
        Documents the current role of the class without changing runtime behavior.
    """
    provider_name = "MOCK"

    def __init__(self, mock_file: str):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `MockHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            mock_file (str): Input associated with mock file.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        path = Path(mock_file)
        if not path.is_absolute():
            path = Path(BASE_DIR) / path
        self.mock_file = path
        self._last_fetch_metadata = {}

    def fetch_headlines(self, *, symbol: str | None = None, limit: int | None = None) -> list[HeadlineRecord]:
        """
        Purpose:
            Fetch headlines for the current evaluation step.
        
        Context:
            Method on `MockHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            symbol (str | None): Underlying symbol or index identifier.
            limit (int | None): Input associated with limit.
        
        Returns:
            list[HeadlineRecord]: Computed value returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        self._last_fetch_metadata = {
            "mock_file": str(self.mock_file),
            "status": "UNKNOWN",
        }

        if not self.mock_file.exists():
            self._last_fetch_metadata["status"] = "MISSING_FILE"
            return []

        try:
            payload = json.loads(self.mock_file.read_text(encoding="utf-8"))
        except Exception:
            self._last_fetch_metadata["status"] = "PARSE_ERROR"
            return []

        records: list[HeadlineRecord] = []
        symbol_upper = str(symbol or "").strip().upper()

        for raw in payload if isinstance(payload, list) else []:
            if not isinstance(raw, dict):
                continue

            ts = coerce_headline_timestamp(raw.get("timestamp"))
            if ts is None:
                continue

            scopes = normalize_scope(raw.get("scope", raw.get("symbol")))
            if symbol_upper and not symbol_scope_matches(symbol_upper, scopes):
                continue

            headline = str(raw.get("headline", "")).strip()
            source = str(raw.get("source", self.provider_name)).strip() or self.provider_name
            if not headline:
                continue

            identifier = (
                str(raw.get("url_or_identifier", "")).strip()
                or str(raw.get("url", "")).strip()
                or str(raw.get("id", "")).strip()
                or headline
            )

            record = HeadlineRecord(
                timestamp=ts,
                source=source,
                headline=headline,
                url_or_identifier=identifier,
                category=str(raw.get("category")).strip() if raw.get("category") is not None else None,
                raw_payload=raw,
            )
            records.append(record)

        records.sort(key=lambda record: record.timestamp, reverse=True)
        self._last_fetch_metadata["status"] = "OK"
        self._last_fetch_metadata["record_count"] = len(records)
        if limit is not None and limit > 0:
            records = records[:limit]
            self._last_fetch_metadata["record_count_after_limit"] = len(records)
        return records

    def last_fetch_metadata(self) -> dict:
        """
        Purpose:
            Process last fetch metadata for downstream use.
        
        Context:
            Method on `MockHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        return dict(self._last_fetch_metadata)


def _first_text(element, names: list[str]) -> str | None:
    """
    Purpose:
        Process first text for downstream use.
    
    Context:
        Internal helper within the news layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        element (Any): Input associated with element.
        names (list[str]): Input associated with names.
    
    Returns:
        str | None: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    for name in names:
        child = element.find(name)
        if child is not None and child.text:
            text = child.text.strip()
            if text:
                return text
    return None


def _parse_rss_timestamp(raw_value):
    """
    Purpose:
        Process parse rss timestamp for downstream use.
    
    Context:
        Internal helper within the news layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        raw_value (Any): Input associated with raw value.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if not raw_value:
        return None

    try:
        parsed = parsedate_to_datetime(raw_value)
        if parsed is not None:
            return coerce_headline_timestamp(parsed.isoformat())
    except Exception:
        pass

    return coerce_headline_timestamp(raw_value)


class RSSHeadlineProvider(HeadlineProvider):
    """
    Purpose:
        Provider adapter responsible for `RSSHeadlineProvider` interactions.

    Context:
        Used within the `providers` module. The module sits in the news layer that converts provider headlines into structured macro state.

    Attributes:
        None: The class primarily groups behavior or protocol methods rather than declared fields.

    Notes:
        Documents the current role of the class without changing runtime behavior.
    """
    provider_name = "RSS"

    def __init__(
        self,
        urls: list[str] | None = None,
        *,
        timeout_seconds: int = HEADLINE_RSS_TIMEOUT_SECONDS,
        user_agent: str = HEADLINE_RSS_USER_AGENT,
    ):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `RSSHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            urls (list[str] | None): Input associated with urls.
            timeout_seconds (int): Input associated with timeout seconds.
            user_agent (str): Input associated with user agent.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        self.urls = [url for url in (urls or HEADLINE_RSS_URLS) if str(url).strip()]
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self._last_fetch_metadata = {}

    def _source_label(self, url: str, root: ET.Element) -> str:
        """
        Purpose:
            Process source label for downstream use.
        
        Context:
            Method on `RSSHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            url (str): Input associated with url.
            root (ET.Element): Input associated with root.
        
        Returns:
            str: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        channel_title = _first_text(root, ["channel/title"])
        if channel_title:
            return channel_title

        parsed = urlparse(url)
        return parsed.netloc or self.provider_name

    def fetch_headlines(self, *, symbol: str | None = None, limit: int | None = None) -> list[HeadlineRecord]:
        """
        Purpose:
            Fetch headlines for the current evaluation step.
        
        Context:
            Method on `RSSHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            symbol (str | None): Underlying symbol or index identifier.
            limit (int | None): Input associated with limit.
        
        Returns:
            list[HeadlineRecord]: Computed value returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        self._last_fetch_metadata = {
            "status": "UNKNOWN",
            "urls": list(self.urls),
            "feed_count": len(self.urls),
            "per_url_status": [],
        }

        if not self.urls:
            self._last_fetch_metadata["status"] = "NO_URLS"
            return []

        headers = {"User-Agent": self.user_agent}
        records: list[HeadlineRecord] = []
        symbol_upper = str(symbol or "").strip().upper()

        for url in self.urls:
            status_entry = {"url": url, "status": "UNKNOWN", "record_count": 0}
            try:
                response = requests.get(url, timeout=self.timeout_seconds, headers=headers)
                response.raise_for_status()
            except Exception as exc:
                status_entry["status"] = f"REQUEST_ERROR:{type(exc).__name__}"
                self._last_fetch_metadata["per_url_status"].append(status_entry)
                continue

            try:
                root = ET.fromstring(response.content)
            except Exception as exc:
                status_entry["status"] = f"PARSE_ERROR:{type(exc).__name__}"
                self._last_fetch_metadata["per_url_status"].append(status_entry)
                continue

            source = self._source_label(url, root)
            items = root.findall("./channel/item")
            # basic Atom fallback
            if not items:
                items = root.findall("{http://www.w3.org/2005/Atom}entry")

            for item in items:
                headline = _first_text(
                    item,
                    ["title", "{http://www.w3.org/2005/Atom}title"],
                )
                if not headline:
                    continue

                if symbol_upper and not headline_mentions_symbol(symbol_upper, headline):
                    status_entry.setdefault("filtered_out", 0)
                    status_entry["filtered_out"] += 1
                    continue

                link = _first_text(
                    item,
                    ["link", "{http://www.w3.org/2005/Atom}id", "guid"],
                )
                if not link:
                    link = headline

                category = _first_text(
                    item,
                    ["category", "{http://www.w3.org/2005/Atom}category"],
                )

                timestamp = _parse_rss_timestamp(
                    _first_text(
                        item,
                        ["pubDate", "{http://www.w3.org/2005/Atom}updated", "{http://www.w3.org/2005/Atom}published"],
                    )
                )
                if timestamp is None:
                    continue

                record = HeadlineRecord(
                    timestamp=timestamp,
                    source=source,
                    headline=headline,
                    url_or_identifier=link,
                    category=category,
                    raw_payload={"feed_url": url},
                )
                records.append(record)
                status_entry["record_count"] += 1

            status_entry["status"] = "OK"
            self._last_fetch_metadata["per_url_status"].append(status_entry)

        records.sort(key=lambda record: record.timestamp, reverse=True)
        self._last_fetch_metadata["status"] = "OK" if records else "NO_RECORDS"
        self._last_fetch_metadata["record_count"] = len(records)
        if limit is not None and limit > 0:
            records = records[:limit]
            self._last_fetch_metadata["record_count_after_limit"] = len(records)
        return records

    def last_fetch_metadata(self) -> dict:
        """
        Purpose:
            Process last fetch metadata for downstream use.
        
        Context:
            Method on `RSSHeadlineProvider` within the news layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        return dict(self._last_fetch_metadata)


def build_headline_provider(
    provider_name: str,
    *,
    mock_file: str,
    rss_urls: list[str] | None = None,
    rss_timeout_seconds: int = HEADLINE_RSS_TIMEOUT_SECONDS,
    rss_user_agent: str = HEADLINE_RSS_USER_AGENT,
) -> HeadlineProvider:
    """
    Purpose:
        Build the headline provider used by downstream components.
    
    Context:
        Public function within the news layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        provider_name (str): Human-readable name for provider.
        mock_file (str): Input associated with mock file.
        rss_urls (list[str] | None): Input associated with rss urls.
        rss_timeout_seconds (int): Input associated with rss timeout seconds.
        rss_user_agent (str): Input associated with rss user agent.
    
    Returns:
        HeadlineProvider: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    provider_name = str(provider_name or "MOCK").strip().upper()

    if provider_name == "MOCK":
        return MockHeadlineProvider(mock_file=mock_file)

    if provider_name == "RSS":
        return RSSHeadlineProvider(
            urls=rss_urls,
            timeout_seconds=rss_timeout_seconds,
            user_agent=rss_user_agent,
        )

    raise ValueError(f"Unsupported headline provider: {provider_name}")
