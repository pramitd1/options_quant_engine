from __future__ import annotations

from typing import Any


def build_raw_event_payloads(records: list[Any]) -> list[dict[str, Any]]:
    """Convert provider headline records into generic event text payloads."""
    out: list[dict[str, Any]] = []
    for record in records or []:
        text = getattr(record, "headline", "")
        out.append(
            {
                "timestamp": getattr(record, "timestamp", None),
                "source": getattr(record, "source", "unknown"),
                "text": str(text or ""),
                "url_or_identifier": getattr(record, "url_or_identifier", ""),
                "raw_payload": getattr(record, "raw_payload", {}) or {},
            }
        )
    return out
