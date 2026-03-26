from __future__ import annotations

import re


def preprocess_event_text(text: str | None) -> str:
    """Normalize noisy headline/event text into a compact parseable form."""
    raw = str(text or "").strip().lower()
    if not raw:
        return ""

    raw = raw.replace("&", " and ")
    raw = re.sub(r"[^a-z0-9.%+\-\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw
