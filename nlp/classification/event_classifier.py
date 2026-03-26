from __future__ import annotations

from nlp.preprocessing.text_normalizer import preprocess_event_text

_EVENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("earnings_result", ("earnings", "q1", "q2", "q3", "q4", "profit beats", "misses estimates")),
    ("guidance_revision", ("guidance", "outlook", "revises", "raises forecast", "cuts forecast")),
    ("large_order_win", ("order win", "order book", "contract award", "wins order")),
    ("regulatory_action", ("sebi", "rbi", "regulator", "probe", "ban", "compliance notice")),
    ("litigation_adverse_order", ("litigation", "court", "tribunal", "penalty", "adverse order")),
    ("management_change", ("ceo", "cfo", "resigns", "appointment", "board change")),
    ("promoter_insider_activity", ("promoter", "pledge", "insider", "stake sale", "stake buy")),
    ("merger_acquisition", ("merger", "acquisition", "acquire", "takeover", "demerger")),
    ("rating_action", ("upgrade", "downgrade", "rating", "outlook revised")),
    ("macro_event_sector_index", ("cpi", "wpi", "gdp", "fed", "us yields", "crude")),
    ("government_policy_event", ("budget", "gst", "policy", "cabinet", "ministry")),
    ("block_bulk_deal", ("block deal", "bulk deal", "large block", "cross deal")),
    ("rumor_unconfirmed_report", ("sources", "rumor", "unconfirmed", "reportedly")),
]


def classify_event_type(text: str | None) -> str:
    normalized = preprocess_event_text(text)
    if not normalized:
        return "unknown"

    for event_type, keywords in _EVENT_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return event_type
    return "unknown"
