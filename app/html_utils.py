"""
Module: html_utils.py

Purpose:
    Provide HTML escaping utilities for safe rendering of dynamic values in
    Streamlit ``unsafe_allow_html`` blocks.

Role in the System:
    Part of the application layer. Prevents XSS by ensuring all user-supplied
    or engine-generated values are escaped before embedding in raw HTML.
"""
from __future__ import annotations

import html as _html


def esc(value) -> str:
    """Escape a value for safe embedding in HTML.

    Converts the value to a string and applies ``html.escape`` to neutralize
    any HTML/JS injection.  Returns ``"-"`` for ``None`` or empty strings.
    """
    if value is None:
        return "-"
    text = str(value)
    if not text:
        return "-"
    return _html.escape(text, quote=True)
