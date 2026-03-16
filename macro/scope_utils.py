"""
Module: scope_utils.py

Purpose:
    Implement scope utils logic used to score scheduled events and macro catalysts.

Role in the System:
    Part of the macro context layer that scores scheduled events and broad market catalysts.

Key Outputs:
    Macro-event state, catalyst scores, and gating diagnostics.

Downstream Usage:
    Consumed by the signal engine, risk overlays, and research diagnostics.
"""

from __future__ import annotations


INDEX_SCOPES = {"INDEX", "INDICES", "NIFTY", "BANKNIFTY", "FINNIFTY"}
INDEX_KEYWORD_MAP = {
    "NIFTY": ["nifty", "nifty 50", "india equities", "indian equities"],
    "BANKNIFTY": ["bank nifty", "banknifty", "bank index", "financials", "banks"],
    "FINNIFTY": ["fin nifty", "finnifty", "financial services"],
}


def normalize_scope(value) -> list[str]:
    """
    Purpose:
        Normalize scope into the repository-standard representation.
    
    Context:
        Public function in the `scope utils` module. It forms part of the macro workflow exposed by this module.
    
    Inputs:
        value (Any): Raw value supplied by the caller.
    
    Returns:
        list[str]: List of results produced by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    if value is None:
        return ["ALL"]

    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, list):
        parts = value
    else:
        return ["ALL"]

    cleaned = []
    for part in parts:
        text = str(part).strip().upper()
        if text:
            cleaned.append(text)

    return cleaned or ["ALL"]


def symbol_scope_matches(symbol: str, scopes: list[str]) -> bool:
    """
    Purpose:
        Process symbol scope matches for downstream use.
    
    Context:
        Public function within the macro context layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        scopes (list[str]): Input associated with scopes.
    
    Returns:
        bool: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    symbol_upper = str(symbol or "").strip().upper()

    if "ALL" in scopes:
        return True

    if symbol_upper in scopes:
        return True

    if symbol_upper in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
        return any(scope in INDEX_SCOPES for scope in scopes)

    if "STOCK" in scopes and symbol_upper not in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
        return True

    return False


def headline_mentions_symbol(symbol: str, headline: str) -> bool:
    """
    Purpose:
        Process headline mentions symbol for downstream use.
    
    Context:
        Public function within the macro context layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        headline (str): Input associated with headline.
    
    Returns:
        bool: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    symbol_upper = str(symbol or "").strip().upper()
    text = str(headline or "").strip().lower()

    if not symbol_upper or not text:
        return True

    if symbol_upper in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
        keywords = INDEX_KEYWORD_MAP.get(symbol_upper, [])
        return any(keyword in text for keyword in keywords)

    simple_aliases = {
        symbol_upper.lower(),
        f"{symbol_upper.lower()}.ns",
        f"nse:{symbol_upper.lower()}",
    }
    return any(alias in text for alias in simple_aliases)
