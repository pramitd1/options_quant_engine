"""
AI Narrative Provider for Daily Signal Research Reports
========================================================

Generates dynamic, data-driven narrative commentary for each section of the
daily research report using an LLM.

Provider fallback chain:
    1. **OpenAI** — if OPENAI_API_KEY is set and has quota.
    2. **Ollama** (local) — if Ollama is running at localhost:11434.
    3. **Skip** — report generates without AI narrative.

Configuration (via .env):
    OPENAI_API_KEY             – API key for OpenAI.
    OPENAI_MODEL               – Model to use (default: gpt-4o).
    OPENAI_BASE_URL            – Optional custom base URL.
    OLLAMA_BASE_URL            – Ollama endpoint (default: http://localhost:11434).
    OLLAMA_MODEL               – Ollama model name (default: llama3).
    REPORT_NARRATIVE_ENABLED   – Set to 'false' to disable entirely.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines the AI's persona and output style
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior quantitative research analyst at an institutional options \
trading desk. You produce concise, insight-rich commentary for internal daily \
signal research reports.

Your role:
- Interpret the quantitative data provided and write a short narrative paragraph.
- Ground every observation in quantitative finance theory — explain WHAT the \
  metrics reveal about market behaviour and WHY, citing relevant theoretical \
  frameworks (e.g., Kyle 1985 for microstructure, Adaptive Markets Hypothesis \
  for regime-dependent alpha, Kelly Criterion for position sizing implications, \
  Brier score decomposition for probability calibration, optimal execution \
  theory for MFE/MAE analysis, dealer gamma hedging flow dynamics, etc.).
- Highlight what the numbers mean for signal quality, model calibration, \
  and potential research actions.
- Be precise and data-driven — cite specific numbers from the data.
- Use professional hedge-fund research language. No fluff, no marketing speak.
- Keep each commentary to 3-5 sentences. Be dense with insight and theory.
- If the data is insufficient or shows no clear pattern, say so directly.
- Never fabricate numbers — only reference data explicitly provided to you.
- Treat the supplied Ground Truth Summary as authoritative. Do not contradict it.
- Do not introduce any numeric value that does not already appear in the Data or
    Ground Truth Summary.
- When theoretical context is provided, use it as a guide but add your own \
  analytical perspective based on the data.
"""


_NUMERIC_TOKEN_RE = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?")


def _extract_numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_TOKEN_RE.findall(text or ""))


def _narrative_uses_only_supported_numbers(ai_text: str, allowed_text: str) -> bool:
    """Reject commentary that invents numeric facts not present in source data.

    This is a lightweight guardrail rather than a full semantic verifier. It
    prevents the most common failure mode in LLM-authored report prose: quoting
    unsupported hit rates, returns, or counts that are not in the table or
    rule-based summary.

    Numeric tokens preserve explicit sign so "-11.0" and "11.0" are treated
    as different claims.
    """
    ai_tokens = _extract_numeric_tokens(ai_text)
    if not ai_tokens:
        return True
    allowed_tokens = _extract_numeric_tokens(allowed_text)
    return ai_tokens.issubset(allowed_tokens)

# Cache the resolved provider so we don't probe on every section call.
_cached_provider: Optional[Tuple[object, str, str]] = None  # (client, model, provider_name)
_provider_resolved = False


def _try_openai():
    """Try to build an OpenAI client. Returns (client, model) or None."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("YOUR_"):
        return None
    try:
        import openai
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=15.0)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o").strip()
        return (client, model, "OpenAI")
    except ImportError:
        logger.info("openai package not installed — skipping OpenAI provider.")
        return None
    except Exception as exc:
        logger.info("OpenAI client init failed: %s — trying fallback.", exc)
        return None


def _try_ollama():
    """Try to connect to a local Ollama instance via the OpenAI-compatible API."""
    try:
        import openai
    except ImportError:
        return None

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_api_url = f"{base_url}/v1"
    model = os.environ.get("OLLAMA_MODEL", "llama3").strip()

    try:
        import urllib.request
        # Quick probe — check if Ollama is running.
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return None
        client = openai.OpenAI(api_key="ollama", base_url=ollama_api_url)
        return (client, model, "Ollama")
    except Exception:
        return None


def _resolve_provider():
    """Resolve the best available LLM provider with fallback chain."""
    global _cached_provider, _provider_resolved
    if _provider_resolved:
        return _cached_provider

    enabled = os.environ.get("REPORT_NARRATIVE_ENABLED", "true").strip().lower()
    if enabled == "false":
        _provider_resolved = True
        _cached_provider = None
        return None

    # Fallback chain: OpenAI → Ollama → None
    for try_fn, name in [(_try_openai, "OpenAI"), (_try_ollama, "Ollama")]:
        result = try_fn()
        if result is not None:
            logger.info("AI narrative provider: %s (model: %s)", result[2], result[1])
            _cached_provider = result
            _provider_resolved = True
            return result

    logger.info("No AI narrative provider available — report will generate without AI commentary.")
    _cached_provider = None
    _provider_resolved = True
    return None


def generate_narrative(
    section_title: str,
    model: Optional[str] = None,
    theory_context: Optional[str] = None,
    ground_truth_summary: Optional[str] = None,
    structured_metrics: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Generate an AI narrative paragraph for a report section.

    Tries OpenAI first, falls back to Ollama, then skips gracefully.

    Parameters
    ----------
    section_title : str
        The section heading (e.g., "Horizon Performance").
    model : str, optional
        Override the model name.
    theory_context : str, optional
        Theoretical framework hint for this section (e.g., which quant
        finance theories apply).  Helps the LLM ground its commentary.
    structured_metrics : dict, optional
        Structured JSON-friendly metrics payload extracted from the section.
        This is the primary input path and is preferred over raw markdown
        because it reduces formatting noise and qualitative drift.

    Returns
    -------
    str or None
        The narrative paragraph, or None if AI is unavailable.
    """
    provider = _resolve_provider()
    if provider is None:
        return None

    client, default_model, provider_name = provider
    use_model = model or default_model

    theory_block = ""
    if theory_context:
        theory_block = (
            f"\nTheoretical context for this section:\n{theory_context}\n\n"
            "Use this theoretical framework to ground your analysis. "
            "Explain what the metrics tell us about market behaviour "
            "from this theoretical perspective.\n"
        )

    summary_block = ""
    if ground_truth_summary:
        summary_block = (
            "\nGround Truth Summary (authoritative, computed directly from the data):\n"
            f"{ground_truth_summary}\n\n"
            "Your commentary must remain fully consistent with this summary.\n"
        )

    metrics_block = ""
    if structured_metrics:
        metrics_block = (
            "\nStructured Metrics Payload (authoritative, machine-extracted from the report section):\n"
            f"{json.dumps(structured_metrics, indent=2, ensure_ascii=True, sort_keys=True)}\n\n"
            "Use this structured payload as the primary evidence base. Do not infer "
            "new metrics from formatting. Only reason from the fields explicitly provided.\n"
        )

    user_prompt = (
        f"Section: {section_title}\n\n"
        f"{metrics_block}"
        f"{summary_block}"
        f"{theory_block}\n"
        "Write a concise analytical commentary (3-5 sentences) interpreting "
        "this data for the daily signal research report. Ground your analysis "
        "in quantitative finance theory — explain what these metrics reveal "
        "about market behaviour, signal quality, and risk. Cite the relevant "
        "theoretical framework and specific numbers from the data."
    )

    try:
        response = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        if content:
            content = content.strip()
            allowed_text = f"{json.dumps(structured_metrics or {}, ensure_ascii=True)}\n{ground_truth_summary or ''}"
            if _narrative_uses_only_supported_numbers(content, allowed_text):
                return content
            logger.warning(
                "AI narrative rejected for '%s': introduced unsupported numeric claims.",
                section_title,
            )
            return None
        return None
    except Exception as exc:
        logger.warning("AI narrative failed for '%s' via %s: %s", section_title, provider_name, type(exc).__name__)
        # If OpenAI fails (e.g. quota), immediately switch to Ollama for this + all future calls.
        if provider_name == "OpenAI":
            return _fallback_to_ollama(section_title, user_prompt)
        return None


def _fallback_to_ollama(section_title: str, user_prompt: str) -> Optional[str]:
    """Runtime fallback: try Ollama if OpenAI call failed."""
    global _cached_provider, _provider_resolved
    ollama = _try_ollama()
    if ollama is None:
        logger.info("Ollama fallback not available — disabling AI narrative for remaining sections.")
        # Permanently disable so subsequent sections don't keep retrying.
        _cached_provider = None
        _provider_resolved = True
        return None

    client, model, _ = ollama
    logger.info("Falling back to Ollama (model: %s) for '%s'.", model, section_title)
    # Update cache so subsequent sections go directly to Ollama.
    _cached_provider = ollama
    _provider_resolved = True

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        if content:
            content = content.strip()
            if _narrative_uses_only_supported_numbers(content, user_prompt):
                return content
            logger.warning(
                "Ollama narrative rejected for '%s': introduced unsupported numeric claims.",
                section_title,
            )
        return None
    except Exception as exc:
        logger.warning("Ollama fallback also failed for '%s': %s", section_title, exc)
        return None


def is_available() -> bool:
    """Check whether any AI narrative provider is available."""
    return _resolve_provider() is not None


def get_provider_name() -> Optional[str]:
    """Return the name of the active provider, or None."""
    provider = _resolve_provider()
    return provider[2] if provider else None
