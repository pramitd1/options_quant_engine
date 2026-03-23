"""
Module: telegram_alert.py

Purpose:
    Send push notifications to a Telegram chat when the signal engine
    transitions to a decision-relevant state.

Role in the System:
    Part of the notifications layer.  Fires on meaningful state changes so
    operators do not need to watch the terminal continuously.

Key Design Decisions:
    - Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment vars so
      no secrets are hardcoded.
    - Completely optional: if env vars are absent or the HTTP call fails, the
      engine continues silently.  Errors are logged at WARNING level only.
    - Alerting is idempotent per-state: the module caches the last alerted
      (trade_status + direction) pair to suppress duplicate messages.

Downstream Usage:
    Called from ``main.py`` or ``engine_runner.py`` after each snapshot result
    when the trade_status changes.  Can also be imported by operator scripts.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

# Per-process state: last alerted signature to suppress redundant messages.
_last_alerted_signature: str | None = None
_last_alert_time: float = 0.0
# Minimum seconds between repeat alerts for the same state (avoids flooding).
_MIN_REPEAT_INTERVAL_SECONDS = 300.0


def _get_telegram_config() -> tuple[str | None, str | None]:
    """Return (bot_token, chat_id) from environment variables."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or None
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or None
    return token, chat_id


def _build_message(trade: dict, prev_status: str | None) -> str:
    """Format a concise, information-rich alert message."""
    trade_status = str(trade.get("trade_status") or "")
    direction = trade.get("direction", "-")
    trade_strength = trade.get("trade_strength", "-")
    conf_level = trade.get("signal_confidence_level", "-")
    gamma_regime = trade.get("gamma_regime", "-")
    symbol = trade.get("symbol", "NIFTY")

    lines = [
        f"📊 *{symbol} — Engine Alert*",
        f"Status : `{trade_status}`",
    ]

    if prev_status:
        lines.append(f"Previous: `{prev_status}`")

    if direction and direction != "None":
        lines.append(f"Direction: `{direction}`")

    lines.append(f"Strength : `{trade_strength}`")
    lines.append(f"Confidence: `{conf_level}`")
    lines.append(f"Gamma regime: `{gamma_regime}`")

    if trade_status == "TRADE":
        strike = trade.get("strike", "-")
        entry = trade.get("entry_price", "-")
        target = trade.get("target", "-")
        stop = trade.get("stop_loss", "-")
        expiry = trade.get("selected_expiry", "-")
        lines += [
            "",
            f"🎯 *Trade Signal*",
            f"Strike : `{strike} {direction}`",
            f"Entry  : `{entry}`",
            f"Target : `{target}`",
            f"Stop   : `{stop}`",
            f"Expiry : `{expiry}`",
        ]

    no_trade_reason = trade.get("no_trade_reason")
    if no_trade_reason and trade_status != "TRADE":
        lines.append(f"Reason : `{no_trade_reason}`")

    return "\n".join(lines)


def _send(token: str, chat_id: str, text: str) -> bool:
    """POST message to Telegram Bot API.  Returns True on success."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "True",
    }).encode()

    req = urllib.request.Request(url, data=params, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception as exc:
        log.warning("telegram_alert: HTTP send failed — %s", exc)
        return False


def maybe_alert(
    trade: dict,
    *,
    prev_status: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Send a Telegram alert if the trade state has changed meaningfully.

    Parameters
    ----------
    trade:
        The engine trade payload dict.
    prev_status:
        The trade_status of the previous snapshot (used to detect transitions).
        When ``None``, any TRADE or state-change will trigger an alert.
    force:
        Send regardless of deduplication state.

    Returns
    -------
    bool: True if a message was dispatched.
    """
    global _last_alerted_signature, _last_alert_time

    if not isinstance(trade, dict):
        return False

    token, chat_id = _get_telegram_config()
    if not token or not chat_id:
        return False  # Silently skip when unconfigured.

    trade_status = str(trade.get("trade_status") or "")
    direction = str(trade.get("direction") or "")
    signature = f"{trade_status}:{direction}"

    # Skip if nothing changed and the repeat window hasn't elapsed.
    now = time.monotonic()
    if not force:
        same_sig = signature == _last_alerted_signature
        too_soon = (now - _last_alert_time) < _MIN_REPEAT_INTERVAL_SECONDS
        if same_sig and too_soon:
            return False

        # Only alert on TRADE entries or on transitions away from TRADE.
        meaningful_transition = (
            trade_status == "TRADE"
            or prev_status == "TRADE"
            or (prev_status and prev_status != trade_status)
        )
        if not meaningful_transition:
            return False

    text = _build_message(trade, prev_status)
    success = _send(token, chat_id, text)

    if success:
        _last_alerted_signature = signature
        _last_alert_time = now
        log.info("telegram_alert: alert sent (status=%s)", trade_status)

    return success
