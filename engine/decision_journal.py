"""
Module: decision_journal.py

Purpose:
    Append-only, immutable audit log for every trade decision emitted by the
    signal engine.  Powered by SQLite with WAL mode so concurrent readers
    (research notebooks, monitoring scripts) never block the engine.

Role in the System:
    Part of the engine layer.  Every call to ``generate_trade`` that produces
    a final payload calls ``append_decision`` so the full reasoning chain is
    permanently preserved — including blocked, watchlist, and executable trades.

Key Properties:
    - Append-only by design: no UPDATE or DELETE SQL is ever executed.
    - WAL journal mode: write-ahead logging keeps reads non-blocking.
    - Schema-stable: adding new payload keys serialises to the ``payload_json``
      column without a migration.
    - Thread-safe: uses a threading lock around the writer connection.

Downstream Usage:
    - Research replay: re-hydrate past decisions from the journal.
    - Drift detection: compare parameter pack used vs live pack.
    - Compliance: every decision has a timestamp, trade_status, and full payload.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Default journal file path (relative to the project root).
_DEFAULT_JOURNAL_PATH = Path(__file__).resolve().parent.parent / "data_store" / "decision_journal.sqlite3"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at       TEXT    NOT NULL,
    trade_status      TEXT    NOT NULL,
    direction         TEXT,
    symbol            TEXT,
    trade_strength    INTEGER,
    signal_confidence TEXT,
    parameter_pack    TEXT,
    payload_json      TEXT    NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO decisions
    (recorded_at, trade_status, direction, symbol, trade_strength,
     signal_confidence, parameter_pack, payload_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

_WAL_PRAGMA = "PRAGMA journal_mode=WAL;"
_SYNC_PRAGMA = "PRAGMA synchronous=NORMAL;"

_lock = threading.Lock()


def _get_journal_path() -> Path:
    """Return the active journal path, creating parent dirs as needed."""
    path = _DEFAULT_JOURNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_WAL_PRAGMA)
    conn.execute(_SYNC_PRAGMA)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()


def append_decision(payload: dict, parameter_pack_name: str | None = None) -> bool:
    """Persist a single trade decision to the journal.

    This is a best-effort write: if the database is unavailable the engine
    continues normally and only a WARNING is logged.

    Parameters
    ----------
    payload:
        The full engine output dict returned by ``generate_trade``.
    parameter_pack_name:
        Name of the active parameter pack (e.g. ``"default"`` or
        ``"aggressive_v2"``).  Stored as a searchable column for drift
        analysis.

    Returns
    -------
    bool: True if the record was written successfully.
    """
    if not isinstance(payload, dict):
        return False

    try:
        now_utc = datetime.now(tz=timezone.utc).isoformat()
        trade_status = str(payload.get("trade_status") or "")
        direction = payload.get("direction")
        symbol = payload.get("symbol")
        trade_strength = payload.get("trade_strength")
        signal_confidence = payload.get("signal_confidence_level")
        pack_name = parameter_pack_name or payload.get("parameter_pack_name") or "default"

        # Serialise full payload (non-serialisable types are stringified).
        try:
            payload_json = json.dumps(payload, default=str)
        except Exception as exc:  # pragma: no cover
            log.warning("decision_journal: payload serialisation failed — %s", exc)
            payload_json = json.dumps({"error": str(exc), "trade_status": trade_status})

        journal_path = _get_journal_path()

        with _lock:
            conn = sqlite3.connect(str(journal_path), timeout=5.0)
            try:
                _init_db(conn)
                conn.execute(
                    _INSERT_SQL,
                    (
                        now_utc,
                        trade_status,
                        direction,
                        symbol,
                        trade_strength,
                        signal_confidence,
                        pack_name,
                        payload_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return True

    except Exception as exc:  # pragma: no cover
        log.warning("decision_journal: write failed — %s", exc)
        return False


def read_recent(limit: int = 100, *, journal_path: Path | None = None) -> list[dict]:
    """Return the *limit* most recent decisions as plain dicts.

    Intended for research / monitoring use; not called by the live engine.
    """
    path = journal_path or _get_journal_path()
    if not path.exists():
        return []

    try:
        conn = sqlite3.connect(str(path), timeout=3.0)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?;", (int(limit),)
            )
            rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        for row in rows:
            try:
                row["payload"] = json.loads(row["payload_json"])
            except Exception:
                row["payload"] = {}
        return rows
    except Exception as exc:
        log.warning("decision_journal: read failed — %s", exc)
        return []
