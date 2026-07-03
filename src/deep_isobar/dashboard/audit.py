"""Append-only audit log for the dashboard API.

Every security-relevant action — logins (success *and* failure), user
management, config changes, manual trade edits — gets one immutable row.
The moment several people and real money touch this system, "who changed
the Kelly fraction and when" must have an answer.

Rows live in the same SQLite database as the users table.  There is no
update or delete path on purpose; the table only grows.

Usage from api.py::

    record_audit(conn, username, "login.success")
    record_audit(conn, admin["username"], "settings.patch",
                 detail="risk.alpha_threshold 0.10 -> 0.12")
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc    TEXT NOT NULL,
    username  TEXT NOT NULL,
    action    TEXT NOT NULL,
    detail    TEXT NOT NULL DEFAULT '',
    ip        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (ts_utc);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log (username);
"""


def init_audit_table(conn: sqlite3.Connection) -> None:
    """Create the audit table if missing (called from API startup)."""
    conn.executescript(_SCHEMA)
    conn.commit()


def record_audit(
    conn: sqlite3.Connection,
    username: str,
    action: str,
    detail: str = "",
    ip: str = "",
) -> None:
    """Append one audit row.  Never raises — auditing must not break the API.

    Args:
        conn: An open connection (caller manages commit/close lifecycles;
            this function commits its own insert).
        username: Acting user ("" becomes "<anonymous>").
        action: Dotted event name, e.g. ``login.success``, ``login.failure``,
            ``users.create``, ``settings.patch``, ``trades.patch``.
        detail: Human-readable specifics (old → new values, target user…).
        ip: Client address when available.
    """
    try:
        conn.execute(
            "INSERT INTO audit_log (ts_utc, username, action, detail, ip) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                username or "<anonymous>",
                action,
                detail,
                ip,
            ),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        logger.exception("audit: failed to record %s by %s", action, username)


def fetch_audit(
    conn: sqlite3.Connection,
    limit: int = 200,
    offset: int = 0,
    username: str | None = None,
) -> list[dict]:
    """Most-recent-first audit rows (admin endpoint)."""
    query = "SELECT * FROM audit_log"
    params: list = []
    if username:
        query += " WHERE username = ?"
        params.append(username)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]
