"""SQLite decision log — the audit trail every stage writes to before
returning. No stage may resolve an item without a Decision row landing here.

The exception report and the HTML report are queries over this table, never
separately maintained lists (ARCHITECTURE.md § Audit trail).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from unbatch.models import Decision

DEFAULT_DB_PATH = Path("out/audit.db")


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the audit database and ensure the decisions
    table exists."""
    raise NotImplementedError


def record(conn: sqlite3.Connection, decision: Decision) -> None:
    """Write one Decision row."""
    raise NotImplementedError


def fetch_decisions(conn: sqlite3.Connection, run_id: str) -> list[Decision]:
    """Return every Decision written during `run_id`."""
    raise NotImplementedError


def fetch_exceptions(conn: sqlite3.Connection, run_id: str) -> list[Decision]:
    """Return every Decision with outcome `exception` for `run_id`, for the
    exception report."""
    raise NotImplementedError
