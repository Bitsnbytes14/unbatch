"""SQLite decision log — the audit trail every stage writes to before
returning. No stage may resolve an item without a Decision row landing here.

The exception report and the HTML report are queries over this table, never
separately maintained lists (ARCHITECTURE.md § Audit trail).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from unbatch.models import Decision, DecisionOutcome, Stage

DEFAULT_DB_PATH = Path("out/audit.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    stage TEXT NOT NULL,
    credit_id TEXT NOT NULL,
    matched_payment_ids TEXT NOT NULL,
    outcome TEXT NOT NULL,
    confidence REAL NOT NULL,
    delta_paise INTEGER NOT NULL,
    reason TEXT NOT NULL,
    rationale TEXT,
    llm_model TEXT,
    llm_cost_paise INTEGER,
    created_at TEXT NOT NULL
)
"""

_CREATE_RUN_ID_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id)"

_INSERT_SQL = """
INSERT INTO decisions (
    run_id, seed, stage, credit_id, matched_payment_ids, outcome,
    confidence, delta_paise, reason, rationale, llm_model, llm_cost_paise,
    created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the audit database and ensure the decisions
    table exists. Idempotent: safe to call on every run, including against
    an existing database — CREATE TABLE/INDEX IF NOT EXISTS never wipes
    prior rows."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_RUN_ID_INDEX_SQL)
    conn.commit()
    return conn


def record(conn: sqlite3.Connection, decision: Decision) -> None:
    """Write one Decision row."""
    conn.execute(
        _INSERT_SQL,
        (
            decision.run_id,
            decision.seed,
            decision.stage.value,
            decision.credit_id,
            json.dumps(decision.matched_payment_ids),
            decision.outcome.value,
            decision.confidence,
            decision.delta_paise,
            decision.reason,
            decision.rationale,
            decision.llm_model,
            decision.llm_cost_paise,
            decision.created_at.isoformat(),
        ),
    )
    conn.commit()


def _row_to_decision(row: sqlite3.Row) -> Decision:
    return Decision(
        run_id=row["run_id"],
        seed=row["seed"],
        stage=Stage(row["stage"]),
        credit_id=row["credit_id"],
        matched_payment_ids=json.loads(row["matched_payment_ids"]),
        outcome=DecisionOutcome(row["outcome"]),
        confidence=row["confidence"],
        delta_paise=row["delta_paise"],
        reason=row["reason"],
        rationale=row["rationale"],
        llm_model=row["llm_model"],
        llm_cost_paise=row["llm_cost_paise"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _query(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> list[Decision]:
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row
    rows = cursor.execute(sql, params).fetchall()
    return [_row_to_decision(row) for row in rows]


def fetch_decisions(conn: sqlite3.Connection, run_id: str) -> list[Decision]:
    """Return every Decision written during `run_id`, in insertion order."""
    return _query(conn, "SELECT * FROM decisions WHERE run_id = ? ORDER BY id", (run_id,))


def fetch_exceptions(conn: sqlite3.Connection, run_id: str | None = None) -> list[Decision]:
    """Return every Decision with outcome `exception`, for the exception
    report. `run_id=None` (the default) queries across every run recorded so
    far, since `unbatch exceptions` has no notion of a "current run" — it is
    a query over the whole table, never a separately maintained list."""
    if run_id is None:
        return _query(
            conn,
            "SELECT * FROM decisions WHERE outcome = ? ORDER BY id",
            (DecisionOutcome.EXCEPTION.value,),
        )
    return _query(
        conn,
        "SELECT * FROM decisions WHERE run_id = ? AND outcome = ? ORDER BY id",
        (run_id, DecisionOutcome.EXCEPTION.value),
    )
