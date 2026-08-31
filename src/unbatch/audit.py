"""SQLite decision log — the audit trail every stage writes to before
returning. No stage may resolve an item without a Decision row landing here.

The exception report and the HTML report are queries over this table, never
separately maintained lists (ARCHITECTURE.md § Audit trail).

`derive_run_id` ties a run's id to both its seed and the actual bytes of the
input CSVs, so a run_id only ever means "this seed against this exact data" —
two runs are safely diffable, and regenerating the fixtures under an
unchanged seed number can never be mistaken for the same run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from unbatch.models import Decision, DecisionOutcome, Stage

DEFAULT_DB_PATH = Path("out/audit.db")
DEFAULT_DATA_DIR = Path("data")

_INPUT_FILENAMES = ("order_ledger.csv", "settlement_report.csv", "bank_statement.csv")

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
    llm_retried INTEGER NOT NULL DEFAULT 0,
    evidence_refs TEXT,
    human_review_required INTEGER,
    created_at TEXT NOT NULL
)
"""

_CREATE_RUN_ID_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id)"

# Columns added after the table's first release — CREATE TABLE IF NOT EXISTS
# only creates a fresh table, so a pre-existing local out/audit.db (gitignored,
# never shipped) needs these added explicitly or every insert against it would
# fail with "table decisions has no column named ...".
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("evidence_refs", "TEXT"),
    ("human_review_required", "INTEGER"),
)

_INSERT_SQL = """
INSERT INTO decisions (
    run_id, seed, stage, credit_id, matched_payment_ids, outcome,
    confidence, delta_paise, reason, rationale, llm_model, llm_cost_paise,
    llm_retried, evidence_refs, human_review_required, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    for name, sql_type in _ADDED_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {name} {sql_type}")


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the audit database and ensure the decisions
    table exists. Idempotent: safe to call on every run, including against
    an existing database — CREATE TABLE/INDEX IF NOT EXISTS never wipes
    prior rows, and `_ensure_columns` adds any column introduced after that
    database was first created rather than erroring on the next insert."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_RUN_ID_INDEX_SQL)
    _ensure_columns(conn)
    conn.commit()
    return conn


def clear_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Delete every Decision previously recorded for `run_id`. `run_id` is
    derived deterministically from seed + input hash (see `derive_run_id`),
    so re-running the same seed against the same data always produces the
    same run_id — without this, a second run would insert a second full set
    of Decisions alongside the first rather than replacing them, silently
    doubling every count anything downstream (metrics.py, `unbatch
    exceptions`) queries. Call before a cascade run starts writing."""
    conn.execute("DELETE FROM decisions WHERE run_id = ?", (run_id,))
    conn.commit()


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
            int(decision.llm_retried),
            None if decision.evidence_refs is None else json.dumps(decision.evidence_refs),
            None if decision.human_review_required is None else int(decision.human_review_required),
            decision.created_at.isoformat(),
        ),
    )
    conn.commit()


def _row_to_decision(row: sqlite3.Row) -> Decision:
    evidence_refs_json = row["evidence_refs"]
    human_review_value = row["human_review_required"]
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
        llm_retried=bool(row["llm_retried"]),
        evidence_refs=None if evidence_refs_json is None else json.loads(evidence_refs_json),
        human_review_required=None if human_review_value is None else bool(human_review_value),
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


def derive_run_id(seed: int, data_dir: Path = DEFAULT_DATA_DIR, *, arm: str = "no_llm") -> str:
    """Derive a run_id deterministically from `seed`, `arm`, and a hash of
    the input CSVs actually present in `data_dir`. Two calls with the same
    seed, arm, and file bytes always produce the same run_id; a changed
    seed, arm, or input data always produces a different one.

    `arm` (e.g. "no_llm", "with_llm", "llm_only") is part of the hash
    because METRICS.md's ablation runs the same seed through three
    different arms and needs to compare all three afterward — without it,
    `--no-llm` and the default (with-LLM) run against the same seed would
    derive the identical run_id, and the second run's `audit.clear_run`
    would silently delete the first arm's results before the ablation ever
    got to compare them.
    """
    hasher = hashlib.sha256()
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(arm.encode("utf-8"))
    for filename in _INPUT_FILENAMES:
        hasher.update(filename.encode("utf-8"))
        hasher.update((data_dir / filename).read_bytes())
    digest = hasher.hexdigest()[:12]
    return f"run_{seed}_{arm}_{digest}"
