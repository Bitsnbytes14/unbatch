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

# Bumped whenever the table's own constraints change in a way SQLite can't
# apply via ALTER TABLE (e.g. adding a CHECK) — see _migrate_schema, which
# rebuilds the table under PRAGMA user_version < _SCHEMA_VERSION rather than
# assuming every existing out/audit.db already has the current constraints.
_SCHEMA_VERSION = 1


def _create_table_sql(table_name: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    stage TEXT NOT NULL,
    credit_id TEXT NOT NULL,
    matched_payment_ids TEXT NOT NULL,
    outcome TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
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


_CREATE_TABLE_SQL = _create_table_sql("decisions")

_CREATE_RUN_ID_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id)"

# A stage returns at most one Decision per credit it processes (documented
# convention, CLAUDE.md § Conventions), so within one run a given
# (run_id, stage, credit_id) triple must be unique — this index makes that a
# schema guarantee instead of an assumption about every stage's own loop.
_CREATE_UNIQUE_CREDIT_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_run_stage_credit "
    "ON decisions(run_id, stage, credit_id)"
)

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


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Bring a pre-existing decisions table up to _SCHEMA_VERSION.

    SQLite has no `ALTER TABLE ... ADD CONSTRAINT`, so adding the confidence
    CHECK to a table that predates it needs the standard SQLite rebuild
    dance: create a new table with the current schema, copy every row across
    by name (so a table that also predates evidence_refs/human_review_required
    still works, since _ensure_columns has already backfilled those before
    this runs), drop the old table, rename the new one into place. Wrapped in
    one transaction so a failure partway through leaves the original table
    untouched rather than half-migrated.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= _SCHEMA_VERSION:
        return
    columns = ", ".join(row[1] for row in conn.execute("PRAGMA table_info(decisions)"))
    conn.execute("BEGIN")
    try:
        conn.execute(_create_table_sql("decisions_new"))
        conn.execute(f"INSERT INTO decisions_new ({columns}) SELECT {columns} FROM decisions")
        conn.execute("DROP TABLE decisions")
        conn.execute("ALTER TABLE decisions_new RENAME TO decisions")
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the audit database and ensure the decisions
    table exists with the current constraints. Idempotent: safe to call on
    every run, including against an existing database — CREATE TABLE/INDEX
    IF NOT EXISTS never wipes prior rows, `_ensure_columns` adds any column
    introduced after that database was first created, and `_migrate_schema`
    rebuilds the table in place (preserving every row) if it predates a
    constraint that can't be added via ALTER TABLE, such as the confidence
    CHECK."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    table_existed = _table_exists(conn, "decisions")
    conn.execute(_CREATE_TABLE_SQL)
    _ensure_columns(conn)
    if table_existed:
        _migrate_schema(conn)
    else:
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    conn.execute(_CREATE_RUN_ID_INDEX_SQL)
    conn.execute(_CREATE_UNIQUE_CREDIT_INDEX_SQL)
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


def _decision_to_params(decision: Decision) -> tuple[object, ...]:
    return (
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
    )


def record(conn: sqlite3.Connection, decision: Decision) -> None:
    """Write one Decision row, committing immediately."""
    conn.execute(_INSERT_SQL, _decision_to_params(decision))
    conn.commit()


def record_many(conn: sqlite3.Connection, decisions: list[Decision]) -> None:
    """Write every Decision in one transaction: a single executemany plus one
    commit, rather than one execute-and-commit per row via `record`.
    `bench --scale 5000` found per-decision commits dominating wall-clock
    (10.49s of 11.45s total) — this is the fix, used by the cascade runner
    in place of calling `record` once per decision within a stage. A
    failure partway through rolls back so the stage's rows are all-or-
    nothing, never half-written."""
    if not decisions:
        return
    try:
        conn.executemany(_INSERT_SQL, [_decision_to_params(d) for d in decisions])
    except Exception:
        conn.rollback()
        raise
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
