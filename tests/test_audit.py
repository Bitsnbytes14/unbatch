"""SQLite decision log: idempotent schema, insert, query, and reopening an
existing database without losing prior rows."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from unbatch import audit
from unbatch.models import Decision, DecisionOutcome, Stage


def _decision(**overrides: object) -> Decision:
    defaults: dict[object, object] = dict(
        run_id="run_42_abc123",
        seed=42,
        stage=Stage.L0,
        credit_id="txn_1",
        matched_payment_ids=["pay_1", "pay_2"],
        outcome=DecisionOutcome.MATCHED,
        confidence=1.0,
        delta_paise=0,
        reason="utr_exact_match",
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


def test_connect_creates_db_file_and_table(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    assert not db_path.exists()

    conn = audit.connect(db_path)
    assert db_path.exists()

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
    ).fetchall()
    assert len(tables) == 1


def test_connect_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn1 = audit.connect(db_path)
    audit.record(conn1, _decision())
    conn1.close()

    # reconnecting must not wipe or duplicate the schema/data
    conn2 = audit.connect(db_path)
    assert len(audit.fetch_decisions(conn2, "run_42_abc123")) == 1


def test_insert_and_query_round_trips_every_field(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    decision = _decision(
        stage=Stage.L4,
        outcome=DecisionOutcome.HUMAN_REVIEW,
        confidence=0.72,
        delta_paise=-150,
        reason="fee_tier_change",
        rationale="fee rate looks like it changed mid-window",
        llm_model="gpt-5-nano",
        llm_cost_paise=12,
    )
    audit.record(conn, decision)

    [fetched] = audit.fetch_decisions(conn, decision.run_id)
    assert fetched == decision


def test_insert_and_query_round_trips_evidence_refs_and_human_review(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    decision = _decision(
        stage=Stage.L4,
        reason="ambiguous_composition",
        evidence_refs=["pay_1", "settle_9"],
        human_review_required=True,
    )
    audit.record(conn, decision)

    [fetched] = audit.fetch_decisions(conn, decision.run_id)
    assert fetched == decision


def test_evidence_refs_and_human_review_default_to_none(tmp_path: Path) -> None:
    """A rules-stage decision (L0-L3) never sets these — round-tripping one
    must come back None, not an empty list or False."""
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision())

    [fetched] = audit.fetch_decisions(conn, "run_42_abc123")
    assert fetched.evidence_refs is None
    assert fetched.human_review_required is None


def test_connect_adds_missing_columns_to_a_pre_existing_database(tmp_path: Path) -> None:
    """A local out/audit.db created before evidence_refs/human_review_required
    existed must not break on the next insert — connect() has to migrate it
    in place rather than assume every existing database already has them."""
    db_path = tmp_path / "audit.db"
    old_conn = sqlite3.connect(db_path)
    old_conn.execute(
        """
        CREATE TABLE decisions (
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
            created_at TEXT NOT NULL
        )
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = audit.connect(db_path)
    audit.record(conn, _decision(evidence_refs=["pay_1"], human_review_required=False))

    [fetched] = audit.fetch_decisions(conn, "run_42_abc123")
    assert fetched.evidence_refs == ["pay_1"]
    assert fetched.human_review_required is False


def test_fetch_decisions_filters_by_run_id(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision(run_id="run_a", credit_id="txn_a"))
    audit.record(conn, _decision(run_id="run_b", credit_id="txn_b"))

    run_a = audit.fetch_decisions(conn, "run_a")
    assert [d.credit_id for d in run_a] == ["txn_a"]


def test_fetch_exceptions_filters_by_outcome(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision(credit_id="txn_matched", outcome=DecisionOutcome.MATCHED))
    audit.record(conn, _decision(credit_id="txn_exception", outcome=DecisionOutcome.EXCEPTION))
    audit.record(
        conn, _decision(credit_id="txn_review", outcome=DecisionOutcome.HUMAN_REVIEW)
    )

    exceptions = audit.fetch_exceptions(conn, "run_42_abc123")
    assert [d.credit_id for d in exceptions] == ["txn_exception"]


def test_fetch_exceptions_with_no_run_id_spans_all_runs(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(
        conn, _decision(run_id="run_a", credit_id="txn_a", outcome=DecisionOutcome.EXCEPTION)
    )
    audit.record(
        conn, _decision(run_id="run_b", credit_id="txn_b", outcome=DecisionOutcome.EXCEPTION)
    )
    audit.record(
        conn, _decision(run_id="run_b", credit_id="txn_c", outcome=DecisionOutcome.MATCHED)
    )

    exceptions = audit.fetch_exceptions(conn)
    assert {d.credit_id for d in exceptions} == {"txn_a", "txn_b"}


def test_reopen_after_close_preserves_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision(credit_id="txn_1"))
    audit.record(conn, _decision(credit_id="txn_2"))
    conn.close()

    reopened = audit.connect(db_path)
    decisions = audit.fetch_decisions(reopened, "run_42_abc123")
    assert {d.credit_id for d in decisions} == {"txn_1", "txn_2"}


def test_clear_run_removes_only_that_runs_rows(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision(run_id="run_a", credit_id="txn_a"))
    audit.record(conn, _decision(run_id="run_b", credit_id="txn_b"))

    audit.clear_run(conn, "run_a")

    assert audit.fetch_decisions(conn, "run_a") == []
    assert [d.credit_id for d in audit.fetch_decisions(conn, "run_b")] == ["txn_b"]


def test_rerunning_the_same_run_id_does_not_duplicate_rows(tmp_path: Path) -> None:
    """Regression: re-running the same seed against the same data derives
    the same run_id (audit.derive_run_id), so without clearing first, a
    second run would insert a second full set of Decisions alongside the
    first — silently doubling every count metrics.py or `unbatch
    exceptions` would later report."""
    conn = audit.connect(tmp_path / "audit.db")
    run_id = "run_42_abc123"

    for _ in range(2):
        audit.clear_run(conn, run_id)
        audit.record(conn, _decision(run_id=run_id, credit_id="txn_1"))
        audit.record(conn, _decision(run_id=run_id, credit_id="txn_2"))

    assert len(audit.fetch_decisions(conn, run_id)) == 2
