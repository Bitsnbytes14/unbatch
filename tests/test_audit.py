"""SQLite decision log: idempotent schema, insert, query, and reopening an
existing database without losing prior rows."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

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


def test_record_many_writes_every_decision_in_one_transaction(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    decisions = [_decision(credit_id="txn_1"), _decision(credit_id="txn_2")]

    audit.record_many(conn, decisions)

    fetched = audit.fetch_decisions(conn, "run_42_abc123")
    assert {d.credit_id for d in fetched} == {"txn_1", "txn_2"}


def test_record_many_with_an_empty_list_is_a_no_op(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    audit.record_many(conn, [])
    assert audit.fetch_decisions(conn, "run_42_abc123") == []


def test_record_many_rolls_back_the_whole_batch_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exception-safety: a failure partway through the batch must not leave
    some of its rows committed and others not — the stage's rows are
    all-or-nothing, same as record() is for a single row."""
    conn = audit.connect(tmp_path / "audit.db")
    good = _decision(credit_id="txn_1")
    also_good = _decision(credit_id="txn_2")

    real_params = audit._decision_to_params
    calls = {"n": 0}

    def _flaky_params(decision: Decision) -> tuple[object, ...]:
        calls["n"] += 1
        if calls["n"] == 2:
            params = list(real_params(decision))
            params[0] = None  # run_id is NOT NULL — forces the 2nd insert to fail
            return tuple(params)
        return real_params(decision)

    monkeypatch.setattr(audit, "_decision_to_params", _flaky_params)

    with pytest.raises(sqlite3.IntegrityError):
        audit.record_many(conn, [good, also_good])

    # the first row must not remain committed just because it came before
    # the one that failed
    assert audit.fetch_decisions(conn, "run_42_abc123") == []


def test_confidence_above_one_is_rejected_at_the_schema_level(tmp_path: Path) -> None:
    """Defense in depth: nothing upstream should ever produce this, but the
    schema itself must refuse it rather than trust every caller forever."""
    conn = audit.connect(tmp_path / "audit.db")
    params = list(audit._decision_to_params(_decision(confidence=0.5)))
    params[6] = 1.5  # confidence column, bypassing the Decision model entirely

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(audit._INSERT_SQL, params)


def test_confidence_below_zero_is_rejected_at_the_schema_level(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    params = list(audit._decision_to_params(_decision(confidence=0.5)))
    params[6] = -0.01

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(audit._INSERT_SQL, params)


def test_confidence_exactly_zero_and_one_are_both_accepted(tmp_path: Path) -> None:
    """BETWEEN is inclusive — L0 writes exactly 1.00 and the --no-llm
    terminal exception writes exactly 0.0; neither may be rejected."""
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision(credit_id="txn_low", confidence=0.0))
    audit.record(conn, _decision(credit_id="txn_high", confidence=1.0))

    fetched = {d.credit_id: d.confidence for d in audit.fetch_decisions(conn, "run_42_abc123")}
    assert fetched == {"txn_low": 0.0, "txn_high": 1.0}


def test_a_required_field_set_to_null_is_rejected_at_the_schema_level(tmp_path: Path) -> None:
    """run_id is NOT NULL on the table, independent of Decision's own typing —
    this checks the column constraint itself, not the pydantic model."""
    conn = audit.connect(tmp_path / "audit.db")
    params = list(audit._decision_to_params(_decision()))
    params[0] = None  # run_id

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(audit._INSERT_SQL, params)


def test_two_decisions_for_the_same_credit_in_the_same_stage_and_run_is_rejected(
    tmp_path: Path,
) -> None:
    """A stage returns at most one Decision per credit it processes
    (CLAUDE.md § Conventions) — the unique index turns that into a schema
    guarantee rather than trusting every stage's own loop forever."""
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision(credit_id="txn_1", stage=Stage.L0))

    with pytest.raises(sqlite3.IntegrityError):
        audit.record(
            conn,
            _decision(credit_id="txn_1", stage=Stage.L0, outcome=DecisionOutcome.EXCEPTION),
        )


def test_same_credit_may_appear_once_per_stage_across_the_cascade(tmp_path: Path) -> None:
    """The unique index is scoped to (run_id, stage, credit_id) — the same
    credit legitimately gets no more than one Decision per stage, but a
    fresh run or a different stage is a different key entirely."""
    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision(credit_id="txn_1", stage=Stage.L0, run_id="run_a"))
    # same credit, different stage, same run: allowed
    audit.record(conn, _decision(credit_id="txn_1", stage=Stage.L1, run_id="run_a"))
    # same credit, same stage, different run: allowed
    audit.record(conn, _decision(credit_id="txn_1", stage=Stage.L0, run_id="run_b"))

    assert len(audit.fetch_decisions(conn, "run_a")) == 2
    assert len(audit.fetch_decisions(conn, "run_b")) == 1


def test_connect_adds_the_confidence_check_to_a_pre_existing_database_without_losing_rows(
    tmp_path: Path,
) -> None:
    """A database created before the CHECK constraint existed must migrate in
    place on the next connect() — SQLite can't ALTER TABLE ADD CONSTRAINT, so
    this exercises the rebuild-and-copy path end to end, including that a
    row already in the table survives the rebuild."""
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
            evidence_refs TEXT,
            human_review_required INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    old_conn.execute(
        audit._INSERT_SQL,
        audit._decision_to_params(_decision(credit_id="txn_preexisting")),
    )
    old_conn.commit()
    old_conn.close()

    conn = audit.connect(db_path)

    # the pre-existing row must have survived the rebuild untouched
    [fetched] = audit.fetch_decisions(conn, "run_42_abc123")
    assert fetched.credit_id == "txn_preexisting"

    # and the new constraints must now be live
    params = list(audit._decision_to_params(_decision(credit_id="txn_bad")))
    params[6] = 2.0
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(audit._INSERT_SQL, params)


def test_migrate_schema_is_a_no_op_once_already_at_current_version(tmp_path: Path) -> None:
    """Idempotence: connecting twice must not rebuild the table a second
    time (and must not error doing nothing)."""
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision(credit_id="txn_1"))
    conn.close()

    reconnected = audit.connect(db_path)
    assert len(audit.fetch_decisions(reconnected, "run_42_abc123")) == 1
    version = reconnected.execute("PRAGMA user_version").fetchone()[0]
    assert version == audit._SCHEMA_VERSION


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
