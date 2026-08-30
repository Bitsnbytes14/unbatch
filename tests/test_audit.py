"""SQLite decision log: idempotent schema, insert, query, and reopening an
existing database without losing prior rows."""

from __future__ import annotations

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
        llm_model="claude-sonnet-5",
        llm_cost_paise=12,
    )
    audit.record(conn, decision)

    [fetched] = audit.fetch_decisions(conn, decision.run_id)
    assert fetched == decision


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
