"""`unbatch exceptions` backed by a query over the audit table — not a
separately maintained list. Tested against a hand-seeded audit DB, per
CLAUDE.md invariant 5."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from unbatch import audit
from unbatch.cli import app
from unbatch.models import Decision, DecisionOutcome, Stage

runner = CliRunner()


def _decision(**overrides: object) -> Decision:
    defaults: dict[object, object] = dict(
        run_id="run_42_abc123",
        seed=42,
        stage=Stage.L4,
        credit_id="txn_1",
        matched_payment_ids=[],
        outcome=DecisionOutcome.EXCEPTION,
        confidence=0.2,
        delta_paise=0,
        reason="pool_too_large",
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


def test_exceptions_against_empty_db_is_correct_not_broken(tmp_path: Path) -> None:
    """Before any stage has ever run, there is nothing in the audit table —
    that's the correct state, not a bug to paper over."""
    db_path = tmp_path / "audit.db"
    result = runner.invoke(app, ["exceptions", "--db", str(db_path)])

    assert result.exit_code == 0
    assert "No exceptions." in result.stdout


def test_exceptions_prints_hand_seeded_exception_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision(credit_id="txn_exception", reason="pool_too_large"))
    audit.record(
        conn, _decision(credit_id="txn_matched", outcome=DecisionOutcome.MATCHED, reason="")
    )
    conn.close()

    result = runner.invoke(app, ["exceptions", "--db", str(db_path)])

    assert result.exit_code == 0
    assert "txn_exception" in result.stdout
    assert "pool_too_large" in result.stdout
    assert "txn_matched" not in result.stdout


def test_exceptions_filters_by_run_id_when_given(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision(run_id="run_a", credit_id="txn_a"))
    audit.record(conn, _decision(run_id="run_b", credit_id="txn_b"))
    conn.close()

    result = runner.invoke(app, ["exceptions", "--db", str(db_path), "--run-id", "run_a"])

    assert result.exit_code == 0
    assert "txn_a" in result.stdout
    assert "txn_b" not in result.stdout


def test_exceptions_is_a_query_not_a_stored_list(tmp_path: Path) -> None:
    """Same audit.db, called twice with no changes in between: identical
    output, because it's re-derived from the table every time."""
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision())
    conn.close()

    first = runner.invoke(app, ["exceptions", "--db", str(db_path)])
    second = runner.invoke(app, ["exceptions", "--db", str(db_path)])

    assert first.stdout == second.stdout
