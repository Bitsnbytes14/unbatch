"""Cascade runner: threads unresolved items through stages in order, writes
every Decision to the audit log immediately, and — under --no-llm —
terminates anything still unresolved after the last real stage as an
exception (L4 stays a stub this session).

Uses fake stage functions rather than the real L0-L3 implementations, so
the runner's own threading/audit behaviour is verified independent of
what any particular stage decides.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from unbatch import audit
from unbatch.cli import run_cascade
from unbatch.models import (
    BankStatementRecord,
    Decision,
    DecisionOutcome,
    RunContext,
    Stage,
    UnresolvedCredit,
)


def _credit(txn_id: str, amount: int = 1000) -> BankStatementRecord:
    return BankStatementRecord(
        txn_id=txn_id,
        value_date=datetime(2024, 1, 1).date(),
        narration="test narration",
        credit_paise=amount,
        debit_paise=None,
        balance_paise=amount,
    )


def _unresolved(txn_id: str) -> UnresolvedCredit:
    return UnresolvedCredit(credit=_credit(txn_id), expected_batches=[], candidates=[])


def _decision_for(u: UnresolvedCredit, stage: Stage, ctx: RunContext) -> Decision:
    return Decision(
        run_id=ctx.run_id,
        seed=ctx.seed,
        stage=stage,
        credit_id=u.credit.txn_id,
        matched_payment_ids=["pay_x"],
        outcome=DecisionOutcome.MATCHED,
        confidence=1.0,
        delta_paise=0,
        reason="test_match",
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _fake_stage(resolves: set[str], stage: Stage):
    def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
        return [_decision_for(u, stage, ctx) for u in unresolved if u.credit.txn_id in resolves]

    return run


def _ctx(no_llm: bool = True) -> RunContext:
    return RunContext(run_id="run_test", seed=1, no_llm=no_llm)


def test_runner_threads_only_unresolved_items_forward(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    ctx = _ctx()
    unresolved = [_unresolved("a"), _unresolved("b"), _unresolved("c")]

    stage_sequence = (
        (Stage.L0, _fake_stage({"a"}, Stage.L0)),
        (Stage.L1, _fake_stage({"b"}, Stage.L1)),
        (Stage.L2, _fake_stage(set(), Stage.L2)),
        (Stage.L3, _fake_stage(set(), Stage.L3)),
    )

    counts = run_cascade(ctx, unresolved, conn, stage_sequence=stage_sequence)

    assert counts["l0"] == 1
    assert counts["l1"] == 1
    assert counts["l2"] == 0
    assert counts["l3"] == 0
    assert counts["terminal_exception"] == 1  # "c" never resolved


def test_every_resolution_writes_an_audit_row(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    ctx = _ctx()
    unresolved = [_unresolved("a")]
    stage_sequence = ((Stage.L0, _fake_stage({"a"}, Stage.L0)),)

    run_cascade(ctx, unresolved, conn, stage_sequence=stage_sequence)

    decisions = audit.fetch_decisions(conn, ctx.run_id)
    assert len(decisions) == 1
    assert decisions[0].credit_id == "a"
    assert decisions[0].outcome == DecisionOutcome.MATCHED


def test_no_llm_terminal_exception_covers_every_unresolved_credit(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    ctx = _ctx(no_llm=True)
    unresolved = [_unresolved("a"), _unresolved("b")]

    counts = run_cascade(ctx, unresolved, conn, stage_sequence=())

    assert counts["terminal_exception"] == 2
    decisions = audit.fetch_decisions(conn, ctx.run_id)
    assert len(decisions) == 2
    assert all(d.outcome == DecisionOutcome.EXCEPTION for d in decisions)
    assert all(d.stage == Stage.L4 for d in decisions)
    assert all(d.reason == "no_llm_unresolved" for d in decisions)


def test_without_no_llm_unresolved_items_get_no_terminal_decision(tmp_path: Path) -> None:
    """Without --no-llm, the runner must not invent an L4 decision — that's
    L4's own job once it exists, not the runner's fallback."""
    conn = audit.connect(tmp_path / "audit.db")
    ctx = _ctx(no_llm=False)
    unresolved = [_unresolved("a")]

    counts = run_cascade(ctx, unresolved, conn, stage_sequence=())

    assert "terminal_exception" not in counts
    assert audit.fetch_decisions(conn, ctx.run_id) == []


def test_stages_never_see_already_resolved_credits(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    ctx = _ctx()
    unresolved = [_unresolved("a")]

    seen_by_second_stage: list[str] = []

    def second_stage(unresolved_list: list[UnresolvedCredit], _ctx: RunContext) -> list[Decision]:
        seen_by_second_stage.extend(u.credit.txn_id for u in unresolved_list)
        return []

    stage_sequence = (
        (Stage.L0, _fake_stage({"a"}, Stage.L0)),
        (Stage.L1, second_stage),
    )
    run_cascade(ctx, unresolved, conn, stage_sequence=stage_sequence)
    assert seen_by_second_stage == []


def test_every_credit_gets_exactly_one_decision_end_to_end(tmp_path: Path) -> None:
    conn = audit.connect(tmp_path / "audit.db")
    ctx = _ctx()
    unresolved = [_unresolved(str(i)) for i in range(5)]
    stage_sequence = (
        (Stage.L0, _fake_stage({"0", "1"}, Stage.L0)),
        (Stage.L1, _fake_stage({"2"}, Stage.L1)),
    )
    run_cascade(ctx, unresolved, conn, stage_sequence=stage_sequence)
    decisions = audit.fetch_decisions(conn, ctx.run_id)
    assert {d.credit_id for d in decisions} == {"0", "1", "2", "3", "4"}
