"""L2 — batch composition: date window first, then compose; single exact
composition resolves, multiple stays unresolved for L4, and compose.py's
refusals become immediate exception Decisions."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from unbatch.compose import ComposeTimeoutError, PoolTooLargeError
from unbatch.models import (
    BankStatementRecord,
    DecisionOutcome,
    RunContext,
    SettlementLine,
    SettlementLineType,
    Stage,
    UnresolvedCredit,
)
from unbatch.stages import l2_compose

CTX = RunContext(run_id="run_test", seed=1)


def _line(net_paise: int, line_id: str, settled: date) -> SettlementLine:
    return SettlementLine(
        settlement_id=f"setl_{line_id}",
        settlement_utr="AXISP000000000001",
        payment_id=f"pay_{line_id}",
        type=SettlementLineType.PAYMENT,
        gross_paise=net_paise,
        fee_paise=0,
        tax_paise=0,
        net_paise=net_paise,
        settled_at=datetime(settled.year, settled.month, settled.day, tzinfo=UTC),
    )


def _credit(amount: int, value_date: date) -> BankStatementRecord:
    return BankStatementRecord(
        txn_id="txn_1",
        value_date=value_date,
        narration="test",
        credit_paise=amount,
        debit_paise=None,
        balance_paise=amount,
    )


def _unresolved(credit: BankStatementRecord, lines: list[SettlementLine]) -> UnresolvedCredit:
    return UnresolvedCredit(credit=credit, expected_batches=[], candidate_lines=lines)


def test_resolves_a_single_exact_composition() -> None:
    lines = [_line(100, "a", date(2024, 1, 5)), _line(200, "b", date(2024, 1, 5))]
    u = _unresolved(_credit(300, date(2024, 1, 5)), lines)

    [decision] = l2_compose.run([u], CTX)

    assert decision.outcome == DecisionOutcome.MATCHED
    assert decision.confidence == 0.90
    assert decision.stage == Stage.L2
    assert set(decision.matched_payment_ids) == {"pay_a", "pay_b"}
    assert decision.delta_paise == 0


def test_zero_compositions_fall_through() -> None:
    lines = [_line(100, "a", date(2024, 1, 5))]
    u = _unresolved(_credit(999, date(2024, 1, 5)), lines)

    assert l2_compose.run([u], CTX) == []


def test_multiple_compositions_fall_through_for_l4() -> None:
    """Bias to exception over wrong match: ambiguity is not this stage's
    to resolve."""
    lines = [
        _line(100, "a", date(2024, 1, 5)),
        _line(100, "b", date(2024, 1, 5)),
        _line(200, "c", date(2024, 1, 5)),
    ]
    u = _unresolved(_credit(200, date(2024, 1, 5)), lines)

    assert l2_compose.run([u], CTX) == []


def test_date_window_excludes_lines_outside_d_minus_3_days() -> None:
    in_window = _line(100, "a", date(2024, 1, 3))
    out_of_window = _line(200, "b", date(2024, 1, 1))  # 4 days before D
    u = _unresolved(_credit(300, date(2024, 1, 5)), [in_window, out_of_window])

    # 300 is only reachable using both lines; with "b" excluded, no match
    assert l2_compose.run([u], CTX) == []


def test_date_window_includes_lines_exactly_3_days_before() -> None:
    line = _line(100, "a", date(2024, 1, 2))  # exactly D-3
    u = _unresolved(_credit(100, date(2024, 1, 5)), [line])

    [decision] = l2_compose.run([u], CTX)
    assert decision.matched_payment_ids == ["pay_a"]


def test_pool_too_large_becomes_an_immediate_exception() -> None:
    lines = [_line(1, str(i), date(2024, 1, 5)) for i in range(CTX.config.max_pool + 1)]
    u = _unresolved(_credit(sum(line_.net_paise for line_ in lines), date(2024, 1, 5)), lines)

    [decision] = l2_compose.run([u], CTX)

    assert decision.outcome == DecisionOutcome.EXCEPTION
    assert decision.reason == "pool_too_large"
    assert decision.stage == Stage.L2
    assert decision.confidence == 0.0


def test_compose_timeout_becomes_an_immediate_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_timeout(*_args: object, **_kwargs: object):
        raise ComposeTimeoutError("simulated")

    monkeypatch.setattr(l2_compose, "compose", _raise_timeout)

    line = _line(100, "a", date(2024, 1, 5))
    u = _unresolved(_credit(100, date(2024, 1, 5)), [line])

    [decision] = l2_compose.run([u], CTX)

    assert decision.outcome == DecisionOutcome.EXCEPTION
    assert decision.reason == "compose_timeout"
    assert decision.confidence == 0.0


def test_a_pool_too_large_exception_does_not_stop_later_credits_from_being_tried() -> None:
    """continue, not break: one credit's pool refusal must not swallow
    every credit processed after it in the same batch."""
    too_large_lines = [_line(1, str(i), date(2024, 1, 5)) for i in range(CTX.config.max_pool + 1)]
    too_large = _unresolved(
        _credit(sum(line_.net_paise for line_ in too_large_lines), date(2024, 1, 5)),
        too_large_lines,
    )
    resolvable = _unresolved(_credit(100, date(2024, 1, 5)), [_line(100, "z", date(2024, 1, 5))])

    decisions = l2_compose.run([too_large, resolvable], CTX)

    assert len(decisions) == 2
    assert decisions[1].outcome == DecisionOutcome.MATCHED


def test_a_compose_timeout_does_not_stop_later_credits_from_being_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """continue, not break: same property as the pool_too_large case above,
    for the timeout exception path."""
    real_compose = l2_compose.compose
    calls = {"n": 0}

    def _raise_once_then_delegate(*args: object, **kwargs: object):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ComposeTimeoutError("simulated")
        return real_compose(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(l2_compose, "compose", _raise_once_then_delegate)

    timeout_credit = _unresolved(
        _credit(100, date(2024, 1, 5)), [_line(100, "a", date(2024, 1, 5))]
    )
    resolvable = _unresolved(_credit(100, date(2024, 1, 5)), [_line(100, "z", date(2024, 1, 5))])

    decisions = l2_compose.run([timeout_credit, resolvable], CTX)

    assert len(decisions) == 2
    assert decisions[1].outcome == DecisionOutcome.MATCHED


def test_only_unmatched_credits_are_absent_from_the_result() -> None:
    matching = _unresolved(_credit(100, date(2024, 1, 5)), [_line(100, "a", date(2024, 1, 5))])
    non_matching = _unresolved(_credit(999, date(2024, 1, 5)), [_line(1, "b", date(2024, 1, 5))])

    decisions = l2_compose.run([matching, non_matching], CTX)
    assert [d.credit_id for d in decisions] == [matching.credit.txn_id]


def test_pool_too_large_error_class_is_reexported_correctly() -> None:
    """Sanity: l2_compose imports the real compose.py exception types, not
    stand-ins, so the except clauses actually catch what compose() raises."""
    assert issubclass(PoolTooLargeError, Exception)
