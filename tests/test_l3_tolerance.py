"""L3 — tolerance: checks each credit against its date-windowed expected
batches directly (no search), resolving only when exactly one batch is
within the fee-structure-derived band. See l3_tolerance.py's docstring for
why this replaced a composition search — FAILURES.md's 2026-08-30 entry."""

from __future__ import annotations

from datetime import date, datetime

from unbatch.models import (
    BankStatementRecord,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    SettlementLine,
    SettlementLineType,
    Stage,
    UnresolvedCredit,
)
from unbatch.stages import l3_tolerance
from unbatch.stages.l3_tolerance import TOLERANCE_FLOOR_PAISE, TOLERANCE_RATE, _tolerance_for

CTX = RunContext(run_id="run_test", seed=1)


def _batch(
    net: int,
    utr: str = "AXISP123456789012",
    payment_ids: list[str] | None = None,
    day: date = date(2024, 1, 5),
) -> ExpectedBatch:
    return ExpectedBatch(
        settlement_utr=utr,
        settlement_ids=["setl_1"],
        payment_ids=payment_ids or ["pay_1", "pay_2"],
        net_paise=net,
        window_start=day,
        window_end=day,
    )


def _credit(amount: int, value_date: date = date(2024, 1, 5)) -> BankStatementRecord:
    return BankStatementRecord(
        txn_id="txn_1",
        value_date=value_date,
        narration="test",
        credit_paise=amount,
        debit_paise=None,
        balance_paise=amount,
    )


def _line(
    net: int,
    payment_id: str = "pay_line",
    settlement_utr: str = "AXISP999999999999",
    day: date = date(2024, 1, 5),
) -> SettlementLine:
    return SettlementLine(
        settlement_id=f"setl_{payment_id}",
        settlement_utr=settlement_utr,
        payment_id=payment_id,
        type=SettlementLineType.PAYMENT,
        gross_paise=net,
        fee_paise=0,
        tax_paise=0,
        net_paise=net,
        settled_at=datetime.combine(day, datetime.min.time()),
    )


def _unresolved(
    credit: BankStatementRecord,
    batches: list[ExpectedBatch],
    candidate_lines: list[SettlementLine] | None = None,
) -> UnresolvedCredit:
    return UnresolvedCredit(
        credit=credit,
        expected_batches=batches,
        candidate_lines=candidate_lines or [],
        candidates=[],
    )


def test_tolerance_for_uses_the_floor_for_small_credits() -> None:
    assert _tolerance_for(1_000) == TOLERANCE_FLOOR_PAISE  # 0.6% of 1000 = 6, below the floor


def test_tolerance_for_uses_the_rate_for_larger_credits() -> None:
    assert _tolerance_for(10_000_000) == round(10_000_000 * TOLERANCE_RATE)


def test_resolves_a_batch_within_tolerance_and_records_the_delta() -> None:
    batch = _batch(net=300)
    u = _unresolved(_credit(305), [batch])  # within the 50-paise floor

    [decision] = l3_tolerance.run([u], CTX)

    assert decision.outcome == DecisionOutcome.MATCHED
    assert decision.confidence == 0.75
    assert decision.stage == Stage.L3
    assert decision.delta_paise == 5  # 305 - 300
    assert decision.matched_payment_ids == batch.payment_ids


def test_exact_match_also_resolves_with_zero_delta() -> None:
    batch = _batch(net=100)
    u = _unresolved(_credit(100), [batch])

    [decision] = l3_tolerance.run([u], CTX)
    assert decision.delta_paise == 0


def test_outside_the_band_falls_through() -> None:
    batch = _batch(net=100)
    u = _unresolved(_credit(100_000), [batch])  # nowhere close

    assert l3_tolerance.run([u], CTX) == []


def test_multiple_batches_within_tolerance_fall_through() -> None:
    """Bias to exception over wrong match: if more than one batch ties
    within the band, this stage is not certain enough to pick one."""
    batch_a = _batch(net=100, utr="AXISP111111111111", payment_ids=["pay_a"])
    batch_b = _batch(net=101, utr="HDFCR222222222222", payment_ids=["pay_b"])
    u = _unresolved(_credit(100), [batch_a, batch_b])

    assert l3_tolerance.run([u], CTX) == []


def test_batch_outside_date_window_is_excluded() -> None:
    out_of_window = _batch(net=100, day=date(2024, 1, 1))  # 4 days before D
    u = _unresolved(_credit(100, date(2024, 1, 5)), [out_of_window])

    assert l3_tolerance.run([u], CTX) == []


def test_batch_exactly_3_days_before_is_included() -> None:
    in_window = _batch(net=100, day=date(2024, 1, 2))  # exactly D-3
    u = _unresolved(_credit(100, date(2024, 1, 5)), [in_window])

    [decision] = l3_tolerance.run([u], CTX)
    assert decision.matched_payment_ids == in_window.payment_ids


def test_only_unmatched_credits_are_absent_from_the_result() -> None:
    matching = _unresolved(_credit(100), [_batch(net=100)])
    non_matching = _unresolved(_credit(999_999), [_batch(net=100)])

    decisions = l3_tolerance.run([matching, non_matching], CTX)
    assert [d.credit_id for d in decisions] == [matching.credit.txn_id]


def test_delta_matching_an_available_line_falls_through_unresolved() -> None:
    """The 2026-08-31 false-accept guard: a within-tolerance delta that exactly
    equals a real settlement line's net is a composition fact (a whole line
    left out), not fee/rounding noise — L3 must decline it, same as zero or
    multiple within-tolerance batches."""
    batch = _batch(net=1_030)
    missing_line = _line(net=30)
    u = _unresolved(_credit(1_000), [batch], candidate_lines=[missing_line])

    assert l3_tolerance.run([u], CTX) == []


def test_ordinary_drift_with_no_matching_line_still_resolves() -> None:
    """The guard must not block a genuine fee-tier/rounding delta just
    because *some* unrelated line exists in the pool — only an exact net
    match disqualifies the batch."""
    batch = _batch(net=1_030)
    unrelated_line = _line(net=45)  # present, but doesn't equal the delta (30)
    u = _unresolved(_credit(1_000), [batch], candidate_lines=[unrelated_line])

    [decision] = l3_tolerance.run([u], CTX)
    assert decision.outcome == DecisionOutcome.MATCHED
    assert decision.delta_paise == -30


def test_matching_line_outside_the_date_window_does_not_block_the_match() -> None:
    batch = _batch(net=1_030)
    out_of_window_line = _line(net=30, day=date(2024, 1, 1))  # 4 days before D
    u = _unresolved(_credit(1_000), [batch], candidate_lines=[out_of_window_line])

    [decision] = l3_tolerance.run([u], CTX)
    assert decision.outcome == DecisionOutcome.MATCHED


def test_delta_matches_a_negative_net_line_via_absolute_value() -> None:
    """net_paise is negative for a REFUND/CHARGEBACK line (fees.py) — the
    guard compares magnitudes so it still catches this shape."""
    batch = _batch(net=1_030)
    refund_line = _line(net=-30)
    u = _unresolved(_credit(1_000), [batch], candidate_lines=[refund_line])

    assert l3_tolerance.run([u], CTX) == []
