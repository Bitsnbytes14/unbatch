"""L1 — amount + date exact: resolves on the same test as L0 minus the UTR
requirement, so narration_mangled (UTR unreliable, amount/date still exact)
resolves here instead."""

from __future__ import annotations

from datetime import date

from unbatch.models import (
    BankStatementRecord,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    Stage,
    UnresolvedCredit,
)
from unbatch.stages import l1_exact

CTX = RunContext(run_id="run_test", seed=1)


def _batch(utr: str = "AXISP123456789012", net: int = 10_000, day: date = date(2024, 1, 5)):
    return ExpectedBatch(
        settlement_utr=utr,
        settlement_ids=["setl_1"],
        payment_ids=["pay_1", "pay_2"],
        net_paise=net,
        window_start=day,
        window_end=day,
    )


def _credit(narration: str, amount: int = 10_000, value_date: date = date(2024, 1, 5)):
    return BankStatementRecord(
        txn_id="txn_1",
        value_date=value_date,
        narration=narration,
        credit_paise=amount,
        debit_paise=None,
        balance_paise=amount,
    )


def test_matches_on_amount_and_date_even_with_truncated_utr() -> None:
    batch = _batch()
    credit = _credit(f"NEFT-{batch.settlement_utr[:10]}...")
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    [decision] = l1_exact.run([u], CTX)

    assert decision.outcome == DecisionOutcome.MATCHED
    assert decision.confidence == 0.98
    assert decision.stage == Stage.L1
    assert decision.matched_payment_ids == batch.payment_ids


def test_matches_with_no_utr_in_narration_at_all() -> None:
    batch = _batch()
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC SETTLEMENT CREDIT")
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert len(l1_exact.run([u], CTX)) == 1


def test_amount_mismatch_falls_through() -> None:
    """rounding_delta / fee_tier_change shape: amount off by a delta."""
    batch = _batch(net=10_000)
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC", amount=10_059)
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l1_exact.run([u], CTX) == []


def test_date_mismatch_falls_through() -> None:
    """date_skew shape: amount ties exactly but the posting date shifts."""
    batch = _batch(day=date(2024, 1, 5))
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC", value_date=date(2024, 1, 6))
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l1_exact.run([u], CTX) == []


def test_multiple_ambiguous_matches_fall_through() -> None:
    batch_a = _batch(utr="AXISP111111111111", net=10_000)
    batch_b = _batch(utr="HDFCR222222222222", net=10_000)
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC")
    u = UnresolvedCredit(credit=credit, expected_batches=[batch_a, batch_b], candidates=[])

    assert l1_exact.run([u], CTX) == []


def test_multi_day_window_excludes_a_date_before_the_start_boundary() -> None:
    """A genuinely ranged window (start != end), with the credit dated
    before the start — must not match even though it's still before the
    end, ruling out a mutant that only checks the end side."""
    batch = ExpectedBatch(
        settlement_utr="AXISP123456789012",
        settlement_ids=["setl_1"],
        payment_ids=["pay_1", "pay_2"],
        net_paise=10_000,
        window_start=date(2024, 1, 3),
        window_end=date(2024, 1, 5),
    )
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC", value_date=date(2024, 1, 1))
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l1_exact.run([u], CTX) == []


def test_multi_day_window_matches_strictly_between_the_boundaries() -> None:
    """Same ranged window, credit dated in the middle — must match, ruling
    out a mutant that requires exact equality with either boundary."""
    batch = ExpectedBatch(
        settlement_utr="AXISP123456789012",
        settlement_ids=["setl_1"],
        payment_ids=["pay_1", "pay_2"],
        net_paise=10_000,
        window_start=date(2024, 1, 3),
        window_end=date(2024, 1, 5),
    )
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC", value_date=date(2024, 1, 4))
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert len(l1_exact.run([u], CTX)) == 1


def test_unrelated_credit_never_matches() -> None:
    batch = _batch()
    credit = _credit("NEFT-000000000000-VENDOR REFUND MISC", amount=25_000)
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l1_exact.run([u], CTX) == []
