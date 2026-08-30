"""L0 — UTR exact match: resolves on UTR + amount + date all tying exactly,
and must never false-match a truncated/absent UTR or a break that only
partially matches (rounding/fee-tier delta, date skew)."""

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
from unbatch.stages import l0_utr

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


def test_matches_when_utr_amount_and_date_all_tie() -> None:
    batch = _batch()
    credit = _credit(f"NEFT-{batch.settlement_utr}-RAZORPAY SOFTWARE PVT")
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    [decision] = l0_utr.run([u], CTX)

    assert decision.outcome == DecisionOutcome.MATCHED
    assert decision.confidence == 1.00
    assert decision.stage == Stage.L0
    assert decision.matched_payment_ids == batch.payment_ids
    assert decision.delta_paise == 0


def test_truncated_utr_does_not_false_match() -> None:
    batch = _batch()
    truncated_narration = f"NEFT-{batch.settlement_utr[:10]}..."
    credit = _credit(truncated_narration)
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l0_utr.run([u], CTX) == []


def test_absent_utr_does_not_false_match() -> None:
    batch = _batch()
    credit = _credit("NEFT-XXXXXXXXXXXX-MISC SETTLEMENT CREDIT")
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l0_utr.run([u], CTX) == []


def test_utr_present_but_amount_mismatched_falls_through() -> None:
    """The rounding_delta / fee_tier_change shape: UTR intact, amount off
    by a delta — must not resolve here, it belongs at L3."""
    batch = _batch(net=10_000)
    credit = _credit(f"NEFT-{batch.settlement_utr}-RAZORPAY SOFTWARE PVT", amount=10_059)
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l0_utr.run([u], CTX) == []


def test_utr_and_amount_match_but_date_mismatched_falls_through() -> None:
    """The date_skew shape: UTR and amount both intact, only the posting
    date shifts — must not resolve here, it belongs at L2's wider window."""
    batch = _batch(day=date(2024, 1, 5))
    credit = _credit(
        f"NEFT-{batch.settlement_utr}-RAZORPAY SOFTWARE PVT", value_date=date(2024, 1, 6)
    )
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l0_utr.run([u], CTX) == []


def test_multiple_ambiguous_matches_fall_through() -> None:
    """Bias to exception over wrong match: if more than one batch ties
    exactly, this stage is not certain enough to pick one."""
    batch_a = _batch(utr="AXISP111111111111", net=10_000)
    batch_b = _batch(utr="HDFCR222222222222", net=10_000)
    narration = f"NEFT-{batch_a.settlement_utr}-{batch_b.settlement_utr}-RAZORPAY"
    credit = _credit(narration)
    u = UnresolvedCredit(credit=credit, expected_batches=[batch_a, batch_b], candidates=[])

    assert l0_utr.run([u], CTX) == []


def test_unrelated_credits_never_match_any_batch() -> None:
    batch = _batch()
    credit = _credit("NEFT-000000000000-VENDOR REFUND MISC", amount=25_000)
    u = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])

    assert l0_utr.run([u], CTX) == []


def test_only_unmatched_credits_are_absent_from_the_result() -> None:
    batch = _batch()
    matching = UnresolvedCredit(
        credit=_credit(f"IMPS/{batch.settlement_utr}/RZPY"),
        expected_batches=[batch],
        candidates=[],
    )
    non_matching = UnresolvedCredit(
        credit=_credit("NEFT-XXXXXXXXXXXX-MISC", amount=99),
        expected_batches=[batch],
        candidates=[],
    )

    decisions = l0_utr.run([matching, non_matching], CTX)

    assert [d.credit_id for d in decisions] == [matching.credit.txn_id]
