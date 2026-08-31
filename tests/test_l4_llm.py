"""Tests for l4_llm.run. `unbatch.adjudicator.adjudicate` is always
monkeypatched here — no real network call, no API key needed."""

from __future__ import annotations

from datetime import date

import pytest

from unbatch.models import (
    AdjudicationResult,
    BankStatementRecord,
    BreakReason,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    UnresolvedCredit,
)
from unbatch.stages import l4_llm

RUN_ID = "run_test"
SEED = 42


def _credit(txn_id: str = "TXN1", value_date: date = date(2026, 3, 1), credit_paise: int = 418332):
    return BankStatementRecord(
        txn_id=txn_id,
        value_date=value_date,
        narration="NEFT-UTR000123-RAZORPAY",
        credit_paise=credit_paise,
        debit_paise=None,
        balance_paise=1000000,
    )


def _batch(
    utr: str = "UTR000123",
    net_paise: int = 419107,
    window_start: date = date(2026, 2, 27),
    window_end: date = date(2026, 3, 1),
    payment_ids: list[str] | None = None,
):
    return ExpectedBatch(
        settlement_utr=utr,
        settlement_ids=[f"setl_{utr}"],
        payment_ids=payment_ids or [f"pay_{utr}"],
        net_paise=net_paise,
        window_start=window_start,
        window_end=window_end,
    )


def _ctx(*, cached: bool = False) -> RunContext:
    return RunContext(run_id=RUN_ID, seed=SEED, cached=cached)


def _stub_adjudicate(monkeypatch, result_and_cost=None, error=None):
    calls = []

    def _fake(credit, expected_batch, delta_paise, candidates, *, cached, cache_dir=None):
        calls.append(
            {
                "credit": credit,
                "expected_batch": expected_batch,
                "delta_paise": delta_paise,
                "candidates": candidates,
                "cached": cached,
            }
        )
        if error is not None:
            raise error
        return result_and_cost

    monkeypatch.setattr(l4_llm.adjudicator, "adjudicate", _fake)
    return calls


def test_picks_the_closest_windowed_batch_as_the_primary_batch(monkeypatch) -> None:
    credit = _credit(credit_paise=418332)
    near = _batch(utr="UTR_NEAR", net_paise=418400)  # delta -68, in window
    far = _batch(
        utr="UTR_FAR",
        net_paise=500000,
        window_start=date(2026, 2, 26),
        window_end=date(2026, 2, 28),
    )
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[far, near], candidates=[])

    calls = _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.ROUNDING_DELTA,
                proposed_resolution="accept",
                confidence=0.95,
                evidence_refs=[],
                human_review_required=False,
            ),
            10,
        ),
    )

    l4_llm.run([unresolved], _ctx())

    assert calls[0]["expected_batch"].settlement_utr == "UTR_NEAR"
    assert calls[0]["delta_paise"] == -68


def test_surfaces_other_windowed_batches_as_candidates(monkeypatch) -> None:
    credit = _credit(credit_paise=418332)
    near = _batch(utr="UTR_NEAR", net_paise=418400)
    also_near = _batch(
        utr="UTR_ALSO",
        net_paise=418500,
        window_start=date(2026, 2, 27),
        window_end=date(2026, 3, 1),
    )
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[near, also_near], candidates=[]
    )
    calls = _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.AMBIGUOUS_COMPOSITION,
                proposed_resolution="review",
                confidence=0.5,
                evidence_refs=[],
                human_review_required=False,
            ),
            10,
        ),
    )

    l4_llm.run([unresolved], _ctx())

    candidate_utrs = [c.hint for c in calls[0]["candidates"]]
    assert any("UTR_ALSO" in hint for hint in candidate_utrs)


def test_falls_back_to_nearest_batch_when_nothing_is_in_the_date_window(monkeypatch) -> None:
    credit = _credit(value_date=date(2026, 3, 1), credit_paise=999999)
    distant = _batch(
        utr="UTR_DISTANT",
        net_paise=1,
        window_start=date(2026, 1, 1),
        window_end=date(2026, 1, 2),
    )
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[distant], candidates=[])
    calls = _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.UNRELATED_CREDIT,
                proposed_resolution="leave as exception",
                confidence=0.95,
                evidence_refs=[],
                human_review_required=False,
            ),
            10,
        ),
    )

    l4_llm.run([unresolved], _ctx())

    assert calls[0]["expected_batch"].settlement_utr == "UTR_DISTANT"


@pytest.mark.parametrize(
    ("confidence", "human_review_required", "expected_outcome"),
    [
        (0.95, False, DecisionOutcome.MATCHED),
        (0.85, False, DecisionOutcome.MATCHED),
        (0.90, True, DecisionOutcome.HUMAN_REVIEW),  # model's own flag downgrades a high band
        (0.70, False, DecisionOutcome.HUMAN_REVIEW),
        (0.60, False, DecisionOutcome.HUMAN_REVIEW),
        (0.59, False, DecisionOutcome.EXCEPTION),
        (0.10, False, DecisionOutcome.EXCEPTION),
    ],
)
def test_confidence_bands_set_the_outcome(
    monkeypatch, confidence, human_review_required, expected_outcome
) -> None:
    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.FEE_TIER_CHANGE,
                proposed_resolution="x",
                confidence=confidence,
                evidence_refs=[],
                human_review_required=human_review_required,
            ),
            10,
        ),
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].outcome == expected_outcome


def test_exception_outcome_has_no_matched_payment_ids(monkeypatch) -> None:
    credit = _credit()
    batch = _batch(payment_ids=["pay_x"])
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.UNRELATED_CREDIT,
                proposed_resolution="x",
                confidence=0.1,
                evidence_refs=[],
                human_review_required=False,
            ),
            10,
        ),
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].matched_payment_ids == []


def test_matched_outcome_uses_the_primary_batchs_payment_ids(monkeypatch) -> None:
    credit = _credit()
    batch = _batch(payment_ids=["pay_x", "pay_y"])
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.FEE_TIER_CHANGE,
                proposed_resolution="x",
                confidence=0.95,
                evidence_refs=[],
                human_review_required=False,
            ),
            10,
        ),
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].matched_payment_ids == ["pay_x", "pay_y"]


def test_adjudication_failed_becomes_an_exception_decision(monkeypatch) -> None:
    from unbatch import adjudicator

    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    _stub_adjudicate(monkeypatch, error=adjudicator.AdjudicationFailedError("still malformed"))

    decisions = l4_llm.run([unresolved], _ctx())

    assert len(decisions) == 1
    assert decisions[0].outcome == DecisionOutcome.EXCEPTION
    assert decisions[0].reason == "adjudication_failed"
    assert decisions[0].matched_payment_ids == []


def test_every_credit_gets_exactly_one_decision(monkeypatch) -> None:
    credits = [
        UnresolvedCredit(
            credit=_credit(txn_id=f"TXN{i}"), expected_batches=[_batch()], candidates=[]
        )
        for i in range(3)
    ]
    _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.OTHER,
                proposed_resolution="x",
                confidence=0.9,
                evidence_refs=[],
                human_review_required=False,
            ),
            5,
        ),
    )

    decisions = l4_llm.run(credits, _ctx())

    assert {d.credit_id for d in decisions} == {"TXN0", "TXN1", "TXN2"}


def test_decision_carries_llm_model_and_cost(monkeypatch) -> None:
    from unbatch import adjudicator

    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.FEE_TIER_CHANGE,
                proposed_resolution="x",
                confidence=0.95,
                evidence_refs=[],
                human_review_required=False,
            ),
            42,
        ),
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].llm_model == adjudicator.MODEL
    assert decisions[0].llm_cost_paise == 42


def test_ctx_cached_is_forwarded_to_adjudicate(monkeypatch) -> None:
    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    calls = _stub_adjudicate(
        monkeypatch,
        result_and_cost=(
            AdjudicationResult(
                break_reason=BreakReason.OTHER,
                proposed_resolution="x",
                confidence=0.9,
                evidence_refs=[],
                human_review_required=False,
            ),
            5,
        ),
    )

    l4_llm.run([unresolved], _ctx(cached=True))

    assert calls[0]["cached"] is True
