"""Tests for l4_llm.run. `unbatch.adjudicator.adjudicate` is always
monkeypatched here — no real network call, no API key needed."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from unbatch.models import (
    AdjudicationResult,
    BankStatementRecord,
    BreakReason,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    SettlementLine,
    SettlementLineType,
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


def _line(
    payment_id: str,
    net_paise: int,
    *,
    utr: str = "UTR_LINE",
    settled_date: date = date(2026, 3, 1),
) -> SettlementLine:
    return SettlementLine(
        settlement_id=f"setl_{payment_id}",
        settlement_utr=utr,
        payment_id=payment_id,
        type=SettlementLineType.PAYMENT,
        gross_paise=net_paise,
        fee_paise=0,
        tax_paise=0,
        net_paise=net_paise,
        settled_at=datetime.combine(settled_date, datetime.min.time(), tzinfo=UTC),
    )


def _ctx(*, cached: bool = False, llm_only: bool = False) -> RunContext:
    return RunContext(run_id=RUN_ID, seed=SEED, cached=cached, llm_only=llm_only)


def _stub_adjudicate(monkeypatch, result_and_cost=None, error=None, retried=False):
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
        result, cost = result_and_cost
        return result, cost, retried

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


def test_decision_carries_evidence_refs_and_human_review_required(monkeypatch) -> None:
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
                evidence_refs=["pay_x", "settle_9"],
                human_review_required=False,
            ),
            10,
        ),
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].evidence_refs == ["pay_x", "settle_9"]
    assert decisions[0].human_review_required is False


def test_adjudication_failed_becomes_an_exception_decision(monkeypatch) -> None:
    from unbatch import adjudicator

    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
    _stub_adjudicate(
        monkeypatch,
        error=adjudicator.AdjudicationFailedError("still malformed", cost_paise=3),
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert len(decisions) == 1
    assert decisions[0].outcome == DecisionOutcome.EXCEPTION
    assert decisions[0].reason == "adjudication_failed"
    assert decisions[0].matched_payment_ids == []
    # the real cost of both calls made (first attempt + retry), not 0 —
    # discarding it would understate spend on exactly the credits that cost
    # the most to adjudicate (see AdjudicationFailedError's docstring)
    assert decisions[0].llm_cost_paise == 3
    # degrading to adjudication_failed only ever happens after one retry
    assert decisions[0].llm_retried is True
    # no classification was ever produced, so there's nothing to report here
    assert decisions[0].evidence_refs is None
    assert decisions[0].human_review_required is None


def test_llm_retried_is_false_when_adjudicate_did_not_need_a_retry(monkeypatch) -> None:
    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
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
        retried=False,
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].llm_retried is False


def test_llm_retried_is_true_when_adjudicate_succeeded_after_a_retry(monkeypatch) -> None:
    credit = _credit()
    batch = _batch()
    unresolved = UnresolvedCredit(credit=credit, expected_batches=[batch], candidates=[])
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
        retried=True,
    )

    decisions = l4_llm.run([unresolved], _ctx())

    assert decisions[0].llm_retried is True


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


def _result(*, confidence: float, human_review_required: bool = False, reason=BreakReason.OTHER):
    return AdjudicationResult(
        break_reason=reason,
        proposed_resolution="x",
        confidence=confidence,
        evidence_refs=[],
        human_review_required=human_review_required,
    )


def test_a_single_exact_composition_is_used_as_the_primary_match(monkeypatch) -> None:
    """Exactly one exact-sum subset (only possible when L2 hasn't already
    claimed it — i.e. --llm-only, where L2 never ran) is a confident,
    rules-established match: matched_payment_ids should be that subset's
    ids, not some unrelated whole-batch guess."""
    credit = _credit(credit_paise=1000)
    lines = [_line("pay_a", 400), _line("pay_b", 600)]  # sums to exactly 1000
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[_batch()], candidate_lines=lines, candidates=[]
    )
    calls = _stub_adjudicate(
        monkeypatch, result_and_cost=(_result(confidence=0.9), 5)
    )

    decisions = l4_llm.run([unresolved], _ctx(llm_only=False))

    assert calls[0]["expected_batch"].payment_ids == ["pay_a", "pay_b"]
    assert calls[0]["delta_paise"] == 0
    assert decisions[0].outcome == DecisionOutcome.MATCHED
    assert decisions[0].matched_payment_ids == ["pay_a", "pay_b"]


def test_two_exact_compositions_force_exception_regardless_of_confidence(monkeypatch) -> None:
    """Provable ambiguity (>=2 exact-sum subsets) must never resolve to a
    specific pick, even if the model reports high confidence and no
    human_review_required — bias to exception over an arbitrary coin flip."""
    credit = _credit(credit_paise=1000)
    lines = [
        _line("pay_a", 400),
        _line("pay_b", 600),  # a+b = 1000
        _line("pay_c", 1000),  # c alone = 1000, a second exact composition
    ]
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[_batch()], candidate_lines=lines, candidates=[]
    )
    _stub_adjudicate(
        monkeypatch,
        result_and_cost=(_result(confidence=0.99, human_review_required=False), 5),
    )

    decisions = l4_llm.run([unresolved], _ctx(llm_only=False))

    assert decisions[0].outcome == DecisionOutcome.EXCEPTION
    assert decisions[0].matched_payment_ids == []
    # the model's classification is still recorded even though the outcome
    # is forced to exception
    assert decisions[0].reason == BreakReason.OTHER.value


def test_two_exact_compositions_surfaces_the_other_as_a_candidate(monkeypatch) -> None:
    credit = _credit(credit_paise=1000)
    lines = [_line("pay_a", 400), _line("pay_b", 600), _line("pay_c", 1000)]
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[_batch()], candidate_lines=lines, candidates=[]
    )
    calls = _stub_adjudicate(monkeypatch, result_and_cost=(_result(confidence=0.5), 5))

    l4_llm.run([unresolved], _ctx(llm_only=False))

    all_candidate_ids = {pid for c in calls[0]["candidates"] for pid in c.payment_ids}
    # whichever subset wasn't picked as primary shows up as a candidate
    assert all_candidate_ids == {"pay_a", "pay_b"} or all_candidate_ids == {"pay_c"}


def test_llm_only_never_runs_the_composition_search(monkeypatch) -> None:
    """Even when candidate_lines would produce an exact match, --llm-only
    must ignore it entirely and fall back to the whole-batch heuristic —
    re-deriving L2's search here would smuggle a rules capability into the
    arm that's supposed to have none (and would be far too slow across all
    ~105 credits with an unpruned pool)."""
    credit = _credit(credit_paise=1000)
    lines = [_line("pay_a", 400), _line("pay_b", 600)]  # would exactly compose 1000
    batch = _batch(payment_ids=["pay_whole_batch"])
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[batch], candidate_lines=lines, candidates=[]
    )
    calls = _stub_adjudicate(monkeypatch, result_and_cost=(_result(confidence=0.9), 5))

    decisions = l4_llm.run([unresolved], _ctx(llm_only=True))

    assert calls[0]["expected_batch"].payment_ids == ["pay_whole_batch"]
    assert decisions[0].matched_payment_ids == ["pay_whole_batch"]


def test_no_exact_composition_falls_back_to_whole_batch_heuristic(monkeypatch) -> None:
    credit = _credit(credit_paise=1000)
    lines = [_line("pay_a", 999)]  # doesn't sum to 1000 -> no exact composition
    batch = _batch(payment_ids=["pay_whole_batch"], net_paise=1000)
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[batch], candidate_lines=lines, candidates=[]
    )
    calls = _stub_adjudicate(monkeypatch, result_and_cost=(_result(confidence=0.9), 5))

    decisions = l4_llm.run([unresolved], _ctx(llm_only=False))

    assert calls[0]["expected_batch"].payment_ids == ["pay_whole_batch"]
    assert decisions[0].matched_payment_ids == ["pay_whole_batch"]


def test_a_pool_too_large_for_compose_falls_back_to_whole_batch_heuristic(monkeypatch) -> None:
    """compose() refusing to search (pool too large) is not evidence of "no
    exact match" — it must fall back exactly like a genuine non-match."""
    credit = _credit(credit_paise=1000)
    lines = [_line(f"pay_{i}", 1) for i in range(60)]  # over MAX_POOL (48)
    batch = _batch(payment_ids=["pay_whole_batch"], net_paise=1000)
    unresolved = UnresolvedCredit(
        credit=credit, expected_batches=[batch], candidate_lines=lines, candidates=[]
    )
    calls = _stub_adjudicate(monkeypatch, result_and_cost=(_result(confidence=0.9), 5))

    decisions = l4_llm.run([unresolved], _ctx(llm_only=False))

    assert calls[0]["expected_batch"].payment_ids == ["pay_whole_batch"]
    assert decisions[0].matched_payment_ids == ["pay_whole_batch"]


def test_synthetic_batch_from_lines_uses_the_subsets_own_sum() -> None:
    lines = [_line("pay_a", 400, utr="UTR_A"), _line("pay_b", 600, utr="UTR_A")]
    batch = l4_llm._synthetic_batch_from_lines(lines)
    assert batch.settlement_utr == "UTR_A"
    assert batch.payment_ids == ["pay_a", "pay_b"]
    assert batch.net_paise == 1000


def test_synthetic_batch_from_lines_joins_utrs_when_lines_span_more_than_one() -> None:
    lines = [_line("pay_a", 400, utr="UTR_A"), _line("pay_b", 600, utr="UTR_B")]
    batch = l4_llm._synthetic_batch_from_lines(lines)
    assert batch.settlement_utr == "UTR_A+UTR_B"
