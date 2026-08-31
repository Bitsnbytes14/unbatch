"""L0 -> L1 -> L2 -> L3 against the real seed-42 fixtures: rounding_delta
(the 1-paise floor case) and fee_tier_change (the ~Rs 59 ceiling case) both
resolve at L3 with their real delta recorded; every ambiguous_composition,
tolerance_ambiguous, and unrelated_credit instance must still not resolve
anywhere in the deterministic cascade — they are the 12 credits meant to
stay unresolved for L4 (D0a, FAILURES.md's 2026-08-31 entry)."""

from __future__ import annotations

import json
from pathlib import Path

from unbatch.cli import build_unresolved_credits, compute_expected_batches, load_input_data
from unbatch.models import BreakType, RunContext
from unbatch.stages import l0_utr, l1_exact, l2_compose, l3_tolerance

CTX = RunContext(run_id="run_test", seed=42)


def _ground_truth_break_types() -> dict[str, str]:
    data = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    return {c["txn_id"]: c["break_type"] for c in data["credits"]}


def _run_full_cascade():
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    all_decisions = []
    consumed: set[str] = set()

    for stage_fn, needs_pool in (
        (l0_utr.run, False),
        (l1_exact.run, False),
        (l2_compose.run, True),
        (l3_tolerance.run, True),  # 2026-09-03: also needs the pool for the missing-line guard
    ):
        if needs_pool:
            available = [line for line in settlements if line.payment_id not in consumed]
            unresolved = [u.model_copy(update={"candidate_lines": available}) for u in unresolved]

        decisions = stage_fn(unresolved, CTX)
        all_decisions.extend(decisions)
        resolved_ids = {d.credit_id for d in decisions}
        consumed.update(pid for d in decisions for pid in d.matched_payment_ids)
        unresolved = [u for u in unresolved if u.credit.txn_id not in resolved_ids]

    return all_decisions, unresolved


def test_l3_resolves_rounding_delta_and_fee_tier_change_with_their_real_delta() -> None:
    decisions, _remaining = _run_full_cascade()
    l3_matched = {d.credit_id: d for d in decisions if d.stage.value == "l3"}

    break_types = _ground_truth_break_types()
    rounding_id = next(t for t, bt in break_types.items() if bt == BreakType.ROUNDING_DELTA.value)
    fee_tier_id = next(
        t for t, bt in break_types.items() if bt == BreakType.FEE_TIER_CHANGE.value
    )

    assert rounding_id in l3_matched, "rounding_delta (the floor case) must resolve at L3"
    assert fee_tier_id in l3_matched, "fee_tier_change (the ceiling case) must resolve at L3"

    rounding_decision = l3_matched[rounding_id]
    fee_tier_decision = l3_matched[fee_tier_id]

    assert rounding_decision.confidence == 0.75
    assert abs(rounding_decision.delta_paise) == 1  # the guaranteed 1-paise gap

    assert fee_tier_decision.confidence == 0.75
    assert abs(fee_tier_decision.delta_paise) == 5900  # the guaranteed Rs 59.00 gap


def test_ambiguous_and_unrelated_credits_all_stay_unresolved() -> None:
    """The 12 credits this deterministic cascade is supposed to leave for
    L4 (D0a): every ambiguous_composition (7) and tolerance_ambiguous (4)
    instance — genuine ambiguities the exactly-one rule must decline rather
    than guess between — plus the one unrelated_credit with nothing to tie
    to at all."""
    _decisions, remaining = _run_full_cascade()
    remaining_ids = {u.credit.txn_id for u in remaining}

    break_types = _ground_truth_break_types()
    ambiguous_ids = {
        t for t, bt in break_types.items() if bt == BreakType.AMBIGUOUS_COMPOSITION.value
    }
    tolerance_ambiguous_ids = {
        t for t, bt in break_types.items() if bt == BreakType.TOLERANCE_AMBIGUOUS.value
    }
    unrelated_ids = {t for t, bt in break_types.items() if bt == BreakType.UNRELATED_CREDIT.value}

    assert len(ambiguous_ids) == 7
    assert len(tolerance_ambiguous_ids) == 4
    assert len(unrelated_ids) == 1
    assert ambiguous_ids <= remaining_ids
    assert tolerance_ambiguous_ids <= remaining_ids
    assert unrelated_ids <= remaining_ids
    assert len(remaining_ids) == 12


def test_no_false_matches_anywhere_in_the_deterministic_cascade() -> None:
    """The single most important property: nothing the cascade resolved
    points at the wrong payment_ids."""
    decisions, _remaining = _run_full_cascade()
    break_types = _ground_truth_break_types()

    gt_by_txn = {
        c["txn_id"]: set(c["payment_ids"])
        for c in json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))[
            "credits"
        ]
    }

    for decision in decisions:
        if decision.outcome.value != "matched":
            continue
        expected = gt_by_txn.get(decision.credit_id)
        assert expected is not None, f"matched a credit with no ground truth: {decision.credit_id}"
        assert set(decision.matched_payment_ids) == expected, (
            f"{decision.credit_id} ({break_types.get(decision.credit_id)}): "
            f"matched {decision.matched_payment_ids}, ground truth wants {expected}"
        )
