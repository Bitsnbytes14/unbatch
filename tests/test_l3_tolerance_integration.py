"""L0 -> L1 -> L2 -> L3 against the real seed-42 fixtures: rounding_delta
(the 1-paise floor case) and fee_tier_change (the ~Rs 59 ceiling case) both
resolve at L3 with their real delta recorded; ambiguous_composition and
unrelated_credit must still not resolve anywhere in the deterministic
cascade — they are the two credits meant to stay unresolved."""

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
        (l3_tolerance.run, False),  # L3 checks expected_batches directly, not lines
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


def test_ambiguous_composition_and_unrelated_credit_stay_unresolved() -> None:
    """The two credits this deterministic cascade is supposed to leave for
    L4: a genuine composition ambiguity, and a credit with nothing to tie to
    at all."""
    _decisions, remaining = _run_full_cascade()
    remaining_ids = {u.credit.txn_id for u in remaining}

    break_types = _ground_truth_break_types()
    ambiguous_id = next(
        t for t, bt in break_types.items() if bt == BreakType.AMBIGUOUS_COMPOSITION.value
    )
    unrelated_id = next(
        t for t, bt in break_types.items() if bt == BreakType.UNRELATED_CREDIT.value
    )

    assert ambiguous_id in remaining_ids
    assert unrelated_id in remaining_ids


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
