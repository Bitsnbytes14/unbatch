"""L0 -> L1 -> L2 against the real seed-42 fixtures: settlement_split,
date_skew, refund_in_window, chargeback_deduction, and duplicate_utr should
all resolve at L2; ambiguous_composition must NOT (multiple compositions
exist by design); fee_tier_change/rounding_delta must NOT (their delta is
real, not zero).

duplicate_utr resolving here is correct, not a false match: L2 composes
from individual settlement lines and never looks at UTR text at all, so the
narration-level ambiguity that stops L0 doesn't stop composition — each
duplicate_utr credit's own lines are still unambiguously its own. DATA_SPEC's
"should resolve at L4" describes where narration-only matching gets stuck,
not a ceiling on how well a smarter composition search is allowed to do.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from unbatch.cli import build_unresolved_credits, compute_expected_batches, load_input_data
from unbatch.models import BreakType, RunContext
from unbatch.stages import l0_utr, l1_exact, l2_compose

CTX = RunContext(run_id="run_test", seed=42)

SHOULD_RESOLVE_AT_L2 = {
    BreakType.SETTLEMENT_SPLIT.value,
    BreakType.DATE_SKEW.value,
    BreakType.REFUND_IN_WINDOW.value,
    BreakType.CHARGEBACK_DEDUCTION.value,
    BreakType.DUPLICATE_UTR.value,
}


def _ground_truth_break_types() -> dict[str, str]:
    data = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    return {c["txn_id"]: c["break_type"] for c in data["credits"]}


def _run_l0_l1_l2():
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    l0_decisions = l0_utr.run(unresolved, CTX)
    remaining = [
        u for u in unresolved if u.credit.txn_id not in {d.credit_id for d in l0_decisions}
    ]

    l1_decisions = l1_exact.run(remaining, CTX)
    remaining = [
        u for u in remaining if u.credit.txn_id not in {d.credit_id for d in l1_decisions}
    ]

    consumed = {pid for d in (*l0_decisions, *l1_decisions) for pid in d.matched_payment_ids}
    available = [line for line in settlements if line.payment_id not in consumed]
    remaining = [u.model_copy(update={"candidate_lines": available}) for u in remaining]

    started = time.monotonic()
    l2_decisions = l2_compose.run(remaining, CTX)
    elapsed = time.monotonic() - started

    return l2_decisions, remaining, elapsed


def test_l2_resolves_every_targeted_break_type() -> None:
    l2_decisions, _remaining, elapsed = _run_l0_l1_l2()
    l2_resolved_ids = {d.credit_id for d in l2_decisions if d.outcome.value == "matched"}

    break_types = _ground_truth_break_types()
    expected_ids = {
        txn_id for txn_id, bt in break_types.items() if bt in SHOULD_RESOLVE_AT_L2
    }

    missing = expected_ids - l2_resolved_ids
    assert missing == set(), f"L2 failed to resolve: {missing}"

    # ambiguous_composition and the tolerance-band breaks must NOT resolve
    # here — exact composition genuinely doesn't exist for the latter, and
    # the former has more than one, which L2 correctly declines to pick.
    must_not_resolve = {
        txn_id
        for txn_id, bt in break_types.items()
        if bt
        in (
            BreakType.AMBIGUOUS_COMPOSITION.value,
            BreakType.FEE_TIER_CHANGE.value,
            BreakType.ROUNDING_DELTA.value,
        )
    }
    wrongly_resolved = must_not_resolve & l2_resolved_ids
    assert wrongly_resolved == set(), f"L2 incorrectly resolved: {wrongly_resolved}"

    assert elapsed < 5.0, "L2 should resolve this dataset's real pools quickly"


def test_l2_never_false_matches_a_non_targeted_credit() -> None:
    l2_decisions, _remaining, _elapsed = _run_l0_l1_l2()
    l2_resolved_ids = {d.credit_id for d in l2_decisions if d.outcome.value == "matched"}

    break_types = _ground_truth_break_types()
    false_matches = {
        txn_id for txn_id in l2_resolved_ids if break_types.get(txn_id) not in SHOULD_RESOLVE_AT_L2
    }
    assert false_matches == set(), f"L2 false-matched: {false_matches}"
