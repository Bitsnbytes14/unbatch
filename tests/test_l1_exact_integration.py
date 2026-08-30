"""L0 then L1 against the real seed-42 fixtures: every `narration_mangled`
credit resolves at L1 (not earlier, not never), and L1 never false-matches
anything else."""

from __future__ import annotations

import json
from pathlib import Path

from unbatch.cli import build_unresolved_credits, compute_expected_batches, load_input_data
from unbatch.models import BreakType, RunContext
from unbatch.stages import l0_utr, l1_exact

CTX = RunContext(run_id="run_test", seed=42)


def _ground_truth_break_types() -> dict[str, str]:
    data = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    return {c["txn_id"]: c["break_type"] for c in data["credits"]}


def test_l1_resolves_every_narration_mangled_credit_and_nothing_else() -> None:
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    l0_decisions = l0_utr.run(unresolved, CTX)
    l0_resolved_ids = {d.credit_id for d in l0_decisions}
    remaining = [u for u in unresolved if u.credit.txn_id not in l0_resolved_ids]

    l1_decisions = l1_exact.run(remaining, CTX)
    l1_resolved_ids = {d.credit_id for d in l1_decisions}

    break_types = _ground_truth_break_types()
    mangled_ids = {
        txn_id for txn_id, bt in break_types.items() if bt == BreakType.NARRATION_MANGLED.value
    }

    false_matches = l1_resolved_ids - mangled_ids
    assert false_matches == set(), f"L1 false-matched: {false_matches}"
    assert mangled_ids <= l1_resolved_ids, "L1 missed a narration_mangled credit"

    # every L1 decision reports a delta of 0 and the L1 confidence
    for decision in l1_decisions:
        assert decision.delta_paise == 0
        assert decision.confidence == 0.98
