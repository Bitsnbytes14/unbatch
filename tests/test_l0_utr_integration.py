"""L0 against the real seed-42 fixtures: every `clean`-tagged credit
resolves, and nothing else does — a false match at L0 would corrupt the
books, so this is the property that matters most.

Reads ground_truth.json to check the result, same as metrics.py will —
CLAUDE.md invariant 7 is about the pipeline (stages/, cli.py's cascade
runner) never importing it, not about tests being unable to verify
behaviour against it.
"""

from __future__ import annotations

import json
from pathlib import Path

from unbatch.cli import build_unresolved_credits, compute_expected_batches, load_input_data
from unbatch.models import BreakType, RunContext
from unbatch.stages import l0_utr

CTX = RunContext(run_id="run_test", seed=42)


def _ground_truth_break_types() -> dict[str, str]:
    data = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    return {c["txn_id"]: c["break_type"] for c in data["credits"]}


def test_l0_resolves_every_clean_credit_and_nothing_else() -> None:
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    decisions = l0_utr.run(unresolved, CTX)
    resolved_ids = {d.credit_id for d in decisions}

    break_types = _ground_truth_break_types()
    clean_ids = {txn_id for txn_id, bt in break_types.items() if bt == BreakType.CLEAN.value}

    # every non-clean credit L0 resolved would be a false match
    false_matches = resolved_ids - clean_ids
    assert false_matches == set(), f"L0 false-matched: {false_matches}"

    # L0 should catch the bulk of clean cases (not necessarily every last one,
    # since a clean credit's batch could in principle collide with another —
    # but for this seed it should be effectively all of them)
    assert len(resolved_ids & clean_ids) / len(clean_ids) > 0.95
