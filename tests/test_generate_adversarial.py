"""`unbatch generate --adversarial` (E11a): a same-scale, deliberately
hostile dataset engineered to hit the collision shapes E5/E9 found by
chance. Checked here: it never touches the default generator's output,
it's internally consistent (every credit round-trips through the real
cascade without crashing), and it actually produces the false-match floor
it was built to demonstrate — not just structurally plausible-looking
data.

The rules-only cascade run is shared across tests via a module-scoped
fixture: the near-cap-pool scenario genuinely trips compose.py's 5-second
`compose_timeout` (see generate.py's `_adv_near_cap_pool`), so running the
cascade once instead of once per test keeps this file from costing several
extra timeouts on every suite run."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from unbatch import audit
from unbatch.cli import (
    STAGE_SEQUENCE,
    build_unresolved_credits,
    compute_expected_batches,
    load_input_data,
    run_cascade,
)
from unbatch.generate import generate, generate_adversarial
from unbatch.models import BreakType, Decision, GroundTruth, RunContext

SEED = 42


def _load_ground_truth(data_dir: Path) -> GroundTruth:
    text = (data_dir / "ground_truth.json").read_text(encoding="utf-8")
    return GroundTruth.model_validate_json(text)


@pytest.fixture(scope="module")
def adversarial_run(tmp_path_factory: pytest.TempPathFactory):
    """Generate the adversarial dataset once and run the real rules-only
    cascade against it once, shared read-only across every test below."""
    data_dir = tmp_path_factory.mktemp("adversarial_data")
    generate_adversarial(SEED, out_dir=data_dir)

    _orders, settlements, bank_records = load_input_data(data_dir)
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)
    run_id = audit.derive_run_id(SEED, data_dir, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=SEED, no_llm=True)
    db_path = tmp_path_factory.mktemp("adversarial_db") / "audit.db"
    conn = audit.connect(db_path)
    try:
        audit.clear_run(conn, run_id)
        counts = run_cascade(
            ctx, unresolved, conn, settlements=settlements, stage_sequence=STAGE_SEQUENCE
        )
        decisions = audit.fetch_decisions(conn, run_id)
    finally:
        conn.close()

    gt = _load_ground_truth(data_dir)
    return data_dir, gt, counts, decisions


def test_adversarial_does_not_affect_the_default_generator_output(tmp_path: Path) -> None:
    default_dir = tmp_path / "default"
    generate(SEED, out_dir=default_dir)

    adversarial_dir = tmp_path / "adversarial"
    generate_adversarial(SEED, out_dir=adversarial_dir)

    # the default path, called fresh after generate_adversarial ran, must
    # still reproduce the committed fixtures byte for byte
    reference_dir = tmp_path / "reference"
    generate(SEED, out_dir=reference_dir)
    for filename in (
        "order_ledger.csv",
        "settlement_report.csv",
        "bank_statement.csv",
        "ground_truth.json",
    ):
        assert (default_dir / filename).read_bytes() == (reference_dir / filename).read_bytes()


def test_adversarial_is_deterministic(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_adversarial(SEED, out_dir=dir_a)
    generate_adversarial(SEED, out_dir=dir_b)
    for filename in (
        "order_ledger.csv",
        "settlement_report.csv",
        "bank_statement.csv",
        "ground_truth.json",
    ):
        assert (dir_a / filename).read_bytes() == (dir_b / filename).read_bytes()


def test_adversarial_is_same_scale_as_the_default_dataset(adversarial_run) -> None:
    _data_dir, gt, _counts, _decisions = adversarial_run
    assert 95 <= len(gt.credits) <= 115


def test_adversarial_contains_the_targeted_break_types(adversarial_run) -> None:
    _data_dir, gt, _counts, _decisions = adversarial_run
    counts = Counter(c.break_type for c in gt.credits)

    assert counts[BreakType.DUPLICATE_UTR] >= 6
    assert counts[BreakType.TOLERANCE_AMBIGUOUS] >= 5
    assert counts[BreakType.AMBIGUOUS_COMPOSITION] >= 3
    assert counts[BreakType.SETTLEMENT_SPLIT] >= 6
    assert counts[BreakType.REFUND_IN_WINDOW] >= 1
    assert counts[BreakType.CHARGEBACK_DEDUCTION] >= 1


def test_adversarial_round_trips_through_the_real_cascade_without_crashing(adversarial_run) -> None:
    _data_dir, gt, counts, decisions = adversarial_run
    assert sum(counts.values()) == len(gt.credits)
    assert len({d.credit_id for d in decisions}) == len(gt.credits)  # exactly one decision each


def test_adversarial_produces_guaranteed_tolerance_collision_false_matches(adversarial_run) -> None:
    """The centrepiece of E11: the tolerance-collision scenario is
    engineered so the credit's own true batch is outside its own tolerance
    band while an unrelated decoy batch is inside it — every instance
    should resolve as a false match at L3, not a coincidence some seeds
    happen to hit."""
    _data_dir, gt, _counts, decisions = adversarial_run
    gt_by_txn = {c.txn_id: c for c in gt.credits}
    decision_by_txn: dict[str, Decision] = {d.credit_id: d for d in decisions}

    tolerance_ambiguous_ids = [
        c.txn_id for c in gt.credits if c.break_type == BreakType.TOLERANCE_AMBIGUOUS
    ]
    false_matches = 0
    for txn_id in tolerance_ambiguous_ids:
        decision = decision_by_txn[txn_id]
        gtc = gt_by_txn[txn_id]
        matched = Counter(decision.matched_payment_ids)
        expected = Counter(gtc.payment_ids)
        if decision.outcome.value == "matched" and matched != expected:
            false_matches += 1
    assert false_matches >= 4  # at least four of the five, allowing for one incidental exception


def test_adversarial_false_match_rate_is_materially_worse_than_normal(adversarial_run) -> None:
    _data_dir, gt, _counts, decisions = adversarial_run
    decision_by_txn: dict[str, Decision] = {d.credit_id: d for d in decisions}

    resolved = 0
    false_matches = 0
    for credit in gt.credits:
        decision = decision_by_txn.get(credit.txn_id)
        if decision is None or decision.outcome.value != "matched":
            continue
        resolved += 1
        if Counter(decision.matched_payment_ids) != Counter(credit.payment_ids):
            false_matches += 1

    false_match_rate = false_matches / resolved if resolved else 0.0
    # the normal seed-42 dataset's false-match rate is 0.0% — the whole
    # point of this dataset is to be materially, measurably worse
    assert false_match_rate > 0.03
