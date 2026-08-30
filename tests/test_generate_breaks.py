"""Break injection and ground_truth.json: every DATA_SPEC.md break type
appears at least once, the mechanics behind each are structurally sound, and
nothing under stages/ ever imports ground truth (CLAUDE.md invariant 7)."""

from __future__ import annotations

import ast
import random
from pathlib import Path

from unbatch.generate import (
    AMBIGUOUS_COMPOSITION_BATCH_INDEX,
    AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX,
    DUPLICATE_UTR_BATCH_INDEX_A,
    DUPLICATE_UTR_BATCH_INDEX_B,
    FEE_TIER_CHANGE_BATCH_INDEX,
    ORPHAN_SETTLEMENT_BATCH_INDEX,
    ROUNDING_DELTA_BATCH_INDEX,
    generate_bank_statement_baseline,
    generate_orders_and_settlements,
    inject_breaks,
    write_ground_truth_json,
)
from unbatch.models import BreakType

SEED = 42


def _full_pipeline(seed: int = SEED):
    orders, _settlements, batches = generate_orders_and_settlements(seed)
    baseline = generate_bank_statement_baseline(random.Random(seed), batches)
    settlements, bank_statement, ground_truth = inject_breaks(
        random.Random(seed), orders, batches, baseline
    )
    return orders, settlements, batches, bank_statement, ground_truth


def test_every_break_type_appears_at_least_once() -> None:
    _, _, _, _, ground_truth = _full_pipeline()
    seen = {c.break_type for c in ground_truth.credits}
    assert seen == set(BreakType) - {BreakType.ORPHAN_SETTLEMENT}
    assert len(ground_truth.orphan_settlements) >= 1


def test_break_injection_is_deterministic() -> None:
    _, _, _, records_a, gt_a = _full_pipeline()
    _, _, _, records_b, gt_b = _full_pipeline()
    assert [r.model_dump() for r in records_a] == [r.model_dump() for r in records_b]
    assert gt_a.model_dump() == gt_b.model_dump()


def test_every_bank_credit_has_exactly_one_ground_truth_entry() -> None:
    _, _, _, bank_statement, ground_truth = _full_pipeline()
    assert {r.txn_id for r in bank_statement} == {c.txn_id for c in ground_truth.credits}
    assert len(ground_truth.credits) == len(bank_statement)


def test_rounding_delta_batch_report_and_credit_diverge_by_a_few_paise() -> None:
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    batch = batches[ROUNDING_DELTA_BATCH_INDEX]
    report_total = sum(line.net_paise for line in batch.lines)

    credit = next(c for c in ground_truth.credits if c.break_type == BreakType.ROUNDING_DELTA)
    record = next(r for r in bank_statement if r.txn_id == credit.txn_id)

    delta = abs(record.credit_paise - report_total)
    assert 0 < delta < 100  # a paise-level rounding gap, not a real mismatch


def test_fee_tier_change_batch_report_and_credit_diverge_by_a_small_amount() -> None:
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    batch = batches[FEE_TIER_CHANGE_BATCH_INDEX]
    report_total = sum(line.net_paise for line in batch.lines)

    credit = next(c for c in ground_truth.credits if c.break_type == BreakType.FEE_TIER_CHANGE)
    record = next(r for r in bank_statement if r.txn_id == credit.txn_id)

    delta = abs(record.credit_paise - report_total)
    assert 0 < delta < report_total * 0.05  # small relative to the batch, not huge


def test_duplicate_utr_credits_share_the_same_utr_in_narration() -> None:
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    dup_credits = [c for c in ground_truth.credits if c.break_type == BreakType.DUPLICATE_UTR]
    assert len(dup_credits) == 2

    records = [next(r for r in bank_statement if r.txn_id == c.txn_id) for c in dup_credits]
    shared_utr = batches[DUPLICATE_UTR_BATCH_INDEX_A].settlement_utr
    assert all(shared_utr in r.narration for r in records)

    utrs_b = {line.settlement_utr for line in batches[DUPLICATE_UTR_BATCH_INDEX_B].lines}
    assert utrs_b == {shared_utr}


def test_settlement_split_produces_two_credits_covering_the_whole_batch() -> None:
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    split_credits = [c for c in ground_truth.credits if c.break_type == BreakType.SETTLEMENT_SPLIT]
    assert len(split_credits) == 2

    records = [next(r for r in bank_statement if r.txn_id == c.txn_id) for c in split_credits]
    dates = sorted(r.value_date for r in records)
    assert (dates[1] - dates[0]).days == 1

    combined_payment_ids = sorted(split_credits[0].payment_ids + split_credits[1].payment_ids)
    original_batch = next(
        b for b in batches if sorted(line.payment_id for line in b.lines) == combined_payment_ids
    )
    assert sum(r.credit_paise for r in records) == sum(
        line.net_paise for line in original_batch.lines
    )


def test_orphan_settlement_has_no_matching_bank_credit() -> None:
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    orphan_batch = batches[ORPHAN_SETTLEMENT_BATCH_INDEX]
    orphan_payment_ids = {line.payment_id for line in orphan_batch.lines}

    assert len(ground_truth.orphan_settlements) == 1
    assert set(ground_truth.orphan_settlements[0].payment_ids) == orphan_payment_ids

    bank_credit_payment_ids = {pid for c in ground_truth.credits for pid in c.payment_ids}
    assert orphan_payment_ids.isdisjoint(bank_credit_payment_ids)


def test_ambiguous_composition_has_a_genuine_alternate_subset() -> None:
    _, _, batches, _bank_statement, ground_truth = _full_pipeline()
    target_batch = batches[AMBIGUOUS_COMPOSITION_BATCH_INDEX]
    decoy_batch = batches[AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX]

    target_total = sum(line.net_paise for line in target_batch.lines)
    decoy_payment_lines = [line for line in decoy_batch.lines if line.type.value == "payment"]
    decoy_pair_sum = decoy_payment_lines[0].net_paise + decoy_payment_lines[1].net_paise

    assert decoy_pair_sum == target_total

    ambiguous_credit = next(
        c for c in ground_truth.credits if c.break_type == BreakType.AMBIGUOUS_COMPOSITION
    )
    assert set(ambiguous_credit.payment_ids) == {line.payment_id for line in target_batch.lines}

    # the decoy batch's own credit still ties out despite the mutated line
    decoy_payment_ids = {ln.payment_id for ln in decoy_batch.lines}
    decoy_credit = next(
        c for c in ground_truth.credits if set(c.payment_ids) == decoy_payment_ids
    )
    assert decoy_credit.break_type == BreakType.CLEAN


def test_unrelated_credit_has_no_settlement_ties_and_is_unresolvable() -> None:
    _, _, _, _, ground_truth = _full_pipeline()
    credit = next(c for c in ground_truth.credits if c.break_type == BreakType.UNRELATED_CREDIT)
    assert credit.settlement_ids == []
    assert credit.payment_ids == []
    assert credit.resolvable is False


def test_bank_statement_running_balance_still_consistent_after_injection() -> None:
    _, _, _, bank_statement, _ = _full_pipeline()
    from unbatch.generate import OPENING_BALANCE_PAISE

    records_sorted = sorted(bank_statement, key=lambda r: r.value_date)
    balance = OPENING_BALANCE_PAISE
    for record in records_sorted:
        balance += record.credit_paise
        assert record.balance_paise == balance


def test_written_ground_truth_json_has_no_carriage_returns_and_round_trips(tmp_path: Path) -> None:
    _, _, _, _, ground_truth = _full_pipeline()
    path = tmp_path / "ground_truth.json"
    write_ground_truth_json(ground_truth, path)

    raw = path.read_bytes()
    assert b"\r" not in raw

    import json

    reloaded = json.loads(raw)
    assert len(reloaded["credits"]) == len(ground_truth.credits)
    assert len(reloaded["orphan_settlements"]) == len(ground_truth.orphan_settlements)


def test_no_stage_module_imports_ground_truth() -> None:
    """CLAUDE.md invariant 7: ground_truth.json is read ONLY by metrics.py."""
    stages_dir = Path(__file__).resolve().parents[1] / "src" / "unbatch" / "stages"
    for path in stages_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [alias.name for alias in node.names]
            assert not any("ground_truth" in name for name in names), (
                f"{path} imports ground_truth — scoring leak"
            )
