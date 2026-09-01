"""Break injection and ground_truth.json: every DATA_SPEC.md break type
appears at least once, and the mechanics behind each are structurally
sound."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from unbatch.generate import (
    AMBIGUOUS_COMPOSITION_DECOY_INDICES,
    AMBIGUOUS_COMPOSITION_TARGET_INDICES,
    DUPLICATE_UTR_BATCH_INDEX_A,
    DUPLICATE_UTR_BATCH_INDEX_B,
    FEE_TIER_CHANGE_BATCH_INDEX,
    ROUNDING_DELTA_BATCH_INDEX,
    SETTLEMENT_SPLIT_BATCH_INDICES,
    TOLERANCE_AMBIGUOUS_P_INDICES,
    TOLERANCE_AMBIGUOUS_Q_INDICES,
    generate_bank_statement_baseline,
    generate_orders_and_settlements,
    inject_breaks,
    write_ground_truth_json,
)
from unbatch.models import BreakType, CascadeConfig

SEED = 42
_TOLERANCE = CascadeConfig()


def _full_pipeline(seed: int = SEED):
    orders, _settlements, batches = generate_orders_and_settlements(seed)
    baseline = generate_bank_statement_baseline(random.Random(seed), batches)
    settlements, bank_statement, ground_truth = inject_breaks(
        random.Random(seed), orders, batches, baseline
    )
    return orders, settlements, batches, bank_statement, ground_truth


def test_distribution_is_a_realistic_long_tail() -> None:
    """~105 credits total. Clean dropped from ~85% to ~75% in D0a
    (FAILURES.md's 2026-08-31 ablation-ceiling entry): ambiguous_composition
    grew from 1 instance to 7 and a new break type, tolerance_ambiguous (4
    instances), was added specifically to give the adjudicator more than one
    genuinely ambiguous case to prove itself on — both counts pulled directly
    from what would otherwise be the generic clean pool, reported honestly
    rather than tuned back toward the old percentage.

    narration_mangled and settlement_split are still clearly more common
    than the one-off duplicate_utr pair — but ambiguous_composition is now
    deliberately the LARGEST non-clean category, not the smallest, so it no
    longer belongs in that same "rare long tail" comparison.
    """
    _, _, _batches, bank_statement, ground_truth = _full_pipeline()

    assert 100 <= len(bank_statement) <= 110

    counts = Counter(c.break_type for c in ground_truth.credits)
    clean_fraction = counts[BreakType.CLEAN] / len(bank_statement)
    assert 0.70 <= clean_fraction <= 0.80

    assert counts[BreakType.NARRATION_MANGLED] > counts[BreakType.DUPLICATE_UTR]
    assert counts[BreakType.SETTLEMENT_SPLIT] > counts[BreakType.DUPLICATE_UTR]
    assert counts[BreakType.AMBIGUOUS_COMPOSITION] == 7
    assert counts[BreakType.TOLERANCE_AMBIGUOUS] == 4


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


def test_settlement_split_produces_two_credits_per_occurrence() -> None:
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    split_credits = [c for c in ground_truth.credits if c.break_type == BreakType.SETTLEMENT_SPLIT]
    # SETTLEMENT_SPLIT_BATCH_INDICES has more than one occurrence deliberately
    # (it's meant to be more common than the one-off duplicate_utr/ambiguous
    # pairs), each occurrence contributing 2 credits.
    assert len(split_credits) == 2 * len(SETTLEMENT_SPLIT_BATCH_INDICES)

    for batch_index in SETTLEMENT_SPLIT_BATCH_INDICES:
        original_batch = batches[batch_index]
        original_payment_id_set = {line.payment_id for line in original_batch.lines}
        original_payment_ids = sorted(original_payment_id_set)

        pair = [c for c in split_credits if set(c.payment_ids) <= original_payment_id_set]
        assert len(pair) == 2, f"expected 2 split credits for batch {batch_index}"

        records = [next(r for r in bank_statement if r.txn_id == c.txn_id) for c in pair]
        dates = sorted(r.value_date for r in records)
        assert (dates[1] - dates[0]).days == 1

        combined_payment_ids = sorted(pair[0].payment_ids + pair[1].payment_ids)
        assert combined_payment_ids == original_payment_ids
        assert sum(r.credit_paise for r in records) == sum(
            line.net_paise for line in original_batch.lines
        )


def test_orphan_settlement_has_no_matching_bank_credit() -> None:
    """One orphan_settlement entry per ambiguous_composition decoy (D0a) —
    each decoy batch is itself an orphan: its lines are never claimed by any
    bank credit, which is exactly what keeps them available in L2's pool for
    the whole run (see FAILURES.md's 2026-08-30 entry)."""
    _, _, batches, _bank_statement, ground_truth = _full_pipeline()
    orphan_payment_ids: set[str] = set()
    for decoy_idx in AMBIGUOUS_COMPOSITION_DECOY_INDICES:
        orphan_payment_ids |= {line.payment_id for line in batches[decoy_idx].lines}

    assert len(ground_truth.orphan_settlements) == len(AMBIGUOUS_COMPOSITION_DECOY_INDICES)
    all_orphan_entry_payment_ids = {
        pid for entry in ground_truth.orphan_settlements for pid in entry.payment_ids
    }
    assert all_orphan_entry_payment_ids == orphan_payment_ids

    bank_credit_payment_ids = {pid for c in ground_truth.credits for pid in c.payment_ids}
    assert orphan_payment_ids.isdisjoint(bank_credit_payment_ids)


def test_ambiguous_composition_has_a_genuine_alternate_subset() -> None:
    """Each of the 7 targets (D0a) covers all-but-the-last line of its own
    batch — the last line is a deliberate, permanently unclaimed leftover.
    Without that, the credit's amount would equal the batch's WHOLE total,
    which is exactly what compute_expected_batches groups by UTR, so L0/L1
    would resolve it via a clean exact match before composition search (L2)
    ever ran. See FAILURES.md's 2026-08-30 entry.

    Each decoy pair lives inside its own dedicated decoy batch rather than
    an independently "clean" one — a clean decoy would have its lines
    consumed by L0 before L2 ever ran (the runner removes matched
    payment_ids between stages), so the coincidence would never actually be
    reachable. Orphan lines are never claimed by any Decision, so they stay
    in the pool for the whole run. See FAILURES.md's second 2026-08-30
    entry for this one, and its 2026-08-31 entry for why each target keeps
    its own decoy rather than sharing one pool."""
    _, _, batches, _bank_statement, ground_truth = _full_pipeline()
    ambiguous_credits = [
        c for c in ground_truth.credits if c.break_type == BreakType.AMBIGUOUS_COMPOSITION
    ]
    assert len(ambiguous_credits) == len(AMBIGUOUS_COMPOSITION_TARGET_INDICES)

    all_credited_payment_ids = {pid for c in ground_truth.credits for pid in c.payment_ids}
    all_orphan_payment_ids = {
        pid for entry in ground_truth.orphan_settlements for pid in entry.payment_ids
    }

    for target_idx, decoy_idx in zip(
        AMBIGUOUS_COMPOSITION_TARGET_INDICES, AMBIGUOUS_COMPOSITION_DECOY_INDICES, strict=True
    ):
        target_batch = batches[target_idx]
        decoy_batch = batches[decoy_idx]

        target_lines = target_batch.lines[:-1]
        target_total = sum(line.net_paise for line in target_lines)
        # The decoy is exactly 2 lines: an anchor (still type=payment) and
        # the manipulated line, whose type is deliberately ADJUSTMENT, not
        # PAYMENT (see FAILURES.md's 2026-08-31 entry — this is what keeps
        # the decoy's own ExpectedBatch.net_paise, which sums PAYMENT lines
        # only, from accidentally equaling this same target_total).
        assert decoy_batch.lines[0].type.value == "payment"
        assert decoy_batch.lines[1].type.value == "adjustment"
        decoy_pair_sum = decoy_batch.lines[0].net_paise + decoy_batch.lines[1].net_paise

        assert decoy_pair_sum == target_total

        # the decoy settles on or before its target, within L2's 3-day
        # backward-looking window — otherwise it would never enter the pool
        # when searching from the target credit's date.
        assert decoy_batch.settled_date <= target_batch.settled_date
        assert (target_batch.settled_date - decoy_batch.settled_date).days <= 3

        target_payment_ids = {line.payment_id for line in target_lines}
        matching_credit = next(
            c for c in ambiguous_credits if set(c.payment_ids) == target_payment_ids
        )
        assert matching_credit is not None

        # the leftover line is never claimed by any ground-truth credit
        leftover_payment_id = target_batch.lines[-1].payment_id
        assert leftover_payment_id not in all_credited_payment_ids

        # the decoy pair's payment_ids are orphaned too — never claimed by
        # any credit, exactly like the rest of their decoy batch
        decoy_payment_ids = {line.payment_id for line in decoy_batch.lines}
        assert decoy_payment_ids.isdisjoint(all_credited_payment_ids)
        assert decoy_payment_ids <= all_orphan_payment_ids


def test_tolerance_ambiguous_has_two_batches_within_band_of_one_credit() -> None:
    """D0a's new break type: P and Q are ordinary resolvable ('clean')
    batches on their own, but their nets both land inside L3's tolerance
    band of one credit priced near P's — L3's exactly-one rule must decline
    both rather than guess which one is real. See FAILURES.md's 2026-08-31
    entry."""
    _, _, batches, bank_statement, ground_truth = _full_pipeline()
    tolerance_credits = [
        c for c in ground_truth.credits if c.break_type == BreakType.TOLERANCE_AMBIGUOUS
    ]
    assert len(tolerance_credits) == len(TOLERANCE_AMBIGUOUS_P_INDICES)

    for p_idx, q_idx, credit in zip(
        TOLERANCE_AMBIGUOUS_P_INDICES, TOLERANCE_AMBIGUOUS_Q_INDICES, tolerance_credits, strict=True
    ):
        p_batch, q_batch = batches[p_idx], batches[q_idx]
        p_net = sum(line.net_paise for line in p_batch.lines)
        q_net = sum(line.net_paise for line in q_batch.lines)
        record = next(r for r in bank_statement if r.txn_id == credit.txn_id)
        tolerance = max(
            _TOLERANCE.tolerance_floor_paise, round(record.credit_paise * _TOLERANCE.tolerance_rate)
        )

        assert abs(record.credit_paise - p_net) <= tolerance
        assert abs(record.credit_paise - q_net) <= tolerance
        assert p_net != q_net  # a genuine near-miss, not a coincidental exact tie
        assert record.credit_paise != p_net  # L1's exact-tie check must still decline

        # both siblings settle on/before the credit's date, inside the same
        # 3-day backward window, so L3 actually sees both as candidates
        assert p_batch.settled_date == q_batch.settled_date == record.value_date

        # P is ground truth's "real" match; both siblings still resolve as
        # ordinary clean credits of their own, unlike the ambiguous_composition
        # decoys, which are never claimed by anything
        assert set(credit.payment_ids) == {line.payment_id for line in p_batch.lines}
        p_own_credit = next(
            c
            for c in ground_truth.credits
            if c.break_type == BreakType.CLEAN
            and set(c.payment_ids) == {line.payment_id for line in p_batch.lines}
        )
        assert p_own_credit.txn_id != credit.txn_id


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
