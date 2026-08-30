"""Seeded synthetic data generation.

Writes data/order_ledger.csv, data/settlement_report.csv,
data/bank_statement.csv, and data/ground_truth.json for a given seed. Must
produce byte-identical CSVs for the same seed on any machine (DATA_SPEC.md).
Prints a break-type distribution summary to stdout so the mix is visible
without opening the files.

Money is computed in paise internally (via unbatch.money / unbatch.fees) and
formatted to 2dp only on write. Every output file is opened with
`newline=""` and, for CSVs, an explicit `lineterminator="\n"` — Python's
default text-mode writing translates `\n` to `os.linesep` on Windows, which
would make the generated files differ byte-for-byte from a Linux run. See
FAILURES.md's 2026-08-30 entry.

All randomness goes through a single `random.Random(seed)` instance passed
explicitly down the call chain — never the `random` module's global state —
so the same seed always produces the same output regardless of what else
has touched the RNG.
"""

from __future__ import annotations

import csv
import json
import random
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from unbatch.fees import (
    FEE_RATES,
    compute_batch_fee_and_tax_paise,
    compute_fee_paise,
    compute_net_paise,
    compute_tax_paise,
)
from unbatch.models import (
    BankStatementRecord,
    BreakType,
    GroundTruth,
    GroundTruthCredit,
    GroundTruthOrphanSettlement,
    OrderLedgerRecord,
    OrderStatus,
    PaymentMethod,
    SettlementLine,
    SettlementLineType,
)
from unbatch.money import format_paise_to_rupees

DEFAULT_OUT_DIR = Path("data")

IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = date(2024, 1, 1)
WINDOW_DAYS = 30

# 103 baseline batches -> 105 final credits after settlement_split adds 2 and
# orphan_settlement removes 1 (see inject_breaks' docstring for the exact
# arithmetic). The 16 non-clean credits that "at least one of every break
# type" structurally requires (see commit 12's FAILURES.md entry) are a fixed
# count regardless of scale, so growing the batch count is what moves the
# clean rate: 16/105 lands at ~84.8%, versus 16/70 at ~77%. It also makes a
# single credit ~0.95% of the dataset, so METRICS.md's sub-1% false-match
# target is finally representable at all (one credit was 1.43% at 70).
# Batch dates are drawn WITH replacement (multiple credits can land on the
# same day, which is realistic for an active merchant) since this many
# batches no longer fits in 30 unique days.
N_BATCHES = 103

AMOUNT_MIN_PAISE = 15_000  # ~Rs 150
AMOUNT_MAX_PAISE = 2_500_000  # ~Rs 25,000
LARGE_VALUE_PAISE = 45_000_000  # ~Rs 4,50,000 — deliberately dwarfs every other line

CHARGEBACK_FEE_PAISE = 50_000  # a flat Rs 500 chargeback penalty, illustrative
CHARGEBACK_ELIGIBLE_MIN_PAISE = 500_000  # chargeback source orders are Rs 5,000+

N_FAILED_ORDERS = 6

# Deterministic guarantees, independent of the random draw: batch REFUND_BATCH
# always contains a refund, CHARGEBACK_BATCH always contains a chargeback,
# LARGE_VALUE_BATCH always contains the one deliberately oversized line. Later
# break injection (generate.py's break-injection step) finds these by
# scanning batch contents, not by importing these indices, so they are only
# meaningful here.
REFUND_BATCH_INDEX = 2
CHARGEBACK_BATCH_INDEX = 3
LARGE_VALUE_BATCH_INDEX = 4

# Same deterministic-by-construction approach for narration variety: batches
# in NARRATION_TRUNCATED_BATCH_INDICES always get a truncated UTR and
# NARRATION_NO_UTR_BATCH_INDEX always gets no UTR at all, so narration_mangled
# has guaranteed instances regardless of the random draw. Two truncated
# instances (rather than one) keeps narration_mangled's total (3) strictly
# ahead of the one-off duplicate_utr pair (2) — see commit 12. Every other
# batch gets a full, correct UTR — chosen NEFT- or IMPS-style at random.
NARRATION_TRUNCATED_BATCH_INDICES = (5, 16)
NARRATION_NO_UTR_BATCH_INDEX = 6

OPENING_BALANCE_PAISE = 10_000_000  # ~Rs 1,00,000 starting balance, illustrative

# Break-type batch assignments (commit 10, rebalanced in commits 12/16).
# Every batch index gets exactly one role; whatever is left over is
# ground-truthed as `clean` (~85% of final credits — see inject_breaks'
# docstring for the exact count). Reusing LARGE_VALUE_BATCH_INDEX for
# rounding_delta is deliberate: it makes the biggest credit in the dataset
# also the one with a rounding quirk, so both "value-weighted diverges from
# count" and "rounding_delta exists" land on the same story.
#
# narration_mangled (3 batches) and settlement_split (2 batches, each
# becoming 2 credits) are deliberately the most common non-clean types,
# ahead of the one-off duplicate_utr pair and ambiguous_composition — a
# realistic long tail, not a flat one.
ROUNDING_DELTA_BATCH_INDEX = LARGE_VALUE_BATCH_INDEX
SETTLEMENT_SPLIT_BATCH_INDICES = (7, 8)
FEE_TIER_CHANGE_BATCH_INDEX = 9
DATE_SKEW_BATCH_INDEX = 10
DUPLICATE_UTR_BATCH_INDEX_A = 11
DUPLICATE_UTR_BATCH_INDEX_B = 12
ORPHAN_SETTLEMENT_BATCH_INDEX = 13
AMBIGUOUS_COMPOSITION_BATCH_INDEX = 14
AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX = 15

# rounding_delta is the floor of L3's tolerance band (a paise-level gap from
# per-line vs per-batch rounding — see fees.py). fee_tier_change is the
# ceiling: a forced Rs 10,000 line with a full half-point rate bump produces
# a delta in the tens of rupees, so L3 gets exercised at both ends rather
# than only at the floor.
FEE_TIER_TARGET_GROSS_PAISE = 1_000_000  # Rs 10,000, forced so the delta is predictable
FEE_TIER_BUMP = Decimal("0.005")  # a 0.5 point rate change -> tens of rupees, not paise
UNRELATED_CREDIT_PAISE = 25_000  # ~Rs 250, a plausible-looking stray inbound transfer

# Every designated batch above is deliberately built by hand, not by the
# generic random draw: organic refund/chargeback/adjustment noise is
# suppressed for all of them (see the `not in _SPECIAL_BATCH_INDICES` guards
# in generate_orders_and_settlements) so their composition stays predictable
# enough to split, subtract, or resum without accidentally going negative.
# That realism belongs in the ordinary "clean" population instead, where it
# doesn't need to be reasoned about precisely.
_SPECIAL_BATCH_INDICES = frozenset(
    {
        REFUND_BATCH_INDEX,
        CHARGEBACK_BATCH_INDEX,
        LARGE_VALUE_BATCH_INDEX,
        *NARRATION_TRUNCATED_BATCH_INDICES,
        NARRATION_NO_UTR_BATCH_INDEX,
        *SETTLEMENT_SPLIT_BATCH_INDICES,
        FEE_TIER_CHANGE_BATCH_INDEX,
        DATE_SKEW_BATCH_INDEX,
        DUPLICATE_UTR_BATCH_INDEX_A,
        DUPLICATE_UTR_BATCH_INDEX_B,
        ORPHAN_SETTLEMENT_BATCH_INDEX,
        AMBIGUOUS_COMPOSITION_BATCH_INDEX,
        AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX,
    }
)

# settlement_split and ambiguous_composition need more than the general 2-4
# line range to split or subtract safely without landing on a near-empty or
# negative group; CHARGEBACK_BATCH_INDEX needs enough other lines to absorb
# the flat CHARGEBACK_FEE_PAISE penalty without the batch total going
# negative; ROUNDING_DELTA_BATCH_INDEX needs room for its two boundary lines
# alongside the large-value line. All forced to a fixed 4 lines instead of
# left to chance.
_FORCED_N_LINES: dict[int, int] = dict.fromkeys(
    (
        *SETTLEMENT_SPLIT_BATCH_INDICES,
        AMBIGUOUS_COMPOSITION_BATCH_INDEX,
        AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX,
        CHARGEBACK_BATCH_INDEX,
        ROUNDING_DELTA_BATCH_INDEX,
    ),
    4,
)

AMBIGUOUS_DECOY_ANCHOR_GROSS_PAISE = AMOUNT_MIN_PAISE  # forced small so the decoy's
# adjusted second line can always be raised to hit the target sum without going negative

# Two lines at this gross, both CARD, each land exactly on a half-paise fee
# boundary (500025 * 2% = 10000.5, rounds up to 10001); their combined gross
# (1000050 * 2% = 20001.0) needs no rounding at all. Per-line sum (20002) vs
# per-batch (20001) is the guaranteed 1-paise rounding_delta gap.
ROUNDING_BOUNDARY_GROSS_PAISE = 500_025

MIN_BATCH_NET_BUFFER_PAISE = 100  # floor for a running batch total after an organic refund

_METHOD_WEIGHTS: dict[PaymentMethod, float] = {
    PaymentMethod.UPI: 0.55,
    PaymentMethod.CARD: 0.25,
    PaymentMethod.NETBANKING: 0.12,
    PaymentMethod.WALLET: 0.08,
}

_ID_ALPHABET = string.ascii_letters + string.digits
_BANK_CODES = ("AXISP", "HDFCR", "ICICR", "SBIN0", "KKBK0")


@dataclass
class Batch:
    """One settlement payout batch: a shared settlement_utr and the
    settlement lines settling on the same date, destined for one bank credit
    before any break injection splits, drops, or duplicates it."""

    index: int
    settled_date: date
    settlement_utr: str
    lines: list[SettlementLine] = field(default_factory=list)


def _random_id(rng: random.Random, prefix: str, length: int = 14) -> str:
    return prefix + "".join(rng.choices(_ID_ALPHABET, k=length))


def _random_utr(rng: random.Random) -> str:
    code = rng.choice(_BANK_CODES)
    digits = "".join(rng.choices(string.digits, k=12))
    return f"{code}{digits}"


def _random_customer_ref(rng: random.Random) -> str:
    return "cust_" + "".join(rng.choices(string.digits, k=8))


def _random_method(rng: random.Random) -> PaymentMethod:
    methods = list(_METHOD_WEIGHTS)
    weights = list(_METHOD_WEIGHTS.values())
    return rng.choices(methods, weights=weights, k=1)[0]


def _random_gross_paise(rng: random.Random) -> int:
    return rng.randint(AMOUNT_MIN_PAISE, AMOUNT_MAX_PAISE)


def _datetime_at(day: date, rng: random.Random) -> datetime:
    return datetime.combine(
        day,
        time(rng.randint(8, 22), rng.randint(0, 59), rng.randint(0, 59)),
        tzinfo=IST,
    )


def _make_payment(
    rng: random.Random,
    captured_date: date,
    settled_date: date,
    settlement_utr: str,
    *,
    gross_paise: int | None = None,
    method: PaymentMethod | None = None,
    status: OrderStatus = OrderStatus.CAPTURED,
) -> tuple[OrderLedgerRecord, SettlementLine]:
    """Build one order and its matching settlement payment line at standard
    (unmutated) fee rates."""
    gross = gross_paise if gross_paise is not None else _random_gross_paise(rng)
    chosen_method = method if method is not None else _random_method(rng)
    payment_id = _random_id(rng, "pay_")
    fee = compute_fee_paise(gross, chosen_method)
    tax = compute_tax_paise(fee)
    net = compute_net_paise(gross, fee, tax)
    order = OrderLedgerRecord(
        order_id=_random_id(rng, "order_"),
        payment_id=payment_id,
        amount_paise=gross,
        currency="INR",
        status=status,
        captured_at=_datetime_at(captured_date, rng),
        customer_ref=_random_customer_ref(rng),
        method=chosen_method,
    )
    line = SettlementLine(
        settlement_id=_random_id(rng, "setl_"),
        settlement_utr=settlement_utr,
        payment_id=payment_id,
        type=SettlementLineType.PAYMENT,
        gross_paise=gross,
        fee_paise=fee,
        tax_paise=tax,
        net_paise=net,
        settled_at=_datetime_at(settled_date, rng),
    )
    return order, line


def _make_refund_line(
    rng: random.Random,
    order: OrderLedgerRecord,
    settlement_utr: str,
    settled_date: date,
    *,
    partial: bool,
) -> SettlementLine:
    """A refund settles in the same batch/date as the order it refunds —
    the simplification that makes `refund_in_window` representable at all:
    it is exactly a refund netting out inside the batch that pays the
    original capture (DATA_SPEC.md's own description of the break)."""
    refund_amount = (
        int(order.amount_paise * rng.uniform(0.2, 0.6)) if partial else order.amount_paise
    )
    return SettlementLine(
        settlement_id=_random_id(rng, "setl_"),
        settlement_utr=settlement_utr,
        payment_id=order.payment_id,
        type=SettlementLineType.REFUND,
        gross_paise=-refund_amount,
        fee_paise=0,
        tax_paise=0,
        net_paise=-refund_amount,
        settled_at=_datetime_at(settled_date, rng),
    )


def _make_chargeback_line(
    rng: random.Random,
    order: OrderLedgerRecord,
    settlement_utr: str,
    settled_date: date,
) -> SettlementLine:
    gross = -order.amount_paise
    fee = CHARGEBACK_FEE_PAISE
    tax = compute_tax_paise(fee)
    net = gross - fee - tax
    return SettlementLine(
        settlement_id=_random_id(rng, "setl_"),
        settlement_utr=settlement_utr,
        payment_id=order.payment_id,
        type=SettlementLineType.CHARGEBACK,
        gross_paise=gross,
        fee_paise=fee,
        tax_paise=tax,
        net_paise=net,
        settled_at=_datetime_at(settled_date, rng),
    )


def _make_adjustment_line(
    rng: random.Random,
    payment_id: str,
    settlement_utr: str,
    settled_date: date,
) -> SettlementLine:
    amount = rng.choice([-500, -200, 200, 500])
    return SettlementLine(
        settlement_id=_random_id(rng, "setl_"),
        settlement_utr=settlement_utr,
        payment_id=payment_id,
        type=SettlementLineType.ADJUSTMENT,
        gross_paise=amount,
        fee_paise=0,
        tax_paise=0,
        net_paise=amount,
        settled_at=_datetime_at(settled_date, rng),
    )


def generate_orders_and_settlements(
    seed: int,
) -> tuple[list[OrderLedgerRecord], list[SettlementLine], list[Batch]]:
    """Generate the economic base data: captured orders grouped into
    N_BATCHES settlement payout batches over a WINDOW_DAYS-day window, each
    batch sharing one settlement_utr across its lines (a UTR is assigned to
    the whole payout, not per transaction). Refunds, a chargeback, and a
    couple of manual adjustments are mixed in for realistic type variety,
    plus a handful of failed orders that never reach settlement at all.

    Deterministic regardless of seed: batch REFUND_BATCH_INDEX contains a
    refund line, batch CHARGEBACK_BATCH_INDEX contains a chargeback line,
    and one line in batch LARGE_VALUE_BATCH_INDEX is deliberately oversized
    (DATA_SPEC.md's "one deliberately large-value credit" requirement) —
    these are guaranteed by construction, not left to a random draw.
    """
    rng = random.Random(seed)

    # Sampled WITH replacement: N_BATCHES (68) exceeds WINDOW_DAYS (30), so
    # several batches necessarily share a settled date — realistic for an
    # active merchant receiving more than one payout a day.
    settled_dates = sorted(
        EPOCH + timedelta(days=rng.randrange(WINDOW_DAYS)) for _ in range(N_BATCHES)
    )
    batches = [
        Batch(index=i, settled_date=d, settlement_utr=_random_utr(rng))
        for i, d in enumerate(settled_dates)
    ]

    orders: list[OrderLedgerRecord] = []

    for batch in batches:
        captured_date = batch.settled_date - timedelta(days=rng.choice([1, 1, 1, 2]))
        n_lines = _FORCED_N_LINES.get(batch.index, rng.randint(2, 4))
        is_special = batch.index in _SPECIAL_BATCH_INDICES
        captured_this_batch: list[OrderLedgerRecord] = []

        for i in range(n_lines):
            gross_paise = None
            method_override = None
            if batch.index == LARGE_VALUE_BATCH_INDEX and i == 0:
                gross_paise = LARGE_VALUE_PAISE
            elif batch.index == ROUNDING_DELTA_BATCH_INDEX and i in (1, 2):
                # Two identical CARD lines sitting exactly on a half-paise fee
                # boundary: each rounds 10000.5 -> 10001 paise individually,
                # but their combined gross rounds to an exact 20001 once —
                # guarantees the per-line-vs-per-batch gap deterministically
                # rather than hoping some random line lands on a boundary.
                gross_paise = ROUNDING_BOUNDARY_GROSS_PAISE
                method_override = PaymentMethod.CARD
            elif batch.index == CHARGEBACK_BATCH_INDEX and i == 0:
                gross_paise = CHARGEBACK_ELIGIBLE_MIN_PAISE + _random_gross_paise(rng)
            elif batch.index == FEE_TIER_CHANGE_BATCH_INDEX and i == 0:
                gross_paise = FEE_TIER_TARGET_GROSS_PAISE
            elif batch.index == AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX and i == 0:
                gross_paise = AMBIGUOUS_DECOY_ANCHOR_GROSS_PAISE

            status = OrderStatus.CAPTURED
            if batch.index == REFUND_BATCH_INDEX and i == 0:
                status = OrderStatus.REFUNDED
            elif gross_paise is None and not is_special:
                roll = rng.random()
                if roll < 0.05:
                    status = OrderStatus.REFUNDED
                elif roll < 0.10:
                    status = OrderStatus.PARTIALLY_REFUNDED

            order, line = _make_payment(
                rng,
                captured_date,
                batch.settled_date,
                batch.settlement_utr,
                gross_paise=gross_paise,
                method=method_override,
                status=status,
            )

            refund_line = None
            is_forced_refund = batch.index == REFUND_BATCH_INDEX and i == 0
            if status is not OrderStatus.CAPTURED:
                refund_line = _make_refund_line(
                    rng,
                    order,
                    batch.settlement_utr,
                    batch.settled_date,
                    partial=status is OrderStatus.PARTIALLY_REFUNDED,
                )
                # A small batch with only a line or two can't always absorb an
                # organic refund without its running total going negative
                # (BankStatementRecord requires credit_paise >= 0). Only the
                # dedicated REFUND_BATCH_INDEX instance is exempt — it is the
                # one guaranteed refund_in_window example and always has
                # enough other lines to stay positive overall.
                projected_total = (
                    sum(existing.net_paise for existing in batch.lines)
                    + line.net_paise
                    + refund_line.net_paise
                )
                if not is_forced_refund and projected_total < MIN_BATCH_NET_BUFFER_PAISE:
                    status = OrderStatus.CAPTURED
                    order = order.model_copy(update={"status": OrderStatus.CAPTURED})
                    refund_line = None

            orders.append(order)
            batch.lines.append(line)
            captured_this_batch.append(order)
            if refund_line is not None:
                batch.lines.append(refund_line)

        if batch.index == CHARGEBACK_BATCH_INDEX:
            batch.lines.append(
                _make_chargeback_line(
                    rng, captured_this_batch[0], batch.settlement_utr, batch.settled_date
                )
            )
        elif not is_special and rng.random() < 0.15:
            eligible = [o for o in captured_this_batch if o.status == OrderStatus.CAPTURED]
            if eligible:
                target = rng.choice(eligible)
                batch.lines.append(
                    _make_chargeback_line(rng, target, batch.settlement_utr, batch.settled_date)
                )

        if not is_special and rng.random() < 0.2 and captured_this_batch:
            ref_order = rng.choice(captured_this_batch)
            batch.lines.append(
                _make_adjustment_line(
                    rng, ref_order.payment_id, batch.settlement_utr, batch.settled_date
                )
            )

    for _ in range(N_FAILED_ORDERS):
        failed_date = EPOCH + timedelta(days=rng.randrange(WINDOW_DAYS))
        orders.append(
            OrderLedgerRecord(
                order_id=_random_id(rng, "order_"),
                payment_id=_random_id(rng, "pay_"),
                amount_paise=_random_gross_paise(rng),
                currency="INR",
                status=OrderStatus.FAILED,
                captured_at=_datetime_at(failed_date, rng),
                customer_ref=_random_customer_ref(rng),
                method=_random_method(rng),
            )
        )

    settlements = [line for batch in batches for line in batch.lines]
    return orders, settlements, batches


def _narration_neft_full(utr: str) -> str:
    return f"NEFT-{utr}-RAZORPAY SOFTWARE PVT"


def _narration_imps_full(utr: str) -> str:
    return f"IMPS/{utr}/RZPY"


def _narration_truncated(utr: str) -> str:
    return f"NEFT-{utr[:10]}..."


def _narration_no_utr() -> str:
    return "NEFT-XXXXXXXXXXXX-MISC SETTLEMENT CREDIT"


def generate_bank_statement_baseline(
    rng: random.Random, batches: list[Batch]
) -> list[BankStatementRecord]:
    """One bank credit per batch: amount = sum(net) of that batch's
    settlement lines, value_date = the batch's settled date (credited the
    same day), narration realistically varied. This is the
    pre-break-injection baseline — the next generation step splits, drops,
    or relabels specific rows from here to inject DATA_SPEC.md's break
    types; this function only knows about realistic narration variety and a
    consistent running balance.
    """
    records: list[BankStatementRecord] = []
    running_balance = OPENING_BALANCE_PAISE

    for batch in batches:
        credit = sum(line.net_paise for line in batch.lines)

        if batch.index in NARRATION_TRUNCATED_BATCH_INDICES:
            narration = _narration_truncated(batch.settlement_utr)
        elif batch.index == NARRATION_NO_UTR_BATCH_INDEX:
            narration = _narration_no_utr()
        elif rng.random() < 0.5:
            narration = _narration_neft_full(batch.settlement_utr)
        else:
            narration = _narration_imps_full(batch.settlement_utr)

        running_balance += credit
        records.append(
            BankStatementRecord(
                txn_id=_random_id(rng, "txn_"),
                value_date=batch.settled_date,
                narration=narration,
                credit_paise=credit,
                debit_paise=None,
                balance_paise=running_balance,
            )
        )

    return records


def _payment_lines(batch: Batch) -> list[SettlementLine]:
    return [line for line in batch.lines if line.type == SettlementLineType.PAYMENT]


def _apply_fee_tier_change(
    batch: Batch, orders_by_payment_id: dict[str, OrderLedgerRecord]
) -> None:
    """Mutate one payment line in place so settlement_report's declared fee
    no longer matches the rate that produced the bank credit already fixed
    by generate_bank_statement_baseline — the "fee rate changed mid-window"
    break. Only the batch's first payment line moves; its gross is forced to
    FEE_TIER_TARGET_GROSS_PAISE so the resulting delta is a predictable
    tens-of-rupees figure, not whatever a random gross happens to produce —
    this is the ceiling case for L3's tolerance band, rounding_delta is the
    floor."""
    target = _payment_lines(batch)[0]
    method = orders_by_payment_id[target.payment_id].method
    bumped_rate = FEE_RATES[method] + FEE_TIER_BUMP
    new_fee = int(
        (Decimal(target.gross_paise) * bumped_rate).to_integral_value(rounding=ROUND_HALF_UP)
    )
    new_tax = compute_tax_paise(new_fee)
    new_net = compute_net_paise(target.gross_paise, new_fee, new_tax)
    updated = target.model_copy(
        update={"fee_paise": new_fee, "tax_paise": new_tax, "net_paise": new_net}
    )
    batch.lines[batch.lines.index(target)] = updated


def _apply_duplicate_utr(batch_a: Batch, batch_b: Batch) -> None:
    """Reassign every line in batch_b to batch_a's settlement_utr — a
    gateway bug where two different payouts got tagged with the same UTR."""
    shared_utr = batch_a.settlement_utr
    batch_b.lines[:] = [
        line.model_copy(update={"settlement_utr": shared_utr}) for line in batch_b.lines
    ]


def _apply_ambiguous_decoy(batch: Batch, target_sum_paise: int) -> None:
    """Adjust the second of this batch's first two payment lines so those
    two lines' net sum exactly matches `target_sum_paise` — a genuine
    coincidental alternate composition for whichever credit target_sum_paise
    belongs to. Bypasses standard fee derivation for the adjusted line
    (fee=tax=0) to hit the target exactly instead of fighting rounding;
    this batch's own credit is recomputed afterward so it still ties out."""
    first, second = _payment_lines(batch)[:2]
    needed_net = target_sum_paise - first.net_paise
    updated = second.model_copy(
        update={
            "gross_paise": needed_net,
            "fee_paise": 0,
            "tax_paise": 0,
            "net_paise": needed_net,
        }
    )
    batch.lines[batch.lines.index(second)] = updated


def _alternate_batch_credit_with_batch_rounding(
    batch: Batch, orders_by_payment_id: dict[str, OrderLedgerRecord]
) -> int:
    """The actual bank credit for the rounding_delta batch: fee/tax applied
    once per payment-method group across the whole batch, instead of summed
    from settlement_report's per-line figures. See fees.py's module
    docstring for why the two totals differ."""
    gross_by_method: dict[PaymentMethod, list[int]] = defaultdict(list)
    other_net_total = 0
    for line in batch.lines:
        if line.type == SettlementLineType.PAYMENT:
            method = orders_by_payment_id[line.payment_id].method
            gross_by_method[method].append(line.gross_paise)
        else:
            other_net_total += line.net_paise

    total = other_net_total
    for method, gross_values in gross_by_method.items():
        fee, tax = compute_batch_fee_and_tax_paise(gross_values, method)
        total += sum(gross_values) - fee - tax
    return total


def _ground_truth_credit(
    txn_id: str,
    lines: list[SettlementLine],
    break_type: BreakType,
    *,
    resolvable: bool = True,
) -> GroundTruthCredit:
    return GroundTruthCredit(
        txn_id=txn_id,
        settlement_ids=[line.settlement_id for line in lines],
        payment_ids=[line.payment_id for line in lines],
        break_type=break_type,
        resolvable=resolvable,
    )


def inject_breaks(
    rng: random.Random,
    orders: list[OrderLedgerRecord],
    batches: list[Batch],
    baseline_records: list[BankStatementRecord],
) -> tuple[list[SettlementLine], list[BankStatementRecord], GroundTruth]:
    """Mutate the baseline economic data and bank statement to inject every
    break type in DATA_SPEC.md's catalogue, and build the matching
    ground_truth.json contents. See the *_BATCH_INDEX constants above for
    which batch plays which role; whatever is left over is ground-truthed
    as `clean`.
    """
    orders_by_payment_id = {o.payment_id: o for o in orders}

    _apply_fee_tier_change(batches[FEE_TIER_CHANGE_BATCH_INDEX], orders_by_payment_id)
    _apply_duplicate_utr(
        batches[DUPLICATE_UTR_BATCH_INDEX_A], batches[DUPLICATE_UTR_BATCH_INDEX_B]
    )
    ambiguous_target = sum(
        line.net_paise for line in batches[AMBIGUOUS_COMPOSITION_BATCH_INDEX].lines
    )
    _apply_ambiguous_decoy(batches[AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX], ambiguous_target)

    settlements = [line for batch in batches for line in batch.lines]

    credits: list[GroundTruthCredit] = []
    orphan_settlements: list[GroundTruthOrphanSettlement] = []
    final_records: list[BankStatementRecord] = []

    for batch in batches:
        record = baseline_records[batch.index]

        if batch.index == ROUNDING_DELTA_BATCH_INDEX:
            actual_credit = _alternate_batch_credit_with_batch_rounding(
                batch, orders_by_payment_id
            )
            record = record.model_copy(update={"credit_paise": actual_credit})
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.ROUNDING_DELTA)
            )

        elif batch.index == DATE_SKEW_BATCH_INDEX:
            record = record.model_copy(
                update={"value_date": batch.settled_date + timedelta(days=1)}
            )
            final_records.append(record)
            credits.append(_ground_truth_credit(record.txn_id, batch.lines, BreakType.DATE_SKEW))

        elif batch.index in SETTLEMENT_SPLIT_BATCH_INDICES:
            midpoint = len(batch.lines) // 2
            group_a, group_b = batch.lines[:midpoint], batch.lines[midpoint:]
            credit_a = BankStatementRecord(
                txn_id=_random_id(rng, "txn_"),
                value_date=batch.settled_date,
                narration=record.narration,
                credit_paise=sum(line.net_paise for line in group_a),
                debit_paise=None,
                balance_paise=0,
            )
            credit_b = BankStatementRecord(
                txn_id=_random_id(rng, "txn_"),
                value_date=batch.settled_date + timedelta(days=1),
                narration=record.narration,
                credit_paise=sum(line.net_paise for line in group_b),
                debit_paise=None,
                balance_paise=0,
            )
            final_records.extend([credit_a, credit_b])
            credits.append(
                _ground_truth_credit(credit_a.txn_id, group_a, BreakType.SETTLEMENT_SPLIT)
            )
            credits.append(
                _ground_truth_credit(credit_b.txn_id, group_b, BreakType.SETTLEMENT_SPLIT)
            )

        elif batch.index in (DUPLICATE_UTR_BATCH_INDEX_A, DUPLICATE_UTR_BATCH_INDEX_B):
            shared_utr = batches[DUPLICATE_UTR_BATCH_INDEX_A].settlement_utr
            record = record.model_copy(update={"narration": _narration_neft_full(shared_utr)})
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.DUPLICATE_UTR)
            )

        elif batch.index == ORPHAN_SETTLEMENT_BATCH_INDEX:
            orphan_settlements.append(
                GroundTruthOrphanSettlement(
                    settlement_ids=[line.settlement_id for line in batch.lines],
                    payment_ids=[line.payment_id for line in batch.lines],
                )
            )

        elif batch.index == AMBIGUOUS_COMPOSITION_BATCH_INDEX:
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.AMBIGUOUS_COMPOSITION)
            )

        elif batch.index == AMBIGUOUS_COMPOSITION_DECOY_BATCH_INDEX:
            # its own credit must still tie to its (now-mutated) lines
            recomputed_credit = sum(line.net_paise for line in batch.lines)
            record = record.model_copy(update={"credit_paise": recomputed_credit})
            final_records.append(record)
            credits.append(_ground_truth_credit(record.txn_id, batch.lines, BreakType.CLEAN))

        elif batch.index in (*NARRATION_TRUNCATED_BATCH_INDICES, NARRATION_NO_UTR_BATCH_INDEX):
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.NARRATION_MANGLED)
            )

        elif batch.index == REFUND_BATCH_INDEX:
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.REFUND_IN_WINDOW)
            )

        elif batch.index == CHARGEBACK_BATCH_INDEX:
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.CHARGEBACK_DEDUCTION)
            )

        elif batch.index == FEE_TIER_CHANGE_BATCH_INDEX:
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, batch.lines, BreakType.FEE_TIER_CHANGE)
            )

        else:
            final_records.append(record)
            credits.append(_ground_truth_credit(record.txn_id, batch.lines, BreakType.CLEAN))

    unrelated_txn_id = _random_id(rng, "txn_")
    unrelated_record = BankStatementRecord(
        txn_id=unrelated_txn_id,
        value_date=EPOCH + timedelta(days=15),
        narration="NEFT-000000000000-VENDOR REFUND MISC",
        credit_paise=UNRELATED_CREDIT_PAISE,
        debit_paise=None,
        balance_paise=0,
    )
    final_records.append(unrelated_record)
    credits.append(
        GroundTruthCredit(
            txn_id=unrelated_txn_id,
            settlement_ids=[],
            payment_ids=[],
            break_type=BreakType.UNRELATED_CREDIT,
            resolvable=False,
        )
    )

    final_records.sort(key=lambda r: r.value_date)
    balance = OPENING_BALANCE_PAISE
    recomputed_records: list[BankStatementRecord] = []
    for r in final_records:
        balance += r.credit_paise
        recomputed_records.append(r.model_copy(update={"balance_paise": balance}))

    ground_truth = GroundTruth(credits=credits, orphan_settlements=orphan_settlements)
    return settlements, recomputed_records, ground_truth


def write_ground_truth_json(ground_truth: GroundTruth, path: Path) -> None:
    """Write data/ground_truth.json. Read ONLY by metrics.py — CLAUDE.md
    invariant 7."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(ground_truth.model_dump(mode="json"), f, indent=2)
        f.write("\n")


def print_break_distribution(ground_truth: GroundTruth) -> None:
    """Print how many credits landed in each break type, plus orphan
    settlements, so the mix is visible without opening the files."""
    counts = Counter(credit.break_type.value for credit in ground_truth.credits)
    print("break-type distribution:")
    for break_type in BreakType:
        if break_type is BreakType.ORPHAN_SETTLEMENT:
            continue
        print(f"  {break_type.value:<24} {counts.get(break_type.value, 0)}")
    print(f"  {'orphan_settlement':<24} {len(ground_truth.orphan_settlements)}")


BANK_STATEMENT_HEADER = [
    "txn_id",
    "value_date",
    "narration",
    "credit",
    "debit",
    "balance",
]


def _bank_statement_row(record: BankStatementRecord) -> list[str]:
    return [
        record.txn_id,
        record.value_date.isoformat(),
        record.narration,
        format_paise_to_rupees(record.credit_paise) if record.credit_paise is not None else "",
        format_paise_to_rupees(record.debit_paise) if record.debit_paise is not None else "",
        format_paise_to_rupees(record.balance_paise),
    ]


def write_bank_statement_csv(records: list[BankStatementRecord], path: Path) -> None:
    """Write data/bank_statement.csv per DATA_SPEC.md."""
    _write_csv(path, BANK_STATEMENT_HEADER, [_bank_statement_row(r) for r in records])


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a CSV with LF-only line endings on every platform. See this
    module's docstring and FAILURES.md's 2026-08-30 entry for why
    `newline=""` and an explicit `lineterminator` are both required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


ORDER_LEDGER_HEADER = [
    "order_id",
    "payment_id",
    "amount",
    "currency",
    "status",
    "captured_at",
    "customer_ref",
    "method",
]

SETTLEMENT_REPORT_HEADER = [
    "settlement_id",
    "settlement_utr",
    "payment_id",
    "type",
    "gross",
    "fee",
    "tax",
    "net",
    "settled_at",
]


def _order_ledger_row(order: OrderLedgerRecord) -> list[str]:
    return [
        order.order_id,
        order.payment_id,
        format_paise_to_rupees(order.amount_paise),
        order.currency,
        order.status.value,
        order.captured_at.isoformat(),
        order.customer_ref,
        order.method.value,
    ]


def _settlement_report_row(line: SettlementLine) -> list[str]:
    return [
        line.settlement_id,
        line.settlement_utr,
        line.payment_id,
        line.type.value,
        format_paise_to_rupees(line.gross_paise),
        format_paise_to_rupees(line.fee_paise),
        format_paise_to_rupees(line.tax_paise),
        format_paise_to_rupees(line.net_paise),
        line.settled_at.isoformat(),
    ]


def write_order_ledger_csv(orders: list[OrderLedgerRecord], path: Path) -> None:
    """Write data/order_ledger.csv per DATA_SPEC.md."""
    _write_csv(path, ORDER_LEDGER_HEADER, [_order_ledger_row(o) for o in orders])


def write_settlement_report_csv(settlements: list[SettlementLine], path: Path) -> None:
    """Write data/settlement_report.csv per DATA_SPEC.md."""
    _write_csv(path, SETTLEMENT_REPORT_HEADER, [_settlement_report_row(s) for s in settlements])


def generate(seed: int, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Generate order_ledger.csv, settlement_report.csv, bank_statement.csv,
    and ground_truth.json under `out_dir` for `seed`, then print the
    break-type distribution to stdout.

    Each phase gets its own `random.Random(seed)`, re-seeded fresh at the
    phase boundary rather than threading one rng through everything — so
    every phase's random draws are reproducible on their own, and a change
    to one phase's random usage can't shift what a later phase draws.
    """
    orders, _baseline_settlements, batches = generate_orders_and_settlements(seed)
    baseline_records = generate_bank_statement_baseline(random.Random(seed), batches)
    settlements, bank_statement, ground_truth = inject_breaks(
        random.Random(seed), orders, batches, baseline_records
    )

    write_order_ledger_csv(orders, out_dir / "order_ledger.csv")
    write_settlement_report_csv(settlements, out_dir / "settlement_report.csv")
    write_bank_statement_csv(bank_statement, out_dir / "bank_statement.csv")
    write_ground_truth_json(ground_truth, out_dir / "ground_truth.json")

    print_break_distribution(ground_truth)
