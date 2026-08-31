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
import re
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
from unbatch.money import format_paise_to_rupees, parse_rupees_to_paise

DEFAULT_OUT_DIR = Path("data")

IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = date(2024, 1, 1)
WINDOW_DAYS = 30

# 105 baseline batches -> 105 final credits (see inject_breaks' docstring for
# the exact arithmetic — settlement_split's +1-per-batch and the 7 ambiguous
# decoys' -1-per-batch happen to cancel against the standalone unrelated_credit
# and tolerance_ambiguous rows, so total credits equals N_BATCHES exactly at
# this mix). Batch dates are drawn WITH replacement (multiple credits can land
# on the same day, which is realistic for an active merchant) since this many
# batches no longer fits in 30 unique days.
#
# Raised from 103 (D0a, FAILURES.md's 2026-08-31 ablation-ceiling entry):
# rules-only was resolving 103/105 credits, leaving the adjudicator exactly
# one genuinely ambiguous case to prove itself on — nowhere near enough to
# measure what an LLM adds. ambiguous_composition went from 1 instance to 7,
# and a new break type, tolerance_ambiguous (4 instances), was added: two
# settlement batches whose nets both land inside L3's tolerance band of one
# credit, so L3's exactly-one rule must correctly decline both rather than
# guess. The 34 non-clean-or-forced credits this now structurally requires
# come out of what would otherwise be N_BATCHES's generic clean pool, so the
# clean rate correspondingly drops — see the module docstring for the actual
# resulting number, reported honestly rather than tuned back toward 85%.
N_BATCHES = 105

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
# 13, 14, 15 are deliberately unassigned — free indices left over once
# ambiguous_composition moved off this single pair (D0a); they fall through
# to the generic clean pool like any other unassigned index.

# D0a (FAILURES.md's 2026-08-31 entry): ambiguous_composition raised from 1
# instance to 7, each with its own dedicated (target, decoy) pair rather than
# 7 targets sharing one decoy pool. A shared pool was considered and rejected:
# its ExpectedBatch window would span every target's anchor date at once,
# wide enough to risk coincidentally falling inside some unrelated credit's
# L1/L3 check purely by chance — exactly the "hoping the odds are favorable"
# trap FAILURES.md already warned about once (2026-08-30, rounding_delta).
# Seven small, narrowly-dated pairs keep each decoy's window scoped to just
# its own target, the same isolation the original single pair had.
AMBIGUOUS_COMPOSITION_N = 7
AMBIGUOUS_COMPOSITION_TARGET_INDICES: tuple[int, ...] = tuple(
    range(17, 17 + AMBIGUOUS_COMPOSITION_N)
)
AMBIGUOUS_COMPOSITION_DECOY_INDICES: tuple[int, ...] = tuple(
    range(17 + AMBIGUOUS_COMPOSITION_N, 17 + 2 * AMBIGUOUS_COMPOSITION_N)
)

# New break type (D0a): two settlement batches' nets both fall inside L3's
# tolerance band of one credit, so L3's "exactly one within tolerance" rule
# must correctly decline rather than guess. Four (P, Q) sibling pairs, each a
# single forced-gross UPI line so the resulting net is exactly predictable —
# see _tolerance_ambiguous_gross_paise. P's net (plus a small offset so L0/L1
# don't tie on it exactly) is the credit amount; Q is a coincidentally close
# decoy. Both siblings resolve as ordinary `clean` credits of their own —
# unlike the ambiguous_composition decoys, nothing about being a tolerance
# rival requires a sibling to go unclaimed.
TOLERANCE_AMBIGUOUS_N = 4
_TOLERANCE_AMBIGUOUS_START = 17 + 2 * AMBIGUOUS_COMPOSITION_N
TOLERANCE_AMBIGUOUS_SIBLING_INDICES: tuple[int, ...] = tuple(
    range(_TOLERANCE_AMBIGUOUS_START, _TOLERANCE_AMBIGUOUS_START + 2 * TOLERANCE_AMBIGUOUS_N)
)
TOLERANCE_AMBIGUOUS_P_INDICES: tuple[int, ...] = TOLERANCE_AMBIGUOUS_SIBLING_INDICES[0::2]
TOLERANCE_AMBIGUOUS_Q_INDICES: tuple[int, ...] = TOLERANCE_AMBIGUOUS_SIBLING_INDICES[1::2]

# One base gross per pair, spread across the realistic amount range so the
# four instances aren't all the same size. DELTA is deliberately small and
# FLAT (not scaled per base): net_delta = gross_delta * (1 - fee_rate*(1+gst))
# is ~constant regardless of base for a fixed method, while L3's tolerance
# band (0.6% of the credit) grows with base — so a flat, small delta clears
# every base's band with margin, verified numerically for this exact mix
# rather than assumed (smallest base: diff_q=473 paise against a 1196-paise
# band). CREDIT_OFFSET keeps the credit off P's net exactly, so L1's exact-tie
# check still declines it instead of resolving it before L3 ever runs.
TOLERANCE_AMBIGUOUS_BASE_GROSS_PAISE: tuple[int, ...] = (200_000, 600_000, 1_000_000, 1_800_000)
TOLERANCE_AMBIGUOUS_SIBLING_DELTA_PAISE = 500
TOLERANCE_AMBIGUOUS_CREDIT_OFFSET_PAISE = 25
TOLERANCE_AMBIGUOUS_NARRATION = "NEFT-UNKNOWN00000000-RAZORPAY SOFTWARE PVT"

# rounding_delta is the floor of L3's tolerance band (a paise-level gap from
# per-line vs per-batch rounding — see fees.py). fee_tier_change is the
# ceiling: a forced Rs 10,000 line with a full half-point rate bump produces
# a delta in the tens of rupees, so L3 gets exercised at both ends rather
# than only at the floor.
FEE_TIER_TARGET_GROSS_PAISE = 1_000_000  # Rs 10,000, forced so the delta is predictable
FEE_TIER_BUMP = Decimal("0.005")  # a 0.5 point rate change -> tens of rupees, not paise
UNRELATED_CREDIT_PAISE = 25_000  # ~Rs 250, a plausible-looking stray inbound transfer

# settlement_split and each ambiguous_composition target need more than the
# general 2-4 line range to split or subtract safely without landing on a
# near-empty or negative group; CHARGEBACK_BATCH_INDEX needs enough other
# lines to absorb the flat CHARGEBACK_FEE_PAISE penalty without the batch
# total going negative; ROUNDING_DELTA_BATCH_INDEX needs room for its two
# boundary lines alongside the large-value line; each tolerance_ambiguous
# sibling is exactly the 1 forced-gross line that defines its net.
#
# Each ambiguous_composition decoy is exactly 2 lines. An earlier version of
# this forced 4-6 lines instead, first to avoid an L1 false-match (see
# FAILURES.md's 2026-08-31 entry) and then padded further to avoid an L3
# false-match — both real bugs, but the extra lines were the wrong fix: with
# 7 decoys now contributing up to 4 padding lines each, L2's candidate pools
# for OTHER unrelated credits nearby swelled to within a few lines of
# MAX_POOL (48), reintroducing the exact clustering risk FAILURES.md's
# 2026-08-30 entry already covered once. `_apply_ambiguous_decoy` below
# fixes the root cause instead: the manipulated line's type, not its
# quantity.
_FORCED_N_LINES: dict[int, int] = {
    **dict.fromkeys(
        (*SETTLEMENT_SPLIT_BATCH_INDICES, CHARGEBACK_BATCH_INDEX, ROUNDING_DELTA_BATCH_INDEX), 4
    ),
    **dict.fromkeys(AMBIGUOUS_COMPOSITION_TARGET_INDICES, 4),
    **dict.fromkeys(AMBIGUOUS_COMPOSITION_DECOY_INDICES, 2),
    **dict.fromkeys(TOLERANCE_AMBIGUOUS_SIBLING_INDICES, 1),
}

# Each ambiguous_composition decoy's first line is forced small so
# `_apply_ambiguous_decoy`'s adjustment of the second line can always be
# raised to hit its (larger, 3-line) target sum without going negative. A
# decoy's lines are never claimed by any Decision — see FAILURES.md's
# 2026-08-30 entry — so they stay available in L2's candidate pool for the
# entire run, which is what makes the coincidental alternate subset
# discoverable at all.
AMBIGUOUS_DECOY_ANCHOR_GROSS_PAISE = AMOUNT_MIN_PAISE

# Two lines at this gross, both CARD, each land exactly on a half-paise fee
# boundary (500025 * 2% = 10000.5, rounds up to 10001); their combined gross
# (1000050 * 2% = 20001.0) needs no rounding at all. Per-line sum (20002) vs
# per-batch (20001) is the guaranteed 1-paise rounding_delta gap.
ROUNDING_BOUNDARY_GROSS_PAISE = 500_025

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


def _tolerance_ambiguous_gross_paise(batch_index: int) -> int:
    """P's gross is this pair's base; Q's is base + a small flat delta (see
    TOLERANCE_AMBIGUOUS_SIBLING_DELTA_PAISE's derivation above)."""
    if batch_index in TOLERANCE_AMBIGUOUS_P_INDICES:
        pair = TOLERANCE_AMBIGUOUS_P_INDICES.index(batch_index)
        return TOLERANCE_AMBIGUOUS_BASE_GROSS_PAISE[pair]
    pair = TOLERANCE_AMBIGUOUS_Q_INDICES.index(batch_index)
    return TOLERANCE_AMBIGUOUS_BASE_GROSS_PAISE[pair] + TOLERANCE_AMBIGUOUS_SIBLING_DELTA_PAISE


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


def generate_orders_and_settlements(
    seed: int,
) -> tuple[list[OrderLedgerRecord], list[SettlementLine], list[Batch]]:
    """Generate the economic base data: captured orders grouped into
    N_BATCHES settlement payout batches over a WINDOW_DAYS-day window, each
    batch sharing one settlement_utr across its lines (a UTR is assigned to
    the whole payout, not per transaction), plus a handful of failed orders
    that never reach settlement at all.

    A refund or chargeback line lives ONLY in its own dedicated batch
    (REFUND_BATCH_INDEX / CHARGEBACK_BATCH_INDEX), never mixed into an
    otherwise-clean one — see FAILURES.md's 2026-08-30 entry for why that
    used to be organic/random and why that was wrong: a "clean" batch
    containing a refund still nets to the same total on both the expected
    and actual side, so L0/L1 would resolve it exactly like a real clean
    batch and refund_in_window's whole point (a naive expectation that
    ignores the refund) would never be exercised.

    Deterministic regardless of seed: batch REFUND_BATCH_INDEX contains a
    refund line, batch CHARGEBACK_BATCH_INDEX contains a chargeback line,
    and one line in batch LARGE_VALUE_BATCH_INDEX is deliberately oversized
    (DATA_SPEC.md's "one deliberately large-value credit" requirement) —
    these are guaranteed by construction, not left to a random draw.
    """
    rng = random.Random(seed)

    # Sampled WITH replacement, deliberately NOT sorted: N_BATCHES exceeds
    # WINDOW_DAYS, so several batches necessarily share a settled date
    # (realistic for an active merchant receiving more than one payout a
    # day) — and leaving the draw order alone means `batch.index` (which
    # *_BATCH_INDEX constants use to assign break-type roles) carries no
    # correlation with chronological position. Sorting used to seem
    # harmless because nothing read `index` as anything but a label, until
    # L2 existed: every special-role batch sat at a low index, which meant
    # every one of them landed in the first few days of the window, which
    # meant their still-unconsumed lines all fell inside each other's L2
    # date windows — pool sizes of 36-43 lines instead of single digits,
    # and a ~20s L2 run instead of a fraction of a second. See FAILURES.md's
    # 2026-08-30 entry. `inject_breaks` re-sorts the final bank_statement
    # rows by date regardless, so nothing downstream depends on this order.
    settled_dates = [
        EPOCH + timedelta(days=rng.randrange(WINDOW_DAYS)) for _ in range(N_BATCHES)
    ]
    batches = [
        Batch(index=i, settled_date=d, settlement_utr=_random_utr(rng))
        for i, d in enumerate(settled_dates)
    ]

    # Each ambiguous_composition decoy needs to land within L2's [D-3d, D]
    # window of its own target on purpose — that proximity is what makes the
    # coincidental alternate subset discoverable at all. The window only
    # looks BACKWARD from a credit's date, so a decoy must settle on or
    # before its target's date, never after, or its lines would never even
    # enter the target's candidate pool. Independently random dates (now that
    # dates no longer correlate with index — see the note above) would only
    # occasionally land a decoy in range at all, let alone on the correct
    # side, so each pair is anchored here explicitly rather than left to
    # chance.
    for target_idx, decoy_idx in zip(
        AMBIGUOUS_COMPOSITION_TARGET_INDICES, AMBIGUOUS_COMPOSITION_DECOY_INDICES, strict=True
    ):
        target_date = batches[target_idx].settled_date
        decoy_offset = rng.choice([-3, -2, -1])
        decoy_date = max(target_date + timedelta(days=decoy_offset), EPOCH)
        batches[decoy_idx].settled_date = decoy_date

    # tolerance_ambiguous needs both siblings dated so a single credit's
    # 3-day backward window covers them both — simplest is the same day.
    for p_idx, q_idx in zip(
        TOLERANCE_AMBIGUOUS_P_INDICES, TOLERANCE_AMBIGUOUS_Q_INDICES, strict=True
    ):
        batches[q_idx].settled_date = batches[p_idx].settled_date

    orders: list[OrderLedgerRecord] = []

    for batch in batches:
        captured_date = batch.settled_date - timedelta(days=rng.choice([1, 1, 1, 2]))
        n_lines = _FORCED_N_LINES.get(batch.index, rng.randint(2, 4))
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
            elif batch.index in AMBIGUOUS_COMPOSITION_DECOY_INDICES and i == 0:
                gross_paise = AMBIGUOUS_DECOY_ANCHOR_GROSS_PAISE
            elif batch.index in TOLERANCE_AMBIGUOUS_SIBLING_INDICES and i == 0:
                gross_paise = _tolerance_ambiguous_gross_paise(batch.index)
                method_override = PaymentMethod.UPI

            status = OrderStatus.CAPTURED
            if batch.index == REFUND_BATCH_INDEX and i == 0:
                status = OrderStatus.REFUNDED

            order, line = _make_payment(
                rng,
                captured_date,
                batch.settled_date,
                batch.settlement_utr,
                gross_paise=gross_paise,
                method=method_override,
                status=status,
            )

            orders.append(order)
            batch.lines.append(line)
            captured_this_batch.append(order)

            if status is not OrderStatus.CAPTURED:
                # Only REFUND_BATCH_INDEX's forced i==0 case reaches here (see
                # above) — refunds/chargebacks/adjustments live exclusively in
                # their own dedicated batch, never mixed into an otherwise
                # `clean` one. A "clean" batch whose lines happen to include a
                # refund would make its own expected-batch total (all lines)
                # equal its own bank credit (also all lines) by construction,
                # same as a genuinely clean batch — no gap for L0/L1 to
                # correctly decline on, silently defeating refund_in_window's
                # whole point. See FAILURES.md's 2026-08-30 entry.
                batch.lines.append(
                    _make_refund_line(
                        rng,
                        order,
                        batch.settlement_utr,
                        batch.settled_date,
                        partial=status is OrderStatus.PARTIALLY_REFUNDED,
                    )
                )

        if batch.index == CHARGEBACK_BATCH_INDEX:
            batch.lines.append(
                _make_chargeback_line(
                    rng, captured_this_batch[0], batch.settlement_utr, batch.settled_date
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
    (fee=tax=0) to hit the target exactly instead of fighting rounding.
    Intended for one of AMBIGUOUS_COMPOSITION_DECOY_INDICES specifically: its
    lines are never claimed by any Decision, so they stay in L2's candidate
    pool for the entire run — a batch that resolves to its own credit would
    have its lines consumed the moment that credit matched, making the
    "coincidence" unreachable by the time anything went looking for it.

    The adjusted line's type is set to ADJUSTMENT, not left as PAYMENT (see
    FAILURES.md's 2026-08-31 entry). L2's composition search doesn't filter
    by line type, so this line is still just as discoverable as a
    composition candidate — but compute_expected_batches's net_paise sums
    PAYMENT lines only, so the decoy's own ExpectedBatch (still the anchor
    line's small, harmless net) no longer accidentally equals the target sum
    it was built from. That equality is exactly what let L1 resolve the real
    target credit against the decoy's own batch as a false exact match."""
    first, second = _payment_lines(batch)[:2]
    needed_net = target_sum_paise - first.net_paise
    updated = second.model_copy(
        update={
            "type": SettlementLineType.ADJUSTMENT,
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
    # Each target's credit only ever accounts for all-but-the-last of its
    # batch's lines — the last line is a deliberate leftover, never claimed
    # by any ground-truth credit. Without this, the credit's amount would
    # equal the WHOLE batch's total, which is exactly what
    # compute_expected_batches groups by UTR, so L0/L1 would resolve it via a
    # clean, unambiguous exact match before composition search (L2) ever ran
    # — silently defeating the entire point of "two subsets compose the same
    # amount." See FAILURES.md's 2026-08-30 entry.
    for target_idx, decoy_idx in zip(
        AMBIGUOUS_COMPOSITION_TARGET_INDICES, AMBIGUOUS_COMPOSITION_DECOY_INDICES, strict=True
    ):
        ambiguous_lines = batches[target_idx].lines[:-1]
        ambiguous_target = sum(line.net_paise for line in ambiguous_lines)
        _apply_ambiguous_decoy(batches[decoy_idx], ambiguous_target)

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

        elif batch.index in AMBIGUOUS_COMPOSITION_DECOY_INDICES:
            orphan_settlements.append(
                GroundTruthOrphanSettlement(
                    settlement_ids=[line.settlement_id for line in batch.lines],
                    payment_ids=[line.payment_id for line in batch.lines],
                )
            )

        elif batch.index in AMBIGUOUS_COMPOSITION_TARGET_INDICES:
            lines = batch.lines[:-1]  # excludes the deliberate leftover line
            record = record.model_copy(
                update={"credit_paise": sum(line.net_paise for line in lines)}
            )
            final_records.append(record)
            credits.append(
                _ground_truth_credit(record.txn_id, lines, BreakType.AMBIGUOUS_COMPOSITION)
            )

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

    # tolerance_ambiguous: a standalone credit, not derived from any batch's
    # own baseline record (same pattern as unrelated_credit below), priced at
    # P's net plus a small offset (so L1's exact-tie check still declines it)
    # and dated to match both siblings so their windows both cover it. P is
    # the "true" match ground truth records; Q is the coincidentally close
    # rival L3 must correctly refuse to pick over it.
    for p_idx in TOLERANCE_AMBIGUOUS_P_INDICES:
        p_line = _payment_lines(batches[p_idx])[0]
        credit_paise = p_line.net_paise + TOLERANCE_AMBIGUOUS_CREDIT_OFFSET_PAISE
        txn_id = _random_id(rng, "txn_")
        final_records.append(
            BankStatementRecord(
                txn_id=txn_id,
                value_date=batches[p_idx].settled_date,
                narration=TOLERANCE_AMBIGUOUS_NARRATION,
                credit_paise=credit_paise,
                debit_paise=None,
                balance_paise=0,
            )
        )
        credits.append(
            _ground_truth_credit(txn_id, batches[p_idx].lines, BreakType.TOLERANCE_AMBIGUOUS)
        )

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV written by `_write_csv` back into row dicts. Opened with
    `newline=""` for symmetry with the writer, even though reading doesn't
    carry the same CRLF risk — csv.reader still expects it."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _order_ledger_from_row(row: dict[str, str]) -> OrderLedgerRecord:
    return OrderLedgerRecord(
        order_id=row["order_id"],
        payment_id=row["payment_id"],
        amount_paise=parse_rupees_to_paise(row["amount"]),
        currency=row["currency"],
        status=OrderStatus(row["status"]),
        captured_at=datetime.fromisoformat(row["captured_at"]),
        customer_ref=row["customer_ref"],
        method=PaymentMethod(row["method"]),
    )


def _settlement_line_from_row(row: dict[str, str]) -> SettlementLine:
    return SettlementLine(
        settlement_id=row["settlement_id"],
        settlement_utr=row["settlement_utr"],
        payment_id=row["payment_id"],
        type=SettlementLineType(row["type"]),
        gross_paise=parse_rupees_to_paise(row["gross"]),
        fee_paise=parse_rupees_to_paise(row["fee"]),
        tax_paise=parse_rupees_to_paise(row["tax"]),
        net_paise=parse_rupees_to_paise(row["net"]),
        settled_at=datetime.fromisoformat(row["settled_at"]),
    )


def _bank_statement_from_row(row: dict[str, str]) -> BankStatementRecord:
    return BankStatementRecord(
        txn_id=row["txn_id"],
        value_date=date.fromisoformat(row["value_date"]),
        narration=row["narration"],
        credit_paise=parse_rupees_to_paise(row["credit"]) if row["credit"] else None,
        debit_paise=parse_rupees_to_paise(row["debit"]) if row["debit"] else None,
        balance_paise=parse_rupees_to_paise(row["balance"]),
    )


def read_order_ledger_csv(path: Path) -> list[OrderLedgerRecord]:
    """Read data/order_ledger.csv back into OrderLedgerRecord models."""
    return [_order_ledger_from_row(row) for row in _read_csv(path)]


def read_settlement_report_csv(path: Path) -> list[SettlementLine]:
    """Read data/settlement_report.csv back into SettlementLine models."""
    return [_settlement_line_from_row(row) for row in _read_csv(path)]


def read_bank_statement_csv(path: Path) -> list[BankStatementRecord]:
    """Read data/bank_statement.csv back into BankStatementRecord models."""
    return [_bank_statement_from_row(row) for row in _read_csv(path)]


# --- Narration noise (E10) ---------------------------------------------
#
# Real bank narrations are noisier than the clean templates above. This is
# a post-processing pass over the *already-generated* bank_statement, not a
# new break type: it only ever rewrites `narration`, never touches
# amount/date/balance/txn_id or any settlement data, so it measures L0/L1
# robustness to messy text, not the cascade's arithmetic. `--noise 0.0`
# (the default) must reproduce the committed seed-42 fixtures byte for
# byte — see `apply_narration_noise`'s own early-return below.
#
# UTRs are always exactly one of `_BANK_CODES` (5 chars) followed by 12
# digits (`_random_utr`), so this pattern finds one reliably wherever a
# clean UTR is still present in a narration; where it isn't (already
# truncated/absent by an earlier break-injection step), the UTR-targeted
# techniques below degrade gracefully to a no-op or a generic fallback.
_UTR_PATTERN = re.compile("(?:" + "|".join(_BANK_CODES) + r")\d{12}")

_NOISE_PREFIXES = ("TRF/", "REF ", "TXN/", "P2A-", "UPI-")
_NOISE_SUFFIXES = (" -SETL", "/CR", " REF", "-CREDIT", " TXN")
_LOOKALIKE_SUBS = {"O": "0", "0": "O", "I": "1", "1": "I", "l": "1"}
_COUNTERPARTY_ONLY_NARRATIONS = (
    "RAZORPAY SOFTWARE PVT LTD",
    "RAZORPAY SOFTWARE PRIVATE LIMITED",
    "SETTLEMENT CREDIT RAZORPAY",
)


def _noise_truncate_mid_utr(narration: str, rng: random.Random) -> str:
    """Truncation at an arbitrary length — landing inside the UTR span when
    one is present, since a cut that stops just short of the UTR (or lands
    past it) wouldn't actually be mid-UTR."""
    match = _UTR_PATTERN.search(narration)
    if match is None or match.end() - match.start() < 2:
        cut = rng.randint(max(1, len(narration) // 3), max(1, len(narration) - 1))
        return narration[:cut]
    start, end = match.span()
    cut = rng.randint(start + 1, end - 1)
    return narration[:cut]


def _noise_transpose_utr_digits(narration: str, rng: random.Random) -> str:
    """Transposed adjacent digits within the UTR — a real UTR string is no
    longer equal to the one on record, so L0's exact substring check must
    correctly decline it."""
    match = _UTR_PATTERN.search(narration)
    if match is None:
        return narration
    utr = match.group()
    swappable = [i for i in range(len(utr) - 1) if utr[i].isdigit() and utr[i + 1].isdigit()]
    if not swappable:
        return narration
    i = rng.choice(swappable)
    swapped = utr[:i] + utr[i + 1] + utr[i] + utr[i + 2 :]
    return narration[: match.start()] + swapped + narration[match.end() :]


def _noise_lookalike_substitution(narration: str, rng: random.Random) -> str:
    """O/0 and I/1/l confusions, applied within the UTR itself — the kind
    of manual re-keying error a bank's own systems introduce."""
    match = _UTR_PATTERN.search(narration)
    if match is None:
        return narration
    utr = match.group()
    positions = [i for i, ch in enumerate(utr) if ch in _LOOKALIKE_SUBS]
    if not positions:
        return narration
    i = rng.choice(positions)
    mutated = utr[:i] + _LOOKALIKE_SUBS[utr[i]] + utr[i + 1 :]
    return narration[: match.start()] + mutated + narration[match.end() :]


def _noise_separators(narration: str, rng: random.Random) -> str:
    """Inconsistent separators and collapsed/doubled whitespace — leaves
    the UTR itself untouched (separators sit between segments, not inside
    an alphanumeric code), so this alone need not defeat L0's match."""
    variants = (
        lambda s: s.replace("-", "_"),
        lambda s: s.replace("-", " "),
        lambda s: s.replace("/", "-"),
        lambda s: s.replace(" ", "  ", 1),
        lambda s: re.sub(r"\s+", "", s),
    )
    return rng.choice(variants)(narration)


def _noise_wrap(narration: str, rng: random.Random) -> str:
    """Bank-specific prefixes/suffixes wrapping the UTR — also leaves the
    UTR substring itself intact, same reasoning as separator noise."""
    prefix = rng.choice(_NOISE_PREFIXES) if rng.random() < 0.5 else ""
    suffix = rng.choice(_NOISE_SUFFIXES) if rng.random() < 0.5 else ""
    if not prefix and not suffix:
        suffix = rng.choice(_NOISE_SUFFIXES)
    return f"{prefix}{narration}{suffix}"


def _noise_case(narration: str, rng: random.Random) -> str:
    """Case inconsistency — lowercasing (or randomly mixing case) changes
    the UTR substring itself, since L0's match is case-sensitive."""
    if rng.random() < 0.5:
        return narration.lower()
    return "".join(
        ch.lower() if ch.isupper() and rng.random() < 0.5 else ch.upper() if ch.islower() else ch
        for ch in narration
    )


def _noise_drop_utr(narration: str, rng: random.Random) -> str:
    """The UTR absent entirely, with only a counterparty name — the
    clearest case: nothing here can satisfy L0's substring check."""
    return rng.choice(_COUNTERPARTY_ONLY_NARRATIONS)


_NOISE_TECHNIQUES = (
    _noise_truncate_mid_utr,
    _noise_transpose_utr_digits,
    _noise_lookalike_substitution,
    _noise_separators,
    _noise_wrap,
    _noise_case,
    _noise_drop_utr,
)


def apply_narration_noise(
    rng: random.Random, bank_statement: list[BankStatementRecord], noise: float
) -> list[BankStatementRecord]:
    """Degrade `noise` (0.0-1.0) of `bank_statement`'s narrations with one
    randomly chosen technique each, seeded and deterministic. `noise <= 0.0`
    returns the input list unchanged (not a copy) — the exact no-op
    `generate(seed, noise=0.0)`'s byte-identical guarantee depends on.
    Every field but `narration` is always left exactly as it was."""
    if noise <= 0.0:
        return bank_statement
    noised: list[BankStatementRecord] = []
    for record in bank_statement:
        if rng.random() < noise:
            technique = rng.choice(_NOISE_TECHNIQUES)
            record = record.model_copy(update={"narration": technique(record.narration, rng)})
        noised.append(record)
    return noised


def generate(seed: int, out_dir: Path = DEFAULT_OUT_DIR, *, noise: float = 0.0) -> None:
    """Generate order_ledger.csv, settlement_report.csv, bank_statement.csv,
    and ground_truth.json under `out_dir` for `seed`, then print the
    break-type distribution to stdout.

    Each phase gets its own `random.Random(seed)`, re-seeded fresh at the
    phase boundary rather than threading one rng through everything — so
    every phase's random draws are reproducible on their own, and a change
    to one phase's random usage can't shift what a later phase draws.

    `noise` (0.0-1.0, default 0.0) degrades bank_statement narrations only
    (see `apply_narration_noise`) — amounts, dates, and settlement data are
    never touched, and every ground-truth answer stays exactly what it was
    without noise. At the default 0.0 this is a strict no-op: output is
    byte-identical to a call with `noise` omitted entirely.
    """
    orders, _baseline_settlements, batches = generate_orders_and_settlements(seed)
    baseline_records = generate_bank_statement_baseline(random.Random(seed), batches)
    settlements, bank_statement, ground_truth = inject_breaks(
        random.Random(seed), orders, batches, baseline_records
    )
    bank_statement = apply_narration_noise(random.Random(seed), bank_statement, noise)

    write_order_ledger_csv(orders, out_dir / "order_ledger.csv")
    write_settlement_report_csv(settlements, out_dir / "settlement_report.csv")
    write_bank_statement_csv(bank_statement, out_dir / "bank_statement.csv")
    write_ground_truth_json(ground_truth, out_dir / "ground_truth.json")

    print_break_distribution(ground_truth)
