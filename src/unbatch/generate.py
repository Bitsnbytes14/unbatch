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
import random
import string
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from unbatch.fees import compute_fee_paise, compute_net_paise, compute_tax_paise
from unbatch.models import (
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
N_BATCHES = 18

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

    settled_dates = sorted(
        EPOCH + timedelta(days=offset) for offset in rng.sample(range(WINDOW_DAYS), N_BATCHES)
    )
    batches = [
        Batch(index=i, settled_date=d, settlement_utr=_random_utr(rng))
        for i, d in enumerate(settled_dates)
    ]

    orders: list[OrderLedgerRecord] = []

    for batch in batches:
        captured_date = batch.settled_date - timedelta(days=rng.choice([1, 1, 1, 2]))
        n_lines = rng.randint(4, 9)
        captured_this_batch: list[OrderLedgerRecord] = []

        for i in range(n_lines):
            gross_paise = None
            if batch.index == LARGE_VALUE_BATCH_INDEX and i == 0:
                gross_paise = LARGE_VALUE_PAISE
            elif batch.index == CHARGEBACK_BATCH_INDEX and i == 0:
                gross_paise = CHARGEBACK_ELIGIBLE_MIN_PAISE + _random_gross_paise(rng)

            status = OrderStatus.CAPTURED
            if batch.index == REFUND_BATCH_INDEX and i == 0:
                status = OrderStatus.REFUNDED
            elif gross_paise is None:
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
                status=status,
            )
            orders.append(order)
            batch.lines.append(line)
            captured_this_batch.append(order)

            if status is not OrderStatus.CAPTURED:
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
        elif rng.random() < 0.15:
            eligible = [o for o in captured_this_batch if o.status == OrderStatus.CAPTURED]
            if eligible:
                target = rng.choice(eligible)
                batch.lines.append(
                    _make_chargeback_line(rng, target, batch.settlement_utr, batch.settled_date)
                )

        if rng.random() < 0.2 and captured_this_batch:
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
    break-type distribution to stdout."""
    raise NotImplementedError
