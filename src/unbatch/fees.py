"""Fee and GST computation. Per-method fee rates, GST on the fee, net =
gross - fee - tax.

Fee rates and the GST rate are illustrative constants for this synthetic
dataset, not a claim about real Razorpay pricing — documented here so the
assumption is visible rather than buried in a magic number.

Rounding is decided and fixed here: fee and tax are rounded to the nearest
paise PER LINE (per settlement_report.csv row), never deferred to a batch
total. This matches how a real gateway actually bills — each transaction is
priced independently; it can't retroactively adjust because a later
transaction in the same payout window rounded the other way. `net_paise` in
settlement_report.csv is therefore always the per-line-rounded figure.

Per-batch rounding (the rate applied once to a batch's summed gross) gives a
different total than summing per-line fees whenever an individual line's fee
or tax lands on a half-paise boundary — the gap is usually 1-2 paise per
batch. That gap is real and is exactly what DATA_SPEC.md's `rounding_delta`
break type injects: generate.py compares this module's per-line total
against `compute_batch_fee_and_tax_paise`'s per-batch total for one picked
batch and uses the per-batch figure as the actual bank credit, so the
settlement_report lines for that batch (authoritative, per-line) no longer
sum to the credit that arrived.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from unbatch.models import PaymentMethod

GST_RATE = Decimal("0.18")

FEE_RATES: dict[PaymentMethod, Decimal] = {
    PaymentMethod.UPI: Decimal("0.003"),
    PaymentMethod.CARD: Decimal("0.020"),
    PaymentMethod.NETBANKING: Decimal("0.020"),
    PaymentMethod.WALLET: Decimal("0.018"),
}


def _round_paise(amount: Decimal) -> int:
    return int(amount.to_integral_value(rounding=ROUND_HALF_UP))


def compute_fee_paise(gross_paise: int, method: PaymentMethod) -> int:
    """Fee for one settlement line, rounded to the nearest paise."""
    return _round_paise(Decimal(gross_paise) * FEE_RATES[method])


def compute_tax_paise(fee_paise: int) -> int:
    """GST on the fee for one settlement line, rounded to the nearest paise."""
    return _round_paise(Decimal(fee_paise) * GST_RATE)


def compute_net_paise(gross_paise: int, fee_paise: int, tax_paise: int) -> int:
    """net = gross - fee - tax. Sign matches DATA_SPEC.md: negative for a
    refund or chargeback line, since gross itself is negative there."""
    return gross_paise - fee_paise - tax_paise


def compute_batch_fee_and_tax_paise(
    gross_paise_values: list[int], method: PaymentMethod
) -> tuple[int, int]:
    """The per-batch alternative: the rate applied once to the summed gross,
    then rounded once. Used only to compute the deliberate rounding_delta
    break — never for settlement_report.csv itself, which is always
    per-line."""
    rate = FEE_RATES[method]
    total_gross = sum(gross_paise_values)
    fee = _round_paise(Decimal(total_gross) * rate)
    tax = _round_paise(Decimal(fee) * GST_RATE)
    return fee, tax
