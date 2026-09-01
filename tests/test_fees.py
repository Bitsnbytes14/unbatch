"""Fee and GST computation, and the per-line vs per-batch rounding decision.
See fees.py's module docstring for why per-line is authoritative."""

from __future__ import annotations

from unbatch.fees import (
    compute_batch_fee_and_tax_paise,
    compute_fee_paise,
    compute_net_paise,
    compute_tax_paise,
)
from unbatch.models import PaymentMethod


def test_compute_fee_paise_uses_per_method_rate() -> None:
    # UPI at 0.3%: 100000 paise * 0.003 = 300 paise exactly.
    assert compute_fee_paise(100_000, PaymentMethod.UPI) == 300
    # Card at 2%: 100000 paise * 0.02 = 2000 paise exactly.
    assert compute_fee_paise(100_000, PaymentMethod.CARD) == 2000


def test_compute_tax_paise_is_gst_on_fee() -> None:
    # 250 paise fee * 18% = 45.0 paise exactly.
    assert compute_tax_paise(250) == 45


def test_compute_net_paise_subtracts_fee_and_tax() -> None:
    assert compute_net_paise(100_000, 2000, 360) == 97_640


def test_compute_net_paise_negative_for_refund_shape() -> None:
    # A refund line: negative gross, zero fee/tax, net mirrors gross.
    assert compute_net_paise(-50_000, 0, 0) == -50_000


def test_fee_rounding_boundary_rounds_half_up() -> None:
    # Card at 2%: 25 paise gross * 0.02 = 0.5 paise exactly -> rounds up to 1.
    assert compute_fee_paise(25, PaymentMethod.CARD) == 1
    # 75 paise gross * 0.02 = 1.5 paise exactly -> rounds up to 2.
    assert compute_fee_paise(75, PaymentMethod.CARD) == 2


def test_tax_rounding_boundary_rounds_half_up() -> None:
    # 25 paise fee * 18% = 4.5 paise exactly -> rounds up to 5.
    assert compute_tax_paise(25) == 5


def test_batch_tax_is_gst_on_the_batch_fee() -> None:
    """compute_batch_fee_and_tax_paise's own tax figure, not just its fee —
    test_per_line_vs_per_batch_rounding_diverges only ever checks batch_fee
    and discards the tax half of the tuple."""
    fee, tax = compute_batch_fee_and_tax_paise([5_000], PaymentMethod.CARD)
    assert fee == 100  # 5000 * 2% = 100 exactly
    assert tax == 18  # 100 * 18% = 18 exactly


def test_per_line_vs_per_batch_rounding_diverges() -> None:
    """This is the exact mechanism rounding_delta exploits: three lines each
    sitting on a half-paise boundary round up individually, but the same
    three lines summed first and rounded once do not round up by as much."""
    gross_values = [25, 25, 25]  # each: 25 * 2% = 0.5 paise, a half-paise boundary

    per_line_fees = [compute_fee_paise(g, PaymentMethod.CARD) for g in gross_values]
    assert sum(per_line_fees) == 3  # 0.5 paise rounds up to 1, three times

    batch_fee, _batch_tax = compute_batch_fee_and_tax_paise(gross_values, PaymentMethod.CARD)
    assert batch_fee == 2  # 75 paise * 2% = 1.5 paise rounds up to 2, once

    assert sum(per_line_fees) != batch_fee
