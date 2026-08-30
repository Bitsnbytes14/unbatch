"""Money handling: paise as int throughout. See CLAUDE.md invariant 1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from unbatch.models import SettlementLine, SettlementLineType
from unbatch.money import format_paise_to_rupees, parse_rupees_to_paise


@pytest.mark.parametrize(
    "rupees",
    ["418.33", "0.00", "1000000.00", "-12.50", "5.00", "0.01"],
)
def test_round_trip(rupees: str) -> None:
    assert format_paise_to_rupees(parse_rupees_to_paise(rupees)) == rupees


def test_negative_amounts() -> None:
    assert parse_rupees_to_paise("-12.50") == -1250
    assert format_paise_to_rupees(-1250) == "-12.50"


@pytest.mark.parametrize("bad", ["abc", "", "12.345", "12.5.6", "₹12.50"])
def test_malformed_input_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_rupees_to_paise(bad)


def test_float_input_to_parse_is_rejected() -> None:
    with pytest.raises(TypeError):
        parse_rupees_to_paise(0.1)  # type: ignore[arg-type]


def test_float_input_to_format_is_rejected() -> None:
    with pytest.raises(TypeError):
        format_paise_to_rupees(41833.0)  # type: ignore[arg-type]


def test_bool_input_to_format_is_rejected() -> None:
    with pytest.raises(TypeError):
        format_paise_to_rupees(True)  # type: ignore[arg-type]


def test_no_float_drift_for_the_classic_case() -> None:
    """The whole reason invariant 1 exists: 0.1 + 0.2 != 0.3 in float."""
    assert 0.1 + 0.2 != 0.3

    ten_paise = parse_rupees_to_paise("0.10")
    twenty_paise = parse_rupees_to_paise("0.20")
    assert ten_paise + twenty_paise == 30
    assert format_paise_to_rupees(ten_paise + twenty_paise) == "0.30"


def test_paise_field_rejects_float_even_when_integral() -> None:
    """A pydantic `int` field would silently coerce 100.0 -> 100. `Paise` must not."""
    with pytest.raises(ValidationError):
        SettlementLine(
            settlement_id="setl_1",
            settlement_utr="UTR1",
            payment_id="pay_1",
            type=SettlementLineType.PAYMENT,
            gross_paise=100.0,  # type: ignore[arg-type]
            fee_paise=2,
            tax_paise=0,
            net_paise=98,
            settled_at="2024-01-02T10:00:00+05:30",
        )
