"""Property-based tests for money.py. CLAUDE.md invariant 1: money is int
paise, never float — these properties hold for every value in their domain,
not just the hand-picked examples in test_money.py."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unbatch.money import format_paise_to_rupees, parse_rupees_to_paise


@given(st.integers(min_value=-(10**12), max_value=10**12))
def test_paise_round_trips_through_rupees_and_back(paise: int) -> None:
    rupees = format_paise_to_rupees(paise)
    assert parse_rupees_to_paise(rupees) == paise


@given(st.integers(min_value=-(10**12), max_value=10**12))
def test_parsed_paise_is_always_a_plain_int_never_a_float(paise: int) -> None:
    rupees = format_paise_to_rupees(paise)
    result = parse_rupees_to_paise(rupees)
    assert type(result) is int


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_no_float_is_ever_silently_accepted_as_an_amount(value: float) -> None:
    """A float reaching this boundary means float arithmetic already
    happened upstream — that's the bug this rejects, not just a type
    mismatch (see money.py's module docstring)."""
    with pytest.raises(TypeError):
        parse_rupees_to_paise(value)  # type: ignore[arg-type]


@given(st.integers(min_value=-(10**12), max_value=10**12))
def test_format_never_produces_more_than_two_decimal_places(paise: int) -> None:
    rupees = format_paise_to_rupees(paise)
    decimal_places = len(rupees.split(".")[1])
    assert decimal_places == 2
