"""Money handling boundary: parse rupee decimal strings to int paise, format
paise back to rupee decimal strings. This is the ONLY place rupee decimal
strings are allowed to exist — everywhere else in the pipeline, money is
`int` paise (CLAUDE.md invariant 1).

`Paise` is the pydantic field type every money field in models.py uses: a
strict `int` that pydantic will not silently coerce from a float, even an
integral one like `41833.0`. A float showing up in a money path means float
arithmetic already happened upstream — that is the bug this guards against,
not just a type mismatch.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated

from pydantic import Field

PAISE_PER_RUPEE = 100

Paise = Annotated[int, Field(strict=True)]


def parse_rupees_to_paise(value: str) -> int:
    """Parse a rupee decimal string (e.g. "418.33", "-12.50") to int paise.

    Raises TypeError if `value` is not a `str` (a `float` here means float
    arithmetic already happened upstream, which is exactly the case this
    function exists to reject). Raises ValueError if the string is not a
    valid decimal or carries more than 2 decimal places.
    """
    if not isinstance(value, str):
        raise TypeError(f"parse_rupees_to_paise requires a str, got {type(value).__name__}")
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"not a valid decimal rupee string: {value!r}") from exc
    exponent = decimal_value.as_tuple().exponent
    if not isinstance(exponent, int) or exponent < -2:
        raise ValueError(f"more than 2 decimal places: {value!r}")
    return int(decimal_value * PAISE_PER_RUPEE)


def format_paise_to_rupees(paise: int) -> str:
    """Format int paise to a 2dp rupee decimal string (e.g. 41833 -> "418.33")."""
    if not isinstance(paise, int) or isinstance(paise, bool):
        raise TypeError(f"format_paise_to_rupees requires an int, got {type(paise).__name__}")
    return str((Decimal(paise) / PAISE_PER_RUPEE).quantize(Decimal("0.01")))
