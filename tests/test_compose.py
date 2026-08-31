"""compose.py: bounded subset-sum composition, standalone (no pipeline
imports beyond models.SettlementLine). Caps must fire as refusals, never
hangs, and never tie-break when multiple compositions exist."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from unbatch.compose import (
    MAX_POOL,
    MAX_SUBSET,
    ComposeTimeoutError,
    PoolTooLargeError,
    compose,
)
from unbatch.models import SettlementLine, SettlementLineType


def _line(net_paise: int, line_id: str = "1") -> SettlementLine:
    return SettlementLine(
        settlement_id=f"setl_{line_id}",
        settlement_utr="AXISP000000000001",
        payment_id=f"pay_{line_id}",
        type=SettlementLineType.PAYMENT,
        gross_paise=net_paise,
        fee_paise=0,
        tax_paise=0,
        net_paise=net_paise,
        settled_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_finds_the_single_composing_subset() -> None:
    candidates = [_line(100, "a"), _line(200, "b"), _line(500, "c")]
    result = compose(300, candidates)

    assert len(result) == 1
    assert {line.payment_id for line in result[0]} == {"pay_a", "pay_b"}


def test_returns_empty_when_nothing_composes_the_target() -> None:
    candidates = [_line(100, "a"), _line(200, "b")]
    assert compose(999, candidates) == []


def test_returns_all_compositions_without_tie_breaking() -> None:
    """DATA_SPEC.md / ARCHITECTURE.md: multiple valid subsets must all come
    back, not be silently resolved to one. Three ways to reach 200 paise
    from {a, b, c} (each 100) plus {d} alone (200) — four valid subsets."""
    candidates = [_line(100, "a"), _line(100, "b"), _line(100, "c"), _line(200, "d")]
    result = compose(200, candidates)

    subsets = [frozenset(line.payment_id for line in subset) for subset in result]
    assert frozenset({"pay_a", "pay_b"}) in subsets
    assert frozenset({"pay_a", "pay_c"}) in subsets
    assert frozenset({"pay_b", "pay_c"}) in subsets
    assert frozenset({"pay_d"}) in subsets
    assert len(subsets) == 4


def test_pool_larger_than_max_pool_is_refused_immediately() -> None:
    """The deliberate blowup case: MAX_POOL+1 candidates must never reach
    the exponential search at all — proving the cap fires rather than the
    search hanging."""
    candidates = [_line(i + 1, str(i)) for i in range(MAX_POOL + 1)]

    started = time.monotonic()
    with pytest.raises(PoolTooLargeError):
        compose(sum(c.net_paise for c in candidates), candidates)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1, "PoolTooLargeError must fire before any search runs"


def test_pool_at_exactly_max_pool_boundary_is_accepted_not_refused() -> None:
    """The refusal is `> max_pool`, not `>= max_pool` — exactly at the cap
    must still be attempted. Uses a small custom max_pool rather than the
    real MAX_POOL=48 so the test stays fast: enumerating a genuine 48-item
    worst case can legitimately take longer than a couple of seconds in
    pure Python, which is exactly what the timeout guard exists for, not
    something a boundary test should need to wait out."""
    candidates = [_line(1, str(i)) for i in range(6)]
    result = compose(1, candidates, max_pool=6)
    assert len(result) == 6  # each single line sums to 1


def test_pool_one_over_a_custom_max_pool_is_still_refused() -> None:
    candidates = [_line(1, str(i)) for i in range(7)]
    with pytest.raises(PoolTooLargeError):
        compose(1, candidates, max_pool=6)


def test_subset_larger_than_max_subset_is_never_returned() -> None:
    """30 identical lines each worth 100 paise; only summing all 30 reaches
    3000, but MAX_SUBSET=25 forbids a subset that large — compose() must
    refuse to return it rather than exceed the cap."""
    candidates = [_line(100, str(i)) for i in range(30)]
    assert compose(3000, candidates, max_subset=MAX_SUBSET) == []
    # sanity: a reachable-within-cap target still works
    assert len(compose(2500, candidates, max_subset=MAX_SUBSET)) >= 1


def test_timeout_raises_rather_than_hangs() -> None:
    candidates = [_line(i + 1, str(i)) for i in range(10)]
    with pytest.raises(ComposeTimeoutError):
        compose(sum(c.net_paise for c in candidates), candidates, timeout_s=-1.0)
