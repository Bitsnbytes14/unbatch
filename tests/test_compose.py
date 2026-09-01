"""compose.py: bounded subset-sum composition, standalone (no pipeline
imports beyond models.SettlementLine). Caps must fire as refusals, never
hangs, and never tie-break when multiple compositions exist."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from unbatch.compose import (
    DEFAULT_TIMEOUT_S,
    MAX_POOL,
    MAX_SUBSET,
    ComposeTimeoutError,
    PoolTooLargeError,
    _enumerate_subset_sums,
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


# --- Mutation testing follow-ups (see bench_mutation.json) ---
#
# These pin down constants and internal contracts that the tests above don't
# touch directly, because cosmic-ray found real gaps: mutating them didn't
# change the outcome of any existing test.


def test_max_pool_is_48() -> None:
    assert MAX_POOL == 48


def test_max_subset_is_25() -> None:
    assert MAX_SUBSET == 25


def test_default_timeout_is_5_seconds() -> None:
    assert DEFAULT_TIMEOUT_S == 5.0


def test_max_pool_etc_must_be_passed_by_keyword() -> None:
    """The `*,` marker enforces this — a positional 3rd argument is almost
    certainly a caller confusing max_pool with something else; TypeError
    should catch that immediately rather than silently accepting it."""
    with pytest.raises(TypeError):
        compose(300, [], 6)  # type: ignore[misc]


def test_enumerate_subset_sums_never_exceeds_max_subset() -> None:
    """compose()'s own oversized-subset filter happens to mask this helper's
    internal pruning at the whole-function level (see the equivalent-mutant
    note in bench_mutation.json), so the pruning contract itself needs a
    direct test of the helper."""
    items = [_line(1, str(i)) for i in range(5)]
    results = _enumerate_subset_sums(items, max_subset=2, deadline=time.monotonic() + 10)

    assert all(len(idxs) <= 2 for _, idxs in results)
    assert any(len(idxs) == 2 for _, idxs in results)  # the boundary is actually reached


def test_enumerate_subset_sums_raises_on_an_already_past_deadline() -> None:
    items = [_line(1, "a")]
    with pytest.raises(ComposeTimeoutError):
        _enumerate_subset_sums(items, max_subset=MAX_SUBSET, deadline=time.monotonic() - 1)


def test_enumerate_subset_sums_does_not_raise_exactly_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check is `>`, not `>=` — being exactly at the deadline (not yet
    past it) must not raise. An already-past deadline doesn't discriminate
    these (both are true), so this pins real-time equality via a mocked
    clock instead."""
    monkeypatch.setattr("unbatch.compose.time.monotonic", lambda: 100.0)
    result = _enumerate_subset_sums([_line(1, "a")], max_subset=MAX_SUBSET, deadline=100.0)
    assert result  # completed without raising


def test_compose_checks_timeout_before_merging_even_with_no_candidates() -> None:
    """Isolates the merge loop's own timeout check from the one inside
    _enumerate_subset_sums: an empty candidate list means enumeration does
    zero iterations on either half (the deadline check lives inside the
    per-item loop, never reached), so only the merge loop's own check can
    catch an already-expired deadline here."""
    with pytest.raises(ComposeTimeoutError):
        compose(0, [], timeout_s=-1.0)


def test_compose_merge_loop_does_not_raise_exactly_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same `>` vs `>=` boundary as the enumeration check above, isolated to
    the merge loop's own separate check via an already-past-vs-exact clock."""
    monkeypatch.setattr("unbatch.compose.time.monotonic", lambda: 100.0)
    assert compose(0, [], timeout_s=0.0) == []  # deadline == 100.0, exactly now


def test_deadline_is_computed_by_adding_timeout_not_another_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins deadline = now + timeout_s against **/* mutants. Real wall-clock
    values don't discriminate these (an already-large monotonic() reading
    swamps any plausible timeout_s under every one of +, **, and * alike),
    so this scripts the clock instead of relying on real timing."""
    calls = {"n": 0}

    def fake_monotonic() -> float:
        calls["n"] += 1
        return 100.0 if calls["n"] == 1 else 111.0

    monkeypatch.setattr("unbatch.compose.time.monotonic", fake_monotonic)

    with pytest.raises(ComposeTimeoutError):
        compose(999, [_line(1, "a")], timeout_s=10.0)


def test_target_zero_never_returns_the_empty_subset() -> None:
    """compose()'s contract is subsets that compose the target, never the
    trivial empty one — a credit is never literally zero, but the function
    itself must still refuse to report "the empty set matches" for target 0."""
    assert compose(0, [_line(1, "a"), _line(2, "b")]) == []


def test_empty_subset_skip_does_not_abort_the_rest_of_the_merge_bucket() -> None:
    """continue, not break: skipping the (structurally always-first) empty
    combination in a sum bucket must not also skip every real composition
    still in that bucket — e.g. two lines whose net cancels to zero."""
    q = _line(50, "q")
    payment = _line(100, "p")
    refund = _line(-100, "r")

    result = compose(0, [q, payment, refund])

    assert len(result) == 1
    assert {line.payment_id for line in result[0]} == {"pay_p", "pay_r"}


def test_oversized_subset_skip_does_not_abort_the_rest_of_the_merge_bucket() -> None:
    """continue, not break: an over-max_subset combination sharing a sum
    bucket with a valid, in-budget one must not shadow the valid one."""
    l0 = _line(1, "l0")
    l1 = _line(12_345, "l1")
    r0 = _line(30, "r0")
    r1 = _line(20, "r1")
    r2 = _line(50, "r2")

    result = compose(51, [l0, l1, r0, r1, r2], max_subset=2)

    assert len(result) == 1
    assert {line.payment_id for line in result[0]} == {"pay_l0", "pay_r2"}
