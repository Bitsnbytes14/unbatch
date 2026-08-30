"""Batch composition — bounded subset-sum over candidate settlement lines.

Given a bank credit's amount and date, finds the settlement-line subset(s)
whose net composes the credit. Standalone: this module imports nothing from
the rest of the pipeline (only `unbatch.models` for the `SettlementLine`
type), so it can be tested and reasoned about in complete isolation from the
cascade that calls it.

Bounded per ARCHITECTURE.md § L2:

- **MAX_POOL** (48): candidate pools larger than this are refused outright
  (`PoolTooLargeError`) before any search runs. Refusing to solve is correct
  behaviour; hanging is not.
- **MAX_SUBSET** (25): no returned subset may use more than this many lines.
- **Meet-in-the-middle**: the pool is split in half, every achievable subset
  sum is enumerated for each half (bounded by MAX_SUBSET), and the two
  halves are merged — an exact hashmap lookup for `compose()`, a sorted
  range scan for `compose_within_tolerance()`. Never a plain recursive
  solver; that is exponential in a way this deliberately is not (each half
  is at most 2^24 enumerated sums, not 2^48).
- **Wall-clock timeout**: checked periodically during enumeration and before
  the merge step. A breach raises `ComposeTimeoutError` rather than running
  indefinitely — the caps bound the *worst case*, not the *typical* case,
  and the timeout is the backstop for when even a capped pool is too slow.

If more than one subset composes the target, `compose()` returns all of
them — the caller (L2) hands every candidate to L4 as competing explanations
rather than picking one.
"""

from __future__ import annotations

import bisect
import time
from collections import defaultdict

from unbatch.models import SettlementLine

MAX_POOL = 48
MAX_SUBSET = 25
# A genuine 48-candidate worst case (2^24 enumerated sums per half) can take
# several seconds in pure Python — measured empirically at over 2s per half
# while building this module's tests. 5s covers that comfortably without
# masking a real hang; this dataset's actual candidate pools (single digits
# to a few dozen within a 3-day window) resolve in a small fraction of this.
DEFAULT_TIMEOUT_S = 5.0


class PoolTooLargeError(Exception):
    """The candidate pool exceeds MAX_POOL; caller should emit a decision
    with reason `pool_too_large` rather than attempt composition."""


class ComposeTimeoutError(Exception):
    """Composition exceeded its wall-clock budget; caller should emit a
    decision with reason `compose_timeout`."""


def _enumerate_subset_sums(
    items: list[SettlementLine], max_subset: int, deadline: float
) -> list[tuple[int, tuple[int, ...]]]:
    """Every (sum, index-tuple) pair achievable from a subset of `items` no
    larger than `max_subset`, including the empty subset. Built by doubling
    rather than by testing every bitmask directly — the doubling step is
    O(1) per existing entry (a tuple-append and an int-add), so the total
    work across all n items is O(2^n), not O(2^n * n).
    """
    results: list[tuple[int, tuple[int, ...]]] = [(0, ())]
    for i, item in enumerate(items):
        extension = [
            (total + item.net_paise, idxs + (i,))
            for total, idxs in results
            if len(idxs) < max_subset
        ]
        results.extend(extension)
        if time.monotonic() > deadline:
            raise ComposeTimeoutError(
                f"composition timed out enumerating subsets ({len(items)} candidates)"
            )
    return results


def _check_pool_size(candidates: list[SettlementLine], max_pool: int) -> None:
    if len(candidates) > max_pool:
        raise PoolTooLargeError(
            f"candidate pool of {len(candidates)} exceeds MAX_POOL={max_pool}"
        )


def compose(
    target_paise: int,
    candidates: list[SettlementLine],
    *,
    max_pool: int = MAX_POOL,
    max_subset: int = MAX_SUBSET,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[list[SettlementLine]]:
    """Return every subset of `candidates` whose net sums to exactly
    `target_paise`, each no larger than `max_subset` lines.

    Raises PoolTooLargeError if len(candidates) > max_pool, ComposeTimeoutError
    if the search exceeds timeout_s. Returns an empty list if no subset
    composes the target — that is a normal, expected outcome, not an error.
    """
    _check_pool_size(candidates, max_pool)
    deadline = time.monotonic() + timeout_s

    midpoint = len(candidates) // 2
    left, right = candidates[:midpoint], candidates[midpoint:]

    left_sums = _enumerate_subset_sums(left, max_subset, deadline)
    right_sums = _enumerate_subset_sums(right, max_subset, deadline)

    right_by_sum: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for total, idxs in right_sums:
        right_by_sum[total].append(idxs)

    results: list[list[SettlementLine]] = []
    for total, left_idxs in left_sums:
        if time.monotonic() > deadline:
            raise ComposeTimeoutError(
                f"composition timed out merging halves ({len(candidates)} candidates)"
            )
        for right_idxs in right_by_sum.get(target_paise - total, []):
            if len(left_idxs) + len(right_idxs) == 0:
                continue  # the empty subset never composes a real credit
            if len(left_idxs) + len(right_idxs) > max_subset:
                continue
            subset_indices = left_idxs + tuple(m + midpoint for m in right_idxs)
            results.append([candidates[i] for i in subset_indices])
    return results


def compose_within_tolerance(
    target_paise: int,
    candidates: list[SettlementLine],
    tolerance_paise: int,
    *,
    max_pool: int = MAX_POOL,
    max_subset: int = MAX_SUBSET,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[tuple[list[SettlementLine], int]]:
    """Return every subset of `candidates` whose net sum falls within
    `tolerance_paise` of `target_paise`, each paired with its signed delta
    (subset_sum - target_paise). Same caps and refusals as `compose()`; used
    by L3 once L2's exact search has already failed.
    """
    _check_pool_size(candidates, max_pool)
    deadline = time.monotonic() + timeout_s

    midpoint = len(candidates) // 2
    left, right = candidates[:midpoint], candidates[midpoint:]

    left_sums = _enumerate_subset_sums(left, max_subset, deadline)
    right_sums = sorted(_enumerate_subset_sums(right, max_subset, deadline), key=lambda x: x[0])
    right_values = [total for total, _ in right_sums]

    results: list[tuple[list[SettlementLine], int]] = []
    for total, left_idxs in left_sums:
        if time.monotonic() > deadline:
            raise ComposeTimeoutError(
                f"composition timed out merging halves ({len(candidates)} candidates)"
            )
        lo = bisect.bisect_left(right_values, target_paise - tolerance_paise - total)
        hi = bisect.bisect_right(right_values, target_paise + tolerance_paise - total)
        for right_total, right_idxs in right_sums[lo:hi]:
            if len(left_idxs) + len(right_idxs) == 0:
                continue
            if len(left_idxs) + len(right_idxs) > max_subset:
                continue
            subset_indices = left_idxs + tuple(m + midpoint for m in right_idxs)
            delta = (total + right_total) - target_paise
            results.append(([candidates[i] for i in subset_indices], delta))
    return results
