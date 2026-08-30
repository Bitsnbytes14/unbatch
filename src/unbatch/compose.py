"""Batch composition — bounded subset-sum over candidate settlement lines.

Given a bank credit's amount and date, finds the payment subset(s) whose net
composes the credit. See ARCHITECTURE.md § L2 for the pruning strategy this
must implement: date-window filtering first, then hard caps on pool and
subset size, meet-in-the-middle (or paise-bucket DP) over what remains, and a
wall-clock timeout guard. Never a plain recursive solver — it is exponential
and will blow up.

If more than one subset composes the target, all of them are returned; the
caller (L2) hands every candidate to L4 as competing explanations rather than
picking one.
"""

from __future__ import annotations

from unbatch.models import SettlementLine

MAX_POOL = 48
MAX_SUBSET = 25


class PoolTooLargeError(Exception):
    """Raised when the candidate pool exceeds MAX_POOL; caller should emit an
    exception decision with reason `pool_too_large` rather than attempt
    composition."""


class ComposeTimeoutError(Exception):
    """Raised when composition exceeds its wall-clock budget; caller should
    emit an exception decision with reason `compose_timeout`."""


def compose(
    target_paise: int,
    candidates: list[SettlementLine],
    *,
    max_pool: int = MAX_POOL,
    max_subset: int = MAX_SUBSET,
    timeout_s: float = 2.0,
) -> list[list[SettlementLine]]:
    """Return every subset of `candidates` whose net sums to `target_paise`.

    Raises PoolTooLargeError if len(candidates) > max_pool, ComposeTimeoutError
    if the search exceeds timeout_s. Returns an empty list if no subset
    composes the target.
    """
    raise NotImplementedError
