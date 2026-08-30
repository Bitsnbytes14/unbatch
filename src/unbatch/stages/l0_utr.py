"""L0 — UTR exact match.

Resolves a credit when the settlement UTR appears verbatim in the bank
narration. Cheapest and most certain stage; confidence 1.00. See
ARCHITECTURE.md § The cascade.
"""

from __future__ import annotations

from unbatch.models import Decision, RunContext, UnresolvedCredit

CONFIDENCE = 1.00


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Match each credit whose narration contains an exact settlement UTR.
    Returns one Decision per credit this stage resolves; any credit with no
    returned Decision remains unresolved for the next stage."""
    raise NotImplementedError
