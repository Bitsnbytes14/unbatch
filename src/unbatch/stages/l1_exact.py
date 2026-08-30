"""L1 — amount + date exact match.

Resolves a credit when it ties exactly to a computed batch net, catching
clean cases where the narration is mangled and L0 couldn't find a UTR.
Confidence 0.98. See ARCHITECTURE.md § The cascade.
"""

from __future__ import annotations

from unbatch.models import Decision, RunContext, UnresolvedCredit

CONFIDENCE = 0.98


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Match each credit whose amount and date tie exactly to one expected
    batch. Returns one Decision per credit this stage resolves; any credit
    with no returned Decision remains unresolved for the next stage."""
    raise NotImplementedError
