"""L3 — tolerance band.

Resolves an L2 composition that lands within a fee/rounding delta band,
recording the delta rather than ignoring it. Catches fee-tier drift and
paise-level rounding. Confidence 0.75. See ARCHITECTURE.md § The cascade.
"""

from __future__ import annotations

from unbatch.models import Decision, RunContext, UnresolvedCredit

CONFIDENCE = 0.75


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Match each credit whose best composition is within tolerance of the
    expected net, recording delta_paise on the Decision. Returns one Decision
    per credit this stage resolves; any credit with no returned Decision
    remains unresolved for L4."""
    raise NotImplementedError
