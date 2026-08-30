"""L2 — batch composition.

Resolves splits and partial settlements by finding the payment subset(s) that
compose a credit, via `unbatch.compose`. Confidence 0.90. If composition
yields more than one valid subset, none is picked — all are attached to the
credit as candidate explanations for L4. See ARCHITECTURE.md § L2.
"""

from __future__ import annotations

from unbatch.models import Decision, RunContext, UnresolvedCredit

CONFIDENCE = 0.90


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Attempt batch composition for each credit. Returns one Decision per
    credit this stage resolves outright; credits with zero or multiple valid
    compositions remain unresolved, carrying candidate explanations forward
    for L3/L4."""
    raise NotImplementedError
