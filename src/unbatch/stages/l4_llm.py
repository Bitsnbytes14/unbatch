"""L4 — LLM adjudication.

The only stage that calls `unbatch.adjudicator`, and therefore the only place
a model sees any data. Receives just what reached L4 unresolved, never the
full batch. Classifies why a break happened and proposes a resolution;
applies the confidence bands from ARCHITECTURE.md § Confidence bands
(>=0.85 auto-accept, 0.60-0.85 human_review_required, <0.60 exception).
"""

from __future__ import annotations

from unbatch.models import Decision, RunContext, UnresolvedCredit

CONFIDENCE_AUTO_ACCEPT = 0.85
CONFIDENCE_HUMAN_REVIEW = 0.60


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Adjudicate every remaining unresolved credit via the LLM boundary and
    apply the confidence bands to route each to matched, human_review, or
    exception. Returns one Decision per credit in `unresolved` — this is the
    terminal stage, so every credit gets a Decision here."""
    raise NotImplementedError
