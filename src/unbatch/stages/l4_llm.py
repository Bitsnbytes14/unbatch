"""L4 — LLM adjudication.

The only stage that calls `unbatch.adjudicator`, and therefore the only place
a model sees any data. Receives just what reached L4 unresolved, never the
full batch. Classifies why a break happened and proposes a resolution;
applies the confidence bands from ARCHITECTURE.md § Confidence bands
(>=0.85 auto-accept, 0.60-0.85 human_review_required, <0.60 exception).

No earlier stage populates `UnresolvedCredit.candidates` (L2/L3 either
resolve a credit outright or drop it unresolved with nothing recorded), so
this stage builds its own top-k candidates the same way L2/L3 already build
their date windows — from `expected_batches`, which every stage already
receives. For each credit, every batch whose settlement window overlaps
`[D - 3d, D]` is scored by `|credit - batch.net|`; the closest becomes the
"expected batch" the adjudicator reasons about, and the next few become
competing CandidateExplanations. A credit with nothing in its date window at
all (the unrelated_credit shape) falls back to the single globally-nearest
batch, so the model always has something concrete to reason against instead
of an empty prompt.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from unbatch import adjudicator
from unbatch.models import (
    CandidateExplanation,
    Decision,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    Stage,
    UnresolvedCredit,
)

CONFIDENCE_AUTO_ACCEPT = 0.85
CONFIDENCE_HUMAN_REVIEW = 0.60

DATE_WINDOW_DAYS = 3
TOP_K_CANDIDATES = 4

REASON_ADJUDICATION_FAILED = "adjudication_failed"


def _scored_batches(unresolved_credit: UnresolvedCredit) -> list[tuple[ExpectedBatch, int]]:
    """Every candidate batch paired with its delta against the credit,
    nearest first. Prefers batches whose window overlaps the same 3-day
    lookback L2/L3 use; falls back to the full batch list only when nothing
    is in that window, so a truly unrelated credit still gets the single
    closest-by-date batch as its point of comparison rather than nothing."""
    credit = unresolved_credit.credit
    window_start = credit.value_date - timedelta(days=DATE_WINDOW_DAYS)
    windowed = [
        batch
        for batch in unresolved_credit.expected_batches
        if batch.window_start <= credit.value_date and window_start <= batch.window_end
    ]
    pool = windowed or unresolved_credit.expected_batches
    scored = [(batch, credit.credit_paise - batch.net_paise) for batch in pool]
    scored.sort(key=lambda pair: (abs(pair[1]), pair[0].settlement_utr))
    return scored


def _pick_batch_and_candidates(
    unresolved_credit: UnresolvedCredit,
) -> tuple[ExpectedBatch, int, list[CandidateExplanation]]:
    scored = _scored_batches(unresolved_credit)
    primary_batch, delta_paise = scored[0]
    candidates = [
        CandidateExplanation(
            payment_ids=batch.payment_ids,
            delta_paise=delta,
            hint=(
                f"alternate batch {batch.settlement_utr}, "
                f"window {batch.window_start}..{batch.window_end}"
            ),
        )
        for batch, delta in scored[1 : 1 + TOP_K_CANDIDATES]
    ]
    return primary_batch, delta_paise, candidates


def _confidence_band_outcome(confidence: float, human_review_required: bool) -> DecisionOutcome:
    """ARCHITECTURE.md's fixed bands decide the default outcome; the model's
    own `human_review_required` flag can only push a confident call down to
    human review, never push an unconfident one up to auto-accept — bias to
    exception/review over a wrong match applies to the model's output just
    as much as to the rules layers."""
    if confidence < CONFIDENCE_HUMAN_REVIEW:
        return DecisionOutcome.EXCEPTION
    if confidence < CONFIDENCE_AUTO_ACCEPT or human_review_required:
        return DecisionOutcome.HUMAN_REVIEW
    return DecisionOutcome.MATCHED


def _decision(
    ctx: RunContext,
    credit_id: str,
    now: datetime,
    *,
    outcome: DecisionOutcome,
    confidence: float,
    delta_paise: int,
    reason: str,
    matched_payment_ids: list[str],
    rationale: str | None,
    llm_cost_paise: int,
    llm_retried: bool,
) -> Decision:
    return Decision(
        run_id=ctx.run_id,
        seed=ctx.seed,
        stage=Stage.L4,
        credit_id=credit_id,
        matched_payment_ids=matched_payment_ids,
        outcome=outcome,
        confidence=confidence,
        delta_paise=delta_paise,
        reason=reason,
        rationale=rationale,
        llm_model=adjudicator.MODEL,
        llm_cost_paise=llm_cost_paise,
        llm_retried=llm_retried,
        created_at=now,
    )


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Adjudicate every remaining unresolved credit via the LLM boundary and
    apply the confidence bands to route each to matched, human_review, or
    exception. Returns one Decision per credit in `unresolved` — this is the
    terminal stage, so every credit gets a Decision here."""
    now = datetime.now(UTC)
    decisions: list[Decision] = []

    for unresolved_credit in unresolved:
        credit = unresolved_credit.credit
        primary_batch, delta_paise, candidates = _pick_batch_and_candidates(unresolved_credit)

        try:
            result, cost_paise, retried = adjudicator.adjudicate(
                credit, primary_batch, delta_paise, candidates, cached=ctx.cached
            )
        except adjudicator.AdjudicationFailedError:
            decisions.append(
                _decision(
                    ctx,
                    credit.txn_id,
                    now,
                    outcome=DecisionOutcome.EXCEPTION,
                    confidence=0.0,
                    delta_paise=delta_paise,
                    reason=REASON_ADJUDICATION_FAILED,
                    matched_payment_ids=[],
                    rationale=None,
                    llm_cost_paise=0,
                    # degrading to adjudication_failed only ever happens
                    # after exactly one retry, by construction (see
                    # adjudicator.adjudicate's docstring)
                    llm_retried=True,
                )
            )
            continue

        outcome = _confidence_band_outcome(result.confidence, result.human_review_required)
        matched_payment_ids = (
            [] if outcome == DecisionOutcome.EXCEPTION else primary_batch.payment_ids
        )
        decisions.append(
            _decision(
                ctx,
                credit.txn_id,
                now,
                outcome=outcome,
                confidence=result.confidence,
                delta_paise=delta_paise,
                reason=result.break_reason.value,
                matched_payment_ids=matched_payment_ids,
                rationale=result.proposed_resolution,
                llm_cost_paise=cost_paise,
                llm_retried=retried,
            )
        )

    return decisions
