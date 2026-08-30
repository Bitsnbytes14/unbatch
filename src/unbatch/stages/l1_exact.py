"""L1 — amount + date exact.

Resolves a credit when it ties exactly, on amount and date, to a batch's
computed net — the same test L0 applies, minus the UTR requirement. This is
what catches narration_mangled: a truncated or absent UTR means L0 never
even considered these credits, but the underlying arithmetic still works out
exactly. Confidence 0.98, one notch below L0 because there is no UTR to
independently confirm identity — only the numbers agreeing.

Like L0, this stage will not resolve rounding_delta or fee_tier_change
(amount is off by a small delta, not exact) or date_skew (amount ties but
the posting date doesn't) — those fall through by the same logic, just
without a UTR check to also fail on.
"""

from __future__ import annotations

from datetime import UTC, datetime

from unbatch.models import (
    Decision,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    Stage,
    UnresolvedCredit,
)

CONFIDENCE = 0.98
REASON = "amount_date_exact_match"


def _find_exact_match(unresolved_credit: UnresolvedCredit) -> ExpectedBatch | None:
    """The one batch (if any) whose net and window both tie to the credit
    exactly, independent of narration. More than one match is an ambiguity
    this stage is not certain enough to resolve — bias to exception over
    wrong match — so it returns None just like finding none."""
    credit = unresolved_credit.credit
    matches = [
        batch
        for batch in unresolved_credit.expected_batches
        if batch.net_paise == credit.credit_paise
        and batch.window_start <= credit.value_date <= batch.window_end
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Match each credit whose amount and date tie exactly to one expected
    batch, regardless of narration. Returns one Decision per credit this
    stage resolves; any credit with no returned Decision remains unresolved
    for the next stage."""
    now = datetime.now(UTC)
    decisions: list[Decision] = []
    for unresolved_credit in unresolved:
        batch = _find_exact_match(unresolved_credit)
        if batch is None:
            continue
        decisions.append(
            Decision(
                run_id=ctx.run_id,
                seed=ctx.seed,
                stage=Stage.L1,
                credit_id=unresolved_credit.credit.txn_id,
                matched_payment_ids=batch.payment_ids,
                outcome=DecisionOutcome.MATCHED,
                confidence=CONFIDENCE,
                delta_paise=0,
                reason=REASON,
                rationale=None,
                llm_model=None,
                llm_cost_paise=None,
                created_at=now,
            )
        )
    return decisions
