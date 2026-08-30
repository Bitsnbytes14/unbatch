"""L0 — UTR exact match.

Resolves a credit when the settlement UTR appears verbatim in the bank
narration AND the credit ties exactly, on both amount and date, to that
same batch. All three conditions are required, not just the UTR:

- A truncated or absent UTR never satisfies the narration substring check,
  so narration_mangled correctly falls through to L1 rather than
  false-matching here — this is the false-match this stage must not make.
- Requiring the amount to match exactly is what keeps rounding_delta and
  fee_tier_change (UTR intact, amount off by a small delta) from resolving
  here; they are meant to surface at L3's tolerance band instead.
- Requiring the date to match exactly is what keeps date_skew (UTR and
  amount both intact, only the posting date shifts by a day) from
  resolving here; it is meant to surface at L2's wider date window.

Cheapest and most certain stage in the cascade; confidence 1.00.
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

CONFIDENCE = 1.00
REASON = "utr_exact_match"


def _find_exact_match(unresolved_credit: UnresolvedCredit) -> ExpectedBatch | None:
    """The one batch (if any) whose UTR appears in the narration and whose
    net and window both tie to the credit exactly. More than one match is
    an ambiguity this stage is not certain enough to resolve — bias to
    exception over wrong match — so it returns None just like finding
    none."""
    credit = unresolved_credit.credit
    matches = [
        batch
        for batch in unresolved_credit.expected_batches
        if batch.settlement_utr in credit.narration
        and batch.net_paise == credit.credit_paise
        and batch.window_start <= credit.value_date <= batch.window_end
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Match each credit whose narration contains the exact settlement UTR
    of a batch it also ties to exactly on amount and date. Returns one
    Decision per credit this stage resolves; any credit with no returned
    Decision remains unresolved for the next stage."""
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
                stage=Stage.L0,
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
