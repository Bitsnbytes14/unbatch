"""L2 — batch composition.

Resolves splits and partial settlements: given whatever L0/L1 left
unresolved, restrict the candidate settlement lines to the date window
`[D - 3d, D]` (D = the credit's value_date) — the dominant prune per
ARCHITECTURE.md, taking the pool from however many lines the runner handed
this credit down to the handful actually plausible — then hand that window
to `compose.compose()`.

- Exactly one subset composes the credit exactly: MATCH, confidence 0.90.
- Zero subsets: falls through to L3, which allows a small tolerance band.
- More than one subset: a genuine composition ambiguity. This stage is not
  certain enough to pick one — bias to exception over wrong match — so it
  writes no Decision and the credit falls through unresolved, same as
  finding zero. (L4 will eventually adjudicate between them; L4 is a stub
  this session, so under --no-llm this ends as a terminal exception.)
- `compose()` refuses rather than searches when the pool exceeds MAX_POOL or
  the search times out; both refusals are terminal here and get their own
  exception Decision immediately, exactly as ARCHITECTURE.md specifies.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from unbatch.compose import ComposeTimeoutError, PoolTooLargeError, compose
from unbatch.models import (
    Decision,
    DecisionOutcome,
    RunContext,
    SettlementLine,
    Stage,
    UnresolvedCredit,
)

CONFIDENCE = 0.90

REASON_MATCH = "single_composition_match"
REASON_POOL_TOO_LARGE = "pool_too_large"
REASON_TIMEOUT = "compose_timeout"


def _candidates_in_window(
    unresolved_credit: UnresolvedCredit, ctx: RunContext
) -> list[SettlementLine]:
    window_start = unresolved_credit.credit.value_date - timedelta(
        days=ctx.config.date_window_days
    )
    window_end = unresolved_credit.credit.value_date
    return [
        line
        for line in unresolved_credit.candidate_lines
        if window_start <= line.settled_at.date() <= window_end
    ]


def _decision(
    ctx: RunContext,
    credit_id: str,
    now: datetime,
    *,
    outcome: DecisionOutcome,
    confidence: float,
    reason: str,
    matched_payment_ids: list[str],
) -> Decision:
    return Decision(
        run_id=ctx.run_id,
        seed=ctx.seed,
        stage=Stage.L2,
        credit_id=credit_id,
        matched_payment_ids=matched_payment_ids,
        outcome=outcome,
        confidence=confidence,
        delta_paise=0,
        reason=reason,
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=now,
    )


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Attempt batch composition for each credit within its date window.
    Returns one Decision per credit this stage resolves outright (a single
    exact composition) or refuses outright (pool_too_large /
    compose_timeout); credits with zero or multiple valid compositions
    remain unresolved, carrying forward for L3."""
    now = datetime.now(UTC)
    decisions: list[Decision] = []

    for unresolved_credit in unresolved:
        pool = _candidates_in_window(unresolved_credit, ctx)
        credit = unresolved_credit.credit

        try:
            results = compose(
                credit.credit_paise,
                pool,
                max_pool=ctx.config.max_pool,
                max_subset=ctx.config.max_subset,
                timeout_s=ctx.config.compose_timeout_s,
            )
        except PoolTooLargeError:
            decisions.append(
                _decision(
                    ctx,
                    credit.txn_id,
                    now,
                    outcome=DecisionOutcome.EXCEPTION,
                    confidence=0.0,
                    reason=REASON_POOL_TOO_LARGE,
                    matched_payment_ids=[],
                )
            )
            continue
        except ComposeTimeoutError:
            decisions.append(
                _decision(
                    ctx,
                    credit.txn_id,
                    now,
                    outcome=DecisionOutcome.EXCEPTION,
                    confidence=0.0,
                    reason=REASON_TIMEOUT,
                    matched_payment_ids=[],
                )
            )
            continue

        if len(results) != 1:
            continue  # zero: falls through to L3. multiple: ambiguous, needs L4.

        decisions.append(
            _decision(
                ctx,
                credit.txn_id,
                now,
                outcome=DecisionOutcome.MATCHED,
                confidence=CONFIDENCE,
                reason=REASON_MATCH,
                matched_payment_ids=[line.payment_id for line in results[0]],
            )
        )

    return decisions
