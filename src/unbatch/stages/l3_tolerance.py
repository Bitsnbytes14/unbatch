"""L3 — tolerance.

Checks, does not search. For each unresolved credit, take the expected
batches (already grouped by settlement UTR — that grouping is known from
compute_expected_batches, not something to rediscover) whose settlement
window overlaps `[D - 3d, D]` (D = the credit's value_date), and compare
each one's precomputed net directly against the credit: `|credit -
batch.net|`. Exactly one batch within tolerance: MATCH, confidence 0.75,
delta recorded. Zero or more than one: falls through unresolved.

This used to be a composition search (`compose_within_tolerance` scanning
every subset of the candidate settlement lines within the band) and that
was the wrong operation, not just mistuned — see FAILURES.md's 2026-08-30
entry on the ~90-match blowup. A pool of even a dozen lines has thousands
of subsets; any band wide enough to admit a real fee-tier drift is wide
enough to admit dozens of coincidental combinations of unrelated lines.
Narrowing the band only moves where the coincidences start — it never
removes them, because the search itself was the wrong tool. Batches are far
fewer than subsets-of-lines, and a batch's net is a real, already-computed
quantity, not an assembled one, so this check has nothing left to be
coincidentally right or wrong about.

**The band, and why it's this size, not tuned to make numbers look good:**
`TOLERANCE_RATE = 0.006` (0.6% of the credit amount), floored at
`TOLERANCE_FLOOR_PAISE = 50` for very small credits. Derived from the fee
structure this project's own fees.py defines, not picked to fit this
dataset's two observed cases:

- Pure rounding noise (per-line vs per-batch GST rounding, see fees.py) is
  bounded by a few paise per batch — utterly swamped by a 50-paise floor.
- A plausible fee-tier drift — a gateway quietly moving a merchant's rate by
  up to half a percentage point, the largest change generate.py's own
  FEE_TIER_BUMP models — costs `0.5% x 1.18` (GST on the extra fee) =
  ~0.59% of the affected line's gross. Rounded up to 0.6% of the *credit*
  (necessarily somewhat diluted by whatever else is in the same batch, so
  0.6% of the whole credit is already generous relative to a single line's
  0.59%).

Wider than this starts accepting deltas explainable only by an actually
wrong batch — a false match. Narrower starts rejecting a legitimate
half-point fee change — a real settlement pushed to exception for no
reason. See ARCHITECTURE.md's § Confidence bands for the full reasoning;
this module only encodes the number.

**2026-09-03 false-accept fix (see FAILURES.md):** `bench --seeds` found
this stage confidently matching `ambiguous_composition` credits on some
seeds — the deliberately-unclaimed line (see l4_llm.py's module docstring
and DATA_SPEC.md) sometimes happens to be small enough that the credit
lands within tolerance of its own whole batch, and this stage had no way to
tell that gap apart from a genuine fee-tier drift. The distinction, derived
from what a tolerance band is *for* rather than tuned to this dataset: a
band absorbs fee and rounding noise, never a whole missing settlement line.
So before accepting the one within-tolerance batch, this stage now checks
whether the delta exactly equals the net of any settlement line still
sitting in the credit's own date-windowed pool — if it does, that gap has a
real, named explanation sitting right there, which is a composition
question L2 already looked at and declined, not tolerance noise. Refusing
here is strictly narrower than before (a batch that used to match no longer
does when a line exactly explains the gap); it never widens or narrows
`TOLERANCE_RATE`/`TOLERANCE_FLOOR_PAISE` themselves.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from unbatch.models import (
    Decision,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    SettlementLine,
    Stage,
    UnresolvedCredit,
)

CONFIDENCE = 0.75
DATE_WINDOW_DAYS = 3

TOLERANCE_RATE = 0.006
TOLERANCE_FLOOR_PAISE = 50

REASON_MATCH = "tolerance_band_match"


def _tolerance_for(credit_paise: int) -> int:
    return max(TOLERANCE_FLOOR_PAISE, round(credit_paise * TOLERANCE_RATE))


def _batches_in_window(unresolved_credit: UnresolvedCredit) -> list[ExpectedBatch]:
    window_start = unresolved_credit.credit.value_date - timedelta(days=DATE_WINDOW_DAYS)
    window_end = unresolved_credit.credit.value_date
    return [
        batch
        for batch in unresolved_credit.expected_batches
        if batch.window_start <= window_end and window_start <= batch.window_end
    ]


def _candidate_lines_in_window(unresolved_credit: UnresolvedCredit) -> list[SettlementLine]:
    window_start = unresolved_credit.credit.value_date - timedelta(days=DATE_WINDOW_DAYS)
    window_end = unresolved_credit.credit.value_date
    return [
        line
        for line in unresolved_credit.candidate_lines
        if window_start <= line.settled_at.date() <= window_end
    ]


def _delta_is_a_missing_line(delta_paise: int, pool: list[SettlementLine]) -> bool:
    """True when some real settlement line in the credit's own date-windowed
    pool exactly explains the gap — a composition fact, not tolerance
    noise. `abs()` on both sides: a missing PAYMENT line leaves the credit
    short (delta negative, line net positive), and net_paise is itself
    negative for a REFUND/CHARGEBACK line (fees.py), so comparing magnitudes
    is what actually generalizes across both."""
    return any(abs(delta_paise) == abs(line.net_paise) for line in pool)


def _decision(
    ctx: RunContext,
    credit_id: str,
    now: datetime,
    *,
    delta_paise: int,
    matched_payment_ids: list[str],
) -> Decision:
    return Decision(
        run_id=ctx.run_id,
        seed=ctx.seed,
        stage=Stage.L3,
        credit_id=credit_id,
        matched_payment_ids=matched_payment_ids,
        outcome=DecisionOutcome.MATCHED,
        confidence=CONFIDENCE,
        delta_paise=delta_paise,
        reason=REASON_MATCH,
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=now,
    )


def run(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]:
    """Check each credit's date-windowed expected batches for exactly one
    within tolerance of the credit's amount. Returns one Decision per credit
    this stage resolves; zero or multiple within-tolerance batches, or a
    delta a real settlement line already exactly explains (a composition
    fact, not tolerance noise — see this module's 2026-09-03 docstring
    entry), leave a credit unresolved."""
    now = datetime.now(UTC)
    decisions: list[Decision] = []

    for unresolved_credit in unresolved:
        credit = unresolved_credit.credit
        tolerance = _tolerance_for(credit.credit_paise)

        within_tolerance = [
            (batch, credit.credit_paise - batch.net_paise)
            for batch in _batches_in_window(unresolved_credit)
            if abs(credit.credit_paise - batch.net_paise) <= tolerance
        ]

        if len(within_tolerance) != 1:
            continue  # zero or multiple: not certain enough, stays unresolved

        batch, delta = within_tolerance[0]
        if _delta_is_a_missing_line(delta, _candidate_lines_in_window(unresolved_credit)):
            continue  # a whole line explains the gap — L2's call, not tolerance

        decisions.append(
            _decision(
                ctx,
                credit.txn_id,
                now,
                delta_paise=delta,
                matched_payment_ids=batch.payment_ids,
            )
        )

    return decisions
