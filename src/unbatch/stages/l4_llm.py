"""L4 — LLM adjudication.

The only stage that calls `unbatch.adjudicator`, and therefore the only place
a model sees any data. Receives just what reached L4 unresolved, never the
full batch. Classifies why a break happened and proposes a resolution;
applies the confidence bands from ARCHITECTURE.md § Confidence bands
(>=0.85 auto-accept, 0.60-0.85 human_review_required, <0.60 exception).

**Exact-composition candidates come first, for the with-LLM arm only**
(2026-08-31 fix — see FAILURES.md's entry on gpt-5-nano/gpt-5-mini both
scoring near-zero break-reason accuracy on `ambiguous_composition`). Under
`ctx.llm_only == False`, this stage re-runs the same bounded search
`compose.compose()` already does for L2, over `candidate_lines` in the
credit's date window — a credit reaching L4 in the with-LLM arm only ever
has 0 or >=2 exact-sum subsets here, since L2 already claimed anything with
exactly 1. Where L2's own composition search used to stop — offering the
model a whole `ExpectedBatch` (every line in a settlement UTR group) as the
"closest" candidate — `ambiguous_composition`'s actual ground-truth answer
is a strict subset of one batch (the generator always leaves one line
permanently unclaimed; see DATA_SPEC.md), which no whole-batch candidate
can ever equal exactly. Offering whole batches made the correct answer
structurally unrepresentable regardless of model capability — not a
prompting or model-choice problem.

**`--llm-only` deliberately skips this search entirely** and always uses
the whole-batch fallback below, for two reasons: re-running L2's
composition machinery here would quietly smuggle a rules capability into
the one arm that's supposed to have none, undermining the ablation's own
point (METRICS.md § the ablation — arm C is supposed to show what an LLM
gets with no cascade behind it); and running `compose()` against a full,
unpruned pool for all ~105 credits instead of the with-LLM arm's ~12
residual ones is a genuine performance cliff, not just a conceptual one.
`--llm-only` scoring worse here, including on exactly this class of credit,
is expected and is itself evidence for the cascade's design.

When >=2 exact-sum subsets tie, that is provable, not merely suspected,
ambiguity — deterministically true regardless of what the model reports.
This stage never auto-accepts or human-reviews a specific pick in that case
(matched_payment_ids would have to name one arbitrary tie over another);
it always exceptions, independent of the model's own confidence — the same
"bias to exception over wrong match" the confidence bands already apply to
model output applies here to a fact the rules layer itself established.

Only when no exact composition exists at all (covers `tolerance_ambiguous`,
`unrelated_credit`, `date_skew`, and anything else compose() can't tie
exactly) does this stage fall back to the whole-batch nearest-neighbor
heuristic: every batch whose settlement window overlaps `[D - 3d, D]` is
scored by `|credit - batch.net|`; the closest becomes the "expected batch"
the adjudicator reasons about, and the next few become competing
CandidateExplanations. A credit with nothing in its date window at all
falls back to the single globally-nearest batch, so the model always has
something concrete to reason against instead of an empty prompt.

A pool too large for `compose()` to search, or a search that times out,
falls back to the same whole-batch heuristic rather than treating "refused
to solve" as "definitely no exact match" — `compose()`'s own refusal is not
evidence either way.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from unbatch import adjudicator
from unbatch.compose import ComposeTimeoutError, PoolTooLargeError, compose
from unbatch.models import (
    CandidateExplanation,
    Decision,
    DecisionOutcome,
    ExpectedBatch,
    RunContext,
    SettlementLine,
    Stage,
    UnresolvedCredit,
)

TOP_K_CANDIDATES = 4

REASON_ADJUDICATION_FAILED = "adjudication_failed"


def _candidate_lines_in_window(
    unresolved_credit: UnresolvedCredit, ctx: RunContext
) -> list[SettlementLine]:
    """Mirrors l2_compose.py's own date-window filter exactly — this stage
    is re-running the same search, not a different one, so it needs the
    same window."""
    credit = unresolved_credit.credit
    window_start = credit.value_date - timedelta(days=ctx.config.date_window_days)
    return [
        line
        for line in unresolved_credit.candidate_lines
        if window_start <= line.settled_at.date() <= credit.value_date
    ]


def _exact_composition_candidates(
    unresolved_credit: UnresolvedCredit, ctx: RunContext
) -> list[list[SettlementLine]] | None:
    """Every exact-sum line subset compose() finds for this credit, or None
    if the pool was too large or the search timed out (a refusal to search,
    not a negative result — the caller must not treat it as "no exact
    composition exists")."""
    pool = _candidate_lines_in_window(unresolved_credit, ctx)
    try:
        return compose(
            unresolved_credit.credit.credit_paise,
            pool,
            max_pool=ctx.config.max_pool,
            max_subset=ctx.config.max_subset,
            timeout_s=ctx.config.compose_timeout_s,
        )
    except (PoolTooLargeError, ComposeTimeoutError):
        return None


def _synthetic_batch_from_lines(lines: list[SettlementLine]) -> ExpectedBatch:
    """An ExpectedBatch-shaped view of one exact-sum subset, for the prompt
    — never registered anywhere, never compared against
    compute_expected_batches' real batches. net_paise is the subset's own
    sum, which by construction (compose() only returns exact matches) always
    equals the credit exactly, so delta_paise for this pick is always 0."""
    utrs = sorted({line.settlement_utr for line in lines})
    dates = [line.settled_at.date() for line in lines]
    return ExpectedBatch(
        settlement_utr=utrs[0] if len(utrs) == 1 else "+".join(utrs),
        settlement_ids=[line.settlement_id for line in lines],
        payment_ids=[line.payment_id for line in lines],
        net_paise=sum(line.net_paise for line in lines),
        window_start=min(dates),
        window_end=max(dates),
    )


def _candidates_from_exact_subsets(
    subsets: list[list[SettlementLine]],
) -> list[CandidateExplanation]:
    return [
        CandidateExplanation(
            payment_ids=[line.payment_id for line in subset],
            delta_paise=0,
            hint=(
                f"another exact-sum composition, {len(subset)} lines, "
                f"batch(es) {sorted({line.settlement_utr for line in subset})}"
            ),
        )
        for subset in subsets
    ]


def _scored_batches(
    unresolved_credit: UnresolvedCredit, ctx: RunContext
) -> list[tuple[ExpectedBatch, int]]:
    """Every candidate batch paired with its delta against the credit,
    nearest first. Prefers batches whose window overlaps the same
    lookback window L2/L3 use; falls back to the full batch list only when
    nothing is in that window, so a truly unrelated credit still gets the
    single closest-by-date batch as its point of comparison rather than
    nothing."""
    credit = unresolved_credit.credit
    window_start = credit.value_date - timedelta(days=ctx.config.date_window_days)
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
    unresolved_credit: UnresolvedCredit, ctx: RunContext
) -> tuple[ExpectedBatch, int, list[CandidateExplanation]]:
    """Whole-batch nearest-neighbor fallback — used only when no exact-sum
    line composition exists at all (see module docstring)."""
    scored = _scored_batches(unresolved_credit, ctx)
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


def _pick_primary_and_candidates(
    unresolved_credit: UnresolvedCredit, ctx: RunContext
) -> tuple[ExpectedBatch, int, list[CandidateExplanation], bool]:
    """Returns (primary_batch, delta_paise, candidates, provably_ambiguous).
    `provably_ambiguous` is True only when compose() itself found >=2
    exact-sum subsets — a rules-established fact, not a model opinion.

    The exact-composition search only runs under `ctx.llm_only == False`.
    For the with-LLM arm this is a legitimate re-derivation of what L2
    already searched over the same (now much smaller, already-pruned)
    pool — L2 found the ambiguity but had nowhere to record it, so L4
    re-runs the identical bounded search rather than needing new
    inter-stage plumbing. For `--llm-only`, which deliberately skips L0-L3
    to measure what an LLM alone can do, silently re-running L2's
    composition search here would smuggle a rules capability into the arm
    that's supposed to have none — undermining exactly the comparison the
    ablation exists to make (METRICS.md § the ablation) — and would also
    run compose() against a full, unpruned pool for every one of the ~105
    credits instead of the ~12 the with-LLM arm actually reaches L4 with,
    which is a real performance cliff, not just a conceptual one."""
    if not ctx.llm_only:
        exact_subsets = _exact_composition_candidates(unresolved_credit, ctx)
        if exact_subsets:
            primary_batch = _synthetic_batch_from_lines(exact_subsets[0])
            candidates = _candidates_from_exact_subsets(exact_subsets[1 : 1 + TOP_K_CANDIDATES])
            return primary_batch, 0, candidates, len(exact_subsets) >= 2

    primary_batch, delta_paise, candidates = _pick_batch_and_candidates(unresolved_credit, ctx)
    return primary_batch, delta_paise, candidates, False


def _confidence_band_outcome(
    confidence: float, human_review_required: bool, ctx: RunContext
) -> DecisionOutcome:
    """ARCHITECTURE.md's fixed bands decide the default outcome; the model's
    own `human_review_required` flag can only push a confident call down to
    human review, never push an unconfident one up to auto-accept — bias to
    exception/review over a wrong match applies to the model's output just
    as much as to the rules layers."""
    if confidence < ctx.config.confidence_human_review:
        return DecisionOutcome.EXCEPTION
    if confidence < ctx.config.confidence_auto_accept or human_review_required:
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
    evidence_refs: list[str] | None = None,
    human_review_required: bool | None = None,
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
        evidence_refs=evidence_refs,
        human_review_required=human_review_required,
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
        primary_batch, delta_paise, candidates, provably_ambiguous = _pick_primary_and_candidates(
            unresolved_credit, ctx
        )

        try:
            result, cost_paise, retried = adjudicator.adjudicate(
                credit, primary_batch, delta_paise, candidates, cached=ctx.cached
            )
        except adjudicator.AdjudicationFailedError as exc:
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
                    # the real cost of both calls actually made (first
                    # attempt + retry), not 0 — see
                    # AdjudicationFailedError's own docstring
                    llm_cost_paise=exc.cost_paise,
                    # degrading to adjudication_failed only ever happens
                    # after exactly one retry, by construction (see
                    # adjudicator.adjudicate's docstring)
                    llm_retried=True,
                )
            )
            continue

        if provably_ambiguous:
            # >=2 exact-sum compositions tie: no pick is any more "right"
            # than another by the numbers alone, so none is ever committed
            # as a match — the model's classification/rationale is still
            # recorded, but the outcome bypasses the confidence bands
            # entirely rather than let a confident-sounding guess win a
            # coin flip.
            outcome = DecisionOutcome.EXCEPTION
            matched_payment_ids: list[str] = []
        else:
            outcome = _confidence_band_outcome(
                result.confidence, result.human_review_required, ctx
            )
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
                evidence_refs=result.evidence_refs,
                human_review_required=result.human_review_required,
            )
        )

    return decisions
