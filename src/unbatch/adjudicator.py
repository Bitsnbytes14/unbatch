"""LLM boundary: prompt construction, response cache, output validation, and
degradation. The only module that talks to the Anthropic API — used solely by
`unbatch.stages.l4_llm`.

The prompt hands the model pre-computed deltas and candidate explanations as
facts; it never asks the model to calculate, sum, or compute anything (CLAUDE.md
invariant 2). Responses are cached in cache/ keyed by a hash of the prompt
payload (including model and prompt version, so a prompt change invalidates
stale entries) so `--cached` runs need no API key.

Malformed JSON is retried once with the validation error appended to the
prompt; if it is still invalid, adjudication degrades to an
`adjudication_failed` outcome rather than crashing the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from unbatch.models import (
    AdjudicationResult,
    BankStatementRecord,
    CandidateExplanation,
    ExpectedBatch,
)

MODEL = "claude-sonnet-5"
PROMPT_VERSION = "v1"
DEFAULT_CACHE_DIR = Path("cache")

SYSTEM_PROMPT = """You are the settlement-reconciliation adjudicator for a payment
gateway's finance-controller pipeline.

Deterministic rules have already tried, and failed, to tie a bank credit to an
expected settlement batch. Every number you are given below — the credit
amount, the batch's net, and the delta between them — is a finished fact the
rules layer already worked out. Do not re-derive, re-check, or adjust any of
these numbers; treat them as given. Your job is judgment, not arithmetic:
decide WHY the two amounts differ and what a human reconciler should do
about it.

Classify the break using exactly one of these reasons:
- refund_in_window: a refund issued inside the settlement window is netted
  into the credit but not the batch.
- chargeback_deduction: a chargeback and its fee reduced the credit below
  the batch net.
- fee_tier_change: the gateway's fee rate shifted, producing a small
  proportional gap.
- rounding_delta: a paise-level gap from GST rounding applied per line
  versus per batch.
- settlement_split: the real payout was split across more than one
  settlement line group.
- date_skew: the amounts agree but the posting date does not.
- duplicate_utr: two real batches share one settlement UTR.
- ambiguous_composition: more than one candidate set of payment lines fits
  equally well.
- unrelated_credit: this bank credit does not correspond to any settlement
  batch at all.
- other: none of the above explain the evidence.

Respond with a single JSON object and nothing else — no prose before or
after it — matching exactly this shape:

{
  "break_reason": "<one of the reasons above>",
  "proposed_resolution": "<one sentence: what a reconciler should do next>",
  "confidence": <float between 0.0 and 1.0>,
  "evidence_refs": ["<payment_id or settlement_id that supports your call>", ...],
  "human_review_required": <true or false>
}
"""


class AdjudicationFailedError(Exception):
    """Raised after a retry still fails to validate; caller (l4_llm) should
    record an exception decision with reason `adjudication_failed`."""


def _format_candidates(candidates: list[CandidateExplanation]) -> str:
    if not candidates:
        return "  (none surfaced)"
    return "\n".join(
        f"  - payment_ids={c.payment_ids}, delta_paise={c.delta_paise}, hint={c.hint!r}"
        for c in candidates
    )


def build_prompt(
    credit: BankStatementRecord,
    expected_batch: ExpectedBatch,
    delta_paise: int,
    candidates: list[CandidateExplanation],
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for one adjudication call.

    Every number here — credit_paise, net_paise, delta_paise — is already
    final, worked out upstream by the rules layer (CLAUDE.md invariant 2:
    the model never does arithmetic). The user turn only presents facts and
    asks for a classification and a recommendation; see
    tests/test_adjudicator_prompt.py for the grep that enforces this.
    """
    user = f"""Unresolved bank credit:
  txn_id: {credit.txn_id}
  value_date: {credit.value_date.isoformat()}
  narration: {credit.narration!r}
  credit_paise: {credit.credit_paise}

Expected settlement batch (already selected by the rules layer):
  settlement_utr: {expected_batch.settlement_utr}
  window: {expected_batch.window_start.isoformat()} to {expected_batch.window_end.isoformat()}
  net_paise: {expected_batch.net_paise}
  payment_ids: {expected_batch.payment_ids}
  settlement_ids: {expected_batch.settlement_ids}

Delta between the credit and the batch net, already final: {delta_paise} paise

Other candidate explanations surfaced by earlier stages:
{_format_candidates(candidates)}

Classify this break and propose a resolution."""
    return SYSTEM_PROMPT, user


def adjudicate(
    credit: BankStatementRecord,
    expected_batch: ExpectedBatch,
    delta_paise: int,
    candidates: list[CandidateExplanation],
    *,
    cached: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> AdjudicationResult:
    """Classify one unresolved break and propose a resolution.

    Checks the cache first; if `cached` is True and no entry exists, raises
    rather than calling the API. On a live call, validates the response
    against AdjudicationResult, retrying once on ValidationError before
    raising AdjudicationFailedError.
    """
    raise NotImplementedError
