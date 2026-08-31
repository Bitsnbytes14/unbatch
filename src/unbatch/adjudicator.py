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

import hashlib
import json
from pathlib import Path

import anthropic
from pydantic import ValidationError

from unbatch.models import (
    AdjudicationResult,
    BankStatementRecord,
    CandidateExplanation,
    ExpectedBatch,
)

MODEL = "claude-sonnet-5"
PROMPT_VERSION = "v1"
DEFAULT_CACHE_DIR = Path("cache")
MAX_TOKENS = 4096

# claude-sonnet-5 rejects temperature/top_p/top_k outright (400) — sampling
# controls were removed for this model family. Determinism instead comes
# from the cache: a `--cached` run always replays the exact bytes recorded
# here, regardless of what a live call would produce this time around.
#
# Anthropic bills this model in USD ($2.00 / $10.00 per 1M input/output
# tokens); the audit log stores every money field in paise (CLAUDE.md
# invariant 1), so a fixed USD->INR rate is needed just to report LLM spend
# in the same unit as everything else. This is billing telemetry, not
# reconciled settlement money — approximate on purpose, and never fed back
# into any matching decision.
USD_TO_INR_RATE = 88
INPUT_COST_USD_PER_MTOK = 2.00
OUTPUT_COST_USD_PER_MTOK = 10.00

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
- tolerance_ambiguous: more than one settlement batch's net lands within the
  tolerance band of this credit, with no single one clearly right.
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


class CacheMissError(Exception):
    """Raised when `cached=True` and no recorded response exists for this
    exact prompt. Deliberately not caught anywhere — an incomplete cache
    under `--cached` is a repo integrity problem (ARCHITECTURE.md's
    reproducibility guarantee is broken), not a per-credit outcome to
    degrade into an exception Decision."""


class _MalformedResponseError(Exception):
    """Internal: the model's text wasn't valid JSON, or didn't validate
    against AdjudicationResult. Carries a message meant to be read back to
    the model on retry."""


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


def _client() -> anthropic.Anthropic:
    """A fresh client per call — this is a low-volume, per-credit boundary
    (METRICS.md targets under 20% of credits ever reaching L4), not a hot
    path worth pooling. Tests monkeypatch this function directly, so no
    real client is ever constructed under pytest."""
    return anthropic.Anthropic()


def _call_model(system: str, user: str) -> tuple[str | None, int, int]:
    """One live call to the Messages API. Returns (response_text,
    input_tokens, output_tokens) — response_text is None if the model
    returned no text block at all (e.g. it hit max_tokens mid-thought),
    which `_require_text` below treats as malformed. No
    temperature/top_p/top_k — claude-sonnet-5 rejects sampling controls
    outright; thinking is left unset, which runs adaptive by default on
    this model, so `response.content` may contain a thinking block ahead
    of the text block."""
    response = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((block.text for block in response.content if block.type == "text"), None)
    return text, response.usage.input_tokens, response.usage.output_tokens


def _cache_key(system: str, user: str) -> str:
    """Hash of the full prompt payload, including model and prompt version —
    changing the prompt template, bumping PROMPT_VERSION, or switching MODEL
    all produce a different key, so a stale cached response can never be
    replayed against a prompt it wasn't generated for."""
    payload = json.dumps(
        {"model": MODEL, "prompt_version": PROMPT_VERSION, "system": system, "user": user},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _get_response(
    system: str, user: str, *, cached: bool, cache_dir: Path
) -> tuple[str | None, int, int]:
    """Cache-backed wrapper around `_call_model`. A hit replays committed
    bytes with no API call at all; `cached=True` on a miss raises
    CacheMissError rather than silently falling through to a live call,
    which is what makes `--cached` runs an honest no-API-key guarantee
    instead of an accidental one."""
    key = _cache_key(system, user)
    path = cache_dir / f"{key}.json"
    if path.exists():
        entry = json.loads(path.read_text(encoding="utf-8"))
        return entry["response_text"], entry["input_tokens"], entry["output_tokens"]
    if cached:
        raise CacheMissError(f"no cached response for prompt key {key!r}; --cached was requested")

    text, input_tokens, output_tokens = _call_model(system, user)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"response_text": text, "input_tokens": input_tokens, "output_tokens": output_tokens},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return text, input_tokens, output_tokens


def _require_text(text: str | None) -> str:
    """A response with no text block at all (e.g. truncated at max_tokens
    before producing any) is malformed in exactly the same sense a bad JSON
    body is — both go through the same retry-then-degrade path."""
    if text is None:
        raise _MalformedResponseError("model response had no text block")
    return text


def _parse_response(text: str) -> AdjudicationResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _MalformedResponseError(f"response is not valid JSON: {exc}") from exc
    try:
        return AdjudicationResult.model_validate(payload)
    except ValidationError as exc:
        raise _MalformedResponseError(
            f"response does not match the required shape: {exc}"
        ) from exc


def _append_validation_error(user: str, error_message: str) -> str:
    return (
        f"{user}\n\n"
        "Your previous reply could not be read as valid JSON in the required "
        f"shape. The reader reported: {error_message}\n"
        "Return only the corrected JSON object, with no other text."
    )


def _cost_paise(input_tokens: int, output_tokens: int) -> int:
    cost_usd = (
        input_tokens / 1_000_000 * INPUT_COST_USD_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_COST_USD_PER_MTOK
    )
    return round(cost_usd * USD_TO_INR_RATE * 100)


def adjudicate(
    credit: BankStatementRecord,
    expected_batch: ExpectedBatch,
    delta_paise: int,
    candidates: list[CandidateExplanation],
    *,
    cached: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[AdjudicationResult, int]:
    """Classify one unresolved break and propose a resolution. Returns
    (result, llm_cost_paise) — the caller (l4_llm) logs both onto the
    Decision it writes.

    Checks the cache first; if `cached` is True and no entry exists, raises
    CacheMissError rather than calling the API. On a live call, validates
    the response against AdjudicationResult; a malformed response is retried
    once with the validation error appended to the prompt, and a second
    failure raises AdjudicationFailedError rather than crashing the cascade.
    """
    system, user = build_prompt(credit, expected_batch, delta_paise, candidates)
    text, input_tokens, output_tokens = _get_response(
        system, user, cached=cached, cache_dir=cache_dir
    )
    cost_paise = _cost_paise(input_tokens, output_tokens)

    try:
        return _parse_response(_require_text(text)), cost_paise
    except _MalformedResponseError as first_error:
        retry_user = _append_validation_error(user, str(first_error))
        retry_text, retry_input, retry_output = _get_response(
            system, retry_user, cached=cached, cache_dir=cache_dir
        )
        cost_paise += _cost_paise(retry_input, retry_output)
        try:
            return _parse_response(_require_text(retry_text)), cost_paise
        except _MalformedResponseError as second_error:
            raise AdjudicationFailedError(str(second_error)) from second_error
