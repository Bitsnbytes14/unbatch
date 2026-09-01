"""LLM boundary: prompt construction, response cache, output validation, and
degradation. The only module that talks to any LLM provider's SDK — used
solely by `unbatch.stages.l4_llm`. No module outside this one imports an SDK;
that boundary is what made the D0.5 provider swap (Anthropic -> OpenAI, this
session's only available key was OPENAI_API_KEY, not an Anthropic one)
a change to this file alone.

**Provider: OpenAI, Chat Completions API** (`client.chat.completions.create`),
not the newer Responses API. OpenAI's current docs recommend Responses for
new projects, but Chat Completions is confirmed still fully supported (not
deprecated) as of 2026-08-31 (checked at developers.openai.com/api/docs, not
assumed from training data) — used here because it is what strict structured
outputs via `response_format={"type": "json_schema", ...}` was originally
documented against, matching this milestone's explicit shape.

**Model: gpt-5-nano** — $0.05 / $0.40 per 1M input/output tokens, OpenAI's
cheapest model at the time of this swap. This took two measurements to
settle, not one: nano first scored 25% break-reason accuracy on the 12 real
L4 credits, confusing `ambiguous_composition` with `tolerance_ambiguous` on
most of them; stepping up to `gpt-5-mini` scored *0%*, strictly worse, plus
a false match on the one `unrelated_credit`. Neither result was actually
about model capability — `l4_llm.py`'s candidate construction at the time
only ever offered whole settlement batches, and `ambiguous_composition`'s
real answer is a strict subset of one batch (see DATA_SPEC.md), which no
whole-batch candidate can equal exactly. No model could have scored well
against candidates that couldn't contain the right answer. Once `l4_llm.py`
was fixed to offer the model's own exact-sum composition search results
instead (see l4_llm.py's module docstring), nano's accuracy jumped to
**83%**, with zero false matches, at roughly 1/6 mini's per-call cost —
mini was never the fix; the candidates were. See FAILURES.md's 2026-08-31
entries for the full chronology, including the one where mini looked like
an improvement it wasn't.

The prompt hands the model pre-computed deltas and candidate explanations as
facts; it never asks the model to calculate, sum, or compute anything (CLAUDE.md
invariant 2). Responses are cached in cache/ keyed by a hash of the prompt
payload (including model and prompt version, so a prompt change invalidates
stale entries) so `--cached` runs need no API key. The cache key already
included the model name before this swap, so switching provider/model here
invalidates every previously-cached (Anthropic) entry automatically — nothing
about the cache format itself needed to change.

**Structured output**: `response_format` is set to a strict JSON Schema
derived directly from `AdjudicationResult.model_json_schema()`, so the API
itself refuses to emit anything that doesn't validate against the pydantic
model's shape — CLAUDE.md invariant 2's "schema-enforced at the API boundary"
is literal here, not just a prompt instruction. This makes malformed output
rare, not impossible: a refusal (`message.refusal` set instead of
`message.content`) or truncation (`finish_reason == "length"`, e.g. a
mid-response cutoff producing invalid JSON) both still bypass the schema
entirely, so the retry-then-degrade path below is unchanged from the
previous provider and just as necessary.

No `temperature` is sent — the previous (Anthropic) integration hit exactly
this wall (see FAILURES.md's 2026-08-31 entry: claude-sonnet-5 rejects
sampling parameters outright) and this project's reproducibility story
doesn't need one either: a `--cached` run always replays the exact bytes
recorded here, regardless of what a live call would produce on any given
day. `reasoning_effort="minimal"` is sent instead — the gpt-5 family are
reasoning-capable models, and this task (single-label classification over an
already-computed delta) doesn't benefit from spending reasoning tokens,
which are billed as output tokens. Kept at "minimal" across the nano ->
mini swap so the model was the only variable that changed between the two
measurements.

Malformed JSON is retried once with the validation error appended to the
prompt; if it is still invalid, adjudication degrades to an
`adjudication_failed` outcome rather than crashing the pipeline.

The schema only proves the response has the right SHAPE — its CONTENT is
still untrusted model output. `_validate_semantics` checks it against the
prompt's own facts: every `evidence_ref` must name a payment_id/settlement_id
the model was actually shown, or the credit's own txn_id (never something it
invented outright); `confidence` must fall inside [0.0, 1.0] (the schema has
no numeric-range keyword to enforce this — OpenAI's strict json_schema mode
doesn't support minimum/maximum); and `break_reason` must be a real
BreakReason value (this one is already unreachable via the normal path
since BreakReason is a StrEnum pydantic validates against directly; kept
anyway as a defense-in-depth assertion against that constraint being
loosened later). A semantic failure raises the same internal error a
schema failure does, so it goes through the identical retry-then-degrade
path above — not a new outcome.

**2026-09-01 measurement, not a guess:** checking every cached with-LLM
response across seeds 42-47 and the adversarial dataset (77 total)
against a stricter evidence_refs check (payment_id/settlement_id only,
no txn_id) found gpt-5-nano citing the credit's own txn_id as evidence in
76 of them (98.7%) — its default habit, not an occasional slip. That's a
real, self-referential identifier, never a wrong pointer, so it's included
in the valid set rather than treated as a rejection; the check still
catches an actually-unrelated or fabricated payment/settlement id, which
is what it exists for.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydantic import ValidationError

from unbatch.models import (
    AdjudicationResult,
    BankStatementRecord,
    BreakReason,
    CandidateExplanation,
    ExpectedBatch,
)

# Loads OPENAI_API_KEY (and anything else) from a .env file at the repo root
# into the environment, so `openai.OpenAI()` below finds it with no shell
# configuration — a judge cloning this repo only needs to drop their own key
# into .env (see .env.example), never export anything. Path is resolved
# relative to this file, not the working directory, so this works
# regardless of where `unbatch` is invoked from. Never overrides a variable
# already set in the real environment (python-dotenv's default) — an
# exported OPENAI_API_KEY still wins over .env. Does nothing, silently, if
# .env doesn't exist (e.g. a `--cached` run needs no key at all).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-nano"
PROMPT_VERSION = "v1"
DEFAULT_CACHE_DIR = Path("cache")
MAX_TOKENS = 4096
REASONING_EFFORT = "minimal"
RESPONSE_SCHEMA_NAME = "adjudication_result"

# OpenAI bills gpt-5-nano in USD ($0.05 / $0.40 per 1M input/output tokens —
# developers.openai.com/api/docs/pricing, 2026-08-31); the audit log stores
# every money field in paise (CLAUDE.md invariant 1), so a fixed USD->INR
# rate is needed just to report LLM spend in the same unit as everything
# else. This is billing telemetry, not reconciled settlement money —
# approximate on purpose, and never fed back into any matching decision.
#
# See the module docstring: nano's first measurement (25% break-reason
# accuracy) and gpt-5-mini's subsequent one (0%, worse) were both measuring
# a broken candidate list, not the models. With l4_llm.py's exact-sum
# candidates fixed, nano re-measured at 83% accuracy with zero false
# matches — the cheapest tier turned out sufficient once it was actually
# being tested fairly.
USD_TO_INR_RATE = 88
INPUT_COST_USD_PER_MTOK = 0.05
OUTPUT_COST_USD_PER_MTOK = 0.40

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


def _client() -> openai.OpenAI:
    """A fresh client per call — this is a low-volume, per-credit boundary
    (METRICS.md targets under 20% of credits ever reaching L4), not a hot
    path worth pooling. Tests monkeypatch this function directly, so no
    real client is ever constructed under pytest."""
    return openai.OpenAI()


def _response_schema() -> dict[str, object]:
    """AdjudicationResult's own pydantic schema, patched with the one field
    OpenAI's strict mode requires that pydantic doesn't set by default:
    `additionalProperties: false`. Every field is already required (pydantic
    has no optional/defaulted fields on this model), which is strict mode's
    other requirement."""
    schema = AdjudicationResult.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def _call_model(system: str, user: str) -> tuple[str | None, int, int]:
    """One live call to Chat Completions. Returns (response_text,
    input_tokens, output_tokens) — response_text is None if the model
    produced no content at all (a refusal, where `message.refusal` is set
    instead, or truncation with an empty completion), which `_require_text`
    below treats as malformed. No `temperature` (see module docstring);
    `reasoning_effort="minimal"` keeps this narrow classification task from
    spending reasoning tokens it doesn't need."""
    response = _client().chat.completions.create(
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": RESPONSE_SCHEMA_NAME,
                "schema": _response_schema(),
                "strict": True,
            },
        },
    )
    message = response.choices[0].message
    return message.content, response.usage.prompt_tokens, response.usage.completion_tokens


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
    """A response with no content at all — a refusal (`message.refusal` set
    instead of `message.content`) or a completion truncated before producing
    anything — is malformed in exactly the same sense a bad JSON body is;
    both go through the same retry-then-degrade path."""
    if text is None:
        raise _MalformedResponseError("model response had no content")
    return text


def _validate_semantics(
    result: AdjudicationResult,
    credit: BankStatementRecord,
    expected_batch: ExpectedBatch,
    candidates: list[CandidateExplanation],
) -> None:
    """Schema validation only proves the response has the right SHAPE;
    treat its CONTENT as untrusted too — a response can be perfectly
    well-formed JSON matching AdjudicationResult and still name evidence
    the model was never shown, or a confidence value outside what
    "confidence" can mean. Any violation here raises the same
    _MalformedResponseError schema failures do, so it goes through the
    identical retry-then-degrade path (adjudicate()) rather than a new
    failure mode.

    `credit.txn_id` is deliberately included in the valid-reference set,
    not just the batch's payment_ids/settlement_ids and each candidate's
    payment_ids: measured against every cached with-LLM response across
    seeds 42-47 and the adversarial dataset, gpt-5-nano cites the credit
    under discussion by its own txn_id in 76 of 77 responses (98.7%) —
    a real, self-referential identifier that's never a wrong pointer, just
    outside the "payment_id or settlement_id" vocabulary the prompt names.
    That is categorically different from citing an unrelated or fabricated
    payment/settlement id, which is what this check exists to catch."""
    valid_refs = (
        {credit.txn_id} | set(expected_batch.payment_ids) | set(expected_batch.settlement_ids)
    )
    for candidate in candidates:
        valid_refs.update(candidate.payment_ids)
    hallucinated_refs = [ref for ref in result.evidence_refs if ref not in valid_refs]
    if hallucinated_refs:
        raise _MalformedResponseError(
            f"evidence_refs names records not in the candidate set: {hallucinated_refs}"
        )

    if not (0.0 <= result.confidence <= 1.0):
        raise _MalformedResponseError(
            f"confidence {result.confidence} is outside the valid [0.0, 1.0] range"
        )

    if result.break_reason not in BreakReason:
        # Unreachable today via the normal path — pydantic already refuses to
        # construct an AdjudicationResult with anything but a real BreakReason
        # member, and OpenAI's strict json_schema mode encodes BreakReason as
        # an enum constraint too. Kept as an explicit, tested assertion
        # anyway: if AdjudicationResult.break_reason were ever loosened to a
        # plain str, this is what would still catch a hallucinated reason.
        raise _MalformedResponseError(
            f"break_reason {result.break_reason!r} is not a known BreakReason value"
        )


def _parse_response(
    text: str,
    credit: BankStatementRecord,
    expected_batch: ExpectedBatch,
    candidates: list[CandidateExplanation],
) -> AdjudicationResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _MalformedResponseError(f"response is not valid JSON: {exc}") from exc
    try:
        result = AdjudicationResult.model_validate(payload)
    except ValidationError as exc:
        raise _MalformedResponseError(
            f"response does not match the required shape: {exc}"
        ) from exc
    _validate_semantics(result, credit, expected_batch, candidates)
    return result


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
) -> tuple[AdjudicationResult, int, bool]:
    """Classify one unresolved break and propose a resolution. Returns
    (result, llm_cost_paise, retried) — the caller (l4_llm) logs all three
    onto the Decision it writes. `retried` is True whenever the first
    response was malformed and a second call was needed to recover, even if
    that second call succeeded — this is METRICS.md's `retry_count` signal.

    Checks the cache first; if `cached` is True and no entry exists, raises
    CacheMissError rather than calling the API. On a live call, validates
    the response against AdjudicationResult (schema) and then against the
    prompt's own facts (semantics: evidence_refs must name something the
    model was actually shown, confidence must be in [0, 1], break_reason
    must be a real BreakReason — see _validate_semantics). A response
    failing either check is retried once with the validation error appended
    to the prompt, and a second failure raises AdjudicationFailedError
    rather than crashing the cascade (a caller catching that should treat
    it as retried=True too — degrading only ever happens after exactly one
    retry, by construction).
    """
    system, user = build_prompt(credit, expected_batch, delta_paise, candidates)
    text, input_tokens, output_tokens = _get_response(
        system, user, cached=cached, cache_dir=cache_dir
    )
    cost_paise = _cost_paise(input_tokens, output_tokens)

    try:
        result = _parse_response(_require_text(text), credit, expected_batch, candidates)
        return result, cost_paise, False
    except _MalformedResponseError as first_error:
        retry_user = _append_validation_error(user, str(first_error))
        retry_text, retry_input, retry_output = _get_response(
            system, retry_user, cached=cached, cache_dir=cache_dir
        )
        cost_paise += _cost_paise(retry_input, retry_output)
        try:
            result = _parse_response(_require_text(retry_text), credit, expected_batch, candidates)
            return result, cost_paise, True
        except _MalformedResponseError as second_error:
            raise AdjudicationFailedError(str(second_error)) from second_error
