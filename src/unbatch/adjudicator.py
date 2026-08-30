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
DEFAULT_CACHE_DIR = Path("cache")


class AdjudicationFailedError(Exception):
    """Raised after a retry still fails to validate; caller (l4_llm) should
    record an exception decision with reason `adjudication_failed`."""


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
