"""Tests for adjudicator.adjudicate's live-call, cache, and retry/degrade
behaviour. The OpenAI client is always stubbed via `adjudicator._client` —
nothing in this file makes a real network call, so these run with no API
key and no network access."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest

from unbatch import adjudicator
from unbatch.models import BankStatementRecord, BreakReason, CandidateExplanation, ExpectedBatch

_CREDIT = BankStatementRecord(
    txn_id="TXN1",
    value_date=date(2026, 3, 1),
    narration="NEFT-UTR000123-RAZORPAY",
    credit_paise=418332,
    debit_paise=None,
    balance_paise=1000000,
)

_BATCH = ExpectedBatch(
    settlement_utr="UTR000123",
    settlement_ids=["setl_a", "setl_b"],
    payment_ids=["pay_a", "pay_b"],
    net_paise=419107,
    window_start=date(2026, 2, 27),
    window_end=date(2026, 3, 1),
)

_OTHER_CREDIT = _CREDIT.model_copy(update={"txn_id": "TXN2", "credit_paise": 999999})

_VALID_JSON = json.dumps(
    {
        "break_reason": "fee_tier_change",
        "proposed_resolution": "accept the tolerance-band match",
        "confidence": 0.9,
        "evidence_refs": ["pay_a", "pay_b"],
        "human_review_required": False,
    }
)

_MALFORMED_JSON = "here is my answer: " + _VALID_JSON  # not valid JSON on its own
_INVALID_SHAPE_JSON = json.dumps({"break_reason": "not_a_real_reason", "confidence": 2.0})


@dataclass
class _FakeMessage:
    content: str | None
    refusal: str | None = None


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


class _FakeCompletions:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("fake client called more times than responses were queued")
        return self._responses.pop(0)


@dataclass
class _FakeChat:
    completions: _FakeCompletions


@dataclass
class _FakeClient:
    chat: _FakeChat = field(default_factory=lambda: _FakeChat(_FakeCompletions([])))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


def _text_response(text: str, *, prompt_tokens: int = 100, completion_tokens: int = 50):
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=text))],
        usage=_FakeUsage(prompt_tokens, completion_tokens),
    )


def _install_fake_client(monkeypatch, responses: list[_FakeResponse]) -> _FakeClient:
    fake = _FakeClient(_FakeChat(_FakeCompletions(responses)))
    monkeypatch.setattr(adjudicator, "_client", lambda: fake)
    return fake


def _no_client_allowed(monkeypatch) -> None:
    def _boom():
        raise AssertionError("no API call should happen — this call should be served from cache")

    monkeypatch.setattr(adjudicator, "_client", _boom)


def test_adjudicate_calls_the_model_with_no_temperature(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])

    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert len(fake.calls) == 1
    kwargs = fake.calls[0]
    assert kwargs["model"] == adjudicator.MODEL
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_adjudicate_requests_strict_structured_output(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])

    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    kwargs = fake.calls[0]
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "break_reason",
        "proposed_resolution",
        "confidence",
        "evidence_refs",
        "human_review_required",
    }


def test_adjudicate_sends_system_and_user_as_chat_messages(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])

    system, user = adjudicator.build_prompt(_CREDIT, _BATCH, -775, [])
    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    messages = fake.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": system}
    assert messages[1] == {"role": "user", "content": user}


def test_adjudicate_parses_a_valid_response(monkeypatch, tmp_path: Path) -> None:
    # gpt-5-nano is cheap enough ($0.05/$0.40 per 1M tokens) that a
    # realistic single-credit call count would round to 0 paise — use a
    # bigger token count here purely so the cost_paise > 0 assertion below
    # is actually exercising the arithmetic, not just its rounding floor.
    _install_fake_client(
        monkeypatch, [_text_response(_VALID_JSON, prompt_tokens=20_000, completion_tokens=8_000)]
    )

    result, cost_paise, retried = adjudicator.adjudicate(
        _CREDIT, _BATCH, -775, [], cache_dir=tmp_path
    )

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert result.confidence == 0.9
    assert cost_paise > 0
    assert retried is False


def test_adjudicate_writes_one_cache_entry_per_prompt(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])

    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1
    assert b"\r" not in entries[0].read_bytes()


def test_cached_replay_makes_no_api_call(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])
    first_result, first_cost, first_retried = adjudicator.adjudicate(
        _CREDIT, _BATCH, -775, [], cache_dir=tmp_path
    )

    _no_client_allowed(monkeypatch)
    second_result, second_cost, second_retried = adjudicator.adjudicate(
        _CREDIT, _BATCH, -775, [], cached=True, cache_dir=tmp_path
    )

    assert second_result == first_result
    assert second_cost == first_cost
    assert second_retried == first_retried


def test_cached_true_raises_on_cache_miss(monkeypatch, tmp_path: Path) -> None:
    _no_client_allowed(monkeypatch)

    with pytest.raises(adjudicator.CacheMissError):
        adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cached=True, cache_dir=tmp_path)


def test_a_different_credit_misses_the_cache(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])
    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    _no_client_allowed(monkeypatch)
    with pytest.raises(adjudicator.CacheMissError):
        adjudicator.adjudicate(_OTHER_CREDIT, _BATCH, -775, [], cached=True, cache_dir=tmp_path)


def test_a_different_candidate_list_misses_the_cache(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])
    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    other_candidates = [CandidateExplanation(payment_ids=["pay_z"], delta_paise=-1, hint="x")]
    _no_client_allowed(monkeypatch)
    with pytest.raises(adjudicator.CacheMissError):
        adjudicator.adjudicate(
            _CREDIT, _BATCH, -775, other_candidates, cached=True, cache_dir=tmp_path
        )


def test_a_different_model_misses_the_cache(monkeypatch, tmp_path: Path) -> None:
    """D0.5: the cache key already includes the model name, so a provider or
    model swap invalidates every previously-cached entry automatically —
    this is what let the Anthropic -> OpenAI swap reuse the same cache
    format with no migration step."""
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])
    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    monkeypatch.setattr(adjudicator, "MODEL", "some-other-model")
    _no_client_allowed(monkeypatch)
    with pytest.raises(adjudicator.CacheMissError):
        adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cached=True, cache_dir=tmp_path)


def test_malformed_then_valid_retries_once_and_succeeds(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(
        monkeypatch, [_text_response(_MALFORMED_JSON), _text_response(_VALID_JSON)]
    )

    result, _cost, retried = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert retried is True
    assert len(fake.calls) == 2
    retry_prompt = fake.calls[1]["messages"][1]["content"]
    assert "could not be read as valid JSON" in retry_prompt


def test_invalid_shape_then_valid_retries_once_and_succeeds(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(
        monkeypatch, [_text_response(_INVALID_SHAPE_JSON), _text_response(_VALID_JSON)]
    )

    result, _cost, retried = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert retried is True
    assert len(fake.calls) == 2


def test_malformed_twice_degrades_to_adjudication_failed(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(
        monkeypatch, [_text_response(_MALFORMED_JSON), _text_response(_MALFORMED_JSON)]
    )

    with pytest.raises(adjudicator.AdjudicationFailedError):
        adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert len(fake.calls) == 2


def test_degraded_adjudication_never_raises_anything_but_adjudication_failed(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_client(
        monkeypatch, [_text_response("not json at all"), _text_response("still not json")]
    )

    try:
        adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)
        raise AssertionError("expected AdjudicationFailedError")
    except adjudicator.AdjudicationFailedError:
        pass


def test_a_response_with_no_content_is_treated_as_malformed(monkeypatch, tmp_path: Path) -> None:
    """Empty content (e.g. truncated before producing anything) goes through
    the same retry path as a bad JSON body."""
    no_content_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=None), finish_reason="length")],
        usage=_FakeUsage(10, 0),
    )
    _install_fake_client(monkeypatch, [no_content_response, _text_response(_VALID_JSON)])

    result, _cost, retried = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert retried is True


def test_a_refusal_is_treated_as_malformed(monkeypatch, tmp_path: Path) -> None:
    """A refusal sets message.refusal instead of message.content — content
    stays None, which _require_text already treats as malformed."""
    refusal_response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=None, refusal="cannot help with that"))],
        usage=_FakeUsage(10, 5),
    )
    _install_fake_client(monkeypatch, [refusal_response, _text_response(_VALID_JSON)])

    result, _cost, retried = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert retried is True
