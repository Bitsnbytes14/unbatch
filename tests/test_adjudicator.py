"""Tests for adjudicator.adjudicate's live-call, cache, and retry/degrade
behaviour. The Anthropic client is always stubbed via `adjudicator._client`
— nothing in this file makes a real network call, so these run with no API
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
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeResponse:
    content: list
    usage: _FakeUsage
    stop_reason: str = "end_turn"


class _FakeMessages:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("fake client called more times than responses were queued")
        return self._responses.pop(0)


@dataclass
class _FakeClient:
    messages: _FakeMessages = field(default_factory=lambda: _FakeMessages([]))


def _text_response(text: str, *, input_tokens: int = 100, output_tokens: int = 50):
    return _FakeResponse(
        content=[_FakeTextBlock(text=text)], usage=_FakeUsage(input_tokens, output_tokens)
    )


def _install_fake_client(monkeypatch, responses: list[_FakeResponse]) -> _FakeClient:
    fake = _FakeClient(_FakeMessages(responses))
    monkeypatch.setattr(adjudicator, "_client", lambda: fake)
    return fake


def _no_client_allowed(monkeypatch) -> None:
    def _boom():
        raise AssertionError("no API call should happen — this call should be served from cache")

    monkeypatch.setattr(adjudicator, "_client", _boom)


def test_adjudicate_calls_the_model_with_no_sampling_params(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])

    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert len(fake.messages.calls) == 1
    kwargs = fake.messages.calls[0]
    assert kwargs["model"] == adjudicator.MODEL
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_adjudicate_parses_a_valid_response(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(
        monkeypatch, [_text_response(_VALID_JSON, input_tokens=200, output_tokens=80)]
    )

    result, cost_paise = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert result.confidence == 0.9
    assert cost_paise > 0


def test_adjudicate_writes_one_cache_entry_per_prompt(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])

    adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    entries = list(tmp_path.glob("*.json"))
    assert len(entries) == 1
    assert b"\r" not in entries[0].read_bytes()


def test_cached_replay_makes_no_api_call(monkeypatch, tmp_path: Path) -> None:
    _install_fake_client(monkeypatch, [_text_response(_VALID_JSON)])
    first_result, first_cost = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    _no_client_allowed(monkeypatch)
    second_result, second_cost = adjudicator.adjudicate(
        _CREDIT, _BATCH, -775, [], cached=True, cache_dir=tmp_path
    )

    assert second_result == first_result
    assert second_cost == first_cost


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


def test_malformed_then_valid_retries_once_and_succeeds(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(
        monkeypatch, [_text_response(_MALFORMED_JSON), _text_response(_VALID_JSON)]
    )

    result, _cost = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert len(fake.messages.calls) == 2
    retry_prompt = fake.messages.calls[1]["messages"][0]["content"]
    assert "could not be read as valid JSON" in retry_prompt


def test_invalid_shape_then_valid_retries_once_and_succeeds(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(
        monkeypatch, [_text_response(_INVALID_SHAPE_JSON), _text_response(_VALID_JSON)]
    )

    result, _cost = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
    assert len(fake.messages.calls) == 2


def test_malformed_twice_degrades_to_adjudication_failed(monkeypatch, tmp_path: Path) -> None:
    fake = _install_fake_client(
        monkeypatch, [_text_response(_MALFORMED_JSON), _text_response(_MALFORMED_JSON)]
    )

    with pytest.raises(adjudicator.AdjudicationFailedError):
        adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert len(fake.messages.calls) == 2


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


def test_a_response_with_no_text_block_is_treated_as_malformed(monkeypatch, tmp_path: Path) -> None:
    no_text_response = _FakeResponse(content=[], usage=_FakeUsage(10, 0), stop_reason="max_tokens")
    _install_fake_client(monkeypatch, [no_text_response, _text_response(_VALID_JSON)])

    result, _cost = adjudicator.adjudicate(_CREDIT, _BATCH, -775, [], cache_dir=tmp_path)

    assert result.break_reason == BreakReason.FEE_TIER_CHANGE
