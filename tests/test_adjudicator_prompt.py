"""Tests for adjudicator.build_prompt. CLAUDE.md invariant 2 says the LLM
never does arithmetic — the grep test below is the actual enforcement
mechanism for that invariant, not just a design aspiration."""

from __future__ import annotations

import re
from datetime import date

from unbatch.adjudicator import build_prompt
from unbatch.models import BankStatementRecord, BreakReason, CandidateExplanation, ExpectedBatch

_BANNED = re.compile(r"calcul|comput|sum", re.IGNORECASE)

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

_CANDIDATES = [
    CandidateExplanation(
        payment_ids=["pay_c"], delta_paise=-775, hint="alternate batch UTR000999"
    ),
]


def test_prompt_has_no_arithmetic_instructions() -> None:
    system, user = build_prompt(_CREDIT, _BATCH, -775, _CANDIDATES)
    combined = system + "\n" + user
    assert not _BANNED.search(combined), combined


def test_prompt_has_no_arithmetic_instructions_with_no_candidates() -> None:
    system, user = build_prompt(_CREDIT, _BATCH, -775, [])
    combined = system + "\n" + user
    assert not _BANNED.search(combined), combined


def test_user_prompt_includes_credit_facts() -> None:
    _system, user = build_prompt(_CREDIT, _BATCH, -775, [])
    assert "TXN1" in user
    assert "418332" in user
    assert "NEFT-UTR000123-RAZORPAY" in user


def test_user_prompt_includes_batch_facts() -> None:
    _system, user = build_prompt(_CREDIT, _BATCH, -775, [])
    assert "UTR000123" in user
    assert "419107" in user
    assert "pay_a" in user
    assert "pay_b" in user


def test_user_prompt_includes_the_precomputed_delta() -> None:
    _system, user = build_prompt(_CREDIT, _BATCH, -775, [])
    assert "-775" in user


def test_user_prompt_includes_candidate_explanations() -> None:
    _system, user = build_prompt(_CREDIT, _BATCH, -775, _CANDIDATES)
    assert "pay_c" in user
    assert "alternate batch UTR000999" in user


def test_user_prompt_states_no_candidates_explicitly_when_empty() -> None:
    _system, user = build_prompt(_CREDIT, _BATCH, -775, [])
    assert "none surfaced" in user


def test_system_prompt_lists_every_break_reason() -> None:
    system, _user = build_prompt(_CREDIT, _BATCH, -775, [])
    for reason in BreakReason:
        assert reason.value in system


def test_system_prompt_demands_json_only_output() -> None:
    system, _user = build_prompt(_CREDIT, _BATCH, -775, [])
    assert "JSON object" in system
    assert "break_reason" in system
    assert "confidence" in system
    assert "evidence_refs" in system
    assert "human_review_required" in system
