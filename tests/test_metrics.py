"""metrics.py: scoring against ground truth. Multiset comparison of
matched_payment_ids (payments and their refunds share a payment_id — see
this module's docstring for why not a set or an ordered list), and every
rate defined in METRICS.md computed from a hand-seeded scenario with known
answers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from unbatch import audit, metrics
from unbatch.models import Decision, DecisionOutcome, Stage

RUN_ID = "run_test"


def _decision(credit_id: str, **overrides: object) -> Decision:
    defaults: dict[object, object] = dict(
        run_id=RUN_ID,
        seed=1,
        stage=Stage.L0,
        credit_id=credit_id,
        matched_payment_ids=[],
        outcome=DecisionOutcome.MATCHED,
        confidence=1.0,
        delta_paise=0,
        reason="test",
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


def _write_ground_truth(path: Path) -> None:
    data = {
        "credits": [
            {
                "txn_id": "txn_correct",
                "settlement_ids": ["setl_a"],
                "payment_ids": ["pay_a", "pay_b"],
                "break_type": "clean",
                "resolvable": True,
            },
            {
                "txn_id": "txn_false",
                "settlement_ids": ["setl_c"],
                "payment_ids": ["pay_c"],
                "break_type": "settlement_split",
                "resolvable": True,
            },
            {
                "txn_id": "txn_missed",
                "settlement_ids": ["setl_d"],
                "payment_ids": ["pay_d"],
                "break_type": "fee_tier_change",
                "resolvable": True,
            },
            {
                "txn_id": "txn_unrelated",
                "settlement_ids": [],
                "payment_ids": [],
                "break_type": "unrelated_credit",
                "resolvable": False,
            },
            {
                "txn_id": "txn_multiset",
                "settlement_ids": ["setl_e", "setl_e_refund"],
                "payment_ids": ["pay_e", "pay_e"],
                "break_type": "refund_in_window",
                "resolvable": True,
            },
        ],
        "orphan_settlements": [{"settlement_ids": ["setl_orphan"], "payment_ids": ["pay_orphan"]}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_bank_statement(path: Path) -> None:
    rows = [
        ("txn_correct", "1000.00"),
        ("txn_false", "2000.00"),
        ("txn_missed", "3000.00"),
        ("txn_unrelated", "500.00"),
        ("txn_multiset", "4000.00"),
    ]
    lines = ["txn_id,value_date,narration,credit,debit,balance"]
    balance = 0
    for txn_id, amount in rows:
        balance += int(float(amount) * 100)
        lines.append(f"{txn_id},2024-01-05,test,{amount},,{balance / 100:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


@pytest.fixture
def scenario(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ground_truth(data_dir / "ground_truth.json")
    _write_bank_statement(data_dir / "bank_statement.csv")

    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision("txn_correct", matched_payment_ids=["pay_a", "pay_b"]))
    audit.record(
        conn,
        _decision("txn_false", stage=Stage.L2, matched_payment_ids=["pay_wrong"]),
    )
    audit.record(
        conn,
        _decision(
            "txn_missed",
            stage=Stage.L4,
            outcome=DecisionOutcome.EXCEPTION,
            matched_payment_ids=[],
        ),
    )
    audit.record(
        conn,
        _decision(
            "txn_unrelated",
            stage=Stage.L4,
            outcome=DecisionOutcome.EXCEPTION,
            matched_payment_ids=[],
        ),
    )
    audit.record(
        conn,
        _decision("txn_multiset", stage=Stage.L2, matched_payment_ids=["pay_e", "pay_e"]),
    )
    return conn, data_dir


def test_total_credits_comes_from_ground_truth(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.total_credits == 5


def test_count_match_rate(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.count_match_rate == pytest.approx(3 / 5)


def test_exception_rate(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.exception_rate == pytest.approx(2 / 5)


def test_false_match_rate(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.false_match_rate == pytest.approx(1 / 3)


def test_precision(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.precision == pytest.approx(2 / 3)


def test_recall_excludes_unresolvable_credits_from_the_denominator(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    # 4 resolvable credits (txn_unrelated excluded); 2 correct
    assert report.recall == pytest.approx(2 / 4)


def test_correctly_rejected_counts_unrelated_credit_and_orphan_settlement(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.correctly_rejected == 2  # txn_unrelated + 1 orphan_settlement


def test_value_weighted_match_rate(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    # resolved: correct(1000) + false(2000) + multiset(4000) = 7000 of 10500
    assert report.value_weighted_match_rate == pytest.approx(7000 / 10500)


def test_stage_funnel_counts_by_stage(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.stage_funnel == {"l0": 1, "l2": 2, "l4": 2}


def test_multiset_comparison_treats_duplicate_payment_id_as_correct(scenario) -> None:
    """A payment and its refund share a payment_id (DATA_SPEC.md); a
    correct match for refund_in_window legitimately has that id twice."""
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.false_match_rate < 1.0  # txn_multiset must not count as a false match


def test_multiset_comparison_flags_missing_duplicate_as_false_match(tmp_path: Path) -> None:
    """The mirror case: matched_payment_ids has the id only once when the
    correct answer needs it twice (e.g. composition found the payment line
    but missed the refund line) — must be a false match, not "close enough"."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ground_truth(data_dir / "ground_truth.json")
    _write_bank_statement(data_dir / "bank_statement.csv")

    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision("txn_correct", matched_payment_ids=["pay_a", "pay_b"]))
    audit.record(conn, _decision("txn_false", matched_payment_ids=["pay_wrong"]))
    audit.record(
        conn, _decision("txn_missed", outcome=DecisionOutcome.EXCEPTION, stage=Stage.L4)
    )
    audit.record(
        conn, _decision("txn_unrelated", outcome=DecisionOutcome.EXCEPTION, stage=Stage.L4)
    )
    # only one pay_e instead of the correct two — missing the refund line
    audit.record(conn, _decision("txn_multiset", matched_payment_ids=["pay_e"]))

    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    # resolved: txn_correct, txn_false, txn_multiset (3); correct: txn_correct
    # only (1); false: txn_false (wrong id) and txn_multiset (missing the
    # duplicate) — 2 of 3 resolved matches are wrong.
    assert report.false_match_rate == pytest.approx(2 / 3)


def test_llm_fields_are_zero_when_no_decision_used_an_llm(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.llm_call_count == 0
    assert report.llm_call_rate == 0.0
    assert report.llm_cost_paise == 0
    assert report.malformed_json_count == 0
    assert report.retry_count == 0
    assert report.adjudication_failed_count == 0


def _write_llm_ground_truth(path: Path) -> None:
    data = {
        "credits": [
            {
                "txn_id": "txn_a",
                "settlement_ids": ["setl_a"],
                "payment_ids": ["pay_a"],
                "break_type": "fee_tier_change",
                "resolvable": True,
            },
            {
                "txn_id": "txn_b",
                "settlement_ids": ["setl_b"],
                "payment_ids": ["pay_b"],
                "break_type": "duplicate_utr",
                "resolvable": True,
            },
            {
                "txn_id": "txn_c",
                "settlement_ids": ["setl_c"],
                "payment_ids": ["pay_c"],
                "break_type": "ambiguous_composition",
                "resolvable": True,
            },
            {
                "txn_id": "txn_d",
                "settlement_ids": ["setl_d"],
                "payment_ids": ["pay_d"],
                "break_type": "tolerance_ambiguous",
                "resolvable": True,
            },
        ],
        "orphan_settlements": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_llm_bank_statement(path: Path) -> None:
    rows = [("txn_a", "1000.00"), ("txn_b", "2000.00"), ("txn_c", "3000.00"), ("txn_d", "4000.00")]
    lines = ["txn_id,value_date,narration,credit,debit,balance"]
    balance = 0
    for txn_id, amount in rows:
        balance += int(float(amount) * 100)
        lines.append(f"{txn_id},2024-01-05,test,{amount},,{balance / 100:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


@pytest.fixture
def llm_scenario(tmp_path: Path):
    """Four credits that all reached L4: two correct classifications, one
    wrong classification, and one adjudication_failed (degraded after a
    retry, per adjudicator.adjudicate's contract) — enough to exercise
    break_reason_accuracy/confusion and the retry/malformed derivation."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_llm_ground_truth(data_dir / "ground_truth.json")
    _write_llm_bank_statement(data_dir / "bank_statement.csv")

    conn = audit.connect(tmp_path / "audit.db")
    audit.record(
        conn,
        _decision(
            "txn_a",
            stage=Stage.L4,
            matched_payment_ids=["pay_a"],
            reason="fee_tier_change",  # correct
            llm_model="claude-sonnet-5",
            llm_cost_paise=10,
            llm_retried=False,
        ),
    )
    audit.record(
        conn,
        _decision(
            "txn_b",
            stage=Stage.L4,
            matched_payment_ids=["pay_b"],
            reason="other",  # wrong — ground truth is duplicate_utr
            llm_model="claude-sonnet-5",
            llm_cost_paise=15,
            llm_retried=True,
        ),
    )
    audit.record(
        conn,
        _decision(
            "txn_c",
            stage=Stage.L4,
            matched_payment_ids=["pay_c"],
            reason="ambiguous_composition",  # correct
            llm_model="claude-sonnet-5",
            llm_cost_paise=12,
            llm_retried=False,
        ),
    )
    audit.record(
        conn,
        _decision(
            "txn_d",
            stage=Stage.L4,
            outcome=DecisionOutcome.EXCEPTION,
            matched_payment_ids=[],
            reason="adjudication_failed",
            llm_model="claude-sonnet-5",
            llm_cost_paise=0,
            llm_retried=True,
        ),
    )
    return conn, data_dir


def test_break_reason_accuracy_excludes_adjudication_failed(llm_scenario) -> None:
    conn, data_dir = llm_scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    # txn_a and txn_c correct, txn_b wrong, txn_d excluded -> 2/3
    assert report.break_reason_accuracy == pytest.approx(2 / 3)


def test_break_reason_confusion_is_keyed_by_ground_truth_then_predicted(llm_scenario) -> None:
    conn, data_dir = llm_scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.break_reason_confusion == {
        "fee_tier_change": {"fee_tier_change": 1},
        "duplicate_utr": {"other": 1},
        "ambiguous_composition": {"ambiguous_composition": 1},
    }


def test_retry_count_comes_from_llm_retried_flag(llm_scenario) -> None:
    conn, data_dir = llm_scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.retry_count == 2  # txn_b and txn_d both retried


def test_malformed_json_count_counts_both_attempts_on_a_double_failure(llm_scenario) -> None:
    conn, data_dir = llm_scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    # txn_b: one malformed response (retried, then succeeded)
    # txn_d: two malformed responses (retried, then still failed)
    assert report.malformed_json_count == 3


def test_adjudication_failed_count_from_llm_scenario(llm_scenario) -> None:
    conn, data_dir = llm_scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.adjudication_failed_count == 1


def test_break_reason_accuracy_is_zero_with_no_llm_decisions(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    assert report.break_reason_accuracy == 0.0
    assert report.break_reason_confusion == {}


def test_exception_break_type_counts_breaks_down_by_ground_truth_type(scenario) -> None:
    conn, data_dir = scenario
    report = metrics.score(conn, RUN_ID, data_dir=data_dir)
    # txn_missed(fee_tier_change) and txn_unrelated(unrelated_credit) are the
    # two exceptions in this scenario
    assert report.exception_break_type_counts == {
        "fee_tier_change": 1,
        "unrelated_credit": 1,
    }


def test_a_missing_decision_counts_as_exception_not_a_crash(tmp_path: Path) -> None:
    """Every credit should have exactly one decision after a full run, but
    scoring a partial/interrupted run must not crash — treat a credit with
    no decision at all the same as an exception."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ground_truth(data_dir / "ground_truth.json")
    _write_bank_statement(data_dir / "bank_statement.csv")

    conn = audit.connect(tmp_path / "audit.db")
    audit.record(conn, _decision("txn_correct", matched_payment_ids=["pay_a", "pay_b"]))
    # every other credit has no decision at all

    report = metrics.score(
        conn, RUN_ID, data_dir=data_dir, ground_truth_path=data_dir / "ground_truth.json"
    )
    assert report.total_credits == 5
    assert report.exception_rate == pytest.approx(4 / 5)
