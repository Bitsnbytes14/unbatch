"""Tests for report.py: the pure data-prep helpers directly, plus one
end-to-end render against a hand-seeded scenario to catch template errors
(a missing/renamed field would raise inside Jinja, not silently render
wrong — jinja2.Environment here uses no `undefined` override, so Jinja's
default Undefined still stringifies quietly in most contexts; the
end-to-end test instead asserts on the actual rendered numbers so a wiring
mistake shows up as a wrong value, not just a successful render)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from unbatch import audit, report
from unbatch.models import Decision, DecisionOutcome, Stage

RUN_ID = "run_test"


def test_format_rupees_positive() -> None:
    assert report.format_rupees(41833) == "₹418.33"


def test_format_rupees_negative_keeps_sign_before_the_symbol() -> None:
    assert report.format_rupees(-500) == "-₹5.00"


def test_format_rupees_zero() -> None:
    assert report.format_rupees(0) == "₹0.00"


def test_format_percent() -> None:
    assert report.format_percent(0.8857142857142857) == "88.6%"


def _decision(credit_id: str, confidence: float, **overrides: object) -> Decision:
    defaults: dict[object, object] = dict(
        run_id=RUN_ID,
        seed=1,
        stage=Stage.L0,
        credit_id=credit_id,
        matched_payment_ids=[],
        outcome=DecisionOutcome.MATCHED,
        confidence=confidence,
        delta_paise=0,
        reason="test",
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


def test_confidence_histogram_buckets_by_tenth() -> None:
    decisions = [
        _decision("a", 0.0),
        _decision("b", 0.05),
        _decision("c", 0.75),
        _decision("d", 0.9),
    ]
    histogram = report._confidence_histogram(decisions)
    counts = dict(histogram)
    assert counts["0.0 to 0.1"] == 2
    assert counts["0.7 to 0.8"] == 1
    assert counts["0.9 to 1.0"] == 1
    assert sum(counts.values()) == 4


def test_confidence_histogram_does_not_overflow_at_exactly_1_0() -> None:
    histogram = report._confidence_histogram([_decision("a", 1.0)])
    counts = dict(histogram)
    assert counts["0.9 to 1.0"] == 1
    assert len(histogram) == 10  # no 11th bucket created


def test_confusion_table_is_none_when_empty() -> None:
    assert report._confusion_table({}) is None


def test_confusion_table_columns_are_the_union_of_predicted_reasons_sorted() -> None:
    confusion = {
        "ambiguous_composition": {"ambiguous_composition": 5, "other": 2},
        "tolerance_ambiguous": {"unrelated_credit": 1},
    }
    table = report._confusion_table(confusion)
    assert table is not None
    assert table.columns == ["ambiguous_composition", "other", "unrelated_credit"]
    rows = dict(table.rows)
    assert rows["ambiguous_composition"] == [5, 2, 0]
    assert rows["tolerance_ambiguous"] == [0, 0, 1]


def test_ambiguity_framing_splits_out_the_named_categories() -> None:
    fake_report = report.metrics_module.MetricsReport(
        run_id=RUN_ID,
        total_credits=105,
        count_match_rate=0.9,
        value_weighted_match_rate=0.9,
        false_match_rate=0.0,
        exception_rate=0.1,
        precision=1.0,
        recall=0.9,
        correctly_rejected=8,
        stage_funnel={},
        llm_call_count=0,
        llm_call_rate=0.0,
        llm_cost_paise=0,
        cost_paise_per_adjudicated_credit=0.0,
        cost_paise_per_exception=0.0,
        malformed_json_count=0,
        retry_count=0,
        adjudication_failed_count=0,
        break_reason_accuracy=0.0,
        break_reason_confusion={},
        exception_break_type_counts={
            "ambiguous_composition": 7,
            "tolerance_ambiguous": 4,
            "unrelated_credit": 1,
            "date_skew": 2,
        },
    )
    framing = report._ambiguity_framing(fake_report)
    assert framing.total_exceptions == 14
    assert framing.correctly_declined == 11
    assert framing.other == 2  # date_skew, not accounted for by the named categories


def _write_ground_truth(path: Path) -> None:
    data = {
        "credits": [
            {
                "txn_id": "txn_ok",
                "settlement_ids": ["setl_a"],
                "payment_ids": ["pay_a"],
                "break_type": "clean",
                "resolvable": True,
            },
            {
                "txn_id": "txn_ambiguous",
                "settlement_ids": ["setl_b"],
                "payment_ids": ["pay_b"],
                "break_type": "ambiguous_composition",
                "resolvable": True,
            },
            {
                "txn_id": "txn_unrelated",
                "settlement_ids": [],
                "payment_ids": [],
                "break_type": "unrelated_credit",
                "resolvable": False,
            },
        ],
        "orphan_settlements": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_bank_statement(path: Path) -> None:
    rows = [("txn_ok", "1000.00"), ("txn_ambiguous", "2000.00"), ("txn_unrelated", "300.00")]
    lines = ["txn_id,value_date,narration,credit,debit,balance"]
    balance = 0
    for txn_id, amount in rows:
        balance += int(float(amount) * 100)
        lines.append(f"{txn_id},2024-01-05,test,{amount},,{balance / 100:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def test_render_end_to_end_with_one_arm_populated(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ground_truth(data_dir / "ground_truth.json")
    _write_bank_statement(data_dir / "bank_statement.csv")
    # derive_run_id hashes all three input CSVs' bytes, even though this
    # test's scoring path never reads order_ledger/settlement_report content
    (data_dir / "order_ledger.csv").write_text(
        "order_id,payment_id,amount,currency,status,captured_at,customer_ref,method\n",
        encoding="utf-8",
        newline="",
    )
    (data_dir / "settlement_report.csv").write_text(
        "settlement_id,settlement_utr,payment_id,type,gross,fee,tax,net,settled_at\n",
        encoding="utf-8",
        newline="",
    )

    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    seed = 7
    no_llm_run_id = audit.derive_run_id(seed, data_dir, arm="no_llm")
    audit.record(
        conn,
        Decision(
            run_id=no_llm_run_id,
            seed=seed,
            stage=Stage.L0,
            credit_id="txn_ok",
            matched_payment_ids=["pay_a"],
            outcome=DecisionOutcome.MATCHED,
            confidence=1.0,
            delta_paise=0,
            reason="utr_exact_match",
            rationale=None,
            llm_model=None,
            llm_cost_paise=None,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )
    for txn_id in ("txn_ambiguous", "txn_unrelated"):
        audit.record(
            conn,
            Decision(
                run_id=no_llm_run_id,
                seed=seed,
                stage=Stage.L4,
                credit_id=txn_id,
                matched_payment_ids=[],
                outcome=DecisionOutcome.EXCEPTION,
                confidence=0.0,
                delta_paise=0,
                reason="no_llm_unresolved",
                rationale=None,
                llm_model=None,
                llm_cost_paise=None,
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        )

    out_path = report.render(seed, data_dir=data_dir, db=db_path, out_path=tmp_path / "report.html")

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert b"\r" not in out_path.read_bytes()

    # arm A populated with real numbers
    assert "33.3%" in html  # count_match_rate = 1/3
    assert "txn_ambiguous" in html
    assert "txn_unrelated" in html

    # arms B and C correctly shown as pending, not faked
    assert "not yet run" in html
    assert "Arm B (rules + LLM) has not been run yet" in html

    # D3's framing text, computed from this scenario's own 2 exceptions
    assert "of the 2 credits reaching L4 in arm A" in html
    assert "1 <code>ambiguous_composition</code>" in html
    assert "plus 1 <code>unrelated_credit</code>" in html


def test_render_with_arm_b_populated_shows_delta_and_break_reason_accuracy(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ground_truth(data_dir / "ground_truth.json")
    _write_bank_statement(data_dir / "bank_statement.csv")
    (data_dir / "order_ledger.csv").write_text(
        "order_id,payment_id,amount,currency,status,captured_at,customer_ref,method\n",
        encoding="utf-8",
        newline="",
    )
    (data_dir / "settlement_report.csv").write_text(
        "settlement_id,settlement_utr,payment_id,type,gross,fee,tax,net,settled_at\n",
        encoding="utf-8",
        newline="",
    )

    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    seed = 7

    def _matched(run_id: str, txn_id: str, payment_ids: list[str]) -> Decision:
        return Decision(
            run_id=run_id,
            seed=seed,
            stage=Stage.L0,
            credit_id=txn_id,
            matched_payment_ids=payment_ids,
            outcome=DecisionOutcome.MATCHED,
            confidence=1.0,
            delta_paise=0,
            reason="utr_exact_match",
            rationale=None,
            llm_model=None,
            llm_cost_paise=None,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

    no_llm_run_id = audit.derive_run_id(seed, data_dir, arm="no_llm")
    audit.record(conn, _matched(no_llm_run_id, "txn_ok", ["pay_a"]))
    for txn_id in ("txn_ambiguous", "txn_unrelated"):
        audit.record(
            conn,
            Decision(
                run_id=no_llm_run_id,
                seed=seed,
                stage=Stage.L4,
                credit_id=txn_id,
                matched_payment_ids=[],
                outcome=DecisionOutcome.EXCEPTION,
                confidence=0.0,
                delta_paise=0,
                reason="no_llm_unresolved",
                rationale=None,
                llm_model=None,
                llm_cost_paise=None,
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        )

    with_llm_run_id = audit.derive_run_id(seed, data_dir, arm="with_llm")
    audit.record(conn, _matched(with_llm_run_id, "txn_ok", ["pay_a"]))
    audit.record(
        conn,
        Decision(
            run_id=with_llm_run_id,
            seed=seed,
            stage=Stage.L4,
            credit_id="txn_ambiguous",
            matched_payment_ids=["pay_b"],
            outcome=DecisionOutcome.MATCHED,
            confidence=0.9,
            delta_paise=0,
            reason="ambiguous_composition",  # correct classification
            rationale="picked the closer batch",
            llm_model="gpt-5-nano",
            llm_cost_paise=10,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )
    audit.record(
        conn,
        Decision(
            run_id=with_llm_run_id,
            seed=seed,
            stage=Stage.L4,
            credit_id="txn_unrelated",
            matched_payment_ids=[],
            outcome=DecisionOutcome.EXCEPTION,
            confidence=0.3,
            delta_paise=0,
            reason="unrelated_credit",
            rationale="nothing to tie to",
            llm_model="gpt-5-nano",
            llm_cost_paise=8,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )

    out_path = report.render(seed, data_dir=data_dir, db=db_path, out_path=tmp_path / "report.html")
    html = out_path.read_text(encoding="utf-8")

    # both credits now resolve in arm B: count_match_rate = 3/3 = 100%
    assert "Value the AI added to match rate (B − A)" in html
    assert "100.0%" in html
    # break-reason accuracy called out separately from the delta
    assert "What the delta doesn&#39;t show" in html or "What the delta doesn't show" in html
    assert "break-reason accuracy on arm B is 100.0%" in html
    # cost per unit: 18 paise total / 2 calls, 18 paise / 1 exception (txn_unrelated)
    assert "9.00 paise" in html
    assert "18.00 paise" in html


def test_render_with_all_three_arms_states_the_c_vs_b_comparison_precisely(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ground_truth(data_dir / "ground_truth.json")
    _write_bank_statement(data_dir / "bank_statement.csv")
    (data_dir / "order_ledger.csv").write_text(
        "order_id,payment_id,amount,currency,status,captured_at,customer_ref,method\n",
        encoding="utf-8",
        newline="",
    )
    (data_dir / "settlement_report.csv").write_text(
        "settlement_id,settlement_utr,payment_id,type,gross,fee,tax,net,settled_at\n",
        encoding="utf-8",
        newline="",
    )

    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    seed = 7

    def _llm_decision(
        run_id: str, txn_id: str, *, outcome: DecisionOutcome, matched: list[str], reason: str
    ) -> Decision:
        return Decision(
            run_id=run_id,
            seed=seed,
            stage=Stage.L4,
            credit_id=txn_id,
            matched_payment_ids=matched,
            outcome=outcome,
            confidence=0.9 if outcome == DecisionOutcome.MATCHED else 0.3,
            delta_paise=0,
            reason=reason,
            rationale="x",
            llm_model="gpt-5-nano",
            llm_cost_paise=5,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

    # arm B: 2 calls (txn_ambiguous, txn_unrelated), both classified correctly
    with_llm_run_id = audit.derive_run_id(seed, data_dir, arm="with_llm")
    audit.record(
        conn,
        Decision(
            run_id=with_llm_run_id,
            seed=seed,
            stage=Stage.L0,
            credit_id="txn_ok",
            matched_payment_ids=["pay_a"],
            outcome=DecisionOutcome.MATCHED,
            confidence=1.0,
            delta_paise=0,
            reason="utr_exact_match",
            rationale=None,
            llm_model=None,
            llm_cost_paise=None,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )
    audit.record(
        conn,
        _llm_decision(
            with_llm_run_id,
            "txn_ambiguous",
            outcome=DecisionOutcome.MATCHED,
            matched=["pay_b"],
            reason="ambiguous_composition",
        ),
    )
    audit.record(
        conn,
        _llm_decision(
            with_llm_run_id,
            "txn_unrelated",
            outcome=DecisionOutcome.EXCEPTION,
            matched=[],
            reason="unrelated_credit",
        ),
    )

    # arm C: 3 calls (every credit), only 1 resolved, 2/3 classified correctly
    llm_only_run_id = audit.derive_run_id(seed, data_dir, arm="llm_only")
    audit.record(
        conn,
        _llm_decision(
            llm_only_run_id,
            "txn_ok",
            outcome=DecisionOutcome.MATCHED,
            matched=["pay_a"],
            reason="clean",
        ),
    )
    audit.record(
        conn,
        _llm_decision(
            llm_only_run_id,
            "txn_ambiguous",
            outcome=DecisionOutcome.EXCEPTION,
            matched=[],
            reason="ambiguous_composition",
        ),
    )
    audit.record(
        conn,
        _llm_decision(
            llm_only_run_id,
            "txn_unrelated",
            outcome=DecisionOutcome.EXCEPTION,
            matched=[],
            reason="tolerance_ambiguous",  # wrong — ground truth is unrelated_credit
        ),
    )

    out_path = report.render(seed, data_dir=data_dir, db=db_path, out_path=tmp_path / "report.html")
    html = out_path.read_text(encoding="utf-8")

    # arm B: 2 calls, 2/3 resolved (txn_unrelated correctly stays an
    # exception) = 66.7%, both LLM classifications correct = 100%
    # arm C: 3 calls, 1/3 resolved (33.3%), 2/3 correct (66.7%)
    assert "Arm C made 3 LLM calls against arm B's 2" in html
    assert "1.5 times as many" in html
    assert "33.3-point" in html  # both gaps happen to be 33.3 points here:
    # match-rate 66.7% - 33.3%, break-reason-accuracy 100.0% - 66.7%
