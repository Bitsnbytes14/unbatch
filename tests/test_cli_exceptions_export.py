"""`unbatch exceptions --export` — the same audit-log query as the plain
`exceptions` command, written out as a CSV work item an analyst can act on
(DATA_SPEC.md/README.md's "honest exception list" requirement) instead of
console lines."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from unbatch import audit
from unbatch import generate as generate_module
from unbatch.cli import _score_rules_only_for_seed, app
from unbatch.models import Decision, DecisionOutcome, Stage

runner = CliRunner()


def _decision(**overrides: object) -> Decision:
    defaults: dict[object, object] = dict(
        run_id="run_42_abc123",
        seed=42,
        stage=Stage.L4,
        credit_id="txn_1",
        matched_payment_ids=[],
        outcome=DecisionOutcome.EXCEPTION,
        confidence=0.2,
        delta_paise=1250,
        reason="pool_too_large",
        rationale=None,
        llm_model=None,
        llm_cost_paise=None,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_export_writes_one_row_per_exception_and_skips_matched(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision(credit_id="txn_exception"))
    audit.record(conn, _decision(credit_id="txn_matched", outcome=DecisionOutcome.MATCHED))
    conn.close()

    out_csv = tmp_path / "exceptions.csv"
    result = runner.invoke(app, ["exceptions", "--db", str(db_path), "--export", str(out_csv)])

    assert result.exit_code == 0, result.output
    assert "wrote 1 rows" in result.output
    rows = _read_csv_rows(out_csv)
    assert [r["txn_id"] for r in rows] == ["txn_exception"]


def test_export_header_has_every_required_column(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision())
    conn.close()

    out_csv = tmp_path / "exceptions.csv"
    runner.invoke(app, ["exceptions", "--db", str(db_path), "--export", str(out_csv)])

    with open(out_csv, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == [
        "txn_id",
        "value_date",
        "credit_amount_rupees",
        "narration",
        "delta_rupees",
        "stage",
        "reason",
        "break_reason",
        "evidence_refs",
        "proposed_resolution",
        "human_review_required",
        "analyst_resolution",
    ]


def test_export_fills_break_reason_and_evidence_only_for_llm_exceptions(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(
        conn,
        _decision(
            credit_id="txn_llm",
            reason="ambiguous_composition",
            rationale="escalate for manual review",
            llm_model="gpt-5-nano",
            llm_cost_paise=1,
            evidence_refs=["pay_1", "settle_2"],
            human_review_required=True,
        ),
    )
    audit.record(conn, _decision(credit_id="txn_rules_only", reason="pool_too_large"))
    conn.close()

    out_csv = tmp_path / "exceptions.csv"
    runner.invoke(app, ["exceptions", "--db", str(db_path), "--export", str(out_csv)])
    rows = {r["txn_id"]: r for r in _read_csv_rows(out_csv)}

    assert rows["txn_llm"]["break_reason"] == "ambiguous_composition"
    assert rows["txn_llm"]["evidence_refs"] == "pay_1;settle_2"
    assert rows["txn_llm"]["proposed_resolution"] == "escalate for manual review"
    assert rows["txn_llm"]["human_review_required"] == "True"
    assert rows["txn_llm"]["analyst_resolution"] == ""

    assert rows["txn_rules_only"]["break_reason"] == ""
    assert rows["txn_rules_only"]["evidence_refs"] == ""
    assert rows["txn_rules_only"]["human_review_required"] == ""


def test_export_looks_up_credit_details_from_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_module.generate(42, out_dir=data_dir)
    bank_records = generate_module.read_bank_statement_csv(data_dir / "bank_statement.csv")
    some_credit = next(r for r in bank_records if r.credit_paise is not None)

    db_path = tmp_path / "audit.db"
    conn = audit.connect(db_path)
    audit.record(conn, _decision(credit_id=some_credit.txn_id))
    conn.close()

    out_csv = tmp_path / "exceptions.csv"
    runner.invoke(
        app,
        [
            "exceptions", "--db", str(db_path), "--data-dir", str(data_dir),
            "--export", str(out_csv),
        ],
    )
    [row] = _read_csv_rows(out_csv)
    assert row["value_date"] == some_credit.value_date.isoformat()
    assert row["narration"] == some_credit.narration


def test_export_row_count_matches_metrics_exception_count(tmp_path: Path) -> None:
    """The round trip the export exists to satisfy: exporting a real run's
    exceptions produces exactly as many rows as metrics.py reports for that
    same run — same audit log, same query shape, no separately drifting
    list."""
    data_dir = tmp_path / "data"
    db_path = tmp_path / "audit.db"
    report = _score_rules_only_for_seed(42, data_dir, db_path)
    expected_exception_count = sum(report.exception_break_type_counts.values())

    run_id = audit.derive_run_id(42, data_dir, arm="no_llm")
    out_csv = tmp_path / "exceptions.csv"
    result = runner.invoke(
        app,
        [
            "exceptions", "--db", str(db_path), "--data-dir", str(data_dir),
            "--run-id", run_id, "--export", str(out_csv),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = _read_csv_rows(out_csv)
    assert len(rows) == expected_exception_count
