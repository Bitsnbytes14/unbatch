"""`unbatch report`: renders out/report.html from whichever arms have real
decisions in the audit DB for this seed's data."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from unbatch import audit
from unbatch.cli import (
    app,
    build_unresolved_credits,
    compute_expected_batches,
    load_input_data,
    run_cascade,
)
from unbatch.models import RunContext

runner = CliRunner()


def test_report_renders_html_from_a_real_no_llm_run(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)
    run_id = audit.derive_run_id(42, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=42, no_llm=True)
    conn = audit.connect(db_path)
    audit.clear_run(conn, run_id)
    run_cascade(ctx, unresolved, conn, settlements=settlements)

    out_path = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--db", str(db_path), "--out", str(out_path)])

    assert result.exit_code == 0
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "unbatch: settlement reconciliation report" in html
    assert b"\r" not in out_path.read_bytes()


def test_report_against_an_empty_audit_db_shows_every_arm_as_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    audit.connect(db_path)
    out_path = tmp_path / "report.html"

    result = runner.invoke(app, ["report", "--db", str(db_path), "--out", str(out_path)])

    assert result.exit_code == 0
    html = out_path.read_text(encoding="utf-8")
    assert html.count("not yet run") >= 3  # each arm's own "not yet run" section note
