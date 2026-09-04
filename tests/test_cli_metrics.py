"""`unbatch metrics`: scores whichever run --seed/--arm point at and prints
JSON; --out also writes it to a file — this is how baseline_rules_only.json
is produced and can be reproduced."""

from __future__ import annotations

import json
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


def _run_no_llm(db_path: Path) -> None:
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)
    run_id = audit.derive_run_id(42, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=42, no_llm=True)
    conn = audit.connect(db_path)
    audit.clear_run(conn, run_id)
    run_cascade(ctx, unresolved, conn, settlements=settlements)


def test_metrics_prints_valid_json_for_a_real_run(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    _run_no_llm(db_path)

    result = runner.invoke(app, ["metrics", "--db", str(db_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_credits"] == 105
    assert payload["false_match_rate"] == 0.0


def test_metrics_out_writes_the_same_json_to_a_file(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    _run_no_llm(db_path)
    out_path = tmp_path / "baseline.json"

    result = runner.invoke(app, ["metrics", "--db", str(db_path), "--out", str(out_path)])

    assert result.exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == json.loads(result.stdout)
    assert b"\r" not in out_path.read_bytes()


def test_metrics_against_an_empty_audit_db_fails_loudly_instead_of_reporting_zeros(
    tmp_path: Path,
) -> None:
    """Before any run has happened, scoring must not silently print a table
    of zeros — a reviewer following the quickstart with the wrong arm (or
    no run at all) needs a clear error, not a project that looks broken."""
    db_path = tmp_path / "audit.db"
    audit.connect(db_path)

    result = runner.invoke(app, ["metrics", "--db", str(db_path)])

    assert result.exit_code != 0
    assert "no runs found" in result.output


def test_metrics_defaults_to_the_most_recently_run_arm(tmp_path: Path) -> None:
    """No --arm given, and the audit log holds a with_llm run, not the old
    hardcoded no_llm default — metrics must report the run that actually
    exists, the way a reviewer running the README quickstart would expect."""
    db_path = tmp_path / "audit.db"
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)
    run_id = audit.derive_run_id(42, arm="with_llm")
    ctx = RunContext(run_id=run_id, seed=42, cached=True)
    conn = audit.connect(db_path)
    audit.clear_run(conn, run_id)
    run_cascade(ctx, unresolved, conn, settlements=settlements)

    result = runner.invoke(app, ["metrics", "--db", str(db_path)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["run_id"] == run_id
    assert payload["total_credits"] == 105


def test_metrics_with_an_arm_that_was_never_run_fails_loudly(tmp_path: Path) -> None:
    """Requesting an arm whose run_id has no decisions must error and name
    the runs that ARE present, not silently score a table of zeros."""
    db_path = tmp_path / "audit.db"
    _run_no_llm(db_path)

    result = runner.invoke(app, ["metrics", "--db", str(db_path), "--arm", "llm_only"])

    assert result.exit_code != 0
    assert "no decisions found" in result.output
    assert "no_llm" in result.output
