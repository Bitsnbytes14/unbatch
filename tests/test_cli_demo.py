"""`unbatch demo`: generate + all three ablation arms + metrics + report in
one command. This tests only the orchestration (progress output, arms all
populating, the cache-miss guard) — the underlying stage/adjudicator/
metrics/report logic each wrapped command calls is already covered by its
own tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import app
from unbatch.models import AdjudicationResult, BreakReason

runner = CliRunner()


def _stub_adjudicate(monkeypatch) -> None:
    """Bypasses the real cache/ lookup entirely, same pattern
    test_cli_run_llm_only.py uses — demo's orchestration is what's under
    test here, not adjudicator.py's own cache/retry behavior."""

    def _fake(credit, expected_batch, delta_paise, candidates, *, cached, cache_dir=None):
        return (
            AdjudicationResult(
                break_reason=BreakReason.OTHER,
                proposed_resolution="reviewed",
                confidence=0.5,
                evidence_refs=[],
                human_review_required=True,
            ),
            5,
            False,
        )

    from unbatch.stages import l4_llm

    monkeypatch.setattr(l4_llm.adjudicator, "adjudicate", _fake)


def test_demo_runs_all_three_arms_and_writes_a_report(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_adjudicate(monkeypatch)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "out" / "report.html"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "not yet run" not in html


def test_demo_prints_progress_for_every_step(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_adjudicate(monkeypatch)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0, result.output
    for expected in (
        "[1/6]",
        "[2/6]",
        "[3/6]",
        "[4/6]",
        "[5/6]",
        "[6/6]",
    ):
        assert expected in result.output


def test_demo_prints_headline_numbers_and_the_report_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_adjudicate(monkeypatch)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0, result.output
    assert "total credits:" in result.output
    assert "match rate:" in result.output
    assert "false-match rate:" in result.output
    assert "exceptions:" in result.output
    assert "LLM cost:" in result.output
    assert "report:" in result.output
    assert "report.html" in result.output


def test_demo_default_seed_is_42(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_adjudicate(monkeypatch)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0, result.output
    assert "generate --seed 42" in result.output


def test_demo_accepts_a_different_seed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_adjudicate(monkeypatch)

    result = runner.invoke(app, ["demo", "--seed", "43"])

    assert result.exit_code == 0, result.output
    assert "generate --seed 43" in result.output


def test_demo_fails_loudly_on_a_cache_miss_instead_of_attempting_a_live_call(
    monkeypatch, tmp_path: Path
) -> None:
    """No adjudicator stub here on purpose — this exercises the real cache
    lookup against a directory with no cache/ at all, which must raise
    CacheMissError and never fall through to a live call. No API key is
    set or needed for this to fail the way it should."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["demo"])

    assert result.exit_code != 0
    assert "never makes a live call" in result.output
    assert "does not cover" in result.output
