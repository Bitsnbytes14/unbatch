"""`unbatch run --llm-only` (D1, METRICS.md's ablation arm C): skips L0-L3
entirely and sends every credit straight to the adjudicator. Guarded against
an accidental ~105-call live run — `adjudicator.adjudicate` is always
monkeypatched here, so a bug in the guard would be caught by the call-count
assertions, never by an actual API call."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from unbatch import audit
from unbatch.cli import app
from unbatch.models import AdjudicationResult, BreakReason

runner = CliRunner()


def _stub_adjudicate(monkeypatch, calls: list):
    def _fake(credit, expected_batch, delta_paise, candidates, *, cached, cache_dir=None):
        calls.append(credit.txn_id)
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


def test_llm_only_without_cached_or_confirm_spend_refuses_to_run(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list = []
    _stub_adjudicate(monkeypatch, calls)

    result = runner.invoke(app, ["run", "--llm-only", "--db", str(tmp_path / "audit.db")])

    assert result.exit_code == 1
    assert "confirm-spend" in result.output.lower() or "confirm_spend" in result.output.lower()
    assert calls == []  # never even started adjudicating


def test_llm_only_with_confirm_spend_proceeds(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _stub_adjudicate(monkeypatch, calls)
    db_path = tmp_path / "audit.db"

    result = runner.invoke(app, ["run", "--llm-only", "--confirm-spend", "--db", str(db_path)])

    assert result.exit_code == 0
    assert len(calls) == 105  # every credit, not just the residue L0-L3 would leave


def test_llm_only_with_cached_does_not_need_confirm_spend(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _stub_adjudicate(monkeypatch, calls)
    db_path = tmp_path / "audit.db"

    result = runner.invoke(app, ["run", "--llm-only", "--cached", "--db", str(db_path)])

    assert result.exit_code == 0
    assert len(calls) == 105


def test_llm_only_writes_one_decision_per_credit(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _stub_adjudicate(monkeypatch, calls)
    db_path = tmp_path / "audit.db"

    runner.invoke(app, ["run", "--llm-only", "--confirm-spend", "--db", str(db_path)])

    conn = audit.connect(db_path)
    run_id = audit.derive_run_id(42, arm="llm_only")
    decisions = audit.fetch_decisions(conn, run_id)
    assert len(decisions) == 105
    assert all(d.stage.value == "l4" for d in decisions)


def test_default_with_llm_arm_does_not_require_confirm_spend(monkeypatch, tmp_path: Path) -> None:
    """The guard is specific to --llm-only's much larger blast radius — the
    default arm's call count is already bounded by the rules layer."""
    calls: list = []
    _stub_adjudicate(monkeypatch, calls)
    db_path = tmp_path / "audit.db"

    result = runner.invoke(app, ["run", "--db", str(db_path)])

    assert result.exit_code == 0
    assert len(calls) == 12  # only what L0-L3 leave unresolved this dataset
