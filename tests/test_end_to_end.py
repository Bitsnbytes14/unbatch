"""End-to-end regression guard: generate --seed 42 -> run --cached (all
three arms) -> metrics, asserting the exact numbers reported in README.md.

Must pass with no OPENAI_API_KEY at all — the whole point of committing
cache/ is that anyone who clones this repo can reproduce every number with
zero credentials (CLAUDE.md invariant 6). `monkeypatch.delenv` removes any
key from *this test's* environment regardless of what the real shell has
set, and the cache-file-count assertions below independently confirm no
live call was attempted, not just that none happened to be needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from unbatch import generate as generate_module
from unbatch.cli import app

runner = CliRunner()

SEED = 42
REAL_DATA_DIR = Path("data")
REAL_CACHE_DIR = Path("cache")


def test_generate_reproduces_the_committed_fixtures_byte_for_byte(tmp_path: Path) -> None:
    """The starting point for everything else this file checks: a fresh
    generate() must match what's actually committed, not just "look
    similar" — otherwise the run --cached step below would be exercising
    different data than what the committed cache/ was ever built against."""
    out_dir = tmp_path / "data"
    generate_module.generate(SEED, out_dir=out_dir)
    for filename in (
        "order_ledger.csv",
        "settlement_report.csv",
        "bank_statement.csv",
        "ground_truth.json",
    ):
        assert (out_dir / filename).read_bytes() == (REAL_DATA_DIR / filename).read_bytes(), (
            filename
        )


def test_end_to_end_no_key_reproduces_the_readme_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_AUTH_TOKEN", raising=False)

    data_dir = tmp_path / "data"
    generate_module.generate(SEED, out_dir=data_dir)
    db_path = tmp_path / "audit.db"

    cache_files_before = set(REAL_CACHE_DIR.glob("*.json"))

    def _run(*flags: str) -> None:
        result = runner.invoke(
            app,
            ["run", "--seed", str(SEED), *flags, "--data-dir", str(data_dir), "--db", str(db_path)],
        )
        assert result.exit_code == 0, result.output

    def _metrics(arm: str) -> dict:
        result = runner.invoke(
            app,
            [
                "metrics", "--seed", str(SEED), "--arm", arm,
                "--data-dir", str(data_dir), "--db", str(db_path),
            ],
        )
        assert result.exit_code == 0, result.output
        return json.loads(result.output)

    _run("--no-llm")
    _run("--cached")
    _run("--llm-only", "--cached")

    # no live call was attempted anywhere above, not just none was needed —
    # if one had been (and somehow not raised for lack of a key), it would
    # have written a new file into the committed cache directory.
    cache_files_after = set(REAL_CACHE_DIR.glob("*.json"))
    assert cache_files_after == cache_files_before

    a = _metrics("no_llm")
    b = _metrics("with_llm")
    c = _metrics("llm_only")

    # Arm A — rules only
    assert a["total_credits"] == 105
    assert a["count_match_rate"] == pytest.approx(93 / 105)
    assert a["false_match_rate"] == 0.0
    assert a["exception_rate"] == pytest.approx(12 / 105)
    assert a["precision"] == 1.0
    assert a["correctly_rejected"] == 8
    assert a["stage_funnel"] == {"l0": 79, "l1": 3, "l2": 9, "l3": 2, "l4": 12}
    assert a["llm_call_count"] == 0
    assert a["llm_cost_paise"] == 0

    # Arm B — rules + LLM: the match-rate delta over A is exactly zero —
    # the 12 credits reaching L4 are ones rules correctly declined, and the
    # model correctly declines almost all of them too (see README.md).
    #
    # 2026-09-01 (FAILURES.md): these numbers moved from an earlier
    # measurement after adjudicator.py gained semantic validation of
    # evidence_refs (schema-shape validation alone wasn't enough — see the
    # module's docstring). Re-running with the stricter check found the
    # model citing a settlement UTR as evidence on several of these 12
    # credits; live retries were made to get a real (not stale) reading.
    # Most retries (6 of 7) did not recover a valid classification even on
    # the second try, which is the real, measured cost of enforcing this
    # check — not a bug in the check or in these numbers.
    assert b["total_credits"] == 105
    assert b["count_match_rate"] == pytest.approx(93 / 105)
    assert b["false_match_rate"] == 0.0
    assert b["stage_funnel"] == {"l0": 79, "l1": 3, "l2": 9, "l3": 2, "l4": 12}
    assert b["llm_call_count"] == 12
    assert b["llm_cost_paise"] == 19
    assert b["break_reason_accuracy"] == pytest.approx(5 / 6)
    assert b["malformed_json_count"] == 13
    assert b["retry_count"] == 7
    assert b["adjudication_failed_count"] == 6

    # Arm C — LLM only: the measured argument for the cascade. Same
    # 2026-09-01 re-measurement as arm B, at much larger scale (all 105
    # credits go to the model here, not just the 12 rules leave unresolved):
    # 33 retries were needed, and one of them landed on a *worse* second
    # answer than the (semantically rejected) first — a real false match,
    # not present before this session. See the note below.
    assert c["total_credits"] == 105
    assert c["count_match_rate"] == pytest.approx(11 / 105)
    assert c["false_match_rate"] == pytest.approx(1 / 11)
    assert c["llm_call_count"] == 105
    assert c["llm_cost_paise"] == 140
    assert c["break_reason_accuracy"] == pytest.approx(2 / 80)

    # The cascade itself (arms A and B) still has zero false matches — the
    # headline safety claim survives. Arm C (llm-only, the deliberately
    # worse baseline this ablation exists to measure) does not: one retry,
    # forced by the new semantic-validation check above, produced a
    # confident-but-wrong classification where the rejected first response
    # would not have (see FAILURES.md's 2026-09-01 entry). This is the
    # measured cost of retrying against an LLM whose second sample is
    # independent of its first, not a defect in the retry-then-degrade
    # design — and it is exactly the kind of risk arm A/B's cascade design
    # (sending only a small, rules-narrowed residual to the model at all)
    # structurally avoids.
    assert a["false_match_rate"] == b["false_match_rate"] == 0.0
    assert c["false_match_rate"] > 0.0
