"""`unbatch bench --adversarial` (E11b): runs every applicable arm against
the hostile dataset and publishes the worst case — including refusing to
make a live LLM call when the cached arm can't cover the new content.

The CLI invocation is shared across tests via a module-scoped fixture: it
regenerates the adversarial dataset and runs the rules-only cascade, which
includes the near-cap-pool scenario's genuine 5-second `compose_timeout`
(see generate.py's `_adv_near_cap_pool`) — running it once instead of once
per test keeps this file from costing several extra timeouts per suite
run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from unbatch.cli import app

runner = CliRunner()

# Captured at collection time, before the shared fixture below ever runs
# the command — the only reliable way to get a true "before" snapshot
# when the command itself is invoked lazily by a module-scoped fixture.
_DATA_DIR_BANK_STATEMENT_BEFORE = (Path("data") / "bank_statement.csv").read_bytes()


@pytest.fixture(scope="module")
def adversarial_bench_result(tmp_path_factory: pytest.TempPathFactory):
    out_path = tmp_path_factory.mktemp("bench_adversarial") / "bench_adversarial.json"
    result = runner.invoke(app, ["bench", "--adversarial", "--adversarial-out", str(out_path)])
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    return result, payload


def test_bench_adversarial_writes_the_rules_only_report(adversarial_bench_result) -> None:
    result, payload = adversarial_bench_result
    assert result.exit_code == 0, result.output
    assert payload["seed"] == 42
    rules_only = payload["rules_only"]
    assert rules_only["total_credits"] >= 95
    assert "false_match_rate" in rules_only
    assert "stage_funnel" in rules_only


def test_bench_adversarial_false_match_rate_is_worse_than_normal(adversarial_bench_result) -> None:
    _result, payload = adversarial_bench_result
    assert payload["rules_only"]["false_match_rate"] > 0.0
    comparison = payload["comparison_vs_normal_dataset"]
    assert comparison["false_match_rate_delta"] > 0.0


def test_bench_adversarial_does_not_attempt_a_live_llm_call(adversarial_bench_result) -> None:
    """The cached with-LLM arm has to fail with a cache-miss error — the
    adversarial credits' prompts are new content, so the only way this
    would come back clean is if a live call quietly happened instead."""
    result, payload = adversarial_bench_result
    assert payload["with_llm_cached"] is None
    assert "no cached response" in payload["with_llm_cached_error"]
    assert "no live calls were made" in result.output


def test_bench_adversarial_does_not_touch_committed_data_dir(adversarial_bench_result) -> None:
    result, _payload = adversarial_bench_result
    assert result.exit_code == 0, result.output
    after = (Path("data") / "bank_statement.csv").read_bytes()
    assert after == _DATA_DIR_BANK_STATEMENT_BEFORE


def test_bench_requires_exactly_one_mode_including_adversarial() -> None:
    assert runner.invoke(app, ["bench"]).exit_code != 0
    assert runner.invoke(app, ["bench", "--seeds", "42", "--adversarial"]).exit_code != 0
