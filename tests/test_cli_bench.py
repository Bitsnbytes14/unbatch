"""`unbatch bench --seeds` — multi-seed variance measurement (rules-only arm
only; the with-LLM arm needs a live call per seed and the committed cache/
only covers seed 42). Checks: writes the expected JSON shape, never touches
the committed data/ fixtures, and refuses to run with no seeds given."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import _BENCH_MULTISEED_METRIC_FIELDS, app

runner = CliRunner()


def test_bench_seeds_writes_per_seed_and_summary_stats(tmp_path: Path) -> None:
    out_path = tmp_path / "bench_multiseed.json"
    result = runner.invoke(app, ["bench", "--seeds", "42,43", "--out", str(out_path)])
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["seeds"] == [42, 43]
    assert payload["arm"] == "no_llm"
    assert set(payload["per_seed"]) == {"42", "43"}

    for field in _BENCH_MULTISEED_METRIC_FIELDS:
        stats = payload["summary"][field]
        values = [payload["per_seed"][seed][field] for seed in ("42", "43")]
        assert stats["min"] == min(values)
        assert stats["max"] == max(values)
        assert stats["mean"] == sum(values) / 2


def test_bench_seeds_does_not_touch_committed_data_dir(tmp_path: Path) -> None:
    data_dir = Path("data")
    before = {name: (data_dir / name).read_bytes() for name in ("bank_statement.csv",)}

    out_path = tmp_path / "bench_multiseed.json"
    result = runner.invoke(app, ["bench", "--seeds", "42", "--out", str(out_path)])
    assert result.exit_code == 0, result.output

    after = {name: (data_dir / name).read_bytes() for name in ("bank_statement.csv",)}
    assert before == after


def test_bench_without_seeds_exits_nonzero() -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code != 0
