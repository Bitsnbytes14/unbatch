"""`unbatch bench --scale` — cascade throughput measurement on a purpose-built
synthetic dataset (NOT generate.py's seeded fixtures, which are tuned for a
fixed 105-batch dataset and specific injected break types; see this
enhancement's FAILURES.md entries). Uses small scales so the suite stays
fast — the actual bench_scale.json artifact was produced at --scale 5000."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import app

runner = CliRunner()


def test_bench_scale_writes_timing_and_counts(tmp_path: Path) -> None:
    out_path = tmp_path / "bench_scale.json"
    result = runner.invoke(app, ["bench", "--scale", "60", "--scale-out", str(out_path)])
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["target_credits"] == 60
    assert payload["actual_credits"] >= 60
    assert payload["total_seconds"] > 0
    assert set(payload["stage_seconds"]) == {"l0", "l1", "l2", "l3"}
    assert set(payload["stage_resolved_counts"]) == {"l0", "l1", "l2", "l3"}
    assert payload["max_pool"] == 48
    assert payload["max_subset"] == 25


def test_bench_scale_stage_seconds_sum_close_to_total(tmp_path: Path) -> None:
    out_path = tmp_path / "bench_scale.json"
    result = runner.invoke(app, ["bench", "--scale", "60", "--scale-out", str(out_path)])
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    stage_sum = sum(payload["stage_seconds"].values())
    assert stage_sum <= payload["total_seconds"] + 0.05


def test_bench_scale_does_not_raise_the_composition_caps(tmp_path: Path) -> None:
    """The whole point of this command is to observe the caps doing their
    job at scale, never to relax them for a nicer-looking number."""
    out_path = tmp_path / "bench_scale.json"
    runner.invoke(app, ["bench", "--scale", "60", "--scale-out", str(out_path)])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["max_pool"] == 48
    assert payload["max_subset"] == 25


def test_bench_scale_does_not_touch_committed_data_dir(tmp_path: Path) -> None:
    data_dir = Path("data")
    before = (data_dir / "bank_statement.csv").read_bytes()

    out_path = tmp_path / "bench_scale.json"
    result = runner.invoke(app, ["bench", "--scale", "60", "--scale-out", str(out_path)])
    assert result.exit_code == 0, result.output

    after = (data_dir / "bank_statement.csv").read_bytes()
    assert before == after


def test_bench_requires_exactly_one_of_seeds_or_scale() -> None:
    assert runner.invoke(app, ["bench"]).exit_code != 0
    assert (
        runner.invoke(app, ["bench", "--seeds", "42", "--scale", "60"]).exit_code != 0
    )
