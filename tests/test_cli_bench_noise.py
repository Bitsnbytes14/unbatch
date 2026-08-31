"""`unbatch bench --noise` — measures how the rules-only cascade degrades
(or doesn't) as bank narrations get noisier, holding seed 42 fixed. Uses a
small level set to keep the suite fast; the actual bench_noise.json
artifact was produced at the full 0.0,0.1,0.25,0.5,0.75,1.0 sweep."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import app

runner = CliRunner()


def test_bench_noise_writes_per_level_results(tmp_path: Path) -> None:
    out_path = tmp_path / "bench_noise.json"
    result = runner.invoke(app, ["bench", "--noise", "0.0,0.5", "--noise-out", str(out_path)])
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["arm"] == "no_llm"
    assert payload["noise_levels"] == [0.0, 0.5]
    assert set(payload["per_level"]) == {"0.0", "0.5"}
    for level_report in payload["per_level"].values():
        assert "count_match_rate" in level_report
        assert "false_match_rate" in level_report
        assert "stage_funnel" in level_report


def test_bench_noise_at_0_matches_the_seed_42_baseline(tmp_path: Path) -> None:
    """noise=0.0 must reproduce exactly what the committed seed-42 fixtures
    score — the same guarantee generate --noise 0.0 makes at the data
    layer, carried through to the metrics this command reports."""
    out_path = tmp_path / "bench_noise.json"
    runner.invoke(app, ["bench", "--noise", "0.0", "--noise-out", str(out_path)])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    report = payload["per_level"]["0.0"]
    assert report["count_match_rate"] == 93 / 105
    assert report["false_match_rate"] == 0.0
    assert report["stage_funnel"] == {"l0": 79, "l1": 3, "l2": 9, "l3": 2, "l4": 12}


def test_bench_noise_shifts_volume_from_l0_to_l1_without_losing_matches(tmp_path: Path) -> None:
    """The result this benchmark exists to check: narration-only noise
    should move resolutions from L0 to L1, not out of the cascade
    entirely, since L1 doesn't look at narration at all."""
    out_path = tmp_path / "bench_noise.json"
    runner.invoke(app, ["bench", "--noise", "0.0,1.0", "--noise-out", str(out_path)])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    clean = payload["per_level"]["0.0"]
    noisy = payload["per_level"]["1.0"]

    assert noisy["count_match_rate"] == clean["count_match_rate"]
    assert noisy["false_match_rate"] == clean["false_match_rate"] == 0.0
    assert noisy["stage_funnel"]["l0"] < clean["stage_funnel"]["l0"]
    assert noisy["stage_funnel"]["l1"] > clean["stage_funnel"]["l1"]
    l0_l1_clean = clean["stage_funnel"]["l0"] + clean["stage_funnel"]["l1"]
    l0_l1_noisy = noisy["stage_funnel"]["l0"] + noisy["stage_funnel"]["l1"]
    assert l0_l1_clean == l0_l1_noisy  # the same credits, just resolved at a different stage


def test_bench_noise_does_not_touch_committed_data_dir(tmp_path: Path) -> None:
    data_dir = Path("data")
    before = (data_dir / "bank_statement.csv").read_bytes()

    out_path = tmp_path / "bench_noise.json"
    result = runner.invoke(app, ["bench", "--noise", "1.0", "--noise-out", str(out_path)])
    assert result.exit_code == 0, result.output

    after = (data_dir / "bank_statement.csv").read_bytes()
    assert before == after


def test_bench_requires_exactly_one_mode() -> None:
    assert runner.invoke(app, ["bench"]).exit_code != 0
    assert runner.invoke(app, ["bench", "--seeds", "42", "--noise", "0.5"]).exit_code != 0
    assert (
        runner.invoke(app, ["bench", "--scale", "60", "--noise", "0.5"]).exit_code != 0
    )
