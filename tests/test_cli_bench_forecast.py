"""`unbatch bench --forecast` — backtests forecast.py across seeds, holding
out everything after as_of and scoring against what actually settled."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import app

runner = CliRunner()


def test_bench_forecast_writes_per_seed_and_summary(tmp_path: Path) -> None:
    out_path = tmp_path / "bench_forecast.json"
    result = runner.invoke(
        app, ["bench", "--forecast", "42,43", "--forecast-out", str(out_path)]
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["seeds"] == [42, 43]
    assert payload["horizon_days"] == 14
    assert set(payload["per_seed"]) == {"42", "43"}

    for seed_key in ("42", "43"):
        row = payload["per_seed"][seed_key]
        assert row["mean_absolute_error_paise"] >= 0
        assert 0.0 <= row["coverage"] <= 1.0
        assert len(row["abs_error_by_horizon_distance_paise"]) == 14

    summary = payload["summary"]
    assert "mean_absolute_error_paise" in summary
    assert "coverage" in summary
    assert "fraction_of_actual_captured" in summary
    assert len(summary["mean_abs_error_by_horizon_distance_paise"]) == 14


def test_bench_forecast_custom_horizon(tmp_path: Path) -> None:
    out_path = tmp_path / "bench_forecast.json"
    result = runner.invoke(
        app,
        [
            "bench", "--forecast", "42", "--forecast-horizon", "5",
            "--forecast-out", str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["horizon_days"] == 5
    assert len(payload["per_seed"]["42"]["abs_error_by_horizon_distance_paise"]) == 5


def test_bench_forecast_does_not_touch_committed_data_dir(tmp_path: Path) -> None:
    before = (Path("data") / "bank_statement.csv").read_bytes()
    out_path = tmp_path / "bench_forecast.json"
    result = runner.invoke(
        app, ["bench", "--forecast", "42", "--forecast-out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    after = (Path("data") / "bank_statement.csv").read_bytes()
    assert before == after


def test_bench_forecast_is_mutually_exclusive_with_other_modes() -> None:
    assert (
        runner.invoke(app, ["bench", "--seeds", "42", "--forecast", "42"]).exit_code != 0
    )
    assert runner.invoke(app, ["bench"]).exit_code != 0
