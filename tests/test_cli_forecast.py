"""`unbatch forecast` — CLI wiring around forecast.py. The projection logic
itself is tested in test_forecast.py; this covers the CLI's defaulting,
--as-of override, and --out writing."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import app

runner = CliRunner()


def test_forecast_default_as_of_reproduces_the_documented_finding(tmp_path: Path) -> None:
    """Against the committed seed-42 fixtures, the default --as-of (last
    settled date minus the horizon) should reproduce the known structural
    result: almost everything the forecaster can see lands on day 1."""
    out_path = tmp_path / "forecast.json"
    result = runner.invoke(app, ["forecast", "--horizon", "14", "--out", str(out_path)])
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2024-01-16"
    assert payload["horizon_days"] == 14
    assert len(payload["daily"]) == 14
    assert payload["unsettled_payment_count"] == 10
    assert payload["daily"][0]["date"] == "2024-01-17"
    assert payload["daily"][0]["payment_count"] == 10
    assert all(d["payment_count"] == 0 for d in payload["daily"][1:])


def test_forecast_as_of_override(tmp_path: Path) -> None:
    out_path = tmp_path / "forecast.json"
    result = runner.invoke(
        app,
        ["forecast", "--horizon", "5", "--as-of", "2024-01-05", "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2024-01-05"
    assert payload["horizon_days"] == 5
    assert len(payload["daily"]) == 5


def test_forecast_money_fields_are_ints(tmp_path: Path) -> None:
    out_path = tmp_path / "forecast.json"
    runner.invoke(app, ["forecast", "--horizon", "14", "--out", str(out_path)])
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    for field in ("total_expected_paise", "total_low_paise", "total_high_paise"):
        assert isinstance(payload[field], int)
    for day in payload["daily"]:
        assert isinstance(day["expected_paise"], int)


def test_forecast_does_not_touch_committed_data_dir(tmp_path: Path) -> None:
    before = (Path("data") / "bank_statement.csv").read_bytes()
    out_path = tmp_path / "forecast.json"
    result = runner.invoke(app, ["forecast", "--horizon", "14", "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    after = (Path("data") / "bank_statement.csv").read_bytes()
    assert before == after
