"""`unbatch generate --noise` — CLI-level validation of the noise dial;
the actual degradation behaviour is tested in test_generate_noise.py."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from unbatch.cli import app

runner = CliRunner()


def test_generate_rejects_noise_above_1(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["generate", "--noise", "1.5"])
    assert result.exit_code != 0


def test_generate_rejects_negative_noise(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["generate", "--noise", "-0.1"])
    assert result.exit_code != 0


def test_generate_accepts_noise_at_the_boundaries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["generate", "--noise", "0.0"]).exit_code == 0
    assert runner.invoke(app, ["generate", "--noise", "1.0"]).exit_code == 0
