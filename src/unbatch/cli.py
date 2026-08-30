"""Typer entrypoints: generate, run, report, exceptions.

`run` orchestrates the cascade directly — L0 -> L1 -> L2 -> L3 -> L4, each
stage consuming only what the previous stages left unresolved, each writing
Decisions to out/audit.db (ARCHITECTURE.md § Data flow summary). There is no
separate orchestration module; the sequence lives here and in
ARCHITECTURE.md's stage table.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="unbatch — settlement reconciliation agent.")


@app.command()
def generate(seed: int = 42) -> None:
    """Write data/ fixtures + ground truth for `seed`."""
    raise NotImplementedError


@app.command()
def run(
    seed: int = 42,
    cached: bool = False,
    no_llm: bool = False,
    llm_only: bool = False,
) -> None:
    """Run the full cascade, writing Decisions to out/audit.db.

    --cached replays L4 responses from cache/ with no API key. --no-llm runs
    the deterministic baseline arm (L0-L3 only). --llm-only runs the ablation
    arm that skips straight to L4 (METRICS.md § the ablation).
    """
    raise NotImplementedError


@app.command()
def report() -> None:
    """Regenerate out/report.html from out/audit.db."""
    raise NotImplementedError


@app.command()
def exceptions() -> None:
    """Print unresolved items and their reasons."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
