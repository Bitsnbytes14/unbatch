"""Typer entrypoints: generate, run, report, exceptions.

`run` orchestrates the cascade directly — L0 -> L1 -> L2 -> L3 -> L4, each
stage consuming only what the previous stages left unresolved, each writing
Decisions to out/audit.db (ARCHITECTURE.md § Data flow summary). There is no
separate orchestration module; the sequence lives here and in
ARCHITECTURE.md's stage table.
"""

from __future__ import annotations

from pathlib import Path

import typer

from unbatch import audit
from unbatch import generate as generate_module

app = typer.Typer(help="unbatch — settlement reconciliation agent.")


@app.command()
def generate(seed: int = 42) -> None:
    """Write data/ fixtures + ground truth for `seed`."""
    generate_module.generate(seed)


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
def exceptions(
    run_id: str | None = None,
    db: Path = audit.DEFAULT_DB_PATH,
) -> None:
    """Print unresolved items and their reasons.

    A query over out/audit.db (audit.fetch_exceptions), never a separately
    maintained list — ARCHITECTURE.md § Audit trail. Empty output before any
    stage has ever run is correct: there is nothing in the table yet, not a
    bug to work around. Omit --run-id to see exceptions across every run
    recorded so far.
    """
    conn = audit.connect(db)
    rows = audit.fetch_exceptions(conn, run_id)
    if not rows:
        typer.echo("No exceptions.")
        return
    for decision in rows:
        typer.echo(f"{decision.credit_id}\t{decision.stage.value}\t{decision.reason}")


if __name__ == "__main__":
    app()
