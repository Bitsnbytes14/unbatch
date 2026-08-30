"""Typer entrypoints: generate, run, report, exceptions.

`run` orchestrates the cascade directly — L0 -> L1 -> L2 -> L3 -> L4, each
stage consuming only what the previous stages left unresolved, each writing
Decisions to out/audit.db (ARCHITECTURE.md § Data flow summary). There is no
separate orchestration module; the sequence lives here and in
ARCHITECTURE.md's stage table.

The stage protocol: every stage is a pure function
`(unresolved: list[UnresolvedCredit], ctx: RunContext) -> list[Decision]`.
A stage returns a Decision only for the credits it resolves — anything
absent from the returned list simply carries forward to the next stage,
unresolved. Stages never call each other and never mutate `unresolved` or
any shared state; `run_cascade` below is the only thing that removes
resolved items between stages and the only thing that writes to the audit
log, so "no stage may resolve an item without an audit row" is enforced in
one place rather than trusted to every stage individually.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from unbatch import audit
from unbatch import generate as generate_module
from unbatch.models import (
    BankStatementRecord,
    Decision,
    DecisionOutcome,
    ExpectedBatch,
    OrderLedgerRecord,
    RunContext,
    SettlementLine,
    SettlementLineType,
    Stage,
    UnresolvedCredit,
)
from unbatch.stages import l0_utr, l1_exact, l2_compose, l3_tolerance

app = typer.Typer(help="unbatch — settlement reconciliation agent.")

# Cheapest-and-most-certain first, per ARCHITECTURE.md's cascade table.
# L4 is intentionally absent this session — it stays a stub, and --no-llm's
# terminal exception step in run_cascade takes its place in the sequence.
STAGE_SEQUENCE: tuple[tuple[Stage, object], ...] = (
    (Stage.L0, l0_utr.run),
    (Stage.L1, l1_exact.run),
    (Stage.L2, l2_compose.run),
    (Stage.L3, l3_tolerance.run),
)


def load_input_data(
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
) -> tuple[list[OrderLedgerRecord], list[SettlementLine], list[BankStatementRecord]]:
    """Load the three seeded CSVs the cascade reconciles."""
    orders = generate_module.read_order_ledger_csv(data_dir / "order_ledger.csv")
    settlements = generate_module.read_settlement_report_csv(data_dir / "settlement_report.csv")
    bank_records = generate_module.read_bank_statement_csv(data_dir / "bank_statement.csv")
    return orders, settlements, bank_records


def compute_expected_batches(settlements: list[SettlementLine]) -> list[ExpectedBatch]:
    """Group settlement lines by settlement_utr into the batches the rules
    layer expects each bank credit to tie back to (ARCHITECTURE.md's data
    flow: settlement_report.csv -> expected settlement batches). A UTR is
    assigned to a whole payout, not per transaction, so grouping by UTR
    recovers each real batch — except where a break deliberately breaks that
    assumption (settlement_split, duplicate_utr), which is exactly the
    ambiguity later stages exist to resolve.

    `net_paise` is the sum of PAYMENT-type lines only — the naive total a
    merchant would expect from captures alone, before any refund or
    chargeback that later lands in the same window. The actual bank credit
    correctly nets those in (ARCHITECTURE.md: "minus refunds issued in the
    window, minus chargebacks and their fees"), so a batch with a refund or
    chargeback line never ties out exactly here — that gap is
    refund_in_window / chargeback_deduction's whole point, and L0/L1 are
    right to decline them; L2 finds the true composition (payment lines
    plus the refund/chargeback line) by searching individual lines directly,
    not through this aggregate. `settlement_ids`/`payment_ids` still list
    every line in the group, payment or not, since those are just bookkeeping
    for whichever stage does resolve the batch as a whole.
    """
    groups: dict[str, list[SettlementLine]] = defaultdict(list)
    for line in settlements:
        groups[line.settlement_utr].append(line)

    batches = []
    for utr, lines in groups.items():
        dates = [line.settled_at.date() for line in lines]
        payment_only_net = sum(
            line.net_paise for line in lines if line.type == SettlementLineType.PAYMENT
        )
        batches.append(
            ExpectedBatch(
                settlement_utr=utr,
                settlement_ids=[line.settlement_id for line in lines],
                payment_ids=[line.payment_id for line in lines],
                net_paise=payment_only_net,
                window_start=min(dates),
                window_end=max(dates),
            )
        )
    return batches


def build_unresolved_credits(
    bank_records: list[BankStatementRecord], expected_batches: list[ExpectedBatch]
) -> list[UnresolvedCredit]:
    """One UnresolvedCredit per bank credit row (debit rows, if any, are not
    the cascade's concern), starting with every expected batch as a
    candidate — narrowing that down is each stage's own job."""
    return [
        UnresolvedCredit(credit=record, expected_batches=expected_batches, candidates=[])
        for record in bank_records
        if record.credit_paise is not None
    ]


_CANDIDATE_LINE_STAGES = frozenset({Stage.L2, Stage.L3})


def run_cascade(
    ctx: RunContext,
    unresolved: list[UnresolvedCredit],
    conn: sqlite3.Connection,
    *,
    settlements: list[SettlementLine] | None = None,
    stage_sequence: tuple[tuple[Stage, object], ...] = STAGE_SEQUENCE,
) -> dict[str, int]:
    """Thread `unresolved` through each stage in `stage_sequence`, in order,
    writing every returned Decision to the audit log immediately and
    removing resolved credits before the next stage runs. A stage never
    sees a credit an earlier stage already claimed.

    Before L2 and L3, `candidate_lines` on every remaining credit is rebuilt
    from `settlements` minus whatever payment_ids any earlier decision this
    run already matched — L0/L1 match whole batches and never touch it, but
    composition needs the individual lines that are actually still up for
    grabs. Pass `settlements=None` (the default) to skip this entirely,
    which is what the runner-behaviour tests do with fake stages that don't
    look at candidate_lines anyway.

    Under `ctx.no_llm`, anything still unresolved after the last stage gets
    a terminal exception Decision (stage=L4, reason="no_llm_unresolved") —
    L4 is a stub this session, and --no-llm's whole point is to stop at L3
    rather than call it, so every credit still ends with exactly one
    Decision by the time this returns.

    Returns per-stage resolution counts keyed by Stage.value, plus
    "terminal_exception" for the --no-llm cleanup step.
    """
    counts: dict[str, int] = {}
    remaining = unresolved
    consumed_payment_ids: set[str] = set()

    for stage, stage_fn in stage_sequence:
        if settlements is not None and stage in _CANDIDATE_LINE_STAGES:
            available = [
                line for line in settlements if line.payment_id not in consumed_payment_ids
            ]
            remaining = [u.model_copy(update={"candidate_lines": available}) for u in remaining]

        decisions = stage_fn(remaining, ctx)
        resolved_ids = {decision.credit_id for decision in decisions}
        for decision in decisions:
            audit.record(conn, decision)
            consumed_payment_ids.update(decision.matched_payment_ids)
        counts[stage.value] = len(decisions)
        remaining = [u for u in remaining if u.credit.txn_id not in resolved_ids]

    if ctx.no_llm:
        now = datetime.now(UTC)
        for u in remaining:
            audit.record(
                conn,
                Decision(
                    run_id=ctx.run_id,
                    seed=ctx.seed,
                    stage=Stage.L4,
                    credit_id=u.credit.txn_id,
                    matched_payment_ids=[],
                    outcome=DecisionOutcome.EXCEPTION,
                    confidence=0.0,
                    delta_paise=0,
                    reason="no_llm_unresolved",
                    rationale=None,
                    llm_model=None,
                    llm_cost_paise=None,
                    created_at=now,
                ),
            )
        counts["terminal_exception"] = len(remaining)

    return counts


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
