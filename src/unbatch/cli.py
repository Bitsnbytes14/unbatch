"""Typer entrypoints: generate, run, metrics, report, exceptions.

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

import contextlib
import csv
import io
import json
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import typer

from unbatch import adjudicator, audit, fees, money
from unbatch import forecast as forecast_module
from unbatch import generate as generate_module
from unbatch import metrics as metrics_module
from unbatch import report as report_module
from unbatch.models import (
    BankStatementRecord,
    CascadeConfig,
    Decision,
    DecisionOutcome,
    ExpectedBatch,
    OrderLedgerRecord,
    PaymentMethod,
    RunContext,
    SettlementLine,
    SettlementLineType,
    Stage,
    UnresolvedCredit,
)
from unbatch.stages import l0_utr, l1_exact, l2_compose, l3_tolerance, l4_llm

app = typer.Typer(help="unbatch — settlement reconciliation agent.")

# Rupee amounts (report_module.format_rupees) print the "₹" sign to the
# terminal — a plain cmd.exe/PowerShell console on Windows defaults to a
# legacy codepage (cp1252 or similar) that cannot encode it, crashing any
# command that echoes one with UnicodeEncodeError. `unbatch demo`'s headline
# summary is the first place this project prints a rupee amount to the
# terminal rather than only into a UTF-8-encoded file, so it's the first
# place this was ever exercised. Reconfiguring to UTF-8 is a no-op where the
# console is already UTF-8 (every non-Windows terminal), so this fixes it
# for every command rather than special-casing just this one.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Cheapest-and-most-certain first, per ARCHITECTURE.md's cascade table.
# --no-llm runs only this sequence, with run_cascade's terminal exception
# step standing in for L4; the default (with-LLM) arm runs FULL_STAGE_SEQUENCE
# below instead, which appends L4 as the real terminal stage.
STAGE_SEQUENCE: tuple[tuple[Stage, object], ...] = (
    (Stage.L0, l0_utr.run),
    (Stage.L1, l1_exact.run),
    (Stage.L2, l2_compose.run),
    (Stage.L3, l3_tolerance.run),
)
FULL_STAGE_SEQUENCE: tuple[tuple[Stage, object], ...] = (*STAGE_SEQUENCE, (Stage.L4, l4_llm.run))
# --llm-only (METRICS.md's ablation arm C): skips L0-L3 entirely and sends
# every credit straight to the adjudicator, deliberately including the ~93
# credits the rules layer would otherwise resolve for free — that's the
# whole point of the arm, not an oversight.
LLM_ONLY_STAGE_SEQUENCE: tuple[tuple[Stage, object], ...] = ((Stage.L4, l4_llm.run),)


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


# L4 needs candidate_lines too (2026-08-31): it re-runs compose() itself to
# find exact-sum candidates before falling back to a whole-batch guess (see
# l4_llm.py's module docstring), so it needs the same up-to-date pool L2
# gets, excluding whatever earlier stages already consumed. Matters most for
# --llm-only, where L4 is the *only* stage and would otherwise never get a
# populated pool at all.
# L3 needs it too (2026-08-31): it checks whether a within-tolerance delta
# exactly equals a real settlement line's net before accepting (see
# l3_tolerance.py's docstring) — rebuilding here excludes whatever L2 itself
# just matched, so that check never sees a line another decision already
# claimed.
_CANDIDATE_LINE_STAGES = frozenset({Stage.L2, Stage.L3, Stage.L4})


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

    Before L2 and before L4, `candidate_lines` on every remaining credit is
    rebuilt from `settlements` minus whatever payment_ids any earlier
    decision this run already matched — L0/L1 match whole batches and never
    touch it, and L3 checks already-computed batch totals directly rather
    than composing from lines, so only L2's and L4's composition searches
    need this (L4 re-runs the same search L2 does, to find exact-sum
    candidates before falling back to a whole-batch guess — see
    l4_llm.py's module docstring). Pass `settlements=None` (the default) to
    skip this entirely, which is what the runner-behaviour tests do with
    fake stages that don't look at candidate_lines anyway.

    Under `ctx.no_llm`, anything still unresolved after the last stage gets
    a terminal exception Decision (stage=L4, reason="no_llm_unresolved") —
    `stage_sequence` for that arm is STAGE_SEQUENCE (L0-L3 only, no real
    L4), and --no-llm's whole point is to stop there rather than call the
    model, so every credit still ends with exactly one Decision by the time
    this returns. The default arm passes FULL_STAGE_SEQUENCE instead, whose
    real L4 (`l4_llm.run`) is itself the terminal stage and resolves
    everything still remaining, so this fallback never fires for it.

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
        audit.record_many(conn, decisions)
        for decision in decisions:
            consumed_payment_ids.update(decision.matched_payment_ids)
        counts[stage.value] = len(decisions)
        remaining = [u for u in remaining if u.credit.txn_id not in resolved_ids]

    if ctx.no_llm:
        now = datetime.now(UTC)
        terminal_decisions = [
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
            )
            for u in remaining
        ]
        audit.record_many(conn, terminal_decisions)
        counts["terminal_exception"] = len(remaining)

    return counts


@app.command()
def generate(seed: int = 42, noise: float = 0.0, adversarial: bool = False) -> None:
    """Write data/ fixtures + ground truth for `seed`.

    --noise (0.0-1.0, default 0.0) degrades bank_statement narrations only
    — amounts, dates, and settlement data are always exact. 0.0 reproduces
    the committed seed-42 fixtures byte for byte.

    --adversarial writes a same-scale, deliberately hostile dataset instead
    (see generate.py's `generate_adversarial`) — engineered to maximize the
    false-match collision shapes E5/E9 found by chance, not the default
    generator. Incompatible with --noise; the default generator path this
    dataset does not touch is unaffected either way.
    """
    if adversarial:
        if noise:
            typer.echo("--adversarial and --noise are mutually exclusive", err=True)
            raise typer.Exit(code=1)
        generate_module.generate_adversarial(seed, out_dir=generate_module.DEFAULT_OUT_DIR)
        return
    if not 0.0 <= noise <= 1.0:
        typer.echo(f"--noise must be between 0.0 and 1.0, got {noise}", err=True)
        raise typer.Exit(code=1)
    generate_module.generate(seed, noise=noise)


def _arm_name(*, no_llm: bool, llm_only: bool) -> str:
    """The ablation arm a run belongs to (METRICS.md § the ablation) — part
    of derive_run_id's hash so the three arms never collide on the same
    seed."""
    if llm_only:
        return "llm_only"
    if no_llm:
        return "no_llm"
    return "with_llm"


def _run_arm(
    seed: int,
    *,
    cached: bool,
    no_llm: bool,
    llm_only: bool,
    data_dir: Path,
    db: Path,
) -> tuple[str, int, dict[str, int]]:
    """Shared body of `run` and `demo`: load input data, run the cascade for
    one arm, return (run_id, credit_count, per-stage counts). Callers that
    need the --llm-only uncached spend guard (the `run` command) check it
    themselves before calling this — `demo` never needs it, since it always
    passes cached=True for llm_only."""
    _orders, settlements, bank_records = load_input_data(data_dir)
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    arm = _arm_name(no_llm=no_llm, llm_only=llm_only)
    run_id = audit.derive_run_id(seed, data_dir, arm=arm)
    ctx = RunContext(run_id=run_id, seed=seed, cached=cached, no_llm=no_llm, llm_only=llm_only)
    conn = audit.connect(db)
    audit.clear_run(conn, run_id)

    if llm_only:
        stage_sequence = LLM_ONLY_STAGE_SEQUENCE
    elif no_llm:
        stage_sequence = STAGE_SEQUENCE
    else:
        stage_sequence = FULL_STAGE_SEQUENCE
    counts = run_cascade(
        ctx, unresolved, conn, settlements=settlements, stage_sequence=stage_sequence
    )
    return run_id, len(unresolved), counts


@app.command()
def run(
    seed: int = 42,
    cached: bool = False,
    no_llm: bool = False,
    llm_only: bool = False,
    confirm_spend: bool = False,
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    db: Path = audit.DEFAULT_DB_PATH,
) -> None:
    """Run the full cascade, writing Decisions to out/audit.db.

    --cached replays L4 responses from cache/ with no API key. --no-llm runs
    the deterministic baseline arm (L0-L3 only, per ARCHITECTURE.md's
    STAGE_SEQUENCE). The default runs the full with-LLM arm
    (FULL_STAGE_SEQUENCE, L0-L4). --llm-only runs the ablation arm that skips
    straight to L4 (LLM_ONLY_STAGE_SEQUENCE, METRICS.md § the ablation) —
    every one of the ~105 credits becomes a live call without --cached, so
    it refuses to run uncached unless --confirm-spend is also passed; that
    guard does not apply to the default with-LLM arm, whose call count is
    bounded by design to whatever the rules layer actually leaves unresolved.
    """
    if llm_only and not cached and not confirm_spend:
        typer.echo(
            f"--llm-only without --cached sends every credit to {adjudicator.MODEL} "
            "live — that's ~105 calls, not the handful the cascade normally "
            "leaves for L4. Pass --confirm-spend to proceed, or add --cached "
            "to replay from the committed cache/ instead.",
            err=True,
        )
        raise typer.Exit(code=1)

    run_id, credit_count, counts = _run_arm(
        seed, cached=cached, no_llm=no_llm, llm_only=llm_only, data_dir=data_dir, db=db
    )

    typer.echo(f"run_id: {run_id}")
    typer.echo(f"credits: {credit_count}")
    for stage_name, count in counts.items():
        typer.echo(f"{stage_name}\t{count}")


@app.command()
def metrics(
    seed: int = 42,
    arm: str | None = None,
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    db: Path = audit.DEFAULT_DB_PATH,
    out: Path | None = None,
) -> None:
    """Score a run's audit log against ground truth and print the result as
    JSON (METRICS.md). `--arm` ("no_llm", "with_llm", or "llm_only") must
    match whichever arm actually produced the run (derive_run_id keys on
    it); without it, this reports whichever run most recently wrote a
    Decision, so a reviewer who just ran `unbatch run --cached` gets that
    run, not a fixed default arm that may never have been run. `--out
    PATH` also writes the same JSON to a file, which is how
    baseline_rules_only.json was produced and can be reproduced.
    """
    conn = audit.connect(db)
    if arm is not None:
        run_id = audit.derive_run_id(seed, data_dir, arm=arm)
    else:
        run_id = audit.latest_run_id(conn)
        if run_id is None:
            typer.echo(
                "no runs found in out/audit.db — run `unbatch run` (or `unbatch demo`) first",
                err=True,
            )
            raise typer.Exit(code=1)

    if not audit.fetch_decisions(conn, run_id):
        present = audit.run_id_summary(conn)
        typer.echo(f"no decisions found for run_id '{run_id}'", err=True)
        if present:
            typer.echo("runs present in the audit log:", err=True)
            for present_run_id, count in present.items():
                typer.echo(f"  {present_run_id} ({count} decisions)", err=True)
            typer.echo("pass --arm to match one of these, or --seed if it differs", err=True)
        else:
            typer.echo("the audit log is empty — run `unbatch run` first", err=True)
        raise typer.Exit(code=1)

    metrics_report = metrics_module.score(conn, run_id, data_dir=data_dir)
    payload = metrics_report.model_dump_json(indent=2)
    if out is not None:
        out.write_text(payload + "\n", encoding="utf-8", newline="")
    typer.echo(payload)


@app.command()
def report(
    seed: int = 42,
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    db: Path = audit.DEFAULT_DB_PATH,
    out: Path = report_module.DEFAULT_OUT_PATH,
) -> None:
    """Regenerate out/report.html from out/audit.db.

    Renders every arm ("no_llm", "with_llm", "llm_only") that has decisions
    for this seed in the audit log; an arm that hasn't been run yet shows as
    pending rather than being silently omitted or faked as all-exceptions.
    """
    out_path = report_module.render(seed, data_dir=data_dir, db=db, out_path=out)
    typer.echo(f"wrote {out_path}")


def _demo_cache_miss_message(arm_label: str, retry_hint: str, exc: Exception) -> str:
    return (
        f"{arm_label} needs a live call that cache/ does not cover: {exc}\n"
        "unbatch demo never makes a live call. Use seed 42 (the default) to "
        f"replay from the committed cache, or set OPENAI_API_KEY and run "
        f"{retry_hint} directly instead of `unbatch demo`."
    )


@app.command()
def demo(seed: int = 42) -> None:
    """Run the whole pipeline end to end with one command: generate the
    fixtures, all three ablation arms, metrics, and the report — then print
    the headline numbers and where the report landed.

    This wraps `generate`, `run --no-llm`, `run --cached`,
    `run --llm-only --cached`, `metrics`, and `report` exactly as they
    already exist; it does not replace them, and running them separately
    (see the README) produces the identical audit log and report. No API
    key is needed for seed 42: the with-LLM and LLM-only arms replay from
    the committed cache/, and this refuses with a clear message on a cache
    miss rather than silently attempting a live call.
    """
    data_dir = generate_module.DEFAULT_OUT_DIR
    db = audit.DEFAULT_DB_PATH

    typer.echo(f"[1/6] generate --seed {seed}")
    generate_module.generate(seed)

    typer.echo("[2/6] run --no-llm (arm A: rules only)")
    _run_arm(seed, cached=False, no_llm=True, llm_only=False, data_dir=data_dir, db=db)

    typer.echo("[3/6] run --cached (arm B: rules + LLM)")
    try:
        _run_arm(seed, cached=True, no_llm=False, llm_only=False, data_dir=data_dir, db=db)
    except adjudicator.CacheMissError as exc:
        typer.echo(
            _demo_cache_miss_message("arm B", f"unbatch run --seed {seed}", exc), err=True
        )
        raise typer.Exit(code=1) from exc

    typer.echo("[4/6] run --llm-only --cached (arm C: LLM only)")
    try:
        _run_arm(seed, cached=True, no_llm=False, llm_only=True, data_dir=data_dir, db=db)
    except adjudicator.CacheMissError as exc:
        typer.echo(
            _demo_cache_miss_message(
                "arm C", f"unbatch run --seed {seed} --llm-only --confirm-spend", exc
            ),
            err=True,
        )
        raise typer.Exit(code=1) from exc

    typer.echo("[5/6] metrics (arm B)")
    conn = audit.connect(db)
    with_llm_run_id = audit.derive_run_id(seed, data_dir, arm="with_llm")
    headline = metrics_module.score(conn, with_llm_run_id, data_dir=data_dir)

    typer.echo("[6/6] report")
    out_path = report_module.render(seed, data_dir=data_dir, db=db)

    typer.echo("")
    typer.echo(f"total credits:    {headline.total_credits}")
    typer.echo(f"match rate:       {report_module.format_percent(headline.count_match_rate)}")
    typer.echo(
        f"false-match rate: {report_module.format_percent(headline.false_match_rate)}"
    )
    typer.echo(f"exceptions:       {report_module.format_percent(headline.exception_rate)}")
    typer.echo(f"LLM cost:         {report_module.format_rupees(headline.llm_cost_paise)}")
    typer.echo(f"report:           {out_path}")


@app.command()
def forecast(
    horizon: int = 14,
    as_of: str | None = None,
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    out: Path | None = None,
) -> None:
    """Project expected settlement inflows for the next `horizon` days from
    the existing settlement report. Pure arithmetic, no LLM — see
    forecast.py's module docstring for why this never touches the
    adjudicator.

    `--as-of` (ISO date, e.g. 2024-01-16) fits the lag and fee-deviation
    distributions on settlements dated on or before it and projects only
    orders unsettled as of that date — the default is the data's last
    settled date minus `horizon` days, so a full horizon of genuinely
    unsettled orders exists to project even though every input file here
    is already fully settled. Pass an earlier `--as-of` to backtest against
    already-known outcomes, same as `bench --forecast` does internally.
    """
    orders, settlements, _bank_records = load_input_data(data_dir)
    if as_of is not None:
        as_of_date = date.fromisoformat(as_of)
    else:
        last_settled = max(
            (
                line.settled_at.date()
                for line in settlements
                if line.type == SettlementLineType.PAYMENT
            ),
            default=None,
        )
        if last_settled is None:
            typer.echo("No settled payments in the data to forecast from.", err=True)
            raise typer.Exit(code=1)
        as_of_date = last_settled - timedelta(days=horizon)

    report_data = forecast_module.forecast(
        orders, settlements, as_of=as_of_date, horizon_days=horizon
    )
    payload = report_data.model_dump_json(indent=2)
    if out is not None:
        out.write_text(payload + "\n", encoding="utf-8", newline="")
    typer.echo(payload)


_EXCEPTION_EXPORT_HEADER = (
    "txn_id",
    "value_date",
    "credit_amount_rupees",
    "narration",
    "delta_rupees",
    "stage",
    "reason",
    "break_reason",
    "evidence_refs",
    "proposed_resolution",
    "human_review_required",
    "analyst_resolution",
)


def _exception_export_row(
    decision: Decision, bank_record: BankStatementRecord | None
) -> list[str]:
    """One CSV row an analyst can act on directly: credit details looked up
    from the bank statement (blank if this run's data_dir doesn't have that
    credit_id — a stale cross-seed audit log degrades gracefully rather than
    crashing), plus the model's own classification when this exception
    actually reached the adjudicator (`decision.llm_model` is None for
    rules-stage exceptions like `pool_too_large` or `no_llm_unresolved`,
    which were never classified at all). `analyst_resolution` is always
    blank — it's the column this export exists to hand someone."""
    reached_llm = decision.llm_model is not None
    return [
        decision.credit_id,
        bank_record.value_date.isoformat() if bank_record else "",
        (
            money.format_paise_to_rupees(bank_record.credit_paise)
            if bank_record and bank_record.credit_paise is not None
            else ""
        ),
        bank_record.narration if bank_record else "",
        money.format_paise_to_rupees(decision.delta_paise),
        decision.stage.value,
        decision.reason,
        decision.reason if reached_llm else "",
        ";".join(decision.evidence_refs) if decision.evidence_refs else "",
        decision.rationale or "",
        "" if decision.human_review_required is None else str(decision.human_review_required),
        "",
    ]


@app.command()
def exceptions(
    run_id: str | None = None,
    db: Path = audit.DEFAULT_DB_PATH,
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    export: Path | None = None,
) -> None:
    """Print unresolved items and their reasons, or export them as a CSV
    work item.

    A query over out/audit.db (audit.fetch_exceptions), never a separately
    maintained list — ARCHITECTURE.md § Audit trail. Empty output before any
    stage has ever run is correct: there is nothing in the table yet, not a
    bug to work around. Omit --run-id to see exceptions across every run
    recorded so far.

    --export PATH writes the same rows as a CSV instead of printing them:
    credit details cross-referenced from `data_dir`'s bank_statement.csv,
    the model's break_reason/evidence_refs/proposed_resolution for whichever
    exceptions actually reached the adjudicator, and a blank
    analyst_resolution column for someone to fill in.
    """
    conn = audit.connect(db)
    rows = audit.fetch_exceptions(conn, run_id)

    if export is not None:
        bank_statement_path = data_dir / "bank_statement.csv"
        bank_by_txn = (
            {r.txn_id: r for r in generate_module.read_bank_statement_csv(bank_statement_path)}
            if bank_statement_path.exists()
            else {}
        )
        export.parent.mkdir(parents=True, exist_ok=True)
        with open(export, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(_EXCEPTION_EXPORT_HEADER)
            for decision in rows:
                writer.writerow(
                    _exception_export_row(decision, bank_by_txn.get(decision.credit_id))
                )
        typer.echo(f"wrote {len(rows)} rows to {export}")
        return

    if not rows:
        typer.echo("No exceptions.")
        return
    for decision in rows:
        typer.echo(f"{decision.credit_id}\t{decision.stage.value}\t{decision.reason}")


_BENCH_MULTISEED_METRIC_FIELDS = (
    "count_match_rate",
    "value_weighted_match_rate",
    "false_match_rate",
    "exception_rate",
    "precision",
    "recall",
)


def _score_rules_only_for_seed(
    seed: int, data_dir: Path, db_path: Path
) -> metrics_module.MetricsReport:
    """Generate `seed`'s own fixtures into `data_dir`, run the rules-only
    (--no-llm) arm against them, and score the result — the one-seed unit of
    work `bench --seeds` repeats per seed. No LLM call is possible on this
    path (STAGE_SEQUENCE stops at L3; run_cascade's ctx.no_llm branch closes
    out whatever's left as a terminal exception), so this needs no API key
    and no cache entry for any seed but 42.
    """
    generate_module.generate(seed, out_dir=data_dir)
    _orders, settlements, bank_records = load_input_data(data_dir)
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    run_id = audit.derive_run_id(seed, data_dir, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=seed, no_llm=True)
    conn = audit.connect(db_path)
    try:
        audit.clear_run(conn, run_id)
        run_cascade(ctx, unresolved, conn, settlements=settlements, stage_sequence=STAGE_SEQUENCE)
        return metrics_module.score(conn, run_id, data_dir=data_dir)
    finally:
        conn.close()


_SCALE_WINDOW_DAYS = 30
_SCALE_SPLIT_FRACTION = 0.05
_SCALE_CLEAN_LINE_RANGE = (2, 4)
_SCALE_SPLIT_LINE_RANGE = (4, 8)
_SCALE_LINE_GROSS_RANGE = (5_000, 50_000)


def _build_settlement_line(
    index: int,
    line_index: int,
    utr: str,
    method: PaymentMethod,
    settled_at: datetime,
    rng: random.Random,
) -> SettlementLine:
    gross = rng.randint(*_SCALE_LINE_GROSS_RANGE)
    fee = fees.compute_fee_paise(gross, method)
    tax = fees.compute_tax_paise(fee)
    return SettlementLine(
        settlement_id=f"scale_setl_{index}_{line_index}",
        settlement_utr=utr,
        payment_id=f"scale_pay_{index}_{line_index}",
        type=SettlementLineType.PAYMENT,
        gross_paise=gross,
        fee_paise=fee,
        tax_paise=tax,
        net_paise=fees.compute_net_paise(gross, fee, tax),
        settled_at=settled_at,
    )


def _synthetic_scale_batch(
    rng: random.Random, index: int, epoch: date
) -> tuple[list[SettlementLine], list[BankStatementRecord]]:
    """One throughput-only batch: either a whole-UTR credit that resolves at
    L0 (narration carries the UTR), or — for `_SCALE_SPLIT_FRACTION` of
    batches — one UTR's lines split into two credits, neither of which ties
    to the whole batch net, so both fall through L0/L1 into L2's composition
    search (the same shape as generate.py's own `settlement_split`, built
    independently here since it needs to scale to thousands of batches over
    a *fixed* calendar window — generate.py's fixture generator is tuned for
    a specific 105-batch dataset with specific injected break types and
    isn't meant to be scaled up; see FAILURES.md's batch-rebalancing
    entries). Every subset composes exactly by construction, so at any scale
    the only thing under test is wall-clock, never match correctness — this
    never touches ground truth or scoring."""
    method = rng.choice(list(PaymentMethod))
    settled_at = datetime.combine(
        epoch + timedelta(days=rng.randrange(_SCALE_WINDOW_DAYS)), datetime.min.time()
    )
    utr = f"SCALEUTR{index:06d}"
    value_date = settled_at.date()

    if rng.random() < _SCALE_SPLIT_FRACTION:
        n_lines = rng.randint(*_SCALE_SPLIT_LINE_RANGE)
        lines = [
            _build_settlement_line(index, i, utr, method, settled_at, rng) for i in range(n_lines)
        ]
        split_at = rng.randint(1, n_lines - 1)
        groups = [lines[:split_at], lines[split_at:]]
        bank_records = [
            BankStatementRecord(
                txn_id=f"scale_txn_{index}_{group_index}",
                value_date=value_date,
                narration="NEFT-MISC SETTLEMENT CREDIT",
                credit_paise=sum(line.net_paise for line in group),
                debit_paise=None,
                balance_paise=sum(line.net_paise for line in group),
            )
            for group_index, group in enumerate(groups)
        ]
        return lines, bank_records

    n_lines = rng.randint(*_SCALE_CLEAN_LINE_RANGE)
    lines = [
        _build_settlement_line(index, i, utr, method, settled_at, rng) for i in range(n_lines)
    ]
    net_total = sum(line.net_paise for line in lines)
    bank_record = BankStatementRecord(
        txn_id=f"scale_txn_{index}",
        value_date=value_date,
        narration=f"NEFT-{utr}-RAZORPAY SOFTWARE PVT",
        credit_paise=net_total,
        debit_paise=None,
        balance_paise=net_total,
    )
    return lines, [bank_record]


def _generate_scale_data(
    seed: int, target_credits: int
) -> tuple[list[SettlementLine], list[BankStatementRecord]]:
    """Enough clean-plus-split batches to reach ~target_credits bank
    credits, all dated within a fixed `_SCALE_WINDOW_DAYS`-day window —
    fixed, not scaled with target_credits, so batch density (and therefore
    L2's date-windowed candidate pool size) genuinely grows with scale
    rather than staying artificially constant."""
    rng = random.Random(seed)
    epoch = date(2024, 1, 1)
    settlements: list[SettlementLine] = []
    bank_records: list[BankStatementRecord] = []
    index = 0
    while len(bank_records) < target_credits:
        lines, records = _synthetic_scale_batch(rng, index, epoch)
        settlements.extend(lines)
        bank_records.extend(records)
        index += 1
    return settlements, bank_records


def _run_rules_only_cascade_timed(
    ctx: RunContext,
    unresolved: list[UnresolvedCredit],
    conn: sqlite3.Connection,
    settlements: list[SettlementLine],
) -> tuple[dict[str, int], dict[str, float]]:
    """The same stage sequence, candidate-line rebuild, and audit-recording
    behaviour as `run_cascade`, timed per stage — a separate function rather
    than adding timing to `run_cascade` itself, since that one is shared by
    every other command and this is purely a benchmarking concern."""
    counts: dict[str, int] = {}
    timings: dict[str, float] = {}
    remaining = unresolved
    consumed_payment_ids: set[str] = set()

    for stage, stage_fn in STAGE_SEQUENCE:
        start = time.perf_counter()
        if stage in _CANDIDATE_LINE_STAGES:
            available = [
                line for line in settlements if line.payment_id not in consumed_payment_ids
            ]
            remaining = [u.model_copy(update={"candidate_lines": available}) for u in remaining]

        decisions = stage_fn(remaining, ctx)
        resolved_ids = {decision.credit_id for decision in decisions}
        audit.record_many(conn, decisions)
        for decision in decisions:
            consumed_payment_ids.update(decision.matched_payment_ids)
        counts[stage.value] = len(decisions)
        remaining = [u for u in remaining if u.credit.txn_id not in resolved_ids]
        timings[stage.value] = time.perf_counter() - start

    return counts, timings


def _bench_seeds(seeds: str, out: Path) -> None:
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    per_seed: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="unbatch-bench-") as tmp:
        tmp_root = Path(tmp)
        for seed in seed_list:
            seed_dir = tmp_root / f"seed_{seed}"
            data_dir = seed_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            report = _score_rules_only_for_seed(seed, data_dir, seed_dir / "audit.db")
            per_seed[str(seed)] = json.loads(report.model_dump_json())

    summary: dict[str, dict[str, float]] = {}
    for field in _BENCH_MULTISEED_METRIC_FIELDS:
        values = [per_seed[str(seed)][field] for seed in seed_list]
        summary[field] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    payload = {
        "seeds": seed_list,
        "arm": "no_llm",
        "config": CascadeConfig().model_dump(),
        "per_seed": per_seed,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")

    typer.echo(f"seeds: {seed_list}")
    for field, stats in summary.items():
        typer.echo(
            f"{field}\tmean={stats['mean']:.4f}\tmin={stats['min']:.4f}"
            f"\tmax={stats['max']:.4f}\tstdev={stats['stdev']:.4f}"
        )
    typer.echo(f"wrote {out}")


_BENCH_NOISE_SEED = 42


def _score_rules_only_for_noise(
    noise: float, data_dir: Path, db_path: Path, *, seed: int = _BENCH_NOISE_SEED
) -> metrics_module.MetricsReport:
    """Same one-unit-of-work shape as `_score_rules_only_for_seed`, but
    varies narration noise instead of the underlying dataset — measures how
    the rules-only cascade degrades as bank narrations get messier, not
    across independent random datasets. `seed` defaults to
    `_BENCH_NOISE_SEED` (42, what `bench --noise` itself always uses);
    `unbatch verify` passes other seeds to reproduce bench_noise_seeds.json,
    E12's noise-sweep re-run on seeds 44 and 46."""
    generate_module.generate(seed, out_dir=data_dir, noise=noise)
    _orders, settlements, bank_records = load_input_data(data_dir)
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    run_id = audit.derive_run_id(seed, data_dir, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=seed, no_llm=True)
    conn = audit.connect(db_path)
    try:
        audit.clear_run(conn, run_id)
        run_cascade(ctx, unresolved, conn, settlements=settlements, stage_sequence=STAGE_SEQUENCE)
        return metrics_module.score(conn, run_id, data_dir=data_dir)
    finally:
        conn.close()


def _bench_noise(noise_levels: str, out: Path) -> None:
    levels = [float(s.strip()) for s in noise_levels.split(",") if s.strip()]

    per_level: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="unbatch-bench-noise-") as tmp:
        tmp_root = Path(tmp)
        for level in levels:
            level_dir = tmp_root / f"noise_{level}"
            data_dir = level_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            report = _score_rules_only_for_noise(level, data_dir, level_dir / "audit.db")
            per_level[str(level)] = json.loads(report.model_dump_json())

    payload = {
        "seed": _BENCH_NOISE_SEED,
        "arm": "no_llm",
        "config": CascadeConfig().model_dump(),
        "noise_levels": levels,
        "per_level": per_level,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")

    typer.echo(f"noise levels: {levels}")
    for level in levels:
        m = per_level[str(level)]
        typer.echo(
            f"noise={level}\tcount_match_rate={m['count_match_rate']:.4f}"
            f"\tfalse_match_rate={m['false_match_rate']:.4f}\tfunnel={m['stage_funnel']}"
        )
    typer.echo(f"wrote {out}")


def _bench_scale(scale: int, out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="unbatch-bench-scale-") as tmp:
        tmp_root = Path(tmp)
        settlements, bank_records = _generate_scale_data(seed=42, target_credits=scale)
        generate_module.write_settlement_report_csv(
            settlements, tmp_root / "settlement_report.csv"
        )
        generate_module.write_bank_statement_csv(bank_records, tmp_root / "bank_statement.csv")

        settlements = generate_module.read_settlement_report_csv(
            tmp_root / "settlement_report.csv"
        )
        bank_records = generate_module.read_bank_statement_csv(tmp_root / "bank_statement.csv")
        expected_batches = compute_expected_batches(settlements)
        unresolved = build_unresolved_credits(bank_records, expected_batches)

        ctx = RunContext(run_id=f"bench_scale_{scale}", seed=42, no_llm=True)
        conn = audit.connect(tmp_root / "audit.db")
        try:
            audit.clear_run(conn, ctx.run_id)
            start = time.perf_counter()
            counts, timings = _run_rules_only_cascade_timed(ctx, unresolved, conn, settlements)
            total_seconds = time.perf_counter() - start
            exceptions = audit.fetch_exceptions(conn, ctx.run_id)
        finally:
            conn.close()

    exception_reason_counts: dict[str, int] = {}
    for decision in exceptions:
        exception_reason_counts[decision.reason] = (
            exception_reason_counts.get(decision.reason, 0) + 1
        )

    payload = {
        "target_credits": scale,
        "actual_credits": len(unresolved),
        "total_seconds": total_seconds,
        "stage_seconds": timings,
        "stage_resolved_counts": counts,
        "exception_count": len(exceptions),
        "exception_reason_counts": exception_reason_counts,
        "config": ctx.config.model_dump(),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")

    typer.echo(f"credits: {len(unresolved)} (target {scale})")
    typer.echo(f"total: {total_seconds:.3f}s")
    for stage_name, seconds in timings.items():
        typer.echo(f"{stage_name}\t{seconds:.3f}s\tresolved={counts.get(stage_name, 0)}")
    typer.echo(f"exceptions: {len(exceptions)} {exception_reason_counts}")
    typer.echo(f"wrote {out}")


_BENCH_ADVERSARIAL_SEED = 42
_BASELINE_RULES_ONLY_PATH = Path("baseline_rules_only.json")


def _bench_adversarial(out: Path) -> None:
    """Generate the adversarial dataset once, run every applicable arm
    against it, and report the worst case honestly. The with-LLM arm is
    only ever attempted `--cached` — as of E13b the committed cache/
    covers the adversarial dataset's prompts too, so this now succeeds; a
    cache miss (e.g. if the generator or seed ever changes) is still
    reported as "not measurable without live calls", never silently
    patched over by making one."""
    seed = _BENCH_ADVERSARIAL_SEED
    with tempfile.TemporaryDirectory(prefix="unbatch-bench-adversarial-") as tmp:
        tmp_root = Path(tmp)
        data_dir = tmp_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        generate_module.generate_adversarial(seed, out_dir=data_dir)

        _orders, settlements, bank_records = load_input_data(data_dir)
        expected_batches = compute_expected_batches(settlements)

        no_llm_run_id = audit.derive_run_id(seed, data_dir, arm="no_llm")
        no_llm_ctx = RunContext(run_id=no_llm_run_id, seed=seed, no_llm=True)
        no_llm_conn = audit.connect(tmp_root / "no_llm_audit.db")
        try:
            audit.clear_run(no_llm_conn, no_llm_run_id)
            unresolved = build_unresolved_credits(bank_records, expected_batches)
            run_cascade(
                no_llm_ctx,
                unresolved,
                no_llm_conn,
                settlements=settlements,
                stage_sequence=STAGE_SEQUENCE,
            )
            no_llm_report = metrics_module.score(no_llm_conn, no_llm_run_id, data_dir=data_dir)
            no_llm_exceptions = audit.fetch_exceptions(no_llm_conn, no_llm_run_id)
        finally:
            no_llm_conn.close()

        with_llm_run_id = audit.derive_run_id(seed, data_dir, arm="with_llm")
        with_llm_ctx = RunContext(run_id=with_llm_run_id, seed=seed, cached=True)
        with_llm_conn = audit.connect(tmp_root / "with_llm_audit.db")
        with_llm_report = None
        with_llm_error: str | None = None
        try:
            audit.clear_run(with_llm_conn, with_llm_run_id)
            unresolved = build_unresolved_credits(bank_records, expected_batches)
            run_cascade(
                with_llm_ctx,
                unresolved,
                with_llm_conn,
                settlements=settlements,
                stage_sequence=FULL_STAGE_SEQUENCE,
            )
            with_llm_report = metrics_module.score(
                with_llm_conn, with_llm_run_id, data_dir=data_dir
            )
        except adjudicator.CacheMissError as exc:
            with_llm_error = str(exc)
        finally:
            with_llm_conn.close()

    exception_reason_counts: dict[str, int] = {}
    for decision in no_llm_exceptions:
        exception_reason_counts[decision.reason] = (
            exception_reason_counts.get(decision.reason, 0) + 1
        )

    baseline = None
    if _BASELINE_RULES_ONLY_PATH.exists():
        baseline = json.loads(_BASELINE_RULES_ONLY_PATH.read_text(encoding="utf-8"))

    stage_funnel_delta = None
    if baseline is not None:
        keys = set(baseline["stage_funnel"]) | set(no_llm_report.stage_funnel)
        stage_funnel_delta = {
            k: no_llm_report.stage_funnel.get(k, 0) - baseline["stage_funnel"].get(k, 0)
            for k in sorted(keys)
        }

    payload = {
        "seed": seed,
        "config": CascadeConfig().model_dump(),
        "rules_only": json.loads(no_llm_report.model_dump_json()),
        "rules_only_exception_reason_counts": exception_reason_counts,
        "with_llm_cached": (
            json.loads(with_llm_report.model_dump_json()) if with_llm_report is not None else None
        ),
        "with_llm_cached_error": with_llm_error,
        "comparison_vs_normal_dataset": {
            "baseline_source": str(_BASELINE_RULES_ONLY_PATH) if baseline is not None else None,
            "false_match_rate_delta": (
                no_llm_report.false_match_rate - baseline["false_match_rate"]
                if baseline is not None
                else None
            ),
            "count_match_rate_delta": (
                no_llm_report.count_match_rate - baseline["count_match_rate"]
                if baseline is not None
                else None
            ),
            "stage_funnel_delta": stage_funnel_delta,
        },
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")

    typer.echo("=== rules-only arm ===")
    typer.echo(f"count_match_rate: {no_llm_report.count_match_rate:.4f}")
    typer.echo(f"value_weighted_match_rate: {no_llm_report.value_weighted_match_rate:.4f}")
    typer.echo(f"false_match_rate: {no_llm_report.false_match_rate:.4f}")
    typer.echo(f"exception_rate: {no_llm_report.exception_rate:.4f}")
    typer.echo(f"stage_funnel: {no_llm_report.stage_funnel}")
    typer.echo(f"exception_reason_counts: {exception_reason_counts}")
    if with_llm_report is not None:
        typer.echo("=== with-LLM (cached) arm ===")
        typer.echo(f"count_match_rate: {with_llm_report.count_match_rate:.4f}")
        typer.echo(f"false_match_rate: {with_llm_report.false_match_rate:.4f}")
        typer.echo(f"break_reason_accuracy: {with_llm_report.break_reason_accuracy:.4f}")
    else:
        typer.echo("=== with-LLM (cached) arm: NOT MEASURABLE ===")
        typer.echo(f"{with_llm_error}")
        typer.echo("no live calls were made")
    typer.echo(f"wrote {out}")


def _last_settled_payment_date(settlements: list[SettlementLine]) -> date | None:
    return max(
        (line.settled_at.date() for line in settlements if line.type == SettlementLineType.PAYMENT),
        default=None,
    )


def _actual_daily_inflow(
    settlements: list[SettlementLine], horizon_dates: list[date]
) -> dict[date, int]:
    """Ground truth for backtest *scoring only* — every PAYMENT-type net
    actually settled on each horizon day, drawn from the full settlement
    report. Never passed into `forecast.forecast()` itself, which only
    ever sees data up to `as_of`; this is the held-out comparison, exactly
    like a scoring module reading ground truth the pipeline never does."""
    daily = dict.fromkeys(horizon_dates, 0)
    for line in settlements:
        if line.type != SettlementLineType.PAYMENT:
            continue
        d = line.settled_at.date()
        if d in daily:
            daily[d] += line.net_paise
    return daily


def _backtest_one_seed(seed: int, horizon: int, data_dir: Path) -> dict:
    generate_module.generate(seed, out_dir=data_dir)
    orders, settlements, _bank_records = load_input_data(data_dir)

    last_settled = _last_settled_payment_date(settlements)
    assert last_settled is not None, f"seed {seed}: no settled payments to backtest against"
    as_of = last_settled - timedelta(days=horizon)

    report = forecast_module.forecast(orders, settlements, as_of=as_of, horizon_days=horizon)
    actual = _actual_daily_inflow(settlements, [day.date for day in report.daily])

    abs_errors_by_offset: dict[str, int] = {}
    in_band_count = 0
    total_abs_error = 0
    total_expected = 0
    for offset, day_forecast in enumerate(report.daily, start=1):
        actual_paise = actual[day_forecast.date]
        error = abs(actual_paise - day_forecast.expected_paise)
        abs_errors_by_offset[str(offset)] = error
        total_abs_error += error
        total_expected += day_forecast.expected_paise
        if day_forecast.low_paise <= actual_paise <= day_forecast.high_paise:
            in_band_count += 1

    mean_absolute_error_paise = total_abs_error / horizon if horizon else 0.0
    mae_pct_of_projected = total_abs_error / total_expected if total_expected else None
    coverage = in_band_count / horizon if horizon else 0.0
    total_actual = sum(actual.values())
    # What fraction of the horizon's real inflow the forecaster's own
    # tracked population (already-captured, still-unsettled orders) could
    # ever explain — low by construction under a 1-2 day settlement lag,
    # since most of a 14-day horizon's money comes from orders not yet
    # captured at all as of `as_of`. This is what actually explains a large
    # MAE/zero coverage: not a bad projection of what it tracks, but a
    # structurally tiny tracked population. See forecast.py's docstring.
    fraction_of_actual_captured = total_expected / total_actual if total_actual else None

    return {
        "seed": seed,
        "as_of": as_of.isoformat(),
        "horizon_days": horizon,
        "unsettled_payment_count": report.unsettled_payment_count,
        "total_expected_paise": report.total_expected_paise,
        "total_actual_paise": total_actual,
        "fraction_of_actual_captured": fraction_of_actual_captured,
        "mean_absolute_error_paise": mean_absolute_error_paise,
        "mae_pct_of_projected": mae_pct_of_projected,
        "coverage": coverage,
        "abs_error_by_horizon_distance_paise": abs_errors_by_offset,
    }


def _bench_forecast(seeds: str, horizon: int, out: Path) -> None:
    """Backtest the forecaster on seeds 42-47 (or whichever are given):
    hold out everything after `as_of = last_settled_date - horizon`, fit
    only on data up to `as_of`, and score the projection against what
    actually settled — the same seeds and the same reasoning as
    `bench --seeds`, applied to the forecaster instead of the cascade."""
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    per_seed: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="unbatch-bench-forecast-") as tmp:
        tmp_root = Path(tmp)
        for seed in seed_list:
            data_dir = tmp_root / f"seed_{seed}" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            per_seed[str(seed)] = _backtest_one_seed(seed, horizon, data_dir)

    mae_values = [per_seed[str(s)]["mean_absolute_error_paise"] for s in seed_list]
    coverage_values = [per_seed[str(s)]["coverage"] for s in seed_list]
    pct_values = [
        per_seed[str(s)]["mae_pct_of_projected"]
        for s in seed_list
        if per_seed[str(s)]["mae_pct_of_projected"] is not None
    ]
    fraction_captured_values = [
        per_seed[str(s)]["fraction_of_actual_captured"]
        for s in seed_list
        if per_seed[str(s)]["fraction_of_actual_captured"] is not None
    ]

    error_by_offset_across_seeds: dict[str, list[int]] = {
        str(offset): [] for offset in range(1, horizon + 1)
    }
    for s in seed_list:
        for offset, error in per_seed[str(s)]["abs_error_by_horizon_distance_paise"].items():
            error_by_offset_across_seeds[offset].append(error)
    mean_error_by_offset = {
        offset: statistics.mean(values) for offset, values in error_by_offset_across_seeds.items()
    }

    summary = {
        "mean_absolute_error_paise": {
            "mean": statistics.mean(mae_values),
            "min": min(mae_values),
            "max": max(mae_values),
            "stdev": statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0,
        },
        "coverage": {
            "mean": statistics.mean(coverage_values),
            "min": min(coverage_values),
            "max": max(coverage_values),
            "stdev": statistics.stdev(coverage_values) if len(coverage_values) > 1 else 0.0,
        },
        "mae_pct_of_projected": (
            {
                "mean": statistics.mean(pct_values),
                "min": min(pct_values),
                "max": max(pct_values),
            }
            if pct_values
            else None
        ),
        "fraction_of_actual_captured": (
            {
                "mean": statistics.mean(fraction_captured_values),
                "min": min(fraction_captured_values),
                "max": max(fraction_captured_values),
            }
            if fraction_captured_values
            else None
        ),
        "mean_abs_error_by_horizon_distance_paise": mean_error_by_offset,
    }

    payload = {
        "seeds": seed_list,
        "horizon_days": horizon,
        "per_seed": per_seed,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")

    typer.echo(f"seeds: {seed_list}, horizon: {horizon}")
    typer.echo(
        f"MAE: mean={summary['mean_absolute_error_paise']['mean']:.1f}p "
        f"stdev={summary['mean_absolute_error_paise']['stdev']:.1f}p"
    )
    typer.echo(
        f"coverage: mean={summary['coverage']['mean']:.3f} "
        f"stdev={summary['coverage']['stdev']:.3f}"
    )
    if summary["fraction_of_actual_captured"] is not None:
        typer.echo(
            f"fraction of actual inflow the tracked population could ever explain: "
            f"mean={summary['fraction_of_actual_captured']['mean']:.4f}"
        )
    typer.echo("mean abs error by horizon distance (paise):")
    for offset, err in mean_error_by_offset.items():
        typer.echo(f"  day {offset}: {err:.1f}")
    typer.echo(f"wrote {out}")


@app.command()
def bench(
    seeds: str | None = None,
    scale: int | None = None,
    noise: str | None = None,
    adversarial: bool = False,
    forecast: str | None = None,
    forecast_horizon: int = 14,
    out: Path = Path("bench_multiseed.json"),
    scale_out: Path = Path("bench_scale.json"),
    noise_out: Path = Path("bench_noise.json"),
    adversarial_out: Path = Path("bench_adversarial.json"),
    forecast_out: Path = Path("bench_forecast.json"),
) -> None:
    """Measure metric stability across seeds, cascade throughput at scale,
    degradation under narration noise, the worst case on hostile data, or
    the cash forecaster's backtest accuracy.

    `--seeds 42,43,44,45,46,47` (comma-separated) generates each seed's own
    fixtures into a fresh temp directory, runs the rules-only (--no-llm) arm
    against them, and reports per-metric mean/min/max/stdev across seeds —
    written to `--out` (default bench_multiseed.json) as well as printed.
    `data/` and the committed seed-42 fixtures are never touched; every temp
    directory is cleaned up before this command returns. Rules-only ONLY:
    the with-LLM arm would need a live API call per credit for every seed
    but 42, and the committed cache/ only covers seed 42 — so this
    deliberately cannot and does not touch L4.

    `--scale 5000` generates ~5000 throughput-only bank credits (NOT
    generate.py's seeded fixtures — a separate, purpose-built dataset over a
    fixed calendar window so batch density actually grows with scale) into a
    temp directory, runs the rules-only cascade once, and reports wall-clock
    total and per-stage timing — written to `--scale-out` (default
    bench_scale.json). Caps (MAX_POOL/MAX_SUBSET) are never raised to hit a
    target number; if they start firing at this scale, the output says so.

    `--noise 0.0,0.1,0.25,0.5,0.75,1.0` (comma-separated) holds seed 42
    fixed and generates it at each narration-noise level (see `unbatch
    generate --noise`) into a fresh temp directory, runs the rules-only arm
    against each, and reports count match rate, false-match rate, and the
    stage funnel per level — written to `--noise-out` (default
    bench_noise.json).

    `--adversarial` generates the hostile dataset (see `unbatch generate
    --adversarial`) into a temp directory and runs every applicable arm
    against it: rules-only always, and with-LLM `--cached` — as of E13b the
    committed cache/ covers the adversarial dataset's prompts too, so this
    now succeeds and reports break-reason accuracy alongside match rate; a
    cache miss (e.g. if the adversarial generator or seed ever changes) is
    still handled by reporting "not measurable without live calls" rather
    than silently making one. Written to `--adversarial-out` (default
    bench_adversarial.json), including a stage-funnel comparison against
    the committed baseline_rules_only.json.

    `--forecast 42,43,44,45,46,47` (comma-separated) backtests `unbatch
    forecast` on each seed: holds out everything after `as_of = last
    settled date - --forecast-horizon` (default 14), fits only on data up
    to `as_of`, and scores the projection against what actually settled —
    mean absolute error in paise, as a percentage of projected inflow,
    error by horizon distance, and coverage (how often the actual falls
    inside the low/high band) — mean/min/max/stdev across seeds, written to
    `--forecast-out` (default bench_forecast.json). Never touches the
    cascade or the audit log.

    Exactly one of --seeds/--scale/--noise/--adversarial/--forecast must be
    given.
    """
    modes = [bool(seeds), bool(scale), bool(noise), adversarial, bool(forecast)]
    if sum(modes) != 1:
        typer.echo(
            "Pass exactly one of --seeds (e.g. 42,43,44), --scale (e.g. 5000), "
            "--noise (e.g. 0.0,0.25,0.5,1.0), --adversarial, or --forecast "
            "(e.g. 42,43,44,45,46,47)",
            err=True,
        )
        raise typer.Exit(code=1)
    if seeds:
        _bench_seeds(seeds, out)
    elif scale:
        _bench_scale(scale, scale_out)
    elif noise:
        _bench_noise(noise, noise_out)
    elif adversarial:
        _bench_adversarial(adversarial_out)
    else:
        assert forecast is not None
        _bench_forecast(forecast, forecast_horizon, forecast_out)


# The permanent fix for the bench_adversarial.json / bench_adjudication.json
# contradiction (see FAILURES.md): a committed artifact drifting from what
# the pipeline actually produces now, silently, because nothing regenerated
# and diffed it. Fields here are exempt from comparison only because they
# record something genuinely not reproducible by re-running the pipeline —
# wall-clock timing, or the historical cost of live calls made when a file
# was first produced (a fresh, fully-cached regeneration correctly reports 0
# new spend, which isn't "wrong", just not the same historical fact) — never
# a blanket "ignore numeric noise".
_VERIFY_IGNORE_KEYS: dict[str, frozenset[str]] = {
    "bench_scale.json": frozenset({"total_seconds", "stage_seconds"}),
    "bench_adjudication.json": frozenset(
        {"new_live_call_cost_paise", "total_new_live_call_cost_paise"}
    ),
    "bench_ablation_seeds.json": frozenset({"new_live_call_cost_paise"}),
}

# Not covered, with why — printed every run rather than silently omitted.
_VERIFY_SKIPPED_ARTIFACTS: dict[str, str] = {
    "bench_mutation.json": (
        "cosmic-ray report, not a pipeline artifact — regenerating it takes "
        "a ~10 minute mutation-testing run, not a bench/metrics call"
    ),
}


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Suppresses the progress echoes each bench/score helper prints to
    stdout while `verify` regenerates artifacts — its own output is the
    per-artifact PASS/DRIFT summary, not each regeneration's own console
    output."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _diff_json(
    committed: object, regenerated: object, ignore_keys: frozenset[str], path: str = "$"
) -> list[str]:
    """Recursively diff two JSON-decoded values. `ignore_keys` is checked
    against dict keys at any depth — bench_scale.json's per-stage timing
    dict is itself skipped by name, so nothing inside it needs comparing
    either."""
    if isinstance(committed, dict) and isinstance(regenerated, dict):
        mismatches: list[str] = []
        for key in sorted(set(committed) | set(regenerated)):
            if key in ignore_keys:
                continue
            child_path = f"{path}.{key}"
            if key not in committed:
                mismatches.append(f"{child_path}: missing from committed file")
            elif key not in regenerated:
                mismatches.append(f"{child_path}: missing from regenerated output")
            else:
                mismatches.extend(
                    _diff_json(committed[key], regenerated[key], ignore_keys, child_path)
                )
        return mismatches
    if isinstance(committed, list) and isinstance(regenerated, list):
        if len(committed) != len(regenerated):
            return [f"{path}: length {len(committed)} != {len(regenerated)}"]
        mismatches = []
        for i, (c, r) in enumerate(zip(committed, regenerated, strict=True)):
            mismatches.extend(_diff_json(c, r, ignore_keys, f"{path}[{i}]"))
        return mismatches
    if committed != regenerated:
        return [f"{path}: committed={committed!r} regenerated={regenerated!r}"]
    return []


def _regenerate_baseline_rules_only() -> dict:
    """Reproduces baseline_rules_only.json against the committed data/
    fixtures for seed 42. Never regenerates data/ itself — that is checked-in
    fixture data, not a benchmark artifact, and must not be touched here."""
    data_dir = generate_module.DEFAULT_OUT_DIR
    seed = 42
    _orders, settlements, bank_records = load_input_data(data_dir)
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)
    run_id = audit.derive_run_id(seed, data_dir, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=seed, no_llm=True)
    with tempfile.TemporaryDirectory(prefix="unbatch-verify-baseline-") as tmp:
        conn = audit.connect(Path(tmp) / "audit.db")
        try:
            audit.clear_run(conn, run_id)
            run_cascade(
                ctx, unresolved, conn, settlements=settlements, stage_sequence=STAGE_SEQUENCE
            )
            report = metrics_module.score(conn, run_id, data_dir=data_dir)
        finally:
            conn.close()
    return json.loads(report.model_dump_json())


def _regenerate_bench_noise_seeds(seeds: tuple[int, ...], noise_levels: list[float]) -> dict:
    """Reproduces bench_noise_seeds.json: E12's noise-sweep re-run on seeds
    other than 42. Rules-only, so this needs no cache and no API key."""
    per_seed: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="unbatch-verify-noise-seeds-") as tmp:
        tmp_root = Path(tmp)
        for seed in seeds:
            per_level: dict[str, dict] = {}
            for level in noise_levels:
                level_dir = tmp_root / f"seed_{seed}_noise_{level}"
                data_dir = level_dir / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                report = _score_rules_only_for_noise(
                    level, data_dir, level_dir / "audit.db", seed=seed
                )
                per_level[str(level)] = json.loads(report.model_dump_json())
            per_seed[str(seed)] = {"per_level": per_level}

    return {
        "seeds": list(seeds),
        "noise_levels": noise_levels,
        "arm": "no_llm",
        "config": CascadeConfig().model_dump(),
        "per_seed": per_seed,
    }


_ADJUDICATION_SEEDS = (42, 43, 44, 45, 46, 47)
_ADJUDICATION_METRICS_FIELDS = (
    "llm_call_count",
    "llm_cost_paise",
    "break_reason_accuracy",
    "break_reason_confusion",
    "malformed_json_count",
    "retry_count",
    "adjudication_failed_count",
)


def _score_with_llm_cached_for_seed(
    seed: int, data_dir: Path, db_path: Path, *, llm_only: bool
) -> metrics_module.MetricsReport:
    """One seed's with-LLM (or LLM-only) cached score — the unit of work
    bench_adjudication.json's and bench_ablation_seeds.json's regeneration
    both repeat per seed. Every prompt these seeds generate must already be
    in cache/ (E13/E13b committed them); a miss raises
    adjudicator.CacheMissError exactly as `run --cached` would."""
    generate_module.generate(seed, out_dir=data_dir)
    _orders, settlements, bank_records = load_input_data(data_dir)
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    arm = "llm_only" if llm_only else "with_llm"
    run_id = audit.derive_run_id(seed, data_dir, arm=arm)
    ctx = RunContext(run_id=run_id, seed=seed, cached=True, llm_only=llm_only)
    stage_sequence = LLM_ONLY_STAGE_SEQUENCE if llm_only else FULL_STAGE_SEQUENCE
    conn = audit.connect(db_path)
    try:
        audit.clear_run(conn, run_id)
        run_cascade(ctx, unresolved, conn, settlements=settlements, stage_sequence=stage_sequence)
        return metrics_module.score(conn, run_id, data_dir=data_dir)
    finally:
        conn.close()


def _regenerate_bench_adjudication() -> dict:
    """Reproduces bench_adjudication.json's organic (pooled across seeds
    42-47) and adversarial break-reason accuracy — entirely cache-driven, no
    live calls, since every seed's with-LLM prompts are already committed to
    cache/ (E13/E13b). new_live_call_cost_paise records what the original
    measurement spent, not a reproducible property of the data, so it's
    allowlisted (_VERIFY_IGNORE_KEYS) rather than recomputed."""
    per_seed: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="unbatch-verify-adjudication-") as tmp:
        tmp_root = Path(tmp)
        for seed in _ADJUDICATION_SEEDS:
            seed_dir = tmp_root / f"seed_{seed}"
            data_dir = seed_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            report = _score_with_llm_cached_for_seed(
                seed, data_dir, seed_dir / "audit.db", llm_only=False
            )
            report_dict = json.loads(report.model_dump_json())
            per_seed[str(seed)] = {
                field: report_dict[field] for field in _ADJUDICATION_METRICS_FIELDS
            }

    classified_total = 0
    correct_total = 0
    pooled_confusion: dict[str, dict[str, int]] = {}
    for seed_data in per_seed.values():
        for actual, predicted_counts in seed_data["break_reason_confusion"].items():
            bucket = pooled_confusion.setdefault(actual, {})
            for predicted, count in predicted_counts.items():
                bucket[predicted] = bucket.get(predicted, 0) + count
                classified_total += count
                if predicted == actual:
                    correct_total += count

    organic = {
        "seeds": list(_ADJUDICATION_SEEDS),
        "per_seed": per_seed,
        "pooled": {
            "classified_total": classified_total,
            "correct_total": correct_total,
            "break_reason_accuracy": (
                correct_total / classified_total if classified_total else 0.0
            ),
            "break_reason_confusion": pooled_confusion,
        },
        "new_live_call_cost_paise": 0,
    }

    with tempfile.TemporaryDirectory(prefix="unbatch-verify-adjudication-adv-") as tmp:
        adv_out = Path(tmp) / "bench_adversarial.json"
        _bench_adversarial(adv_out)
        adv_payload = _load_json(adv_out)
    assert isinstance(adv_payload, dict)
    with_llm_cached = adv_payload["with_llm_cached"]
    adversarial = None
    if with_llm_cached is not None:
        adversarial = {
            "seed": adv_payload["seed"],
            **{field: with_llm_cached[field] for field in _ADJUDICATION_METRICS_FIELDS},
            "count_match_rate": with_llm_cached["count_match_rate"],
            "false_match_rate": with_llm_cached["false_match_rate"],
        }

    return {
        "config": CascadeConfig().model_dump(),
        "organic": organic,
        "adversarial": adversarial,
        "total_new_live_call_cost_paise": 0,
    }


_ABLATION_SEEDS = (42, 43, 44, 45, 46, 47)


def _regenerate_bench_ablation_seeds() -> dict:
    """Reproduces bench_ablation_seeds.json's per-seed reports and pooled
    match/false-match rates for arms B (with_llm) and C (llm_only) across
    all six seeds — cache-driven, no live calls. Pooled rates are
    reconstructed from each seed's own rate and total_credits (an exact
    round-trip: these are small integer ratios), not recomputed from raw
    decisions, since MetricsReport itself doesn't expose the raw counts."""
    per_seed: dict[str, dict[str, dict]] = {"with_llm": {}, "llm_only": {}}
    with tempfile.TemporaryDirectory(prefix="unbatch-verify-ablation-") as tmp:
        tmp_root = Path(tmp)
        for seed in _ABLATION_SEEDS:
            for arm_name, llm_only in (("with_llm", False), ("llm_only", True)):
                seed_dir = tmp_root / f"seed_{seed}_{arm_name}"
                data_dir = seed_dir / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                report = _score_with_llm_cached_for_seed(
                    seed, data_dir, seed_dir / "audit.db", llm_only=llm_only
                )
                per_seed[arm_name][str(seed)] = json.loads(report.model_dump_json())

    pooled: dict[str, dict] = {}
    for arm_name in ("with_llm", "llm_only"):
        total_credits = 0
        resolved_total = 0
        false_matched_total = 0
        llm_cost_total = 0
        llm_calls_total = 0
        for seed in _ABLATION_SEEDS:
            r = per_seed[arm_name][str(seed)]
            total_credits += r["total_credits"]
            resolved = round(r["count_match_rate"] * r["total_credits"])
            resolved_total += resolved
            false_matched_total += round(r["false_match_rate"] * resolved)
            llm_cost_total += r["llm_cost_paise"]
            llm_calls_total += r["llm_call_count"]
        pooled[arm_name] = {
            "total_credits": total_credits,
            "pooled_count_match_rate": (
                resolved_total / total_credits if total_credits else 0.0
            ),
            "pooled_false_match_rate": (
                false_matched_total / resolved_total if resolved_total else 0.0
            ),
            "total_llm_cost_paise": llm_cost_total,
            "total_llm_call_count": llm_calls_total,
        }

    return {
        "seeds": list(_ABLATION_SEEDS),
        "config": CascadeConfig().model_dump(),
        "per_seed": per_seed,
        "pooled": pooled,
        "new_live_call_cost_paise": {"with_llm": 0, "llm_only": 0, "total": 0},
    }


@app.command()
def verify() -> None:
    """Regenerate every committed benchmark/report artifact into a temp
    directory and diff against what's committed, exiting non-zero on any
    mismatch — the permanent fix for the bench_adversarial.json /
    bench_adjudication.json contradiction found and hand-fixed once already
    (FAILURES.md): a committed artifact silently drifting from what the
    pipeline actually produces, because nothing regenerated and diffed it.

    Each committed file's own recorded parameters (seeds, noise levels,
    scale) are read back out of it and fed to the same code path that
    produced it, so this can never drift from what `bench` itself accepts.
    Every check here is cache-driven or rules-only — no live API calls are
    ever made. Only a small, explicit set of fields that record something
    genuinely non-reproducible (wall-clock timing; the historical cost of
    live calls made when a file was first produced) are exempt, via
    _VERIFY_IGNORE_KEYS, never a blanket ignore.

    Not covered — see _VERIFY_SKIPPED_ARTIFACTS, printed every run with its
    reason rather than silently omitted.
    """
    checks: list[tuple[str, object, object]] = []

    with tempfile.TemporaryDirectory(prefix="unbatch-verify-") as tmp, _quiet():
        tmp_root = Path(tmp)

        baseline_path = _BASELINE_RULES_ONLY_PATH
        if baseline_path.exists():
            checks.append(
                (
                    str(baseline_path),
                    _load_json(baseline_path),
                    _regenerate_baseline_rules_only(),
                )
            )

        multiseed_path = Path("bench_multiseed.json")
        if multiseed_path.exists():
            committed = _load_json(multiseed_path)
            assert isinstance(committed, dict)
            seeds_str = ",".join(str(s) for s in committed["seeds"])
            out_path = tmp_root / "bench_multiseed.json"
            _bench_seeds(seeds_str, out_path)
            checks.append((str(multiseed_path), committed, _load_json(out_path)))

        noise_path = Path("bench_noise.json")
        if noise_path.exists():
            committed = _load_json(noise_path)
            assert isinstance(committed, dict)
            levels_str = ",".join(str(v) for v in committed["noise_levels"])
            out_path = tmp_root / "bench_noise.json"
            _bench_noise(levels_str, out_path)
            checks.append((str(noise_path), committed, _load_json(out_path)))

        scale_path = Path("bench_scale.json")
        if scale_path.exists():
            committed = _load_json(scale_path)
            assert isinstance(committed, dict)
            out_path = tmp_root / "bench_scale.json"
            _bench_scale(committed["target_credits"], out_path)
            checks.append((str(scale_path), committed, _load_json(out_path)))

        adversarial_path = Path("bench_adversarial.json")
        if adversarial_path.exists():
            committed = _load_json(adversarial_path)
            out_path = tmp_root / "bench_adversarial.json"
            _bench_adversarial(out_path)
            checks.append((str(adversarial_path), committed, _load_json(out_path)))

        forecast_path = Path("bench_forecast.json")
        if forecast_path.exists():
            committed = _load_json(forecast_path)
            assert isinstance(committed, dict)
            seeds_str = ",".join(str(s) for s in committed["seeds"])
            out_path = tmp_root / "bench_forecast.json"
            _bench_forecast(seeds_str, committed["horizon_days"], out_path)
            checks.append((str(forecast_path), committed, _load_json(out_path)))

        noise_seeds_path = Path("bench_noise_seeds.json")
        if noise_seeds_path.exists():
            committed = _load_json(noise_seeds_path)
            assert isinstance(committed, dict)
            seeds = tuple(committed["seeds"])
            levels = committed["noise_levels"]
            checks.append(
                (
                    str(noise_seeds_path),
                    committed,
                    _regenerate_bench_noise_seeds(seeds, levels),
                )
            )

        adjudication_path = Path("bench_adjudication.json")
        if adjudication_path.exists():
            checks.append(
                (
                    str(adjudication_path),
                    _load_json(adjudication_path),
                    _regenerate_bench_adjudication(),
                )
            )

        ablation_path = Path("bench_ablation_seeds.json")
        if ablation_path.exists():
            checks.append(
                (
                    str(ablation_path),
                    _load_json(ablation_path),
                    _regenerate_bench_ablation_seeds(),
                )
            )

    failures: dict[str, list[str]] = {}
    for path_str, committed_value, regenerated_value in checks:
        diff = _diff_json(
            committed_value,
            regenerated_value,
            ignore_keys=_VERIFY_IGNORE_KEYS.get(path_str, frozenset()),
        )
        if diff:
            failures[path_str] = diff
            typer.echo(f"DRIFT  {path_str}")
        else:
            typer.echo(f"PASS   {path_str}")

    if _VERIFY_SKIPPED_ARTIFACTS:
        typer.echo("not covered:")
        for name, reason in _VERIFY_SKIPPED_ARTIFACTS.items():
            typer.echo(f"  {name}: {reason}")

    if failures:
        typer.echo("")
        for path_str, diffs in failures.items():
            typer.echo(f"DRIFT in {path_str}:", err=True)
            for d in diffs:
                typer.echo(f"  {d}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"verify: all {len(checks)} covered artifacts match freshly regenerated output")


if __name__ == "__main__":
    app()
