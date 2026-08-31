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

import csv
import json
import sqlite3
import statistics
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from unbatch import adjudicator, audit, money
from unbatch import generate as generate_module
from unbatch import metrics as metrics_module
from unbatch import report as report_module
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
from unbatch.stages import l0_utr, l1_exact, l2_compose, l3_tolerance, l4_llm

app = typer.Typer(help="unbatch — settlement reconciliation agent.")

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
_CANDIDATE_LINE_STAGES = frozenset({Stage.L2, Stage.L4})


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


def _arm_name(*, no_llm: bool, llm_only: bool) -> str:
    """The ablation arm a run belongs to (METRICS.md § the ablation) — part
    of derive_run_id's hash so the three arms never collide on the same
    seed."""
    if llm_only:
        return "llm_only"
    if no_llm:
        return "no_llm"
    return "with_llm"


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

    typer.echo(f"run_id: {run_id}")
    typer.echo(f"credits: {len(unresolved)}")
    for stage_name, count in counts.items():
        typer.echo(f"{stage_name}\t{count}")


@app.command()
def metrics(
    seed: int = 42,
    arm: str = "no_llm",
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    db: Path = audit.DEFAULT_DB_PATH,
    out: Path | None = None,
) -> None:
    """Score a run's audit log against ground truth and print the result as
    JSON (METRICS.md). `--arm` must match whichever arm actually produced
    the run (derive_run_id keys on it) — "no_llm", "with_llm", or
    "llm_only". `--out PATH` also writes the same JSON to a file, which is
    how baseline_rules_only.json was produced and can be reproduced.
    """
    run_id = audit.derive_run_id(seed, data_dir, arm=arm)
    conn = audit.connect(db)
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


@app.command()
def bench(
    seeds: str | None = None,
    out: Path = Path("bench_multiseed.json"),
) -> None:
    """Measure metric stability across independent seeds.

    `--seeds 42,43,44,45,46,47` (comma-separated) generates each seed's own
    fixtures into a fresh temp directory, runs the rules-only (--no-llm) arm
    against them, and reports per-metric mean/min/max/stdev across seeds —
    written to `--out` (default bench_multiseed.json) as well as printed.
    `data/` and the committed seed-42 fixtures are never touched; every temp
    directory is cleaned up before this command returns.

    Rules-only ONLY. The with-LLM arm would need a live API call per credit
    for every seed but 42, and the committed cache/ only covers seed 42 — so
    this deliberately cannot and does not touch L4.
    """
    if not seeds:
        typer.echo("Pass --seeds, e.g. --seeds 42,43,44,45,46,47", err=True)
        raise typer.Exit(code=1)
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

    payload = {"seeds": seed_list, "arm": "no_llm", "per_seed": per_seed, "summary": summary}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")

    typer.echo(f"seeds: {seed_list}")
    for field, stats in summary.items():
        typer.echo(
            f"{field}\tmean={stats['mean']:.4f}\tmin={stats['min']:.4f}"
            f"\tmax={stats['max']:.4f}\tstdev={stats['stdev']:.4f}"
        )
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
