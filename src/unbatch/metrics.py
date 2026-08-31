"""Scoring vs ground truth. The ONLY module permitted to read
data/ground_truth.json — CLAUDE.md invariant 7. If a module under
`unbatch.stages`, `unbatch.cli`, `unbatch.compose`, or anywhere else in the
pipeline ever imports ground truth, that is a scoring leak; see
tests/test_no_scoring_leak.py for the AST check enforcing this across the
whole package, not just stages/.

Computes every figure defined in METRICS.md: count and value-weighted match
rate, false-match rate, exception rate, precision, recall (excluding
correctly-unresolvable break types from the denominator), and the stage
funnel. LLM-related figures (call count/rate/cost, malformed-JSON/retry/
adjudication-failed counts) are computed too — correctly zero for a
`--no-llm` run, since no LLM call ever happened — and will start reporting
real numbers once L4 exists.

**Comparing `matched_payment_ids` to ground truth `payment_ids`**: both are
built the same way, one payment_id per contributing settlement line — so a
payment and its own refund (which share a payment_id; see DATA_SPEC.md) are
each supposed to contribute their own entry, and a genuinely correct match
for e.g. refund_in_window has that payment_id appearing *twice*. Compared
as **multisets** (`collections.Counter`), never as:

- plain sets — would silently discard the duplicate and call a match
  "correct" when it's actually missing the refund line's contribution;
- ordered lists — the order lines are discovered in (composition search,
  batch grouping) isn't guaranteed to match the order ground truth recorded
  them in, so an order-sensitive comparison would flag correct matches as
  wrong purely on ordering.

Latency (p50/p95 per stage) is not computed this milestone: nothing in the
audit schema records per-decision timing, and inventing percentiles from a
single per-stage wall-clock measurement would be fabricating precision the
data doesn't have. Per METRICS.md's own rule ("if a number in the README
does not have a line in metrics.py producing it, delete the number"), it is
left out entirely rather than reported as a fake zero.

**Break-reason classification accuracy** (D0a/D0b — see METRICS.md's own
section on this): count and value-weighted match rate score WHETHER a
credit resolved and to what payment_ids, never WHY. Every credit that
reaches L4 already ties to *some* expected batch by construction (L4 always
picks the closest one, per l4_llm.py) — the model's real job there is
naming the right `break_reason`, which the match-rate numbers above cannot
see at all, since they only look at `matched_payment_ids`. This is scored
separately: `decision.reason` (the model's `BreakReason`) against the
matching `GroundTruthCredit.break_type` for every credit whose Decision
carries an `llm_model` (i.e. actually reached the adjudicator) and did not
degrade to `adjudication_failed` — that outcome means no classification was
produced at all, so it is excluded from the accuracy denominator and left
visible only in `adjudication_failed_count`. `BreakReason` and `BreakType`
intentionally share their string values for the categories that can reach
L4 (see models.py), so `decision.reason == ground_truth.break_type.value`
is a direct, correct comparison — not a mapping to maintain.

**`exception_break_type_counts`** exists so report.py can state D3's
ablation framing ("of the N credits reaching L4, M are cases the rules
correctly declined") without importing ground truth itself — that would be
exactly the scoring leak this module's docstring warns about, just one
layer up. It is a ground-truth `break_type` -> count breakdown, computed
here where ground truth is already loaded, over every credit this run left
as an exception (regardless of arm), so report.py only ever reads already
computed MetricsReport fields.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from unbatch import audit
from unbatch import generate as generate_module
from unbatch.models import DecisionOutcome, GroundTruth

DEFAULT_DATA_DIR = Path("data")


class MetricsReport(BaseModel):
    """Every number METRICS.md permits to be quoted anywhere in the README,
    HTML report, or pitch video."""

    run_id: str
    total_credits: int

    count_match_rate: float
    value_weighted_match_rate: float
    false_match_rate: float
    exception_rate: float
    precision: float
    recall: float
    correctly_rejected: int

    stage_funnel: dict[str, int]

    llm_call_count: int
    llm_call_rate: float
    llm_cost_paise: int
    malformed_json_count: int
    retry_count: int
    adjudication_failed_count: int

    break_reason_accuracy: float
    break_reason_confusion: dict[str, dict[str, int]]

    exception_break_type_counts: dict[str, int]


def _load_ground_truth(path: Path) -> GroundTruth:
    return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))


def _is_correct_match(matched_payment_ids: list[str], ground_truth_payment_ids: list[str]) -> bool:
    """Multiset equality — see this module's docstring for why not a plain
    set or an ordered list."""
    return Counter(matched_payment_ids) == Counter(ground_truth_payment_ids)


def score(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    ground_truth_path: Path | None = None,
) -> MetricsReport:
    """Compute the full metrics report for one run against ground truth.
    `ground_truth_path` defaults to `data_dir / "ground_truth.json"`;
    override it only to point at a ground truth file that doesn't live
    alongside the rest of that run's data (e.g. in a test fixture)."""
    ground_truth = _load_ground_truth(ground_truth_path or data_dir / "ground_truth.json")
    bank_records = generate_module.read_bank_statement_csv(data_dir / "bank_statement.csv")
    credit_amount_by_id = {
        record.txn_id: record.credit_paise
        for record in bank_records
        if record.credit_paise is not None
    }

    decisions = audit.fetch_decisions(conn, run_id)
    decision_by_credit = {decision.credit_id: decision for decision in decisions}
    gt_by_txn = {credit.txn_id: credit for credit in ground_truth.credits}

    total_credits = len(ground_truth.credits)
    resolvable_ids = {credit.txn_id for credit in ground_truth.credits if credit.resolvable}

    resolved_ids: set[str] = set()
    correct_ids: set[str] = set()
    false_match_ids: set[str] = set()
    exception_ids: set[str] = set()

    for credit in ground_truth.credits:
        decision = decision_by_credit.get(credit.txn_id)
        if decision is None or decision.outcome == DecisionOutcome.EXCEPTION:
            exception_ids.add(credit.txn_id)
            continue

        resolved_ids.add(credit.txn_id)
        if _is_correct_match(decision.matched_payment_ids, credit.payment_ids):
            correct_ids.add(credit.txn_id)
        else:
            false_match_ids.add(credit.txn_id)

    correctly_rejected_credits = {
        txn_id for txn_id in exception_ids if txn_id not in resolvable_ids
    }
    correctly_rejected = len(correctly_rejected_credits) + len(ground_truth.orphan_settlements)

    total_amount = sum(credit_amount_by_id.get(c.txn_id, 0) for c in ground_truth.credits)
    resolved_amount = sum(credit_amount_by_id.get(txn_id, 0) for txn_id in resolved_ids)

    count_match_rate = len(resolved_ids) / total_credits if total_credits else 0.0
    value_weighted_match_rate = resolved_amount / total_amount if total_amount else 0.0
    false_match_rate = len(false_match_ids) / len(resolved_ids) if resolved_ids else 0.0
    exception_rate = len(exception_ids) / total_credits if total_credits else 0.0
    precision = len(correct_ids) / len(resolved_ids) if resolved_ids else 0.0
    recall = (
        len(correct_ids & resolvable_ids) / len(resolvable_ids) if resolvable_ids else 0.0
    )

    stage_funnel: dict[str, int] = {}
    for decision in decisions:
        stage_funnel[decision.stage.value] = stage_funnel.get(decision.stage.value, 0) + 1

    llm_decisions = [d for d in decisions if d.llm_model is not None]
    llm_call_count = len(llm_decisions)
    llm_call_rate = llm_call_count / total_credits if total_credits else 0.0
    llm_cost_paise = sum(d.llm_cost_paise or 0 for d in llm_decisions)
    adjudication_failed_count = sum(1 for d in llm_decisions if d.reason == "adjudication_failed")
    retry_count = sum(1 for d in llm_decisions if d.llm_retried)
    # Every retried credit means one malformed first response; a credit that
    # retried AND still failed means both attempts were malformed.
    malformed_json_count = retry_count + sum(
        1 for d in llm_decisions if d.llm_retried and d.reason == "adjudication_failed"
    )

    exception_break_type_counts: dict[str, int] = {}
    for txn_id in sorted(exception_ids):
        gt_credit = gt_by_txn.get(txn_id)
        if gt_credit is None:
            continue
        bt = gt_credit.break_type.value
        exception_break_type_counts[bt] = exception_break_type_counts.get(bt, 0) + 1

    break_reason_confusion: dict[str, dict[str, int]] = {}
    break_reason_correct = 0
    break_reason_total = 0
    for decision in llm_decisions:
        if decision.reason == "adjudication_failed":
            continue  # no classification was produced at all
        gt_credit = gt_by_txn.get(decision.credit_id)
        if gt_credit is None:
            continue
        actual = gt_credit.break_type.value
        predicted = decision.reason
        by_actual = break_reason_confusion.setdefault(actual, {})
        by_actual[predicted] = by_actual.get(predicted, 0) + 1
        break_reason_total += 1
        if predicted == actual:
            break_reason_correct += 1
    break_reason_accuracy = (
        break_reason_correct / break_reason_total if break_reason_total else 0.0
    )

    return MetricsReport(
        run_id=run_id,
        total_credits=total_credits,
        count_match_rate=count_match_rate,
        value_weighted_match_rate=value_weighted_match_rate,
        false_match_rate=false_match_rate,
        exception_rate=exception_rate,
        precision=precision,
        recall=recall,
        correctly_rejected=correctly_rejected,
        stage_funnel=stage_funnel,
        llm_call_count=llm_call_count,
        llm_call_rate=llm_call_rate,
        llm_cost_paise=llm_cost_paise,
        malformed_json_count=malformed_json_count,
        retry_count=retry_count,
        adjudication_failed_count=adjudication_failed_count,
        break_reason_accuracy=break_reason_accuracy,
        break_reason_confusion=break_reason_confusion,
        exception_break_type_counts=exception_break_type_counts,
    )
