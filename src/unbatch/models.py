"""Pydantic schemas — the contract for every record, decision, and LLM output
in the pipeline. Read this file first.

All money fields use `Paise` (see `unbatch.money`): a strict `int`, never
float, never `Decimal`. See CLAUDE.md invariant 1.

Field shapes mirror DATA_SPEC.md (source records and ground_truth.json) and
ARCHITECTURE.md (the audit row and the L4 adjudication output).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from unbatch.money import Paise


class OrderStatus(StrEnum):
    CAPTURED = "captured"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    FAILED = "failed"


class PaymentMethod(StrEnum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class SettlementLineType(StrEnum):
    PAYMENT = "payment"
    REFUND = "refund"
    CHARGEBACK = "chargeback"
    ADJUSTMENT = "adjustment"


class Stage(StrEnum):
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    L4 = "l4"


class DecisionOutcome(StrEnum):
    MATCHED = "matched"
    HUMAN_REVIEW = "human_review"
    EXCEPTION = "exception"


class BreakReason(StrEnum):
    """What the L4 adjudicator classifies a break as. See DATA_SPEC.md's
    injected break-type catalogue — this is the subset the model is asked to
    name, not the generator's ground-truth label."""

    REFUND_IN_WINDOW = "refund_in_window"
    CHARGEBACK_DEDUCTION = "chargeback_deduction"
    FEE_TIER_CHANGE = "fee_tier_change"
    ROUNDING_DELTA = "rounding_delta"
    SETTLEMENT_SPLIT = "settlement_split"
    DATE_SKEW = "date_skew"
    DUPLICATE_UTR = "duplicate_utr"
    AMBIGUOUS_COMPOSITION = "ambiguous_composition"
    TOLERANCE_AMBIGUOUS = "tolerance_ambiguous"
    UNRELATED_CREDIT = "unrelated_credit"
    OTHER = "other"


class BreakType(StrEnum):
    """The generator's ground-truth label — the full catalogue from
    DATA_SPEC.md's injected break-type table. Distinct from `BreakReason`,
    which is the narrower vocabulary the L4 model is asked to classify
    against; `metrics.py` compares one to the other, nothing under stages/
    ever sees this enum."""

    CLEAN = "clean"
    NARRATION_MANGLED = "narration_mangled"
    SETTLEMENT_SPLIT = "settlement_split"
    REFUND_IN_WINDOW = "refund_in_window"
    CHARGEBACK_DEDUCTION = "chargeback_deduction"
    FEE_TIER_CHANGE = "fee_tier_change"
    ROUNDING_DELTA = "rounding_delta"
    DATE_SKEW = "date_skew"
    DUPLICATE_UTR = "duplicate_utr"
    UNRELATED_CREDIT = "unrelated_credit"
    ORPHAN_SETTLEMENT = "orphan_settlement"
    AMBIGUOUS_COMPOSITION = "ambiguous_composition"
    TOLERANCE_AMBIGUOUS = "tolerance_ambiguous"


class OrderLedgerRecord(BaseModel):
    """One row of data/order_ledger.csv."""

    order_id: str
    payment_id: str
    amount_paise: Paise
    currency: str
    status: OrderStatus
    captured_at: datetime
    customer_ref: str
    method: PaymentMethod


class SettlementLine(BaseModel):
    """One row of data/settlement_report.csv."""

    settlement_id: str
    settlement_utr: str
    payment_id: str
    type: SettlementLineType
    gross_paise: Paise
    fee_paise: Paise
    tax_paise: Paise
    net_paise: Paise
    settled_at: datetime


class BankStatementRecord(BaseModel):
    """One row of data/bank_statement.csv. `credit_paise` is None for debit
    rows; only credit rows are candidates for reconciliation. Exactly one of
    credit_paise/debit_paise is set, matching DATA_SPEC.md's "blank for
    debits" / "blank for credits" convention — never both, never neither."""

    txn_id: str
    value_date: date
    narration: str
    credit_paise: Paise | None
    debit_paise: Paise | None
    balance_paise: Paise

    @model_validator(mode="after")
    def _exactly_one_of_credit_or_debit(self) -> BankStatementRecord:
        if (self.credit_paise is None) == (self.debit_paise is None):
            raise ValueError("exactly one of credit_paise or debit_paise must be set")
        if self.credit_paise is not None and self.credit_paise < 0:
            raise ValueError("credit_paise must be non-negative")
        if self.debit_paise is not None and self.debit_paise < 0:
            raise ValueError("debit_paise must be non-negative")
        return self


class ExpectedBatch(BaseModel):
    """A settlement batch the rules computed from order_ledger +
    settlement_report: the net amount a bank credit should match.

    `settlement_utr` is the UTR a real payout carries — needed for L0 to
    check whether it appears in a bank credit's narration. Grouping
    settlement lines by UTR (see cli.compute_expected_batches) is itself
    what makes duplicate_utr and settlement_split show up as ambiguity
    rather than something a stage needs to special-case: two real batches
    sharing a UTR collapse into one ExpectedBatch here, so neither credit's
    amount ties to it exactly and L0 correctly declines both."""

    settlement_utr: str
    settlement_ids: list[str]
    payment_ids: list[str]
    net_paise: Paise
    window_start: date
    window_end: date


class CandidateExplanation(BaseModel):
    """One candidate explanation for a break, surfaced by L2/L3 and handed to
    L4 as a fact, never as an arithmetic task."""

    payment_ids: list[str]
    delta_paise: Paise
    hint: str


class UnresolvedCredit(BaseModel):
    """A bank credit still unresolved after earlier stages, paired with
    whatever candidates the cascade has assembled so far. This is the `Item`
    in the stage contract `(unresolved: list[Item], ctx) -> list[Decision]`.

    `candidate_lines` is the pool of individual settlement lines still
    available for L2/L3 to compose from — the cascade runner rebuilds it
    before each of those stages, excluding whatever earlier decisions
    already matched, so a stage never has to reach outside its own inputs
    to know what's left. L0/L1 match whole expected batches and ignore it.
    """

    credit: BankStatementRecord
    expected_batches: list[ExpectedBatch]
    candidate_lines: list[SettlementLine] = Field(default_factory=list)
    candidates: list[CandidateExplanation] = Field(default_factory=list)


class AdjudicationResult(BaseModel):
    """Pydantic-validated JSON the L4 adjudicator must return. See
    ARCHITECTURE.md § L4."""

    break_reason: BreakReason
    proposed_resolution: str
    confidence: float
    evidence_refs: list[str]
    human_review_required: bool


class Decision(BaseModel):
    """One row of the audit log. Every stage writes one of these before
    returning — no exceptions. See ARCHITECTURE.md § Audit trail."""

    run_id: str
    seed: int
    stage: Stage
    credit_id: str
    matched_payment_ids: list[str]
    outcome: DecisionOutcome
    confidence: float
    delta_paise: Paise
    reason: str
    rationale: str | None
    llm_model: str | None
    llm_cost_paise: Paise | None
    llm_retried: bool = False
    evidence_refs: list[str] | None = None
    human_review_required: bool | None = None
    created_at: datetime


class RunContext(BaseModel):
    """Config threaded through every stage of one cascade run."""

    run_id: str
    seed: int
    cached: bool = False
    no_llm: bool = False
    llm_only: bool = False
    max_pool: int = 48
    max_subset: int = 25


class GroundTruthCredit(BaseModel):
    """One ground-truth entry for a bank credit line, keyed by `txn_id`.

    `settlement_ids` is a list, not DATA_SPEC.md's single `settlement_id`: a
    settlement batch is normally many settlement_report.csv rows (one per
    payment), so a credit composed from a real batch needs more than one id.
    Empty for `unrelated_credit`, which ties to no settlement at all."""

    txn_id: str
    settlement_ids: list[str]
    payment_ids: list[str]
    break_type: BreakType
    resolvable: bool


class GroundTruthOrphanSettlement(BaseModel):
    """orphan_settlement is the one break type with no bank credit to key on
    — settlement_report.csv line(s) that never got paid out — so it cannot
    live in `GroundTruth.credits`, which is keyed by txn_id."""

    settlement_ids: list[str]
    payment_ids: list[str]


class GroundTruth(BaseModel):
    """The full contents of data/ground_truth.json. Read ONLY by
    metrics.py — CLAUDE.md invariant 7. If any module under stages/ imports
    this, that is a scoring leak."""

    credits: list[GroundTruthCredit]
    orphan_settlements: list[GroundTruthOrphanSettlement]


class DailyForecast(BaseModel):
    """One day's projected settlement inflow within a forecast horizon —
    see forecast.py's module docstring. `low_paise`/`high_paise` bound the
    observed drift between a settlement's actual net and what fees.py's own
    fee schedule would compute for it; `payment_count` is how many
    unsettled orders were projected onto this date."""

    date: date
    expected_paise: Paise
    low_paise: Paise
    high_paise: Paise
    payment_count: int


class ForecastReport(BaseModel):
    """Output of `unbatch forecast` — a projection over the existing
    settlement report, not a new data source. Pure arithmetic (CLAUDE.md
    invariant 2); never touches the adjudicator."""

    as_of: date
    horizon_days: int
    daily: list[DailyForecast]
    total_expected_paise: Paise
    total_low_paise: Paise
    total_high_paise: Paise
    unsettled_payment_count: int
    historical_payment_count: int
    historical_lag_distribution: dict[str, int]
    historical_deviation_stdev: float
