"""Pydantic schemas — the contract for every record, decision, and LLM output
in the pipeline. Read this file first.

All money fields are `int` paise. Never float, never `Decimal`. See CLAUDE.md
invariant 1.

Field shapes mirror DATA_SPEC.md (source records) and ARCHITECTURE.md
(the audit row and the L4 adjudication output).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel


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
    UNRELATED_CREDIT = "unrelated_credit"
    OTHER = "other"


class OrderLedgerRecord(BaseModel):
    """One row of data/order_ledger.csv."""

    order_id: str
    payment_id: str
    amount_paise: int
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
    gross_paise: int
    fee_paise: int
    tax_paise: int
    net_paise: int
    settled_at: datetime


class BankStatementRecord(BaseModel):
    """One row of data/bank_statement.csv. `credit_paise` is None for debit
    rows; only credit rows are candidates for reconciliation."""

    txn_id: str
    value_date: date
    narration: str
    credit_paise: int | None
    debit_paise: int | None
    balance_paise: int


class ExpectedBatch(BaseModel):
    """A settlement batch the rules computed from order_ledger +
    settlement_report: the net amount a bank credit should match."""

    settlement_ids: list[str]
    payment_ids: list[str]
    net_paise: int
    window_start: date
    window_end: date


class CandidateExplanation(BaseModel):
    """One candidate explanation for a break, surfaced by L2/L3 and handed to
    L4 as a fact, never as an arithmetic task."""

    payment_ids: list[str]
    delta_paise: int
    hint: str


class UnresolvedCredit(BaseModel):
    """A bank credit still unresolved after earlier stages, paired with
    whatever candidates the cascade has assembled so far. This is the `Item`
    in the stage contract `(unresolved: list[Item], ctx) -> list[Decision]`."""

    credit: BankStatementRecord
    expected_batches: list[ExpectedBatch]
    candidates: list[CandidateExplanation]


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
    delta_paise: int
    reason: str
    rationale: str | None
    llm_model: str | None
    llm_cost_paise: int | None
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
