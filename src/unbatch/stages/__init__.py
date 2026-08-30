"""The reconciliation cascade, cheapest-and-most-certain stage first: l0_utr,
l1_exact, l2_compose, l3_tolerance, l4_llm.

Every stage is a pure function `(unresolved: list[UnresolvedCredit], ctx:
RunContext) -> list[Decision]`. Stages never mutate global state and never
call each other — cli.py orchestrates the sequence, feeding each stage's
leftovers to the next. Never reorder without updating ARCHITECTURE.md.
"""
