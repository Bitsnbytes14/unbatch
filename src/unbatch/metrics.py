"""Scoring vs ground truth. The ONLY module permitted to read
data/ground_truth.json — CLAUDE.md invariant 7. If a stage under
`unbatch.stages` ever imports ground truth, that is a scoring leak.

Computes every figure defined in METRICS.md: count and value-weighted match
rate, false-match rate, exception rate, precision, recall (excluding
correctly-unresolvable break types from the denominator), the stage funnel,
LLM call count and cost, and p50/p95 latency per stage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

DEFAULT_GROUND_TRUTH_PATH = Path("data/ground_truth.json")


class MetricsReport(BaseModel):
    """Every number METRICS.md permits to be quoted anywhere in the README,
    HTML report, or pitch video."""

    run_id: str
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
    latency_p50_ms: dict[str, float]
    latency_p95_ms: dict[str, float]
    malformed_json_count: int
    retry_count: int
    adjudication_failed_count: int


def score(
    conn: sqlite3.Connection,
    run_id: str,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> MetricsReport:
    """Compute the full metrics report for one run against ground truth."""
    raise NotImplementedError
