"""metrics.score against a real --no-llm cascade run over the seed-42
fixtures: no false matches (verified independently in
test_l3_tolerance_integration.py), and every rate lines up with the known
break-type distribution."""

from __future__ import annotations

from pathlib import Path

from unbatch import audit, metrics
from unbatch.cli import (
    build_unresolved_credits,
    compute_expected_batches,
    load_input_data,
    run_cascade,
)
from unbatch.models import RunContext

SEED = 42


def _run_no_llm_cascade(db_path: Path) -> str:
    _orders, settlements, bank_records = load_input_data()
    expected_batches = compute_expected_batches(settlements)
    unresolved = build_unresolved_credits(bank_records, expected_batches)

    run_id = audit.derive_run_id(SEED, arm="no_llm")
    ctx = RunContext(run_id=run_id, seed=SEED, no_llm=True)
    conn = audit.connect(db_path)
    audit.clear_run(conn, run_id)
    run_cascade(ctx, unresolved, conn, settlements=settlements)
    return run_id


def test_no_false_matches_on_the_real_dataset(tmp_path: Path) -> None:
    run_id = _run_no_llm_cascade(tmp_path / "audit.db")
    conn = audit.connect(tmp_path / "audit.db")

    report = metrics.score(conn, run_id)

    assert report.false_match_rate == 0.0
    assert report.total_credits == 105


def test_count_and_exception_rate_match_the_known_distribution(tmp_path: Path) -> None:
    """clean(79) + narration_mangled(3) + settlement_split(4) + date_skew(1)
    + refund_in_window(1) + chargeback_deduction(1) + duplicate_utr(2) +
    rounding_delta(1) + fee_tier_change(1) = 93 resolved; ambiguous_
    composition(7) + tolerance_ambiguous(4) + unrelated_credit(1) = 12
    exceptions. D0a (FAILURES.md's 2026-08-31 entry) deliberately raised
    ambiguous_composition from 1 to 7 and added tolerance_ambiguous (4) so
    the adjudicator has more than one genuinely ambiguous case to prove
    itself on — clean dropped from ~85% to ~75% as a direct result, reported
    honestly rather than tuned back."""
    run_id = _run_no_llm_cascade(tmp_path / "audit.db")
    conn = audit.connect(tmp_path / "audit.db")

    report = metrics.score(conn, run_id)

    assert report.count_match_rate == 93 / 105
    assert report.exception_rate == 12 / 105
    assert report.precision == 1.0  # every resolved match is correct


def test_correctly_rejected_is_unrelated_credit_plus_the_orphan_settlements(
    tmp_path: Path,
) -> None:
    run_id = _run_no_llm_cascade(tmp_path / "audit.db")
    conn = audit.connect(tmp_path / "audit.db")

    report = metrics.score(conn, run_id)

    # unrelated_credit(1) + 7 orphan_settlement entries (one per
    # ambiguous_composition decoy, D0a)
    assert report.correctly_rejected == 8


def test_stage_funnel_matches_the_per_stage_resolution_counts(tmp_path: Path) -> None:
    run_id = _run_no_llm_cascade(tmp_path / "audit.db")
    conn = audit.connect(tmp_path / "audit.db")

    report = metrics.score(conn, run_id)

    assert report.stage_funnel["l0"] == 79
    assert report.stage_funnel["l1"] == 3
    assert report.stage_funnel["l2"] == 9
    assert report.stage_funnel["l3"] == 2
    assert report.stage_funnel["l4"] == 12  # the terminal no_llm_unresolved exceptions


def test_value_weighted_match_rate_is_within_a_few_points_of_count_rate(
    tmp_path: Path,
) -> None:
    """METRICS.md's target shape: value-weighted should track count rate
    closely, not be dragged down by the one deliberately large credit
    landing in the unresolved 2%."""
    run_id = _run_no_llm_cascade(tmp_path / "audit.db")
    conn = audit.connect(tmp_path / "audit.db")

    report = metrics.score(conn, run_id)

    assert abs(report.value_weighted_match_rate - report.count_match_rate) < 0.05


def test_no_llm_arm_makes_zero_llm_calls(tmp_path: Path) -> None:
    run_id = _run_no_llm_cascade(tmp_path / "audit.db")
    conn = audit.connect(tmp_path / "audit.db")

    report = metrics.score(conn, run_id)

    assert report.llm_call_count == 0
    assert report.llm_cost_paise == 0
