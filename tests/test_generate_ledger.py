"""order_ledger.csv and settlement_report.csv generation: determinism,
structure, and the guarantees generate_orders_and_settlements makes by
construction (a refund, a chargeback, one deliberately large line)."""

from __future__ import annotations

from pathlib import Path

from unbatch.generate import (
    CHARGEBACK_BATCH_INDEX,
    LARGE_VALUE_BATCH_INDEX,
    LARGE_VALUE_PAISE,
    N_BATCHES,
    REFUND_BATCH_INDEX,
    generate_orders_and_settlements,
    write_order_ledger_csv,
    write_settlement_report_csv,
)
from unbatch.models import SettlementLineType
from unbatch.money import parse_rupees_to_paise

SEED = 42


def test_generation_is_deterministic_for_the_same_seed() -> None:
    orders_a, settlements_a, _ = generate_orders_and_settlements(SEED)
    orders_b, settlements_b, _ = generate_orders_and_settlements(SEED)

    assert [o.model_dump() for o in orders_a] == [o.model_dump() for o in orders_b]
    assert [s.model_dump() for s in settlements_a] == [s.model_dump() for s in settlements_b]


def test_different_seeds_produce_different_data() -> None:
    orders_42, _, _ = generate_orders_and_settlements(42)
    orders_43, _, _ = generate_orders_and_settlements(43)
    assert [o.order_id for o in orders_42] != [o.order_id for o in orders_43]


def test_produces_the_documented_batch_count() -> None:
    _, _, batches = generate_orders_and_settlements(SEED)
    assert len(batches) == N_BATCHES
    assert len({b.settled_date for b in batches}) == N_BATCHES, "batch dates must be distinct"


def test_batch_shares_one_utr_across_its_lines() -> None:
    _, _, batches = generate_orders_and_settlements(SEED)
    for batch in batches:
        utrs = {line.settlement_utr for line in batch.lines}
        assert utrs == {batch.settlement_utr}


def test_scale_is_at_least_the_documented_floor() -> None:
    _, settlements, _ = generate_orders_and_settlements(SEED)
    assert len(settlements) >= 50  # DATA_SPEC.md's floor; generator targets ~150


def test_guarantees_at_least_one_refund_line() -> None:
    _, settlements, batches = generate_orders_and_settlements(SEED)
    assert any(s.type == SettlementLineType.REFUND for s in settlements)
    assert any(
        line.type == SettlementLineType.REFUND for line in batches[REFUND_BATCH_INDEX].lines
    )


def test_guarantees_at_least_one_chargeback_line() -> None:
    _, settlements, batches = generate_orders_and_settlements(SEED)
    assert any(s.type == SettlementLineType.CHARGEBACK for s in settlements)
    assert any(
        line.type == SettlementLineType.CHARGEBACK
        for line in batches[CHARGEBACK_BATCH_INDEX].lines
    )


def test_guarantees_one_deliberately_large_value_line() -> None:
    _, settlements, batches = generate_orders_and_settlements(SEED)
    assert any(s.gross_paise == LARGE_VALUE_PAISE for s in settlements)
    assert any(
        line.gross_paise == LARGE_VALUE_PAISE for line in batches[LARGE_VALUE_BATCH_INDEX].lines
    )
    # it must actually dwarf ordinary lines for the value-weighted metric to diverge
    ordinary = [s.gross_paise for s in settlements if s.gross_paise != LARGE_VALUE_PAISE]
    assert LARGE_VALUE_PAISE > 10 * max(ordinary)


def test_every_settlement_line_net_equals_gross_minus_fee_minus_tax() -> None:
    _, settlements, _ = generate_orders_and_settlements(SEED)
    for line in settlements:
        assert line.net_paise == line.gross_paise - line.fee_paise - line.tax_paise


def test_refund_and_chargeback_lines_have_negative_gross() -> None:
    _, settlements, _ = generate_orders_and_settlements(SEED)
    for line in settlements:
        if line.type in (SettlementLineType.REFUND, SettlementLineType.CHARGEBACK):
            assert line.gross_paise < 0


def test_written_csvs_have_no_carriage_returns(tmp_path: Path) -> None:
    orders, settlements, _ = generate_orders_and_settlements(SEED)
    ledger_path = tmp_path / "order_ledger.csv"
    settlement_path = tmp_path / "settlement_report.csv"

    write_order_ledger_csv(orders, ledger_path)
    write_settlement_report_csv(settlements, settlement_path)

    assert b"\r" not in ledger_path.read_bytes()
    assert b"\r" not in settlement_path.read_bytes()


def test_written_csv_bytes_are_identical_across_runs(tmp_path: Path) -> None:
    orders_a, settlements_a, _ = generate_orders_and_settlements(SEED)
    orders_b, settlements_b, _ = generate_orders_and_settlements(SEED)

    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    write_order_ledger_csv(orders_a, path_a)
    write_order_ledger_csv(orders_b, path_b)

    assert path_a.read_bytes() == path_b.read_bytes()

    path_a2 = tmp_path / "a2.csv"
    path_b2 = tmp_path / "b2.csv"
    write_settlement_report_csv(settlements_a, path_a2)
    write_settlement_report_csv(settlements_b, path_b2)

    assert path_a2.read_bytes() == path_b2.read_bytes()


def test_written_csv_amount_round_trips_through_money_parsing(tmp_path: Path) -> None:
    orders, _, _ = generate_orders_and_settlements(SEED)
    path = tmp_path / "order_ledger.csv"
    write_order_ledger_csv(orders, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    amount_col = header.index("amount")
    first_row = lines[1].split(",")

    assert parse_rupees_to_paise(first_row[amount_col]) == orders[0].amount_paise
