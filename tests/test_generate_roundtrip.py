"""CSV read functions round-trip what write functions produce — the
runner's only way to get seeded data back into pydantic models."""

from __future__ import annotations

from pathlib import Path

from unbatch.cli import load_input_data
from unbatch.generate import (
    generate_orders_and_settlements,
    read_bank_statement_csv,
    read_order_ledger_csv,
    read_settlement_report_csv,
    write_order_ledger_csv,
    write_settlement_report_csv,
)

SEED = 42


def test_order_ledger_round_trips(tmp_path: Path) -> None:
    orders, _settlements, _batches = generate_orders_and_settlements(SEED)
    path = tmp_path / "order_ledger.csv"
    write_order_ledger_csv(orders, path)

    reloaded = read_order_ledger_csv(path)
    assert [o.model_dump() for o in reloaded] == [o.model_dump() for o in orders]


def test_settlement_report_round_trips(tmp_path: Path) -> None:
    _orders, settlements, _batches = generate_orders_and_settlements(SEED)
    path = tmp_path / "settlement_report.csv"
    write_settlement_report_csv(settlements, path)

    reloaded = read_settlement_report_csv(path)
    assert [s.model_dump() for s in reloaded] == [s.model_dump() for s in settlements]


def test_load_input_data_reads_the_real_generated_fixtures() -> None:
    """Against the actual committed data/ fixtures, not a fresh generation —
    this is what `unbatch run` will actually read."""
    orders, settlements, bank_records = load_input_data()
    assert len(orders) > 0
    assert len(settlements) > 0
    assert len(bank_records) > 0
    assert all(r.credit_paise is not None for r in bank_records)


def test_bank_statement_round_trips_via_the_real_fixture() -> None:
    records = read_bank_statement_csv(Path("data/bank_statement.csv"))
    assert len(records) > 0
    for r in records:
        assert r.credit_paise is not None
        assert r.debit_paise is None
