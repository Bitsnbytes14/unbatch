"""bank_statement.csv baseline generation: narration variety and a
consistent running balance, before any break injection."""

from __future__ import annotations

import random
from pathlib import Path

from unbatch.generate import (
    NARRATION_NO_UTR_BATCH_INDEX,
    NARRATION_TRUNCATED_BATCH_INDEX,
    OPENING_BALANCE_PAISE,
    generate_bank_statement_baseline,
    generate_orders_and_settlements,
    write_bank_statement_csv,
)

SEED = 42


def _baseline():
    _, _, batches = generate_orders_and_settlements(SEED)
    records = generate_bank_statement_baseline(random.Random(SEED), batches)
    return batches, records


def test_one_credit_per_batch() -> None:
    batches, records = _baseline()
    assert len(records) == len(batches)


def test_credit_amount_equals_sum_of_batch_net() -> None:
    batches, records = _baseline()
    for batch, record in zip(batches, records, strict=True):
        assert record.credit_paise == sum(line.net_paise for line in batch.lines)


def test_value_date_matches_batch_settled_date() -> None:
    batches, records = _baseline()
    for batch, record in zip(batches, records, strict=True):
        assert record.value_date == batch.settled_date


def test_running_balance_is_internally_consistent() -> None:
    _, records = _baseline()
    balance = OPENING_BALANCE_PAISE
    for record in records:
        balance += record.credit_paise
        assert record.balance_paise == balance


def test_balance_strictly_increases_with_each_credit() -> None:
    _, records = _baseline()
    balances = [r.balance_paise for r in records]
    assert balances == sorted(balances)
    assert len(set(balances)) == len(balances)


def test_txn_ids_are_unique() -> None:
    _, records = _baseline()
    assert len({r.txn_id for r in records}) == len(records)


def test_narration_templates_vary_realistically() -> None:
    batches, records = _baseline()

    truncated = records[NARRATION_TRUNCATED_BATCH_INDEX]
    no_utr = records[NARRATION_NO_UTR_BATCH_INDEX]
    full_utr_records = [
        r
        for i, r in enumerate(records)
        if i not in (NARRATION_TRUNCATED_BATCH_INDEX, NARRATION_NO_UTR_BATCH_INDEX)
    ]

    truncated_utr = batches[NARRATION_TRUNCATED_BATCH_INDEX].settlement_utr
    assert truncated_utr not in truncated.narration
    assert truncated.narration.startswith(f"NEFT-{truncated_utr[:10]}")

    no_utr_utr = batches[NARRATION_NO_UTR_BATCH_INDEX].settlement_utr
    assert no_utr_utr not in no_utr.narration

    # every other batch carries its real, full UTR verbatim
    for batch, record in zip(
        [b for i, b in enumerate(batches) if i not in (5, 6)], full_utr_records, strict=True
    ):
        assert batch.settlement_utr in record.narration

    # both NEFT and IMPS full-UTR flavors appear somewhere
    assert any(r.narration.startswith("NEFT-") for r in full_utr_records)
    assert any(r.narration.startswith("IMPS/") for r in full_utr_records)


def test_written_csv_has_blank_debit_column_and_no_carriage_returns(tmp_path: Path) -> None:
    _, records = _baseline()
    path = tmp_path / "bank_statement.csv"
    write_bank_statement_csv(records, path)

    raw = path.read_bytes()
    assert b"\r" not in raw

    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    debit_col = header.index("debit")
    for line in lines[1:]:
        assert line.split(",")[debit_col] == ""


def test_written_csv_bytes_identical_across_runs(tmp_path: Path) -> None:
    _, records_a = _baseline()
    _, records_b = _baseline()

    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    write_bank_statement_csv(records_a, path_a)
    write_bank_statement_csv(records_b, path_b)

    assert path_a.read_bytes() == path_b.read_bytes()
