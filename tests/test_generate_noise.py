"""Narration noise (E10a): a post-processing pass over an already-generated
bank_statement that only ever rewrites `narration` — amounts, dates,
balances, and txn_ids stay exact, and `noise=0.0` is a strict no-op so the
committed seed-42 fixtures stay byte-for-byte reproducible."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

from unbatch.generate import (
    _noise_drop_utr,
    _noise_lookalike_substitution,
    _noise_transpose_utr_digits,
    _noise_truncate_mid_utr,
    apply_narration_noise,
    generate,
)
from unbatch.models import BankStatementRecord

SEED = 42


def _record(narration: str, txn_id: str = "txn_1") -> BankStatementRecord:
    return BankStatementRecord(
        txn_id=txn_id,
        value_date=date(2024, 1, 5),
        narration=narration,
        credit_paise=100_000,
        debit_paise=None,
        balance_paise=100_000,
    )


def test_generate_with_noise_0_is_byte_identical_to_no_noise_argument(tmp_path: Path) -> None:
    default_dir = tmp_path / "default"
    explicit_dir = tmp_path / "explicit"
    generate(SEED, out_dir=default_dir)
    generate(SEED, out_dir=explicit_dir, noise=0.0)

    for filename in (
        "order_ledger.csv",
        "settlement_report.csv",
        "bank_statement.csv",
        "ground_truth.json",
    ):
        assert (default_dir / filename).read_bytes() == (explicit_dir / filename).read_bytes()


def test_apply_narration_noise_0_returns_input_unchanged() -> None:
    records = [_record("NEFT-AXISP123456789012-RAZORPAY SOFTWARE PVT")]
    result = apply_narration_noise(random.Random(SEED), records, 0.0)
    assert result is records


def _utr_digits(i: int) -> str:
    """12 digits with no two adjacent digits equal — consecutive positions
    differ by exactly 1 mod 10, so a digit-transposition noise technique
    can never coincidentally no-op on this corpus the way it legitimately
    can on a real UTR that happens to have a repeated digit pair."""
    return "".join(str((d + i) % 10) for d in range(12))


def test_apply_narration_noise_only_ever_touches_narration() -> None:
    records = [
        _record(f"NEFT-AXISP{_utr_digits(i)}-RAZORPAY SOFTWARE PVT", f"txn_{i}") for i in range(20)
    ]
    noised = apply_narration_noise(random.Random(SEED), records, 1.0)

    for before, after in zip(records, noised, strict=True):
        assert after.txn_id == before.txn_id
        assert after.value_date == before.value_date
        assert after.credit_paise == before.credit_paise
        assert after.debit_paise == before.debit_paise
        assert after.balance_paise == before.balance_paise


def test_apply_narration_noise_is_deterministic() -> None:
    records = [
        _record(f"NEFT-AXISP{_utr_digits(i)}-RAZORPAY SOFTWARE PVT", f"txn_{i}") for i in range(20)
    ]
    first = apply_narration_noise(random.Random(SEED), records, 0.5)
    second = apply_narration_noise(random.Random(SEED), records, 0.5)
    assert [r.narration for r in first] == [r.narration for r in second]


def test_apply_narration_noise_1_0_changes_almost_every_narration() -> None:
    """Not literally every one: e.g. the separator technique's `/` -> `-`
    variant is a legitimate no-op on a NEFT-style narration that never had
    a `/` to begin with — realistic noise can genuinely have no visible
    effect on a given input, same as in a real bank's own formatting."""
    records = [
        _record(f"NEFT-AXISP{_utr_digits(i)}-RAZORPAY SOFTWARE PVT", f"txn_{i}") for i in range(30)
    ]
    noised = apply_narration_noise(random.Random(SEED), records, 1.0)
    changed = sum(
        1
        for before, after in zip(records, noised, strict=True)
        if after.narration != before.narration
    )
    assert changed >= len(records) - 2


def test_generate_at_higher_noise_still_produces_valid_credits(tmp_path: Path) -> None:
    """Noise never touches amounts/dates, so every credit must still be
    exactly what generate() without noise would have produced."""
    clean_dir = tmp_path / "clean"
    noisy_dir = tmp_path / "noisy"
    generate(SEED, out_dir=clean_dir)
    generate(SEED, out_dir=noisy_dir, noise=0.8)

    assert (clean_dir / "settlement_report.csv").read_bytes() == (
        noisy_dir / "settlement_report.csv"
    ).read_bytes()
    assert (clean_dir / "order_ledger.csv").read_bytes() == (
        noisy_dir / "order_ledger.csv"
    ).read_bytes()
    assert (clean_dir / "ground_truth.json").read_bytes() == (
        noisy_dir / "ground_truth.json"
    ).read_bytes()
    assert (clean_dir / "bank_statement.csv").read_bytes() != (
        noisy_dir / "bank_statement.csv"
    ).read_bytes()


def test_noise_truncate_mid_utr_cuts_inside_the_utr_span() -> None:
    narration = "NEFT-AXISP123456789012-RAZORPAY SOFTWARE PVT"
    result = _noise_truncate_mid_utr(narration, random.Random(1))
    assert "AXISP123456789012" not in result
    assert result == narration[: len(result)]  # still a prefix — a true truncation


def test_noise_transpose_utr_digits_changes_the_utr() -> None:
    narration = "NEFT-AXISP123456789012-RAZORPAY SOFTWARE PVT"
    result = _noise_transpose_utr_digits(narration, random.Random(1))
    assert "AXISP123456789012" not in result
    # only the UTR span should differ; prefix/suffix text is untouched
    assert result.startswith("NEFT-AXISP")
    assert result.endswith("-RAZORPAY SOFTWARE PVT")


def test_noise_lookalike_substitution_changes_the_utr() -> None:
    narration = "NEFT-AXISP123456789012-RAZORPAY SOFTWARE PVT"
    result = _noise_lookalike_substitution(narration, random.Random(1))
    assert "AXISP123456789012" not in result


def test_noise_drop_utr_removes_the_utr_entirely() -> None:
    narration = "NEFT-AXISP123456789012-RAZORPAY SOFTWARE PVT"
    result = _noise_drop_utr(narration, random.Random(1))
    assert "AXISP123456789012" not in result
    assert "123456789012" not in result


def test_noise_functions_degrade_gracefully_with_no_utr_present() -> None:
    """A narration that's already had its UTR removed by an earlier
    break-injection step (narration_mangled's no-UTR case) shouldn't crash
    the UTR-targeted techniques — they should just leave it as-is or fall
    back to a generic transform."""
    narration = "NEFT-XXXXXXXXXXXX-MISC SETTLEMENT CREDIT"
    for technique in (_noise_transpose_utr_digits, _noise_lookalike_substitution):
        assert technique(narration, random.Random(1)) == narration
    # truncation still produces *something* shorter, just not UTR-targeted
    truncated = _noise_truncate_mid_utr(narration, random.Random(1))
    assert len(truncated) < len(narration)
