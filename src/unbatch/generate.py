"""Seeded synthetic data generation.

Writes data/order_ledger.csv, data/settlement_report.csv,
data/bank_statement.csv, and data/ground_truth.json for a given seed. Must
produce byte-identical CSVs for the same seed on any machine (DATA_SPEC.md).
Prints a break-type distribution summary to stdout so the mix is visible
without opening the files.

Money is computed in paise internally and formatted to 2dp only on write.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_OUT_DIR = Path("data")


def generate(seed: int, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Generate order_ledger.csv, settlement_report.csv, bank_statement.csv,
    and ground_truth.json under `out_dir` for `seed`, then print the
    break-type distribution to stdout."""
    raise NotImplementedError
