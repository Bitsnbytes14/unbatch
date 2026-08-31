"""derive_run_id: reproducible across identical inputs, non-colliding across
different seeds or different underlying data."""

from __future__ import annotations

from pathlib import Path

from unbatch.audit import derive_run_id

_FILES = {
    "order_ledger.csv": "order_id,payment_id,amount\norder_1,pay_1,100.00\n",
    "settlement_report.csv": "settlement_id,payment_id,net\nsetl_1,pay_1,98.00\n",
    "bank_statement.csv": "txn_id,value_date,credit\ntxn_1,2024-01-01,98.00\n",
}


def _write_data_dir(base: Path, files: dict[str, str] | None = None) -> Path:
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (files or _FILES).items():
        (data_dir / name).write_text(content, encoding="utf-8", newline="")
    return data_dir


def test_same_seed_and_data_yields_the_same_run_id(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path)
    assert derive_run_id(42, data_dir) == derive_run_id(42, data_dir)


def test_different_seeds_do_not_collide(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path)
    assert derive_run_id(42, data_dir) != derive_run_id(43, data_dir)


def test_different_input_data_does_not_collide_even_with_the_same_seed(tmp_path: Path) -> None:
    dir_a = _write_data_dir(tmp_path / "a")
    changed = dict(_FILES)
    changed["order_ledger.csv"] = _FILES["order_ledger.csv"].replace("100.00", "999.00")
    dir_b = _write_data_dir(tmp_path / "b", changed)

    assert derive_run_id(42, dir_a) != derive_run_id(42, dir_b)


def test_run_id_contains_the_seed_for_human_readability(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path)
    assert derive_run_id(42, data_dir).startswith("run_42_")


def test_different_arms_do_not_collide(tmp_path: Path) -> None:
    """The whole reason `arm` is part of the hash: --no-llm and the default
    with-LLM run against the same seed must never derive the same run_id,
    or the second run's audit.clear_run would delete the first arm's
    results before the ablation could compare them."""
    data_dir = _write_data_dir(tmp_path)
    no_llm_id = derive_run_id(42, data_dir, arm="no_llm")
    with_llm_id = derive_run_id(42, data_dir, arm="with_llm")
    llm_only_id = derive_run_id(42, data_dir, arm="llm_only")

    assert len({no_llm_id, with_llm_id, llm_only_id}) == 3


def test_arm_appears_in_the_run_id_for_human_readability(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path)
    assert "no_llm" in derive_run_id(42, data_dir, arm="no_llm")
    assert "llm_only" in derive_run_id(42, data_dir, arm="llm_only")
