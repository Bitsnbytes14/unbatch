"""`unbatch verify`'s diff logic: the recursive comparison that decides
whether a regenerated artifact matches what's committed. This is the part
that actually needs correctness guarantees — a full `unbatch verify` run
regenerates real cached/rules-only artifacts end to end (manually confirmed:
PASS on the real committed files, DRIFT when one is deliberately corrupted,
reverting cleanly afterward), which is too slow to run on every test
invocation."""

from __future__ import annotations

from unbatch.cli import _diff_json


def test_identical_values_produce_no_diff() -> None:
    committed = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}
    regenerated = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}

    assert _diff_json(committed, regenerated, ignore_keys=frozenset()) == []


def test_nested_dict_mismatch_reports_the_full_path() -> None:
    committed = {"outer": {"inner": {"value": 1}}}
    regenerated = {"outer": {"inner": {"value": 2}}}

    diff = _diff_json(committed, regenerated, ignore_keys=frozenset())

    assert len(diff) == 1
    assert "$.outer.inner.value" in diff[0]
    assert "committed=1" in diff[0]
    assert "regenerated=2" in diff[0]


def test_list_length_mismatch_is_reported_without_descending_into_elements() -> None:
    committed = {"items": [1, 2, 3]}
    regenerated = {"items": [1, 2]}

    diff = _diff_json(committed, regenerated, ignore_keys=frozenset())

    assert len(diff) == 1
    assert "$.items" in diff[0]
    assert "length 3 != 2" in diff[0]


def test_list_element_mismatch_reports_the_index() -> None:
    committed = {"items": [1, 2, 3]}
    regenerated = {"items": [1, 99, 3]}

    diff = _diff_json(committed, regenerated, ignore_keys=frozenset())

    assert len(diff) == 1
    assert "$.items[1]" in diff[0]


def test_ignore_keys_skips_the_field_at_any_depth() -> None:
    committed = {"total_seconds": 11.4, "stage_seconds": {"l0": 10.5}, "count": 5}
    regenerated = {"total_seconds": 3.5, "stage_seconds": {"l0": 3.2}, "count": 5}

    diff = _diff_json(
        committed, regenerated, ignore_keys=frozenset({"total_seconds", "stage_seconds"})
    )

    assert diff == []


def test_ignore_keys_does_not_suppress_other_mismatches() -> None:
    committed = {"total_seconds": 11.4, "count": 5}
    regenerated = {"total_seconds": 3.5, "count": 6}

    diff = _diff_json(committed, regenerated, ignore_keys=frozenset({"total_seconds"}))

    assert len(diff) == 1
    assert "$.count" in diff[0]


def test_a_key_missing_from_the_regenerated_output_is_reported() -> None:
    committed = {"a": 1, "b": 2}
    regenerated = {"a": 1}

    diff = _diff_json(committed, regenerated, ignore_keys=frozenset())

    assert len(diff) == 1
    assert "$.b" in diff[0]
    assert "missing from regenerated output" in diff[0]


def test_a_key_missing_from_the_committed_file_is_reported() -> None:
    committed = {"a": 1}
    regenerated = {"a": 1, "b": 2}

    diff = _diff_json(committed, regenerated, ignore_keys=frozenset())

    assert len(diff) == 1
    assert "$.b" in diff[0]
    assert "missing from committed file" in diff[0]
