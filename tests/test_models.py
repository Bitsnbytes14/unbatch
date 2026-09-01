"""CascadeConfig — every tunable cascade parameter collected in one place
(models.py), validated on construction rather than trusted as bare module
constants scattered across compose.py/l2_compose.py/l3_tolerance.py/l4_llm.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from unbatch.models import CascadeConfig, RunContext


def test_defaults_match_the_original_scattered_constants() -> None:
    """The whole point of this refactor is that no value moved, only where
    it lives — this pins the exact numbers that used to be module
    constants."""
    config = CascadeConfig()
    assert config.max_pool == 48
    assert config.max_subset == 25
    assert config.compose_timeout_s == 5.0
    assert config.date_window_days == 3
    assert config.tolerance_rate == 0.006
    assert config.tolerance_floor_paise == 50
    assert config.confidence_auto_accept == 0.85
    assert config.confidence_human_review == 0.60


@pytest.mark.parametrize("bad_value", [0, -1])
def test_max_pool_must_be_positive(bad_value: int) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(max_pool=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_max_subset_must_be_positive(bad_value: int) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(max_subset=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1.0])
def test_compose_timeout_must_be_positive(bad_value: float) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(compose_timeout_s=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_date_window_days_must_be_positive(bad_value: int) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(date_window_days=bad_value)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_tolerance_rate_must_be_in_zero_one(bad_value: float) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(tolerance_rate=bad_value)


def test_tolerance_rate_exactly_zero_and_one_are_both_accepted() -> None:
    assert CascadeConfig(tolerance_rate=0.0).tolerance_rate == 0.0
    assert CascadeConfig(tolerance_rate=1.0).tolerance_rate == 1.0


def test_tolerance_floor_paise_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(tolerance_floor_paise=-1)


def test_tolerance_floor_paise_zero_is_accepted() -> None:
    assert CascadeConfig(tolerance_floor_paise=0).tolerance_floor_paise == 0


def test_confidence_bands_must_be_ordered_human_review_below_auto_accept() -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(confidence_auto_accept=0.5, confidence_human_review=0.6)


def test_confidence_bands_equal_is_accepted() -> None:
    """`<=` in the validator, not `<` — a degenerate but not contradictory
    config (every human-reviewed call is also auto-accepted) is legal."""
    config = CascadeConfig(confidence_auto_accept=0.7, confidence_human_review=0.7)
    assert config.confidence_auto_accept == config.confidence_human_review == 0.7


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_confidence_auto_accept_must_be_in_zero_one(bad_value: float) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(confidence_auto_accept=bad_value, confidence_human_review=0.0)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_confidence_human_review_must_be_in_zero_one(bad_value: float) -> None:
    with pytest.raises(ValidationError):
        CascadeConfig(confidence_human_review=bad_value)


def test_run_context_defaults_to_a_valid_cascade_config() -> None:
    ctx = RunContext(run_id="run_test", seed=1)
    assert ctx.config == CascadeConfig()
