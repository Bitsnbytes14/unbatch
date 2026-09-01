"""Property-based tests for the matching engine (compose.py / l2_compose.py).

The example-based tests in test_compose.py and test_l2_compose.py pin down
specific, hand-picked shapes. These properties hold for *every* input in
their domain, which is the guarantee CLAUDE.md invariant 4 (bias to
exception over wrong match) actually needs: no example-based test can prove
a false match never slips through, but a property that holds under
hundreds of generated inputs is much closer to that proof."""

from __future__ import annotations

from datetime import UTC, date, datetime

from hypothesis import given
from hypothesis import strategies as st

from unbatch.compose import compose
from unbatch.models import (
    BankStatementRecord,
    RunContext,
    SettlementLine,
    SettlementLineType,
    UnresolvedCredit,
)
from unbatch.stages import l2_compose

CTX = RunContext(run_id="run_test", seed=1)


def _line(net_paise: int, line_id: str, day: date = date(2024, 1, 5)) -> SettlementLine:
    return SettlementLine(
        settlement_id=f"setl_{line_id}",
        settlement_utr="AXISP000000000001",
        payment_id=f"pay_{line_id}",
        type=SettlementLineType.PAYMENT,
        gross_paise=net_paise,
        fee_paise=0,
        tax_paise=0,
        net_paise=net_paise,
        settled_at=datetime(day.year, day.month, day.day, tzinfo=UTC),
    )


_values = st.lists(st.integers(min_value=1, max_value=5_000), min_size=1, max_size=8)


@given(values=_values, data=st.data())
def test_a_constructed_matching_subset_is_always_found(
    values: list[int], data: st.DataObject
) -> None:
    """A batch built to exactly equal a credit must be found."""
    candidates = [_line(v, str(i)) for i, v in enumerate(values)]
    subset_indices = data.draw(
        st.lists(st.sampled_from(range(len(values))), min_size=1, max_size=len(values), unique=True)
    )
    target = sum(values[i] for i in subset_indices)
    expected = frozenset(f"pay_{i}" for i in subset_indices)

    result = compose(target, candidates)

    found = {frozenset(line.payment_id for line in subset) for subset in result}
    assert expected in found


@given(values=st.lists(st.integers(min_value=1, max_value=5_000), min_size=0, max_size=10))
def test_a_negative_target_is_never_composed(values: list[int]) -> None:
    """If no valid composition exists, no match may be produced. Every
    candidate's net is strictly positive, so no non-empty subset can ever
    sum to a negative target — this is true by construction, not luck."""
    candidates = [_line(v, str(i)) for i, v in enumerate(values)]
    assert compose(-1, candidates) == []


@given(value=st.integers(min_value=1, max_value=100_000))
def test_two_equally_valid_compositions_are_both_returned(value: int) -> None:
    """If two exact compositions exist, compose() must return both — never
    silently pick a winner. (What happens with that ambiguity is the
    caller's decision — see test_two_equally_valid_compositions_never_resolve_at_l2.)"""
    line_a = _line(value, "a")
    line_b = _line(value, "b")

    result = compose(value, [line_a, line_b])

    found = {frozenset(line.payment_id for line in subset) for subset in result}
    assert {"pay_a"} in found
    assert {"pay_b"} in found


@given(value=st.integers(min_value=1, max_value=100_000))
def test_two_equally_valid_compositions_never_resolve_at_l2(value: int) -> None:
    """Bias to exception over wrong match: L2 must decline to pick between
    two equally valid exact compositions rather than silently choosing
    one — the credit must stay unresolved (an exception downstream), not
    resolve to whichever one happened to sort first."""
    line_a = _line(value, "a")
    line_b = _line(value, "b")
    credit = BankStatementRecord(
        txn_id="txn_1",
        value_date=date(2024, 1, 5),
        narration="test",
        credit_paise=value,
        debit_paise=None,
        balance_paise=value,
    )
    u = UnresolvedCredit(credit=credit, expected_batches=[], candidate_lines=[line_a, line_b])

    assert l2_compose.run([u], CTX) == []


@given(values=_values, data=st.data())
def test_permuting_candidates_does_not_change_the_compositions_found(
    values: list[int], data: st.DataObject
) -> None:
    """Permuting settlement lines must not change the decision — the
    algorithm's meet-in-the-middle split is index-based, so this is exactly
    the kind of bug an off-by-one in the split point would produce."""
    target = sum(values)
    indexed = list(enumerate(values))
    shuffled_indexed = data.draw(st.permutations(indexed))

    original = compose(target, [_line(v, str(i)) for i, v in indexed])
    permuted = compose(target, [_line(v, str(i)) for i, v in shuffled_indexed])

    original_sets = {frozenset(line.payment_id for line in subset) for subset in original}
    permuted_sets = {frozenset(line.payment_id for line in subset) for subset in permuted}
    assert original_sets == permuted_sets


@given(values=_values)
def test_repeated_execution_on_the_same_input_gives_the_same_result(values: list[int]) -> None:
    """Repeated execution on the same input must give the same result —
    no reliance on set/dict iteration order or other incidental state."""
    candidates = [_line(v, str(i)) for i, v in enumerate(values)]
    target = sum(values)

    first = compose(target, candidates)
    second = compose(target, candidates)

    first_sets = [frozenset(line.payment_id for line in subset) for subset in first]
    second_sets = [frozenset(line.payment_id for line in subset) for subset in second]
    assert first_sets == second_sets
