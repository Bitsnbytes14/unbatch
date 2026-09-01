"""forecast.py: a projection over the existing settlement report, fit only
on data already settled by `as_of`. Pure arithmetic — no LLM, no new data
source, no randomness of its own to seed (see the module docstring)."""

from __future__ import annotations

from datetime import date, datetime

from unbatch import fees
from unbatch.forecast import forecast
from unbatch.models import (
    OrderLedgerRecord,
    OrderStatus,
    PaymentMethod,
    SettlementLine,
    SettlementLineType,
)

AS_OF = date(2024, 1, 10)


def _order(
    payment_id: str,
    amount: int,
    captured_date: date,
    *,
    method: PaymentMethod = PaymentMethod.CARD,
    status: OrderStatus = OrderStatus.CAPTURED,
) -> OrderLedgerRecord:
    return OrderLedgerRecord(
        order_id=f"order_{payment_id}",
        payment_id=payment_id,
        amount_paise=amount,
        currency="INR",
        status=status,
        captured_at=datetime.combine(captured_date, datetime.min.time()),
        customer_ref="cust_1",
        method=method,
    )


def _settlement_line(
    payment_id: str,
    gross: int,
    settled_date: date,
    *,
    method: PaymentMethod = PaymentMethod.CARD,
    net_override: int | None = None,
) -> SettlementLine:
    fee = fees.compute_fee_paise(gross, method)
    tax = fees.compute_tax_paise(fee)
    net = fees.compute_net_paise(gross, fee, tax) if net_override is None else net_override
    return SettlementLine(
        settlement_id=f"setl_{payment_id}",
        settlement_utr="AXISP123456789012",
        payment_id=payment_id,
        type=SettlementLineType.PAYMENT,
        gross_paise=gross,
        fee_paise=fee,
        tax_paise=tax,
        net_paise=net,
        settled_at=datetime.combine(settled_date, datetime.min.time()),
    )


def test_an_already_settled_order_is_not_projected() -> None:
    order = _order("pay_1", 100_000, date(2024, 1, 8))
    line = _settlement_line("pay_1", 100_000, date(2024, 1, 9))  # settled before as_of

    report = forecast([order], [line], as_of=AS_OF, horizon_days=7)

    assert report.unsettled_payment_count == 0
    assert report.total_expected_paise == 0


def test_an_unsettled_order_is_projected_at_the_fee_schedule_net() -> None:
    # historical fit: one already-settled T+1 payment, so lag_mode=1
    historical_order = _order("pay_hist", 50_000, date(2024, 1, 5))
    historical_line = _settlement_line("pay_hist", 50_000, date(2024, 1, 6))

    unsettled_order = _order("pay_2", 200_000, date(2024, 1, 10))  # captured on as_of itself

    report = forecast(
        [historical_order, unsettled_order],
        [historical_line],
        as_of=AS_OF,
        horizon_days=7,
    )

    assert report.unsettled_payment_count == 1
    expected_day = date(2024, 1, 11)  # captured_at + lag_mode(1)
    [day] = [d for d in report.daily if d.payment_count == 1]
    assert day.date == expected_day

    fee = fees.compute_fee_paise(200_000, PaymentMethod.CARD)
    tax = fees.compute_tax_paise(fee)
    expected_net = fees.compute_net_paise(200_000, fee, tax)
    assert day.expected_paise == expected_net
    assert report.total_expected_paise == expected_net


def test_lag_mode_is_the_most_common_historical_lag() -> None:
    """Three T+2 lags, one T+1 -> mode is 2. Captured exactly on as_of, so
    the natural projection (as_of + 2) actually lands past the first
    horizon day and isn't masked by the overdue floor — unlike lag_mode=1,
    where every unsettled order (captured_at <= as_of by definition)
    floors to day 1 regardless, since captured_at + 1 can never exceed
    as_of + 1. See test_overdue_order_floors_to_the_first_horizon_day and
    forecast.py's docstring on why day 1 dominates on real, T+1-heavy
    data."""
    pairs = [
        ("pay_a", date(2024, 1, 1), date(2024, 1, 3)),
        ("pay_b", date(2024, 1, 2), date(2024, 1, 4)),
        ("pay_c", date(2024, 1, 3), date(2024, 1, 5)),
        ("pay_d", date(2024, 1, 4), date(2024, 1, 5)),  # T+1
    ]
    orders = [_order(pid, 10_000, captured) for pid, captured, _settled in pairs]
    lines = [_settlement_line(pid, 10_000, settled) for pid, _captured, settled in pairs]

    unsettled = _order("pay_new", 10_000, AS_OF)
    report = forecast([*orders, unsettled], lines, as_of=AS_OF, horizon_days=5)

    assert report.historical_lag_distribution == {"1": 1, "2": 3}
    [day] = [d for d in report.daily if d.payment_count == 1]
    assert day.date == date(2024, 1, 12)  # as_of + mode lag (2), not floored


def test_every_unsettled_order_lands_on_day_one_when_lag_mode_is_one() -> None:
    """A provable structural property, not an empirical curiosity: 'unsettled
    as of as_of' means captured_at <= as_of by definition, so
    captured_at + 1 can never exceed as_of + 1 (the first horizon day) —
    every such order floors to day 1 whenever the fitted lag mode is 1,
    regardless of how many days before as_of it was actually captured.
    This is the whole reason a T+1-dominated business has essentially no
    multi-day forward visibility from settlement data alone."""
    historical_order = _order("pay_hist", 50_000, date(2024, 1, 5))
    historical_line = _settlement_line("pay_hist", 50_000, date(2024, 1, 6))  # T+1

    captured_today = _order("pay_today", 10_000, AS_OF)
    captured_a_week_ago = _order("pay_old", 10_000, date(2024, 1, 3))

    report = forecast(
        [historical_order, captured_today, captured_a_week_ago],
        [historical_line],
        as_of=AS_OF,
        horizon_days=7,
    )

    assert report.historical_lag_distribution == {"1": 1}
    assert report.unsettled_payment_count == 2
    populated_days = {d.date for d in report.daily if d.payment_count > 0}
    assert populated_days == {date(2024, 1, 11)}  # as_of + 1, both orders
    [day_one] = [d for d in report.daily if d.payment_count > 0]
    assert day_one.payment_count == 2


def test_overdue_order_floors_to_the_first_horizon_day() -> None:
    """An unsettled order whose naive captured_at + lag_mode already falls
    on or before as_of (it's running late) must not be dropped or placed
    in the past — it lands on the first day of the horizon instead."""
    historical_order = _order("pay_hist", 50_000, date(2024, 1, 5))
    historical_line = _settlement_line("pay_hist", 50_000, date(2024, 1, 6))

    overdue_order = _order("pay_overdue", 30_000, date(2023, 12, 20))  # long overdue

    report = forecast(
        [historical_order, overdue_order], [historical_line], as_of=AS_OF, horizon_days=7
    )

    assert report.unsettled_payment_count == 1
    [day] = [d for d in report.daily if d.payment_count == 1]
    assert day.date == date(2024, 1, 11)  # as_of + 1, the first horizon day


def test_an_order_settling_beyond_the_horizon_is_excluded() -> None:
    historical_order = _order("pay_hist", 50_000, date(2024, 1, 5))
    historical_line = _settlement_line("pay_hist", 50_000, date(2024, 1, 6))
    # captured right at as_of with lag_mode=1 -> would land at as_of+1, fine
    # but with a short horizon of 0 days there's nowhere for it to land
    far_order = _order("pay_far", 20_000, AS_OF)

    report = forecast(
        [historical_order, far_order], [historical_line], as_of=AS_OF, horizon_days=0
    )

    assert report.daily == []
    assert report.total_expected_paise == 0


def test_deviation_stdev_widens_the_band() -> None:
    # one historical line whose actual net deviates from the fee-schedule
    # theoretical net by a known amount
    gross = 100_000
    fee = fees.compute_fee_paise(gross, PaymentMethod.CARD)
    tax = fees.compute_tax_paise(fee)
    theoretical_net = fees.compute_net_paise(gross, fee, tax)
    actual_net = theoretical_net - 1_000  # a deliberate deviation

    deviating_order = _order("pay_dev", gross, date(2024, 1, 5))
    deviating_line = _settlement_line(
        "pay_dev", gross, date(2024, 1, 6), net_override=actual_net
    )
    # a second historical line at zero deviation so pstdev has 2 points
    clean_order = _order("pay_clean", gross, date(2024, 1, 5))
    clean_line = _settlement_line("pay_clean", gross, date(2024, 1, 6))

    unsettled = _order("pay_new", gross, date(2024, 1, 10))
    report = forecast(
        [deviating_order, clean_order, unsettled],
        [deviating_line, clean_line],
        as_of=AS_OF,
        horizon_days=5,
    )

    assert report.historical_deviation_stdev > 0.0
    [day] = [d for d in report.daily if d.payment_count == 1]
    assert day.low_paise < day.expected_paise < day.high_paise


def test_no_historical_data_falls_back_to_default_lag_and_zero_band() -> None:
    unsettled = _order("pay_new", 50_000, date(2024, 1, 10))
    report = forecast([unsettled], [], as_of=AS_OF, horizon_days=5)

    assert report.historical_payment_count == 0
    assert report.historical_deviation_stdev == 0.0
    [day] = [d for d in report.daily if d.payment_count == 1]
    assert day.date == date(2024, 1, 11)  # default T+1
    assert day.low_paise == day.expected_paise == day.high_paise


def test_refunded_and_failed_orders_are_never_projected() -> None:
    refunded = _order("pay_r", 10_000, date(2024, 1, 9), status=OrderStatus.REFUNDED)
    failed = _order("pay_f", 10_000, date(2024, 1, 9), status=OrderStatus.FAILED)

    report = forecast([refunded, failed], [], as_of=AS_OF, horizon_days=5)

    assert report.unsettled_payment_count == 0
    assert report.total_expected_paise == 0


def test_amounts_stay_int_paise() -> None:
    unsettled = _order("pay_new", 123_457, date(2024, 1, 10))
    report = forecast([unsettled], [], as_of=AS_OF, horizon_days=5)

    for day in report.daily:
        assert isinstance(day.expected_paise, int)
        assert isinstance(day.low_paise, int)
        assert isinstance(day.high_paise, int)
