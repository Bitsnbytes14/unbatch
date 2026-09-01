"""Forward cash forecasting — a projection over the existing settlement
report and order ledger, not a new data source. No new dependency: this
module imports only `unbatch.fees` and the stdlib (`statistics`,
`collections.Counter`, `datetime`).

**This is arithmetic, not judgment — it never touches the adjudicator.**
CLAUDE.md invariant 2 says the LLM classifies *why* a break happened; a
forecast answers *when* and *how much* money is expected to arrive, which
is a computation over historical timing and fee data, with a single
correct procedure given the inputs. There is nothing here for a model to
judge, so `adjudicator` is not imported and no L4 call is possible from
this code path.

**Method:** fit two distributions from settlement lines already settled on
or before `as_of` — the fit set, exactly analogous to a train/test split:

1. **Settlement lag** (days from `captured_at` to `settled_at`) — the mode
   of the observed distribution is the point estimate for when an
   unsettled order will land (DATA_SPEC.md's generator draws T+1 three
   times as often as T+2, so the fit mode is 1 day on real data).
2. **Deviation from the fee schedule** — for each fit-set settlement line,
   how far its actual `net_paise` sits from what `fees.py`'s own
   `compute_fee_paise`/`compute_tax_paise`/`compute_net_paise` would
   compute for that gross and method, as a fraction of gross. This is the
   real residual — refund/chargeback risk, a fee-tier change, rounding —
   rather than an invented distribution: on the committed seed-42 data
   this is a population stdev of ~0.03%, because the fee schedule is
   deterministic and only one injected `fee_tier_change` line deviates
   from it at all. A forecaster that reports a wide band here would be
   overstating its own uncertainty; a narrow one is the honest finding.

Every unsettled captured order (an `OrderLedgerRecord` with no PAYMENT-type
settlement line dated on or before `as_of`) is projected onto
`max(captured_at + lag_mode, as_of + 1 day)` — the floor exists so an
already-overdue order lands on the first day of the horizon instead of
being silently dropped or placed in the past — using the exact fee-schedule
net as the point estimate and the fit deviation stdev (scaled by the
order's own gross) as the band width. Orders whose projected date falls
outside `[as_of + 1, as_of + horizon_days]` are excluded from that
forecast; a longer `--horizon` would include them.

Deterministic given its inputs — there is no randomness of its own to
seed; whatever variance appears in the output is `generate.py`'s own
seeded randomness, already baked into the committed settlement data by the
time this module ever runs.
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import date, timedelta

from unbatch import fees
from unbatch.models import (
    DailyForecast,
    ForecastReport,
    OrderLedgerRecord,
    OrderStatus,
    SettlementLine,
    SettlementLineType,
)

DEFAULT_LAG_DAYS = 1  # T+1 — used only when the fit set has no history at all


def _fit_pairs(
    orders_by_payment_id: dict[str, OrderLedgerRecord],
    settlements: list[SettlementLine],
    as_of: date,
) -> list[tuple[OrderLedgerRecord, SettlementLine]]:
    """Payments already settled on or before `as_of` — the only data this
    module is allowed to fit on. A backtest passes the full settlement
    list; this filter is what keeps it from peeking at the future."""
    pairs = []
    for line in settlements:
        if line.type != SettlementLineType.PAYMENT or line.settled_at.date() > as_of:
            continue
        order = orders_by_payment_id.get(line.payment_id)
        if order is not None:
            pairs.append((order, line))
    return pairs


def _lag_days(order: OrderLedgerRecord, line: SettlementLine) -> int:
    return (line.settled_at.date() - order.captured_at.date()).days


def _fit_lag_mode(pairs: list[tuple[OrderLedgerRecord, SettlementLine]]) -> int:
    if not pairs:
        return DEFAULT_LAG_DAYS
    counts = Counter(_lag_days(order, line) for order, line in pairs)
    # ties broken toward the shorter lag — the optimistic, sooner-money read
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _fit_deviation_stdev(pairs: list[tuple[OrderLedgerRecord, SettlementLine]]) -> float:
    deviations = []
    for order, line in pairs:
        if line.gross_paise == 0:
            continue
        fee = fees.compute_fee_paise(line.gross_paise, order.method)
        tax = fees.compute_tax_paise(fee)
        theoretical_net = fees.compute_net_paise(line.gross_paise, fee, tax)
        deviations.append((line.net_paise - theoretical_net) / line.gross_paise)
    if len(deviations) < 2:
        return 0.0
    return statistics.pstdev(deviations)


def _settled_payment_ids(settlements: list[SettlementLine], as_of: date) -> set[str]:
    return {
        line.payment_id
        for line in settlements
        if line.type == SettlementLineType.PAYMENT and line.settled_at.date() <= as_of
    }


def _unsettled_orders(
    orders: list[OrderLedgerRecord], settled_payment_ids: set[str], as_of: date
) -> list[OrderLedgerRecord]:
    return [
        order
        for order in orders
        if order.status == OrderStatus.CAPTURED
        and order.payment_id not in settled_payment_ids
        and order.captured_at.date() <= as_of
    ]


def forecast(
    orders: list[OrderLedgerRecord],
    settlements: list[SettlementLine],
    *,
    as_of: date,
    horizon_days: int,
) -> ForecastReport:
    """Project expected settlement inflows for `horizon_days` after
    `as_of`, fitting only on data already settled by `as_of`. See this
    module's docstring for the method."""
    orders_by_payment_id = {order.payment_id: order for order in orders}
    fit_pairs = _fit_pairs(orders_by_payment_id, settlements, as_of)
    lag_mode = _fit_lag_mode(fit_pairs)
    deviation_stdev = _fit_deviation_stdev(fit_pairs)
    lag_distribution = Counter(_lag_days(order, line) for order, line in fit_pairs)

    settled_ids = _settled_payment_ids(settlements, as_of)
    unsettled = _unsettled_orders(orders, settled_ids, as_of)

    horizon_dates = [as_of + timedelta(days=i) for i in range(1, horizon_days + 1)]
    earliest_horizon_date = horizon_dates[0] if horizon_dates else as_of
    daily_expected = dict.fromkeys(horizon_dates, 0)
    daily_low = dict.fromkeys(horizon_dates, 0)
    daily_high = dict.fromkeys(horizon_dates, 0)
    daily_count = dict.fromkeys(horizon_dates, 0)

    for order in unsettled:
        projected_date = max(
            order.captured_at.date() + timedelta(days=lag_mode), earliest_horizon_date
        )
        if projected_date not in daily_expected:
            continue  # settles beyond this horizon — a longer one would include it

        fee = fees.compute_fee_paise(order.amount_paise, order.method)
        tax = fees.compute_tax_paise(fee)
        expected_net = fees.compute_net_paise(order.amount_paise, fee, tax)
        band_width = round(order.amount_paise * deviation_stdev)

        daily_expected[projected_date] += expected_net
        daily_low[projected_date] += expected_net - band_width
        daily_high[projected_date] += expected_net + band_width
        daily_count[projected_date] += 1

    daily = [
        DailyForecast(
            date=d,
            expected_paise=daily_expected[d],
            low_paise=daily_low[d],
            high_paise=daily_high[d],
            payment_count=daily_count[d],
        )
        for d in horizon_dates
    ]

    return ForecastReport(
        as_of=as_of,
        horizon_days=horizon_days,
        daily=daily,
        total_expected_paise=sum(daily_expected.values()),
        total_low_paise=sum(daily_low.values()),
        total_high_paise=sum(daily_high.values()),
        unsettled_payment_count=len(unsettled),
        historical_payment_count=len(fit_pairs),
        historical_lag_distribution={
            str(k): v for k, v in sorted(lag_distribution.items())
        },
        historical_deviation_stdev=deviation_stdev,
    )
