"""jinja2 -> out/report.html from the audit DB. Static file, no server, no
CDN dependencies — everything (CSS included) is inlined into one HTML file
so it survives being emailed, screen-recorded, or opened straight off disk
on a judge's machine with no build step and no network access.

This module never touches data/ground_truth.json. Every number it renders
comes from `metrics.score()` (CLAUDE.md invariant 7) or directly from the
audit log via `unbatch.audit` (the audit log is the pipeline's own output,
not ground truth — reading it here is the same thing `unbatch exceptions`
already does). The one exception-shaped breakdown by ground-truth break
type (`exception_break_type_counts`, for the ablation framing in
METRICS.md) is computed inside metrics.py and only ever read here as an
already-computed MetricsReport field.

One `unbatch report` renders every arm that has actually been run
(`unbatch run --no-llm` / `unbatch run` / `unbatch run --llm-only ...`) —
each is looked up independently by its own `derive_run_id`, and an arm with
zero decisions in the audit log is rendered as "not yet run" rather than
faked as all-exceptions. Money fields are formatted to rupees only here,
at the report layer (CLAUDE.md invariant 1) — every other layer keeps paise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import jinja2

from unbatch import audit
from unbatch import forecast as forecast_module
from unbatch import generate as generate_module
from unbatch import metrics as metrics_module
from unbatch.metrics import MetricsReport
from unbatch.models import Decision, SettlementLineType
from unbatch.money import format_paise_to_rupees

DEFAULT_FORECAST_HORIZON_DAYS = 14

DEFAULT_OUT_PATH = Path("out/report.html")
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.jinja2"

# Ordered A -> B -> C to match METRICS.md's ablation table.
ARM_ORDER: tuple[str, ...] = ("no_llm", "with_llm", "llm_only")
ARM_LABELS: dict[str, str] = {
    "no_llm": "A — rules only",
    "with_llm": "B — rules + LLM",
    "llm_only": "C — LLM only",
}
ARM_COMMANDS: dict[str, str] = {
    "no_llm": "unbatch run --no-llm",
    "with_llm": "unbatch run",
    "llm_only": "unbatch run --llm-only",
}

# 10 fixed-width bands from 0.0 to 1.0 — coarse enough to read as a shape,
# fine enough to show real spread across the model's own reported values.
# Fed only the decisions that actually went through the adjudicator (see
# build_arm_views): L0-L3's confidences are fixed constants (1.00/0.98/
# 0.90/0.75), so including them piles almost every decision into one or
# two bands and the rest render as empty — not a chart bug, just a chart
# with nothing to show for stages whose confidence never varies.
_CONFIDENCE_BAND_WIDTH = 0.1
_CONFIDENCE_BAND_COUNT = 10

# Fixed L0 -> L4 order for the funnel — stage_funnel only contains keys for
# stages that resolved at least one credit (metrics.py), so arm C's l0-l3
# would otherwise be silently absent instead of shown as zero.
_FUNNEL_STAGE_ORDER: tuple[str, ...] = ("l0", "l1", "l2", "l3", "l4")


def format_rupees(paise: int) -> str:
    """Paise to a signed rupee string with a currency mark, e.g. 41833 ->
    "₹418.33", -500 -> "-₹5.00". The only place this module does money
    formatting — everywhere else in report.py, amounts stay in paise until
    the template needs to show one."""
    text = format_paise_to_rupees(paise)
    if text.startswith("-"):
        return f"-₹{text[1:]}"
    return f"₹{text}"


def format_percent(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


@dataclass
class ExceptionRow:
    credit_id: str
    stage: str
    reason: str
    rationale: str | None
    confidence: float
    delta_rupees: str


@dataclass
class ConfusionTable:
    """A break_reason confusion matrix laid out for a template: one column
    per predicted reason the model actually used (discovered from the data,
    not a fixed enum list — a model that never predicts a given reason
    shouldn't get an empty column), one row per ground-truth break type."""

    columns: list[str]
    rows: list[tuple[str, list[int]]]  # (ground_truth_break_type, counts per column)


@dataclass
class FunnelRow:
    """One stage's row in the funnel chart — count is always rendered as
    text next to the bar, never only inside it, so a stage resolving a
    handful of credits (L1/L3 on seed 42) is still legible next to L0's
    79 and doesn't depend on the bar being wide enough to hold a label."""

    stage: str
    count: int
    percent_of_total: float


def _funnel_rows(stage_funnel: dict[str, int], total_credits: int) -> list[FunnelRow]:
    """Every cascade stage in fixed L0 -> L4 order. metrics.py's
    stage_funnel only has a key for a stage that resolved at least one
    credit, so arm C's l0-l3 (it skips straight to L4) would otherwise be
    silently absent instead of shown as zero."""
    return [
        FunnelRow(
            stage=stage,
            count=stage_funnel.get(stage, 0),
            percent_of_total=(
                stage_funnel.get(stage, 0) / total_credits * 100 if total_credits else 0.0
            ),
        )
        for stage in _FUNNEL_STAGE_ORDER
    ]


@dataclass
class ArmView:
    """Everything the template needs for one ablation arm. `ran=False`
    means this arm's run_id has zero decisions in the audit log — rendered
    as "not yet run", never faked as all-exceptions."""

    key: str
    label: str
    command: str
    run_id: str
    ran: bool
    metrics: MetricsReport | None = None
    exceptions: list[ExceptionRow] = field(default_factory=list)
    funnel_rows: list[FunnelRow] = field(default_factory=list)
    confidence_histogram: list[tuple[str, int]] = field(default_factory=list)
    confidence_histogram_max: int = 1
    confusion_table: ConfusionTable | None = None
    llm_cost_rupees: str = "₹0.00"
    cost_per_adjudicated_credit_paise: str = "0.00"
    cost_per_exception_paise: str = "0.00"
    has_llm_calls: bool = False
    # llm_call_count minus adjudication_failed_count: credits that actually
    # received a usable break_reason, the denominator break_reason_accuracy
    # is computed against. Shown alongside the accuracy figure so a small
    # sample (e.g. 83.3% on 6) is never read as a precise measurement.
    classified_count: int = 0


@dataclass
class AmbiguityFraming:
    """D3's required framing, computed from arm A's own exceptions so the
    prose in the report always matches whatever the actual dataset is —
    never hardcoded counts. See METRICS.md § the ablation."""

    total_exceptions: int
    ambiguous_composition: int
    tolerance_ambiguous: int
    unrelated_credit: int
    other: int

    @property
    def correctly_declined(self) -> int:
        return self.ambiguous_composition + self.tolerance_ambiguous


def _confidence_histogram(decisions: list[Decision]) -> list[tuple[str, int]]:
    counts = [0] * _CONFIDENCE_BAND_COUNT
    for decision in decisions:
        band = int(decision.confidence / _CONFIDENCE_BAND_WIDTH)
        band = min(band, _CONFIDENCE_BAND_COUNT - 1)  # confidence == 1.0 lands in the top band
        counts[band] += 1
    labels = [
        f"{i * _CONFIDENCE_BAND_WIDTH:.1f}–{(i + 1) * _CONFIDENCE_BAND_WIDTH:.1f}"
        for i in range(_CONFIDENCE_BAND_COUNT)
    ]
    return list(zip(labels, counts, strict=True))


def _exception_rows(decisions: list[Decision]) -> list[ExceptionRow]:
    return [
        ExceptionRow(
            credit_id=d.credit_id,
            stage=d.stage.value,
            reason=d.reason,
            rationale=d.rationale,
            confidence=d.confidence,
            delta_rupees=format_rupees(d.delta_paise),
        )
        for d in decisions
    ]


def _confusion_table(confusion: dict[str, dict[str, int]]) -> ConfusionTable | None:
    if not confusion:
        return None
    columns = sorted({predicted for by_actual in confusion.values() for predicted in by_actual})
    rows = [
        (actual, [confusion[actual].get(column, 0) for column in columns])
        for actual in sorted(confusion)
    ]
    return ConfusionTable(columns=columns, rows=rows)


def _ambiguity_framing(report: MetricsReport) -> AmbiguityFraming:
    counts = report.exception_break_type_counts
    ambiguous = counts.get("ambiguous_composition", 0)
    tolerance = counts.get("tolerance_ambiguous", 0)
    unrelated = counts.get("unrelated_credit", 0)
    total = sum(counts.values())
    return AmbiguityFraming(
        total_exceptions=total,
        ambiguous_composition=ambiguous,
        tolerance_ambiguous=tolerance,
        unrelated_credit=unrelated,
        other=total - ambiguous - tolerance - unrelated,
    )


@dataclass
class ForecastDayView:
    """One day of the forecast table, rupee-formatted for display."""

    date: str
    expected_rupees: str
    low_rupees: str
    high_rupees: str
    payment_count: int


@dataclass
class ForecastView:
    """The forward cash forecast — a separate loop from reconciliation,
    over the same settlement report. See forecast.py's module docstring
    for the method and bench_forecast.json for the measured backtest this
    section's caveat text quotes."""

    as_of: str
    horizon_days: int
    daily: list[ForecastDayView]
    total_expected_rupees: str
    total_low_rupees: str
    total_high_rupees: str
    unsettled_payment_count: int
    historical_payment_count: int
    historical_deviation_stdev: float


def build_forecast_view(
    data_dir: Path, *, horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS
) -> ForecastView | None:
    """Run the forecaster against `data_dir`'s own settlement report, same
    default `as_of` as `unbatch forecast` itself (last settled date minus
    the horizon). Returns None only if there's no settled payment at all
    to forecast from — an empty data_dir, not a normal outcome."""
    orders = generate_module.read_order_ledger_csv(data_dir / "order_ledger.csv")
    settlements = generate_module.read_settlement_report_csv(data_dir / "settlement_report.csv")
    last_settled = max(
        (
            line.settled_at.date()
            for line in settlements
            if line.type == SettlementLineType.PAYMENT
        ),
        default=None,
    )
    if last_settled is None:
        return None
    as_of = last_settled - timedelta(days=horizon_days)
    forecast_report = forecast_module.forecast(
        orders, settlements, as_of=as_of, horizon_days=horizon_days
    )
    return ForecastView(
        as_of=forecast_report.as_of.isoformat(),
        horizon_days=forecast_report.horizon_days,
        daily=[
            ForecastDayView(
                date=day.date.isoformat(),
                expected_rupees=format_paise_to_rupees(day.expected_paise),
                low_rupees=format_paise_to_rupees(day.low_paise),
                high_rupees=format_paise_to_rupees(day.high_paise),
                payment_count=day.payment_count,
            )
            for day in forecast_report.daily
        ],
        total_expected_rupees=format_paise_to_rupees(forecast_report.total_expected_paise),
        total_low_rupees=format_paise_to_rupees(forecast_report.total_low_paise),
        total_high_rupees=format_paise_to_rupees(forecast_report.total_high_paise),
        unsettled_payment_count=forecast_report.unsettled_payment_count,
        historical_payment_count=forecast_report.historical_payment_count,
        historical_deviation_stdev=forecast_report.historical_deviation_stdev,
    )


def build_arm_views(
    seed: int, data_dir: Path, db: Path
) -> dict[str, ArmView]:
    """Score every arm that has data in the audit log; arms with none are
    still returned (ran=False) so the template can render them as pending
    rather than silently omit them."""
    conn = audit.connect(db)
    views: dict[str, ArmView] = {}
    for arm in ARM_ORDER:
        run_id = audit.derive_run_id(seed, data_dir, arm=arm)
        decisions = audit.fetch_decisions(conn, run_id)
        if not decisions:
            views[arm] = ArmView(
                key=arm, label=ARM_LABELS[arm], command=ARM_COMMANDS[arm], run_id=run_id, ran=False
            )
            continue

        report = metrics_module.score(conn, run_id, data_dir=data_dir)
        # Only decisions that actually went through the adjudicator carry a
        # model name (l4_llm.py is the sole call site that sets llm_model —
        # the --no-llm terminal exception explicitly leaves it None), so
        # this excludes L0-L3's fixed confidences and arm A's declined
        # items rather than letting them swamp the one stage whose
        # confidence is genuinely model-reported.
        adjudicated_decisions = [d for d in decisions if d.llm_model is not None]
        histogram = _confidence_histogram(adjudicated_decisions)
        views[arm] = ArmView(
            key=arm,
            label=ARM_LABELS[arm],
            command=ARM_COMMANDS[arm],
            run_id=run_id,
            ran=True,
            metrics=report,
            exceptions=_exception_rows(audit.fetch_exceptions(conn, run_id)),
            funnel_rows=_funnel_rows(report.stage_funnel, report.total_credits),
            confidence_histogram=histogram,
            confidence_histogram_max=max((count for _label, count in histogram), default=1) or 1,
            confusion_table=_confusion_table(report.break_reason_confusion),
            llm_cost_rupees=format_rupees(report.llm_cost_paise),
            cost_per_adjudicated_credit_paise=f"{report.cost_paise_per_adjudicated_credit:.2f}",
            cost_per_exception_paise=f"{report.cost_paise_per_exception:.2f}",
            has_llm_calls=report.llm_call_count > 0,
            classified_count=report.llm_call_count - report.adjudication_failed_count,
        )
    return views


def render(
    seed: int = 42,
    *,
    data_dir: Path = generate_module.DEFAULT_OUT_DIR,
    db: Path = audit.DEFAULT_DB_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
) -> Path:
    """Render out/report.html (or `out_path`) from whichever arms have been
    run against `data_dir`'s seed-42 fixtures. Returns the path written."""
    arms = build_arm_views(seed, data_dir, db)
    forecast_view = build_forecast_view(data_dir)

    framing = None
    if arms["no_llm"].ran and arms["no_llm"].metrics is not None:
        framing = _ambiguity_framing(arms["no_llm"].metrics)

    delta_b_minus_a = None
    if arms["no_llm"].ran and arms["with_llm"].ran:
        a_metrics = arms["no_llm"].metrics
        b_metrics = arms["with_llm"].metrics
        assert a_metrics is not None and b_metrics is not None
        delta_b_minus_a = b_metrics.count_match_rate - a_metrics.count_match_rate

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    env.filters["percent"] = format_percent
    template = env.get_template(TEMPLATE_NAME)
    html = template.render(
        seed=seed,
        arm_order=ARM_ORDER,
        arms=arms,
        framing=framing,
        delta_b_minus_a=delta_b_minus_a,
        forecast=forecast_view,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8", newline="")
    return out_path
