"""report.py's forecast section (E14c): build_forecast_view wraps
forecast.py's output for the template, and render() includes it as its
own section, separate from the reconciliation arms."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from unbatch import audit, generate, report
from unbatch.forecast import forecast as run_forecast
from unbatch.models import SettlementLineType
from unbatch.money import format_paise_to_rupees

SEED = 42


def test_build_forecast_view_matches_the_forecast_module(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate.generate(SEED, out_dir=data_dir)

    view = report.build_forecast_view(data_dir, horizon_days=14)
    assert view is not None

    orders = generate.read_order_ledger_csv(data_dir / "order_ledger.csv")
    settlements = generate.read_settlement_report_csv(data_dir / "settlement_report.csv")
    last_settled = max(
        line.settled_at.date() for line in settlements if line.type == SettlementLineType.PAYMENT
    )
    as_of = last_settled - timedelta(days=14)
    raw = run_forecast(orders, settlements, as_of=as_of, horizon_days=14)

    assert view.as_of == raw.as_of.isoformat()
    assert view.horizon_days == raw.horizon_days
    assert view.unsettled_payment_count == raw.unsettled_payment_count
    assert view.total_expected_rupees == format_paise_to_rupees(raw.total_expected_paise)
    assert len(view.daily) == len(raw.daily)
    assert view.daily[0].date == raw.daily[0].date.isoformat()
    assert view.daily[0].expected_rupees == format_paise_to_rupees(raw.daily[0].expected_paise)
    assert view.daily[0].payment_count == raw.daily[0].payment_count


def test_build_forecast_view_returns_none_with_no_settled_payments(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "order_ledger.csv").write_text(
        "order_id,payment_id,amount,currency,status,captured_at,customer_ref,method\n",
        encoding="utf-8",
        newline="",
    )
    (data_dir / "settlement_report.csv").write_text(
        "settlement_id,settlement_utr,payment_id,type,gross,fee,tax,net,settled_at\n",
        encoding="utf-8",
        newline="",
    )

    assert report.build_forecast_view(data_dir) is None


def test_render_includes_the_forecast_section(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate.generate(SEED, out_dir=data_dir)
    db_path = tmp_path / "audit.db"
    audit.connect(db_path)  # empty audit log — the forecast section doesn't need decisions

    out_path = report.render(SEED, data_dir=data_dir, db=db_path, out_path=tmp_path / "report.html")
    html = out_path.read_text(encoding="utf-8")

    assert "Forward cash forecast" in html
    assert "unbatch forecast --horizon 14" in html
    assert "Measured limitation, not a bug" in html
    assert "Daily breakdown" in html
