"""Renders out/report.html from a MetricsReport via jinja2. Static file, no
server — zero runtime risk during the pitch video, nothing to break on a
judge's machine.

Contents: stage funnel, count and value-weighted match rates, false-match
rate, confidence distribution, the full exception table with reasons (no
truncation), and the rules-only vs rules+LLM vs LLM-only comparison
(METRICS.md § the ablation).
"""

from __future__ import annotations

from pathlib import Path

from unbatch.metrics import MetricsReport

DEFAULT_OUT_PATH = Path("out/report.html")
TEMPLATE_NAME = "report.html.j2"


def render(report: MetricsReport, out_path: Path = DEFAULT_OUT_PATH) -> None:
    """Render the HTML report for one run's metrics to `out_path`."""
    raise NotImplementedError
