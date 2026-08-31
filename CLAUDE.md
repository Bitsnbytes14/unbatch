# CLAUDE.md

Context for Claude Code. Read this before any task. Keep it under 200 lines — it loads every session.

## What this is

`unbatch` — settlement reconciliation agent. Matches **bank credit lines** against **expected settlement batches** derived from an **order ledger**. Many-to-one: one bank credit = many payments, minus fees, tax, refunds, chargebacks.

Built for the Razorpay AI Buildathon, Track 04 (AI Finance Controller). Judged on: problem taste, build quality, AI judgment, failure recovery. Deadline is hard.

## Non-negotiable invariants

1. **Money is `int` paise. Never float. Never `Decimal` in storage.** Parse to paise at the boundary, format to rupees only in the report layer. A float in a money path is a bug even if tests pass.
2. **The LLM never does arithmetic.** Rules compute deltas and compose batches. The model classifies *why* a break happened and proposes a resolution. If a prompt contains "calculate", "sum", or "compute", it is wrong.
3. **The LLM never sees the full batch.** Only unresolved items that reached L4. If the call count approaches the record count, the cascade is broken.
4. **Bias to exception over wrong match.** A false match silently corrupts books; an exception costs ten minutes. Optimise false-match rate toward zero, not match rate toward 100%.
5. **Every decision writes an audit row.** No stage may resolve an item without a `Decision` in SQLite. The exception report is a *query over that table*, never a separate list.
6. **Runs are reproducible.** Seeded data, seeded run_id, cached LLM responses committed to the repo. `--cached` must reproduce reported metrics with no API key.
7. **Ground truth is never read by the pipeline.** Only by `metrics.py`. If a stage imports ground truth, that is a scoring leak.

## Stack

- Python 3.11+, `uv` for deps and running
- pandas, rapidfuzz, pydantic v2, typer, jinja2, pytest
- SQLite (stdlib `sqlite3`) for the audit log
- OpenAI SDK, `gpt-5-nano`, strict `response_format` JSON schema, validated by pydantic (D0.5 — swapped from Anthropic/claude-sonnet-5 when only an OpenAI key was available; see adjudicator.py's module docstring)

Not used, deliberately: Docker, Postgres, LangChain/LlamaIndex, embeddings/vector DB, FastAPI, Streamlit, any frontend framework. Do not add dependencies without asking.

## Commands

```bash
uv sync                          # install
uv run unbatch generate --seed 42  # write data/ fixtures + ground truth
uv run unbatch run --seed 42       # full cascade, hits API on L4 misses
uv run unbatch run --cached        # replay from cache/, no API key needed
uv run unbatch run --no-llm        # deterministic baseline arm
uv run unbatch report              # regenerate out/report.html from audit.db
uv run unbatch exceptions          # print unresolved items + reasons
uv run unbatch exceptions --export out/exceptions.csv  # same, as an analyst work item
uv run unbatch bench --seeds 42,43,44,45,46,47  # rules-only metric stability across seeds
uv run unbatch bench --scale 5000  # rules-only cascade throughput at scale
uv run unbatch bench --noise 0.0,0.1,0.25,0.5,0.75,1.0  # degradation curve under narration noise
uv run unbatch generate --adversarial  # write a same-scale, deliberately hostile dataset
uv run unbatch bench --adversarial     # measure and publish the worst case on it
uv run pytest -q
```

## Layout

```
src/unbatch/
  models.py        pydantic schemas — the contract, read first
  money.py         paise parsing/formatting — the money-handling boundary
  fees.py          per-method fee rates, GST, net; the rounding decision
  generate.py      seeded synthetic data + ground_truth.json
  compose.py       batch composition / bounded subset-sum
  stages/          l0_utr.py l1_exact.py l2_compose.py l3_tolerance.py l4_llm.py
  adjudicator.py   LLM boundary: prompt, cache, validate, degrade
  audit.py         SQLite decision log
  metrics.py       scoring vs ground truth — ONLY file that reads it
  report.py        jinja2 -> out/report.html
  templates/       report.html.jinja2
  cli.py           typer entrypoints
tests/
data/              generated fixtures, committed
cache/             LLM responses keyed by prompt hash, committed
out/               report.html, audit.db — gitignored
```

See `ARCHITECTURE.md` for the stage cascade, `DATA_SPEC.md` for record shapes and injected break types, `METRICS.md` for what we report.

## Conventions

- Every stage is a pure function: `(unresolved: list[Item], ctx: Context) -> list[Decision]`. Stages never mutate global state and never call each other.
- Stages are ordered cheapest-and-most-certain first. Never reorder without updating `ARCHITECTURE.md`.
- All LLM output goes through a pydantic model. A `ValidationError` is a handled outcome (retry once, then escalate to human review), never a crash.
- Type hints everywhere. `ruff` clean.
- Tests live beside behaviour, not implementation. Test the cascade's outcomes, not internal call order.
- Commit messages: imperative, one line, no emoji.

## Failure logging — do this every time

When something breaks — a subset-sum blowup, malformed model JSON, a date-window off-by-one, a rounding mismatch — append two lines to `FAILURES.md` **at the moment it happens**, before fixing it. Format is in that file.

This is not housekeeping. The application form asks "what broke, and how you got out" and states it is the first thing they read. A log written live reads completely differently from one reconstructed at the end. Do not skip it, do not batch it up, do not let me skip it.

## Working style

- Ask before adding a dependency, changing an invariant above, or restructuring `models.py`.
- Prefer boring code. This gets read by a hiring panel; cleverness that needs a paragraph of explanation is a liability.
- If a task is ambiguous, state the assumption in one line and proceed — don't stall.
- Do not write README.md yet. It gets written last, once real numbers exist.

## Commit conventions

This repo's history is part of the submission — commit like it will be read.

- Commit in logical units, not one dump. Split scaffold/tooling, package skeleton, docs, and CLI changes into separate commits even within one session.
- Messages: imperative mood, lowercase, under 60 chars, no emoji, no "Generated with Claude Code" footer, no Co-Authored-By trailer.
- Never `git add -A` blindly; stage files deliberately by name.
- Push after each completed milestone, not after every commit.
