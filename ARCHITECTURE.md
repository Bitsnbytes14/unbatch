# ARCHITECTURE.md

## The problem

A payment gateway does not pay a merchant per transaction. It **batches**: many captured payments over a cutoff window, minus per-transaction fees, minus GST on those fees, minus refunds issued in the window, minus chargebacks and their fees, settled T+1 or T+2 as **one lump credit** into the merchant's bank account with a single UTR.

The merchant's finance team sees a bank statement line for ₹4,18,332 and has to answer: which orders is this, and why isn't it the ₹4,19,107 we expected?

That is the loop this closes. Three sources, many-to-one, real money math.

```
order_ledger.csv     ──┐
                       ├──> expected settlement batches ──┐
settlement_report.csv ─┘                                  ├──> matched / exception
                                                          │
bank_statement.csv ───────────> bank credit lines ────────┘
```

Why this pair and not invoice-vs-payment 1:1: 1:1 matching is a solved problem and leaves the model nothing real to do. Many-to-one produces genuine ambiguity, which is where AI judgment can be demonstrated instead of asserted.

## The cascade

Ordered cheapest-and-most-certain first. Each stage consumes only what the previous stages left unresolved.

| Stage | Method | Resolves | Confidence |
|---|---|---|---|
| **L0** | UTR exact — settlement UTR appears in bank narration | bulk of clean cases | 1.00 |
| **L1** | Amount + date exact — credit ties exactly to a computed batch net | clean cases with mangled narration | 0.98 |
| **L2** | Batch composition — find the payment subset that composes the credit | splits, partial settlements | 0.90 |
| **L3** | Tolerance — credit checked directly against date-windowed expected batches, delta recorded | fee-tier drift, rounding | 0.75 |
| **L4** | LLM adjudication — classify the break, propose resolution | everything left | model-reported |

Anything L4 cannot resolve with confidence lands in the exception list. That is a **success outcome**, not a failure.

### L2 — batch composition

The algorithmic centrepiece. Given a credit of amount `X` on date `D`, find the subset of unsettled payments whose net composes `X`.

Naive subset-sum over all payments is exponential and will blow up. Bounded as follows:

- **Date window first.** Only payments with `capture_date` in `[D - 3d, D]` are candidates. This is the dominant pruning step — it takes the candidate pool from hundreds to tens.
- **Hard cap on candidate pool.** If the window yields more than `MAX_POOL` (48) candidates, do not attempt composition — emit an exception with reason `pool_too_large`. Refusing to solve is correct behaviour; hanging is not.
- **Hard cap on subset size** (`MAX_SUBSET`, 25).
- **Meet-in-the-middle** over the pooled candidates, or DP on paise buckets if the pool is small. Either is fine; do not write a plain recursive solver.
- **Deterministic tie-break.** If more than one subset composes `X`, do not pick one. Emit all candidates to L4 as competing explanations. Multiple valid compositions is exactly the ambiguity the model exists to adjudicate.
- **Timeout guard.** Wall-clock budget per credit; on breach, exception with reason `compose_timeout`.

### L3 — tolerance

L3 checks; it does not search. For each credit L2 left unresolved, take the expected batches (already grouped by settlement UTR, one net per batch — that grouping is computed once, upstream of every stage) whose settlement window overlaps `[D - 3d, D]`, and compare each one's already-computed net directly against the credit: `|credit − batch.net|`. Exactly one batch within tolerance resolves at 0.75 with the delta recorded on the Decision. Zero or more than one leaves the credit unresolved.

**This was originally a composition search** — `compose_within_tolerance`, the same meet-in-the-middle machinery as L2 but accepting any subset within a band instead of an exact sum. It was wrong, not mistuned: composing exactly is a strong filter (almost no subset of a random pool sums to precisely a target), but composing within a band turns "sums to X" into "sums to approximately X", and an 11-line candidate pool has 2048 subsets to draw near-misses from. Run against the real fee_tier_change credit it returned roughly ninety within-tolerance subsets. No band width fixes this — wide enough to admit a real fee-tier drift is wide enough to admit dozens of coincidental line combinations; narrow enough to exclude the coincidences is narrow enough to reject the real drift too. See FAILURES.md's 2026-08-30 entry. Comparing against batches instead of composing from lines removes the assembly step that produced the coincidences in the first place — a batch net is a real, already-computed quantity, not something built out of whatever happens to be lying around.

**The band, set from the fee structure, not tuned to fit this dataset:**

`tolerance = max(50 paise, 0.6% × credit amount)`.

- Pure rounding noise (per-line vs per-batch GST rounding — see `fees.py`) is bounded at a few paise per batch, swamped by the 50-paise floor.
- A plausible fee-tier drift — a gateway moving a merchant's rate by up to half a percentage point, the largest shift `generate.py`'s own `FEE_TIER_BUMP` models — costs `0.5% × 1.18` (GST on the extra fee) ≈ 0.59% of the affected line's gross. Rounded up to 0.6% of the *credit* (diluted by whatever else shares the batch, so already generous relative to one line's 0.59%).

Wider admits deltas only explainable by a genuinely wrong batch — a false match. Narrower rejects a legitimate half-point fee change — a real settlement pushed to exception for nothing. This number is derived once from the fee structure and left alone; it is not adjusted to make any run's numbers look better.

### L4 — the LLM boundary

This is the only place a model is called, and its job is narrow. Provider: OpenAI, model `gpt-5-nano` — the cheapest model on OpenAI's own pricing page at the time this was chosen (D0.5), picked deliberately for narrow single-label classification at ablation scale (~117 calls total across the with-LLM and llm-only arms), not defaulted to a mid-tier model. Structured output is schema-enforced at the API boundary (`response_format` strict JSON Schema, generated straight from `AdjudicationResult`'s own pydantic schema) — see adjudicator.py's module docstring for the full reasoning and for why this was originally Anthropic/claude-sonnet-5 until only an OpenAI key was available.

**Input:** one unresolved bank credit, a primary candidate, the numeric delta, and up to four competing candidates as alternate explanations — built in one of two ways (2026-08-31 fix; see FAILURES.md). For the with-LLM arm, L4 first re-runs L2's own bounded exact-sum search (`compose.compose()`) over the credit's date-windowed settlement lines: a credit reaching L4 here only ever has 0 or ≥2 exact-sum subsets (L2 already claimed anything with exactly 1). ≥2 is *provable* ambiguity — L4 forces the outcome to exception regardless of the model's reported confidence, since no pick among exact ties is more correct than another. Only when no exact subset exists at all (covers `tolerance_ambiguous`, `unrelated_credit`, `date_skew`) does L4 fall back to comparing whole `ExpectedBatch` objects by `|credit − batch.net|`, same as before. `--llm-only` always uses this whole-batch fallback and never the exact-sum search — re-deriving L2's search there would smuggle a rules capability into the one arm meant to have none, and would run compose() against a full, unpruned pool for all ~105 credits instead of ~12.

**Output:** pydantic-validated JSON —

```
break_reason           enum
proposed_resolution    str
confidence             float 0-1
evidence_refs          list[str]   ids of records supporting the call
human_review_required  bool
```

**It receives no arithmetic task.** Deltas are pre-computed by rules and handed to it as facts. The model reasons over *why*, never *how much*. This is the single most important design line in the project and it is the direct answer to the "AI judgment — the right tool in the right place, and where you chose not to use one" criterion.

**Degradation:** malformed JSON → retry once with the validation error appended → still bad → exception with reason `adjudication_failed`. The pipeline never crashes on model output.

**Caching:** responses stored in `cache/` keyed by a hash of the prompt payload (including the model name — a provider or model swap invalidates every prior entry automatically, no migration step needed). Committed to the repo. `--cached` replays a full measured run with no API key, which makes the reported metrics independently verifiable by anyone who clones it.

**Reproducibility comes from this committed cache, not from sampling configuration.** No `temperature` (or equivalent) is sent to the model at all — both providers used in this project's history reject or ignore sampling controls on their current model families, and it wouldn't help anyway: a `--cached` run always replays the exact bytes recorded here, regardless of what a live call would produce on any given day. Determinism is a property of the cache file, never of the request.

## Confidence bands

| Band | Action |
|---|---|
| ≥ 0.85 | auto-accept |
| 0.60 – 0.85 | matched, flagged `human_review_required` |
| < 0.60 | exception, unresolved |

Deliberately conservative. In finance a false match is far more expensive than an unresolved item: the wrong match silently corrupts the books and is discovered weeks later, while an exception costs an analyst ten minutes today. We report false-match rate separately and drive it to zero even at the cost of headline match rate.

The bands are the default outcome from `confidence` alone. The model's own `human_review_required` flag can only push a call *down* — a confident call the model itself flags for review still lands in the middle band — never up: a low-confidence call is never auto-accepted just because the model didn't ask for review. Bias to caution applies to the model's self-assessment the same way it applies to the rules layers.

## Audit trail

Every stage writes a `Decision` row before returning. No exceptions.

```
run_id · seed · stage · credit_id · matched_payment_ids · outcome
confidence · delta_paise · reason · rationale · llm_model · llm_cost_paise · llm_retried
evidence_refs · human_review_required · created_at
```

`run_id` is derived from the seed, so runs are reproducible and two runs are diffable. The exception report and the HTML report are **queries over this table**, never separately maintained lists — which is what makes the audit trail load-bearing rather than decorative. `evidence_refs`/`human_review_required` are the two `AdjudicationResult` fields the original schema left unpersisted — set only when a Decision actually reached the adjudicator, `None` for every rules-stage (L0-L3) decision and for `adjudication_failed`, since no classification exists to record in either case. `unbatch exceptions --export` is what reads them back out.

## Reporting

`report.py` renders `out/report.html` from the audit DB via jinja2 (template in `src/unbatch/templates/report.html.jinja2`). Static file, no server, no CDN — every byte (CSS included) is inlined into the one HTML file. Contents: the three-arm ablation table (every METRICS.md rate, per arm — including false-match rate and LLM cost per arm, not just overall), the B−A delta with the D3 framing prose explaining why a small delta is the correct outcome, break-reason confusion tables, a confidence-value histogram, and the full per-arm exception table (reasons and rationales, no truncation).

An arm with no decisions yet in the audit log for the current seed renders as "not yet run", never faked as all-exceptions — `unbatch report` works correctly against a partial ablation (e.g. only `--no-llm` has been run) and against an empty database. Money fields are formatted to rupees only inside `report.py` (CLAUDE.md invariant 1); everywhere upstream, including the audit log itself, stays in paise.

Static HTML over a dashboard framework on purpose: zero runtime risk during the pitch video, and nothing to break on a judge's machine.

## Data flow summary

```
generate.py --seed 42
      ├──> data/order_ledger.csv
      ├──> data/settlement_report.csv
      ├──> data/bank_statement.csv
      └──> data/ground_truth.json      (read ONLY by metrics.py)

cli run --seed 42
      ├──> L0 -> L1 -> L2 -> L3 -> L4   each writing Decisions
      ├──> out/audit.db
      └──> out/report.html
```

`ground_truth.json` is never imported by any module under `stages/`. If it is, that is a scoring leak and the metrics are worthless.

## Deliberately excluded

Multi-source expansion beyond these three files, real-time ingestion, auth, multi-tenant, a web UI, forecasting. One loop closed properly beats three half-closed. The brief says a cherry-picked match proves nothing; it does not ask for breadth.
