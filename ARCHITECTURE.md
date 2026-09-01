# ARCHITECTURE.md

This file describes how the system is put together: the matching cascade, the LLM boundary, the audit trail, and the forecasting loop. Read it if you want to know how a credit gets matched or why a stage is ordered the way it is, rather than what was measured (see METRICS.md) or why a feature is absent (see DECISIONS.md).

## The problem

A payment gateway does not pay a merchant per transaction. It **batches**: many captured payments over a cutoff window, minus per-transaction fees, minus GST (Goods and Services Tax, India's VAT) on those fees, minus refunds issued in the window, minus chargebacks and their fees, settled T+1 or T+2 (one or two business days after capture) as **one lump credit** into the merchant's bank account with a single UTR (Unique Transaction Reference, the bank's own reference number for a transfer).

The merchant's finance team sees a bank statement line for ₹4,18,332 and has to answer: which orders is this, and why isn't it the ₹4,19,107 we expected?

That is the loop this closes. Three input files, a many-to-one relationship between them, real money math.

Data flow, top to bottom:

```
 order_ledger.csv, settlement_report.csv, bank_statement.csv
                              v
      expected settlement batches  +  bank credit lines
                              v
         +-----------------------------------------+
         | L0  UTR exact match                1.00 |
         +-----------------------------------------+
                              v  unresolved
         +-----------------------------------------+
         | L1  amount + date exact            0.98 |
         +-----------------------------------------+
                              v  unresolved
         +-----------------------------------------+
         | L2  batch composition              0.90 |
         +-----------------------------------------+
                              v  unresolved
         +-----------------------------------------+
         | L3  tolerance band                 0.75 |
         +-----------------------------------------+
                              v  unresolved: 12 of 105 credits (seed 42)
         +-----------------------------------------+
         | L4  LLM adjudication     model-reported |
         +-----------------------------------------+
                              v
       audit log (SQLite, one Decision row per credit)
                              v
out/report.html, unbatch metrics, unbatch exceptions --export
```

Why this pair and not invoice-vs-payment 1:1: 1:1 matching is a solved problem and leaves the model nothing real to do. Many-to-one produces genuine ambiguity, which is where AI judgment can be demonstrated instead of asserted.

## The cascade

Ordered cheapest-and-most-certain first. Each stage consumes only what the previous stages left unresolved.

| Stage | Method | Resolves | Confidence |
|---|---|---|---|
| **L0** | UTR exact: settlement UTR appears in bank narration | bulk of clean cases | 1.00 |
| **L1** | Amount + date exact: credit ties exactly to a computed batch net | clean cases with mangled narration | 0.98 |
| **L2** | Batch composition: find the payment subset that composes the credit | splits, partial settlements | 0.90 |
| **L3** | Tolerance: credit checked directly against date-windowed expected batches, delta recorded | fee-tier drift, rounding | 0.75 |
| **L4** | LLM adjudication: classify the break, propose resolution | everything left | model-reported |

Anything L4 cannot resolve with confidence lands in the exception list. That is a **success outcome**, not a failure.

### L2: batch composition

The algorithmic centrepiece. Given a credit of amount `X` on date `D`, find the subset of unsettled payments whose net composes `X`.

Naive subset-sum over all payments is exponential and will blow up. Bounded as follows:

- **Date window first.** Only payments with `capture_date` in `[D - 3d, D]` are candidates. This is the dominant pruning step; it takes the candidate pool from hundreds to tens.
- **Hard cap on candidate pool.** If the window yields more than `MAX_POOL` (48) candidates, do not attempt composition. Emit an exception with reason `pool_too_large` instead. Refusing to solve is correct behaviour; hanging is not.
- **Hard cap on subset size** (`MAX_SUBSET`, 25).
- **Meet-in-the-middle** (split the candidate pool in half, enumerate each half's subset sums separately, then look for a pair that adds to the target, far cheaper than enumerating every subset of the whole pool) over the pooled candidates, or DP (dynamic programming) on paise (1/100 of a rupee) buckets if the pool is small. Either is fine; do not write a plain recursive solver.
- **Deterministic tie-break.** If more than one subset composes `X`, do not pick one. Emit all candidates to L4 as competing explanations. Multiple valid compositions is exactly the ambiguity the model exists to adjudicate.
- **Timeout guard.** Wall-clock budget per credit; on breach, exception with reason `compose_timeout`.

**Measured at scale** (`unbatch bench --scale 5000`, see bench_scale.json and FAILURES.md's 2026-08-31 entry): at production scale (105 credits) this is all invisible. At synthetic scale, two bottlenecks showed up, neither the one this section originally predicted. First, `pool_too_large` genuinely fires once date-windowed pools exceed 48 (all 448 of scale 5000's split-batch credits hit it), which is the cap doing exactly its documented job. Second, and unpredicted: pools that land *just under* 48 rather than over it are legal but expensive. The meet-in-the-middle's ~2^24-per-half worst case is real, and a run with several such credits back to back can take minutes even though every individual call stays within its own timeout (at scale 1000, six credits landed at exactly 48). The per-credit timeout bounds one call; nothing currently bounds a run that queues up many expensive-but-legal calls. Left unfixed deliberately: closing it is a cascade-behaviour decision, not a bug fix.

### L3: tolerance

L3 checks; it does not search. For each credit L2 left unresolved, take the expected batches (already grouped by settlement UTR, one net per batch; that grouping is computed once, upstream of every stage) whose settlement window overlaps `[D - 3d, D]`, and compare each one's already-computed net directly against the credit: `|credit - batch.net|`. Exactly one batch within tolerance resolves at 0.75 with the delta recorded on the Decision. Zero or more than one leaves the credit unresolved.

**This was originally a composition search:** `compose_within_tolerance`, the same meet-in-the-middle machinery as L2 but accepting any subset within a band instead of an exact sum. It was wrong, not mistuned. Composing exactly is a strong filter (almost no subset of a random pool sums to precisely a target), but composing within a band turns "sums to X" into "sums to approximately X", and an 11-line candidate pool has 2048 subsets to draw near-misses from. Run against the real fee_tier_change credit it returned roughly ninety within-tolerance subsets. No band width fixes this: wide enough to admit a real fee-tier drift is wide enough to admit dozens of coincidental line combinations, and narrow enough to exclude the coincidences is narrow enough to reject the real drift too. See FAILURES.md's 2026-08-30 entry. Comparing against batches instead of composing from lines removes the assembly step that produced the coincidences in the first place; a batch net is a real, already-computed quantity, not something built out of whatever happens to be lying around.

**The band, set from the fee structure, not tuned to fit this dataset:**

`tolerance = max(50 paise, 0.6% of credit amount)`.

- Pure rounding noise (per-line vs per-batch GST rounding, see `fees.py`) is bounded at a few paise per batch, swamped by the 50-paise floor.
- A plausible fee-tier drift (a gateway moving a merchant's rate by up to half a percentage point, the largest shift `generate.py`'s own `FEE_TIER_BUMP` models) costs `0.5% times 1.18` (GST on the extra fee), about 0.59% of the affected line's gross. Rounded up to 0.6% of the *credit* (diluted by whatever else shares the batch, so already generous relative to one line's 0.59%).

Wider admits deltas only explainable by a genuinely wrong batch: a false match. Narrower rejects a legitimate half-point fee change: a real settlement pushed to exception for nothing. This number is derived once from the fee structure and left alone; it is not adjusted to make any run's numbers look better.

**False-accept guard (2026-08-31, see FAILURES.md):** before accepting the one within-tolerance batch, L3 now refuses if the delta exactly equals the net of a real settlement line still in the credit's own date-windowed pool. That gap is a composition fact (a whole line, not fee/rounding noise) that L2 already looked at and declined. Verified against real seed-42 data; `rounding_delta` and `fee_tier_change` still resolve here. It does **not** close every false-accept path `bench --seeds` measures: the seed 44/45/46 false matches actually resolve against a *different, unrelated* batch whose net coincidentally lands within tolerance of the credit, the same "two unrelated quantities collide within a narrow target" shape as L2's exact-sum collision below, not a missing-line case. That one is left open, per the same reasoning as L2's: not fixable without either widening/narrowing the band (rejected) or inventing a disambiguation the system has no information for.

### L4: the LLM boundary

This is the only place a model is called, and its job is narrow. Provider: OpenAI, model `gpt-5-nano`, the cheapest model on OpenAI's own pricing page at the time this was chosen (milestone D0.5), picked deliberately for narrow single-label classification at ablation scale, not defaulted to a mid-tier model. Seed 42's own ablation sends 117 credits to the model across the with-LLM and llm-only arms combined; counting retries (added by the semantic-validation check described below, see FAILURES.md's 2026-09-01 entries), the raw call count is 159. Structured output is schema-enforced at the API boundary (`response_format` strict JSON Schema, generated straight from `AdjudicationResult`'s own pydantic schema); see the module docstring in `adjudicator.py` (the module that calls the LLM) for the full reasoning and for why this was originally Anthropic/claude-sonnet-5 until only an OpenAI key was available.

**Input:** one unresolved bank credit, a primary candidate, the numeric delta, and up to four competing candidates as alternate explanations, built in one of two ways (2026-08-31 fix; see FAILURES.md). For the with-LLM arm, L4 first re-runs L2's own bounded exact-sum search (`compose.compose()`) over the credit's date-windowed settlement lines: a credit reaching L4 here only ever has 0 or 2+ exact-sum subsets (L2 already claimed anything with exactly 1). 2 or more is *provable* ambiguity. L4 forces the outcome to exception regardless of the model's reported confidence, since no pick among exact ties is more correct than another. Only when no exact subset exists at all (covers `tolerance_ambiguous`, `unrelated_credit`, `date_skew`) does L4 fall back to comparing whole `ExpectedBatch` objects by `|credit - batch.net|`, same as before. `--llm-only` always uses this whole-batch fallback and never the exact-sum search: re-deriving L2's search there would smuggle a rules capability into the one arm meant to have none, and would run compose() against a full, unpruned pool for all ~105 credits instead of ~12.

**Output:** pydantic-validated JSON:

```
break_reason           enum
proposed_resolution    str
confidence             float 0-1
evidence_refs          list[str]   ids of records supporting the call
human_review_required  bool
```

**It receives no arithmetic task.** Deltas are pre-computed by rules and handed to it as facts. The model reasons over *why*, never *how much*. This is the design line the "AI judgment, and where you chose not to use it" criterion depends on most directly.

**Degradation:** malformed JSON leads to one retry with the validation error appended; if that is still bad, the outcome is an exception with reason `adjudication_failed`. The pipeline never crashes on model output.

**Caching:** responses stored in `cache/` keyed by a hash of the prompt payload (including the model name, so a provider or model swap invalidates every prior entry automatically, no migration step needed). Committed to the repo. `--cached` replays a full measured run with no API key, which makes the reported metrics independently verifiable by anyone who clones it.

**Reproducibility comes from this committed cache, not from sampling configuration.** No `temperature` (or equivalent) is sent to the model at all. Both providers used in this project's history reject or ignore sampling controls on their current model families, and it wouldn't help anyway: a `--cached` run always replays the exact bytes recorded here, regardless of what a live call would produce on any given day. Determinism is a property of the cache file, never of the request.

## Narration robustness (E10)

The synthetic data's bank narrations came from clean templates, which gave L0's UTR substring match an easy ride. `generate.py`'s `apply_narration_noise` (seeded, deterministic, `--noise 0.0` a strict no-op) degrades narrations only: truncation mid-UTR, transposed adjacent digits, O/0 and I/1/l lookalike substitution, inconsistent separators, bank-specific wrapping text, case inconsistency, or the UTR dropped entirely for a bare counterparty name. Amounts, dates, and settlement data are never touched, so this measures narration robustness in isolation, not arithmetic.

`unbatch bench --noise 0.0,0.1,0.25,0.5,0.75,1.0` (see bench_noise.json) holds seed 42 fixed and sweeps noise from clean to maximum. The result: **count match rate is exactly 0.8857 and false-match rate is exactly 0.0 at every single level.** What moves is the stage funnel: L0 falls from 79 to 21 resolutions as noise rises to 1.0, and L1 (amount + date exact, which never looks at narration at all) picks up precisely the slack, rising from 3 to 61. The two stages' combined total is identical at every noise level. This is the redundancy the cascade was designed to have, measured rather than assumed: a credit that a noisy narration knocks off L0 does not fall further than L1, because L1's own criterion never depended on narration being clean in the first place.

**A partial-UTR fuzzy-match path (rapidfuzz similarity, confidence below L0's 1.00) was considered and explicitly not built.** The stated bar was to build it only if the curve showed L0 dropping credits that a fuzzy match would have caught more safely than falling through. It doesn't: every credit L0 loses to noise is caught by L1 at the same rate, with zero false matches, all the way to noise 1.0. Adding a fuzzy matcher here would be solving a problem this measurement shows the cascade doesn't have, at the cost of a new lower-confidence path and a new way to be wrong. Left out on the evidence, not by default.

## Confidence bands

| Band | Action |
|---|---|
| >= 0.85 | auto-accept |
| 0.60 to 0.85 | matched, flagged `human_review_required` |
| < 0.60 | exception, unresolved |

Deliberately conservative. In finance a false match is far more expensive than an unresolved item: the wrong match silently corrupts the books and is discovered weeks later, while an exception costs an analyst ten minutes today. False-match rate is reported separately and driven toward zero even at the cost of headline match rate.

The bands are the default outcome from `confidence` alone. The model's own `human_review_required` flag can only push a call *down*. A confident call the model itself flags for review still lands in the middle band; it can never push a call up. A low-confidence call is never auto-accepted just because the model didn't ask for review. Bias to caution applies to the model's self-assessment the same way it applies to the rules layers.

## Audit trail

Every stage writes a `Decision` row before returning. No exceptions.

```
run_id · seed · stage · credit_id · matched_payment_ids · outcome
confidence · delta_paise · reason · rationale · llm_model · llm_cost_paise · llm_retried
evidence_refs · human_review_required · created_at
```

`run_id` is derived from the seed, so runs are reproducible and two runs are diffable. The exception report and the HTML report are **queries over this table**, never separately maintained lists, which is what makes the audit trail load-bearing rather than decorative. `evidence_refs`/`human_review_required` are the two `AdjudicationResult` fields the original schema left unpersisted. They are set only when a Decision actually reached the adjudicator, and `None` for every rules-stage (L0-L3) decision and for `adjudication_failed`, since no classification exists to record in either case. `unbatch exceptions --export` is what reads them back out.

`audit.record` commits after every row: durability per decision, invisible at 105 credits. `bench --scale` found this is actually the dominant wall-clock cost once decision *counts* run into the hundreds-plus, ahead of anything in the matching logic itself (see FAILURES.md's 2026-08-31 entry). Left as-is: batching commits is a real behaviour change to a stage every other command depends on, not something to decide unilaterally mid-benchmark.

## Reporting

`report.py` renders `out/report.html` from the audit DB via jinja2 (a Python templating library; template in `src/unbatch/templates/report.html.jinja2`). Static file, no server, no CDN: every byte (CSS included) is inlined into the one HTML file. Contents: the three-arm ablation table (every METRICS.md rate, per arm, including false-match rate and LLM cost per arm, not just overall), the B minus A delta with the D3 framing prose explaining why a small delta is the correct outcome, break-reason confusion tables, a confidence-value histogram, and the full per-arm exception table (reasons and rationales, no truncation).

An arm with no decisions yet in the audit log for the current seed renders as "not yet run", never faked as all-exceptions. `unbatch report` works correctly against a partial ablation (e.g. only `--no-llm` has been run) and against an empty database. Money fields are formatted to rupees only inside `report.py` (CLAUDE.md invariant 1); everywhere upstream, including the audit log itself, stays in paise.

Static HTML over a dashboard framework on purpose: zero runtime risk during the pitch video, and nothing to break on a judge's machine.

## Data flow summary

The same flow as the diagram at the top, with the exact commands and file paths:

```
generate.py --seed 42
      +--> data/order_ledger.csv
      +--> data/settlement_report.csv
      +--> data/bank_statement.csv
      +--> data/ground_truth.json      (read ONLY by metrics.py)

cli run --seed 42
      +--> L0 -> L1 -> L2 -> L3 -> L4   each writing Decisions
      +--> out/audit.db
      +--> out/report.html
```

`ground_truth.json` is never imported by any module under `stages/`. If it is, that is a scoring leak and the metrics are worthless.

## Forward cash forecast (E14)

The track brief asks to run the books *and* the cash position. Reconciliation is the first loop; this is the second: `unbatch forecast --horizon N` and `forecast.py`, entirely separate from the cascade above. **Pure arithmetic, not judgment:** projecting *when* and *how much* money arrives has one correct procedure given the inputs, with nothing for a model to adjudicate. `forecast.py` does not import `adjudicator` and no L4 call is reachable from this code path (CLAUDE.md invariant 2). It is also a projection over data the pipeline already has: `order_ledger.csv` and `settlement_report.csv`, no new data source, no new dependency. `forecast.py` imports only `unbatch.fees` and the stdlib.

**Method.** Fit two distributions from settlement lines already settled on or before `as_of` (the same train/test discipline as the backtest below): the *lag mode* (days from `captured_at` to `settled_at`, 1 on real data, since the generator draws T+1 three times as often as T+2) and the *fee-schedule deviation stdev* (how far each historical line's actual net sits from what `fees.py`'s own fee/GST computation would produce for that gross and method, as a fraction of gross: the real residual from refunds, chargebacks, or an injected fee-tier change, not an invented distribution). Every captured order with no PAYMENT-type settlement line dated on or before `as_of` is projected onto `max(captured_at + lag_mode, as_of + 1 day)`. The floor exists so an already-overdue order lands on the first horizon day instead of vanishing or landing in the past. It projects at the exact fee-schedule net, banded by the deviation stdev scaled to that order's own gross.

**A provable structural property, not a tuning choice:** "unsettled as of `as_of`" means `captured_at <= as_of` by definition, so `captured_at + lag_mode` can never exceed `as_of + lag_mode`. When the fitted lag mode is 1 (the normal case here), that is never later than `as_of + 1`, the first horizon day, so *every* unsettled order floors to day 1, regardless of how long ago it was actually captured. This is not an artifact of the floor logic; it is a direct consequence of a short settlement lag combined with projecting only already-captured orders, and it is the reason the backtest below finds almost no signal past day 1.

**Default `as_of`** is the data's last settled date minus the horizon, so a full horizon of genuinely unsettled orders exists to project by default; `--as-of` overrides it, which is how the backtest below holds out a window without touching `generate.py`.

**Backtest (E14b):** `unbatch bench --forecast 42,43,44,45,46,47` (bench_forecast.json) splits each seed's 30-day window at `as_of = last settled date - 14`, fits only on data up to `as_of`, and scores the projection against what actually settled in the held-out 14 days. It never touches the cascade or the audit log. See the operating envelope below for the numbers.

## Operating envelope (E11c, updated E12, E13, E14, a post-E14 ablation-pooling pass, and a 2026-09-01 semantic-validation update)

This section states the system's limits as measured, rather than hedged with prose. Four independent measurements, each stressing a different axis, define where this system is reliable and where, and by how much, it degrades.

**Reliable against narration noise specifically, and confirmed independent of the collision floor:** `bench --noise 0.0,0.1,0.25,0.5,0.75,1.0` (bench_noise.json) holds count match rate and false-match rate exactly flat across the entire noise range on seed 42, whose organic false-match rate is 0.0%. L0 resolutions fall from 79 to 21 as noise rises to maximum, and L1 (which never reads narration) picks up exactly the slack, rising from 3 to 61. That alone only proved the redundancy holds on the one seed with nothing to interact with. **E12** re-ran the same sweep (`bench --noise 0.0,0.25,0.5,0.75,1.0`, bench_noise_seeds.json) on seeds 44 and 46, the two highest organic false-match rates in the multi-seed set, 1.06% and 2.11%. Both stayed exactly as flat: count match rate and false-match rate are bit-for-bit constant across every noise level on both seeds, with the identical L0-falls/L1-rises shift (seed 44: 79 to 30 resolutions at L0, 3 to 52 at L1; seed 46: 79 to 20 at L0, 3 to 62 at L1) and the *same* false-match rate the seed already had at noise 0.0, never higher. **The noise axis and the collision axis are independent.** Narration quality changes only which stage resolves a credit, never whether a false match occurs, regardless of how collision-prone that seed's underlying dataset already is.

**Degrades with dataset composition, not with narration:** `bench --seeds 42,43,44,45,46,47` (bench_multiseed.json) holds narration and scale fixed and varies the random dataset instead. Count match rate stays tight (88.57% to 90.48%, mean 89.05%, stdev 0.80 points), but false-match rate ranges **0.0% to 2.11%**: 0.0% on 2 of 6 seeds, nonzero on the other 4. Tracing the actual false matches (not just the rate) found every one is a coincidental collision between two *unrelated* quantities landing within a narrow target of each other, an exact-sum coincidence in L2's composition pool, or a tolerance-band coincidence between a credit and an unrelated batch in L3, never a credit resolving against its own genuinely-broken batch. A guard closing the one L3 false-accept shape that *was* a real logic gap (see FAILURES.md's 2026-08-31 entries) left this range completely unchanged, confirming the remaining floor is coincidence, not a fixable defect.

**The measured worst case:** `bench --adversarial` (bench_adversarial.json) is data built specifically to maximize that same coincidence shape: many more duplicate-UTR batches, amounts deliberately clustered inside each other's tolerance bands, multi-way exact composition ties, and a composition search pushed to its cap. On this dataset, at the same ~105-credit scale, false-match rate reaches **5.21%**, more than double the worst organically-observed seed, because the collisions there are engineered rather than left to chance. This is the number to quote as the floor under genuinely hostile input, not the 0.0% any single clean seed reports.

**Scale:** `bench --scale 5000` (bench_scale.json, see also FAILURES.md) found the audit log's per-decision commit, not the matching logic, becomes the dominant wall-clock cost as decision counts grow past a few hundred, and that L2 candidate pools sitting just under `MAX_POOL` (48) can time out outright rather than merely run slow. Neither was fixed; both are documented costs of deliberate designs (durability-per-decision, a hard pool cap), not defects.

**The ablation axis, pooled:** the three-arm ablation (rules-only vs. rules+LLM vs. LLM-only) was originally measured on seed 42 alone: 88.6% / 88.6% / 13.3% match rate, 0.0% false-match rate on all three arms, while everything around it was already pooled across six seeds, exactly the "one cherry-picked match" shape the brief warns against. `bench_ablation_seeds.json` extends arms B and C across all six seeds (arm A: rules only, arm B: rules + LLM, arm C: LLM only; arm A was already pooled in bench_multiseed.json); this needed live calls for arm C on seeds 43-47 (arm B's cache already covered all six from milestone E13), reported below.

Pooled across 630 credits: arm B holds at 89.4% match / 0.89% false-match, indistinguishable from arm A's own pooled 89.05% / 0.89%, confirming the near-zero B minus A delta is not a seed-42 artifact. Arm C is a fourth instance of the headline being seed-specific, and in the direction that matters most: pooled match rate is 8.7% (55/630), worse than seed 42's standalone rate (see below), with a per-seed range of 6.7% to 10.5%, and pooled false-match rate is 3.64% (2 of 55 resolved), not 0.0%, split between seeds 42 and 46, each contributing one false match at a 9.1% rate (1 of 11 resolved). Seed 42 alone made LLM-only look worse and more expensive; pooled, it is also measurably less safe, a claim the single-seed table never made and never could have.

**2026-09-01 update (see FAILURES.md):** a semantic-validation check added to the adjudicator (evidence_refs must reference a record the model was actually shown) forced live retries across every seed and arm above. Arm A and arm B's own false-match rates did not move; the numbers in this section already reflect the post-fix state. Seed 42's standalone ablation table, quoted for historical framing here, changed as a direct result: match rate for arm C moved from 13.3% (14/105) to 10.5% (11/105), and arm C's false-match rate moved from 0.0% to 9.1% (1 of 11 resolved), where it had been exactly 0.0% before. Arms A and B's seed-42 figures (88.6% match, 0.0% false-match) are unchanged.

**The adjudication axis (E13):** break-reason accuracy was originally measured on twelve L4 credits from one seed: 91.7%, the number this whole "AI judgment" argument leans on. `bench_adjudication.json` pools the same measurement across all six organic seeds (42-47, live calls made for 43-47 and committed to cache/ so every seed now replays keylessly). At the time of that original pooling, before the semantic-validation fix below, pooled accuracy was 89.9% (62 of 69 classified credits), close to, not identical to, the seed-42 figure, with individual seeds swinging from 80.0% to 100.0% on a sample of 10-12 classifications each. Seed 42's 91.7% was never a stable estimate on its own; it happened to land in the upper half of that range.

**2026-09-01 update (see FAILURES.md):** a semantic-validation check added to the adjudicator (evidence_refs must reference a record the model was actually shown, not an unrelated identifier such as a settlement UTR it read off a candidate hint) rejected most of the responses behind the numbers above, since citing an out-of-scope identifier turned out to be the model's default habit on this prompt, not an occasional slip. Retrying with the validation error appended mostly did not recover a classification: across all six seeds, only 35 of the original 69 credits (now with retries filled in from live calls) still receive a real classification, the rest degrading to `adjudication_failed`. Pooled accuracy on that smaller, now-correct set is 85.7% (30 of 35). The accuracy figure did not just drop from the earlier 89.9%: the population it is measured over shrank by half at the same time, from 69 classified credits to 35, because the credits the validation check now excludes are exactly the ones whose evidence citations could never be checked in the first place, not a random subset of the original 69. Reporting 85.7% next to 89.9% without saying that the denominator halved would understate what changed. Seed 42 alone moved from 91.7% (11 of 12) to 83.3% (5 of 6), with 6 of its 12 L4 credits now ending in `adjudication_failed` where none did before. The pooled confusion table on the current data: `ambiguous_composition` is classified reliably (18 of 20 where classified), `tolerance_ambiguous` similarly (9 of 11), `unrelated_credit` perfectly (3 of 3), and the single `duplicate_utr` instance that reached a classification was misclassified. On the adversarial dataset (also run for real, also cached), only 3 of 8 credits still classify at all after the same fix, and none of the 3 are correct: accuracy is 0.0%, down from 37.5% (3 of 8) before the fix. A sample of 3 is too small to support any claim about the model's adversarial accuracy in either direction; 0.0% is reported here for completeness, not read as a finding the way the pooled 85.7% is. The confusion is consistent with the earlier finding rather than new: `duplicate_utr` credits are still called `ambiguous_composition`.

**The forecasting axis (E14), measured against the same discipline:** `bench --forecast 42,43,44,45,46,47` (bench_forecast.json) backtests the cash forecaster on all six seeds, holding out the last 14 days of each 30-day window and scoring the projection against what actually settled. The result is not a gentle accuracy curve. It is a near-total miss, reported plainly: mean absolute error is **~₹1.29 lakh/day**, and **coverage (actual inside the low/high band) is 0.0% on every seed**. The explanation is the structural property above, not a broken projection: the forecaster's own tracked population (already-captured, still-unsettled orders) can only ever account for **~10.0% of actual 14-day inflow on average** (7.3% to 16.7% across seeds). The remaining ~90% comes from orders not yet captured as of the forecast date, which is unknowable from settlement data alone, by construction, not by an oversight. Reported as measured rather than tuned to look better or quietly narrowed to a horizon where it would look good: this is a real, bounded technique (project the known settlement pipeline) that answers a real but narrow question, and it should not be read as a general-purpose cash-position forecast.

**In one sentence:** this cascade is reliable against messy *text* at any measured scale and independently of how collision-prone the underlying data is; it is unreliable, to a small but non-zero and quantified degree, against data where independently-generated amounts happen to collide, a risk that triples to quintuples under deliberately hostile construction; the ablation itself holds up pooled (89.4% vs. seed 42's 88.6%) while the LLM-only counter-experiment turns out worse *and* less safe pooled than it looked on one seed (8.7% match, 3.64% false-match vs. seed 42's original 13.3%/0.0%); the model's break-reason classification, after a semantic-validation check found it frequently citing evidence outside the vocabulary it was given, resolves fewer credits than before (35 classified, down from 69) and pooled accuracy on that smaller, now-correctly-scored set is 85.7%, a figure that moved together with its denominator rather than in place of it; on hostile data only 3 of 8 credits still classify at all, too few to read the resulting 0.0% as a finding rather than a small-sample artifact; and the forward cash forecast, honestly backtested rather than shipped on faith, captures only about 10% of future inflow that a short settlement lag makes visible at all. These numbers, not the single cleanest one, are the reason the exception list, not the match-rate headline, is this project's real deliverable.

## Deliberately excluded

Multi-source expansion beyond these three files, real-time ingestion, auth, multi-tenant, a web UI. One loop closed properly beats three half-closed. The brief says a cherry-picked match proves nothing; it does not ask for breadth.

Forecasting was on this list until E14 closed it too, at the same narrow scope as reconciliation: a projection over these same three files, no new data source, backtested and reported honestly rather than shipped on faith. See "Forward cash forecast" above.
