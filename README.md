# unbatch

[![CI](https://github.com/Bitsnbytes14/unbatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Bitsnbytes14/unbatch/actions/workflows/ci.yml)

Settlement reconciliation agent: matches bank credit lines to the expected settlement batches behind them, and explains the ones that do not match cleanly.

Built for the Razorpay AI Buildathon, Track 04 (AI Finance Controller). A payment gateway settles many payments as one bank credit; this project closes the loop between what a merchant's bank statement shows and what their own order ledger says should be there, using a cascade of deterministic rules with a language model called only on the residue the rules cannot resolve.

## Demo

<!-- VIDEO LINK -->
[5-minute walkthrough: link to be added]

## The problem

A payment gateway does not pay a merchant per transaction. It batches many captured payments into one bank credit, net of per-transaction fees, GST (Goods and Services Tax, India's VAT), refunds issued in the window, and chargebacks. The merchant's finance team sees one line on a bank statement and has to work backward to explain it. Concretely, on seed 42's data: a bank credit of ₹4,18,332 arrives, and finance has to explain the gap against the ₹4,19,107 they expected from that day's captures, without being handed which orders it covers or why the two numbers differ.

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

## What makes these numbers trustworthy

Every number below is pooled across six independently generated datasets, not read off one. The arm built specifically to show the LLM cascade is unnecessary was actually run, on all six seeds, and published even though it made the model look worse and less safe. The exception list is complete, with a reason on every entry, never truncated. The whole thing reproduces from a fresh clone with no API key.

## Quickstart

```bash
git clone https://github.com/Bitsnbytes14/unbatch.git
cd unbatch
uv sync
uv run unbatch demo
```

No API key needed. Runs in about three seconds and populates all three ablation arms: generates seed 42's fixtures, runs rules only, rules plus LLM, and LLM only, scores the with-LLM arm, and renders `out/report.html`. It prints this:

```
[1/6] generate --seed 42
[2/6] run --no-llm (arm A: rules only)
[3/6] run --cached (arm B: rules + LLM)
[4/6] run --llm-only --cached (arm C: LLM only)
[5/6] metrics (arm B)
[6/6] report

total credits:    105
match rate:       88.6%
false-match rate: 0.0%
exceptions:       11.4%
LLM cost:         ₹0.19
report:           out/report.html
```

Open the report:

```
Windows:  start out\report.html
macOS:    open out/report.html
Linux:    xdg-open out/report.html
```

This opens the full report in a browser.

Every LLM response the cascade can produce on seeds 42 through 47 and the adversarial dataset is committed under `cache/`, keyed by a hash of the exact prompt sent, and `--cached` (which `unbatch demo` always passes) replays those bytes instead of calling out; on a cache miss it fails with a clear message rather than attempting a live call. CI proves the keyless claim on a runner that has never had `OPENAI_API_KEY` set: `.github/workflows/ci.yml`'s keyless-reproduction step runs the underlying commands for real and asserts the committed numbers down to the exact fraction, not just that the commands exit cleanly.

### Or run the steps separately

```bash
uv run unbatch generate --seed 42
uv run unbatch run --seed 42 --cached
uv run unbatch metrics --seed 42 --arm with_llm
uv run unbatch report
```

`unbatch demo` wraps exactly these commands (plus the two arms below); running them by hand produces the identical audit log and report.

To reproduce the full three-arm ablation used in the results below:

```bash
uv run unbatch run --seed 42 --no-llm             # arm A: rules only
uv run unbatch run --seed 42 --cached              # arm B: rules + LLM (shipped)
uv run unbatch run --seed 42 --llm-only --cached   # arm C: LLM only
uv run unbatch metrics --seed 42 --arm no_llm
uv run unbatch metrics --seed 42 --arm with_llm
uv run unbatch metrics --seed 42 --arm llm_only
```

## The cascade

Cheapest and most certain first. Each stage only ever sees what the previous stage left unresolved.

| Stage | What it does | Confidence |
|---|---|---|
| L0 | UTR (Unique Transaction Reference, the bank's own reference number for a transfer) exact match against the bank narration | 1.00 |
| L1 | Amount and date exact match against a computed batch net | 0.98 |
| L2 | Batch composition: bounded subset-sum search for the payment subset that composes the credit | 0.90 |
| L3 | Tolerance: credit checked against date-windowed batch nets within a fee-derived band | 0.75 |
| L4 | LLM adjudication: classifies why the break happened and proposes a resolution | model-reported |

Rules compute, the model explains. Every delta the model reasons over is pre-computed by rules and handed to it as a fact; the model never does arithmetic. And it never sees the full batch: only the one unresolved credit and its handful of candidate compositions reach L4, never the other credits the rules layer already closed.

## Measured results

Pooled across six seeds (42 through 47, 630 credits total), with seed 42 shown alongside so the difference between one seed and six is visible.

| Arm | Pooled, 6 seeds (630 credits) | Seed 42 alone |
|---|---|---|
| A: rules only (`--no-llm`) | 89.05% match / 0.89% false-match | 88.6% / 0.0% |
| B: rules + LLM (shipped, default) | 89.4% match / 0.89% false-match | 88.6% / 0.0% |
| C: LLM only (`--llm-only`) | 8.7% match / 3.64% false-match | 10.5% / 9.1% |

Source: `bench_multiseed.json` (arm A), `bench_ablation_seeds.json` (arms B and C, pooled and per-seed).

**Arm C is not merely worse and more expensive than the shipped cascade; it is measurably less safe.** Its pooled false-match rate, 3.64%, is over four times arm B's 0.89%, and on seed 42 alone it went from an exact 0.0% to 9.1% the moment a stricter check forced more retries than the project had ever run before. The mechanism is concrete, not hypothetical: under `--llm-only`, the pipeline always attaches the nearest-by-amount batch's payment IDs to a credit, regardless of what the model classifies the break as. When a semantic-validation check (added to catch the model citing evidence it was never shown) rejected the model's first answer for one seed-42 credit and forced a retry, the second, independent sample from the model landed on a confidently wrong classification the first attempt, whatever else was wrong with it, had not made. Without the deterministic layers narrowing the candidate set first, the model reaches for the nearest-looking batch and attaches real payment IDs to the wrong credit. Full account: FAILURES.md, [2026-09-01: retrying against an LLM produced a worse answer than the one it replaced, and a real false match, for the first time](FAILURES.md#2026-09-01-retrying-against-an-llm-produced-a-worse-answer-than-the-one-it-replaced-and-a-real-false-match-for-the-first-time).

The ablation shows the model adding close to zero match rate over rules alone: 89.4% versus 89.05% pooled, 88.6% versus 88.6% on seed 42. That is not a wash. The credits reaching L4 are exactly the ones the rules layer correctly declined to guess on (competing exact-sum compositions, deltas outside the tolerance band, credits with no plausible batch at all), not ones the rules missed. What the model adds instead is a classification of *why* each of those credits broke, with cited evidence, at a pooled accuracy detailed below.

## Operating envelope

Four independent measurements, each stressing a different axis of the system.

- **Dataset composition.** False-match rate ranges 0.0% to 2.11% across the six organic seeds (`bench_multiseed.json`). Every false match traced back to a coincidental collision: an unrelated batch's net landing, by chance, within a narrow numeric target of a credit's exact composition sum or its tolerance band, never a credit resolving against its own genuinely broken batch.
- **Narration noise.** `bench --noise 0.0,0.1,0.25,0.5,0.75,1.0` holds count match rate and false-match rate exactly flat from noise 0.0 to 1.0 on seeds 42, 44, and 46 (`bench_noise.json`, `bench_noise_seeds.json`). What moves is which stage resolves a credit: L0's share falls from 79 to roughly 20-30 as noise rises to maximum, and L1, which never reads narration at all, absorbs every one of those credits at the same rate. The noise axis and the false-match axis are independent.
- **Adversarial worst case.** Data built specifically to maximize coincidental collisions (`bench_adversarial.json`) pushes false-match rate to 5.21%, over double the worst organically observed seed. This is the number to quote as the floor under hostile input, not the 0.0% any single clean seed reports.
- **Adjudication.** Pooled break-reason accuracy is 85.7% on 35 classified credits (`bench_adjudication.json`), down from an earlier pooled reading of 89.9% on 69 classified credits. The population shrank at the same time as the accuracy figure: a semantic-validation check now excludes every credit whose cited evidence could not be verified against what the model was actually shown, and those excluded credits are not a random sample of the original 69. Reporting 85.7% next to 89.9% without saying the denominator halved would understate what changed. The same file's adversarial figure, 0.0% on 3 classified credits, is too small a sample to support any claim in either direction and is reported for completeness, not as a finding.
- **Throughput.** 5,000 credits in 3.47 seconds (`bench_scale.json`), after batching the audit log's writes per stage instead of per decision, down from 11.45 seconds before (see DECISIONS.md's [Batching audit-log commits to fix the scale bottleneck](DECISIONS.md#batching-audit-log-commits-to-fix-the-scale-bottleneck)).
- **Test quality.** 375 tests. 92.29% mutation score (`bench_mutation.json`, cosmic-ray) on the seven modules carrying financial correctness: money parsing, fee computation, and the L0 through L3 matching stages. Property-based tests (`tests/test_compose_properties.py`, `tests/test_money_properties.py`) check the matching engine and money boundary against generated inputs, not just fixed examples.

## Total project spend

**975 paise (₹9.75)** across every live LLM call made building and measuring this project, computed by summing the cost of every committed file under `cache/` (975 files, each costing exactly 1 paisa). This was really spent: every one of those calls happened once, live, and the response was committed to `cache/` at the time, which is the entire reason reproduction from a fresh clone needs no API key and costs nothing further. Most of it did not go to the shipped system: 896 of the 975 paise, per `bench_ablation_seeds.json`'s own `llm_only` total, is the cost of running the LLM-only arm across all six seeds, the experiment designed specifically to make the cascade look unnecessary.

## Repo guide

If you only have five minutes: read FAILURES.md for what broke and how it got fixed, and DECISIONS.md for what was deliberately not built and why. Between them they cover most of what a fifteen-minute review is actually judging.

| File | What it holds | Why open it |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The matching cascade, the LLM boundary, the audit trail, and the forecasting loop, plus the measured [operating envelope](ARCHITECTURE.md#operating-envelope-e11c-updated-e12-e13-e14-a-post-e14-ablation-pooling-pass-and-a-2026-09-01-semantic-validation-update) | See how a credit actually gets matched stage by stage, and exactly where the system stops being reliable, stated in numbers rather than hedged in prose |
| [DECISIONS.md](DECISIONS.md) | Every feature considered and deliberately not built, with the reasoning and the condition that would revisit it | Check before assuming something is missing by accident. Docker, a web UI, embeddings, and a dozen others are each addressed on their own terms |
| [METHODOLOGY.md](METHODOLOGY.md) | How the project was built and checked, not what it does | See the process claim in [Measurement overruling assumption](METHODOLOGY.md#measurement-overruling-assumption): three cases where a real measurement, not an assumption, decided what shipped |
| [FAILURES.md](FAILURES.md) | A live log of every real break, written at the moment it happened, in strict chronological order, unedited afterward | This is what the Buildathon's "failure recovery" criterion is asking for. Read it to see the wrong first guesses too, not just the fixes |
| [DATA_SPEC.md](DATA_SPEC.md) | The shape of the three input CSVs and the [thirteen break types](DATA_SPEC.md#injected-break-types) the generator deliberately injects | See exactly what "realistic synthetic data" means here, field by field, and why the break-type mix is weighted the way it is |
| [METRICS.md](METRICS.md) | The definition of every number this project reports, and the [reporting rules](METRICS.md#reporting-rules) that govern how they get published | Check what a figure actually means before trusting it, or see the rule (never report a match rate without its false-match rate) this README follows throughout |
| [CLAUDE.md](CLAUDE.md) | The project's own operating rules for building with AI assistance | See the [non-negotiable invariants](CLAUDE.md#non-negotiable-invariants) (money is paise, the LLM never does arithmetic) that constrain every other file in this table |

Raw evidence behind the numbers above, all committed and all reproducible with `--cached`:

| Artifact | Backs |
|---|---|
| [`baseline_rules_only.json`](baseline_rules_only.json) | The rules-only baseline, locked in before the adjudicator existed |
| [`bench_multiseed.json`](bench_multiseed.json) | Arm A's pooled match rate and false-match rate across six seeds |
| [`bench_ablation_seeds.json`](bench_ablation_seeds.json) | Arms B and C's pooled and per-seed match rate, false-match rate, and cost |
| [`bench_adjudication.json`](bench_adjudication.json) | Pooled break-reason accuracy, organic and adversarial |
| [`bench_noise.json`](bench_noise.json) / [`bench_noise_seeds.json`](bench_noise_seeds.json) | The narration-noise sweep on seeds 42, 44, and 46 |
| [`bench_adversarial.json`](bench_adversarial.json) | The adversarial worst-case false-match rate |
| [`bench_scale.json`](bench_scale.json) | Throughput at 5,000 credits |
| [`bench_forecast.json`](bench_forecast.json) | The forecast backtest: mean absolute error, coverage, and fraction of inflow captured |
| [`bench_mutation.json`](bench_mutation.json) | The mutation-testing score |

## Limitations

What this does not do yet, and what it would take.

- **Synthetic data throughout.** The open question is not whether this works on real data in the abstract, but which break types a real merchant settlement file contains that these thirteen do not.
- **One reconciliation pair.** Closed properly rather than splitting the same effort across two closed only partially; see DECISIONS.md's [A second reconciliation pair](DECISIONS.md#a-second-reconciliation-pair-bank-statement-to-invoices) for why.
- **The false-match floor is irreducible without an upstream identifying signal.** The mechanism is a coincidental collision between unrelated amounts landing within the same narrow target, which the numbers alone cannot distinguish from a genuine match.
- **The forecaster reaches about 10% of future inflow.** This is a provable ceiling given T+1 settlement rather than a modelling failure, since settlement data structurally cannot see orders not yet captured; see ARCHITECTURE.md's [Forward cash forecast](ARCHITECTURE.md#forward-cash-forecast-e14).
- **Evidence-grounding gap.** Semantic validation found the model citing references outside the vocabulary the prompt specified; the fix is a prompt change precise enough to stop the habit, plus a full cache regeneration once that prompt changes.
- **Keyless reproduction is scoped to the committed cache.** New data (a new seed, noise level, or adversarial scenario) needs live calls before it reproduces the same way.
- **Single-tenant, batch, not streaming.** One merchant, static CSV files read once per run, with no live gateway ingestion.

## Measurement overruled assumption

Five cases where a real measurement, not an assumption, decided what shipped.

- **The ablation ceiling**, found by measuring the rules-only baseline before the adjudicator existed at all: 11 of 12 credits reaching L4 were deterministically unresolvable by design, capping the LLM's possible contribution to match rate near one point before a single live call was made. See METHODOLOGY.md's [Measurement overruling assumption](METHODOLOGY.md#measurement-overruling-assumption).
- **The L3 false-accept diagnosis, proven wrong by its own follow-up measurement.** A guard was built for a plausible-sounding hypothesis (a small, deliberately unclaimed line landing inside the tolerance band); tracing the actual false matches individually found every one was a coincidental collision with a completely unrelated batch instead. See FAILURES.md, [2026-08-31: a headline metric was a property of one dataset, and the fix for it targeted the wrong mechanism](FAILURES.md#2026-08-31-a-headline-metric-was-a-property-of-one-dataset-and-the-fix-for-it-targeted-the-wrong-mechanism).
- **Break-reason accuracy at three different sample sizes told three different stories.** 91.7% on 12 classified credits from one seed, 89.9% pooled across 69 credits from six seeds (see METHODOLOGY.md's [Measurement overruling assumption](METHODOLOGY.md#measurement-overruling-assumption)), then 85.7% on the 35 credits that survive semantic validation (see FAILURES.md, [2026-09-01: semantic validation of evidence_refs rejects 76 of 77 cached responses, and 51 still fail after the one obvious fix](FAILURES.md#2026-09-01-semantic-validation-of-evidence_refs-rejects-76-of-77-cached-responses-and-51-still-fail-after-the-one-obvious-fix)). No single number in that sequence was a stable estimate on its own.
- **The LLM-only arm's safety record held until retries were forced at volume**, then did not. Zero false matches across six seeds looked like a property of the design; it turned out to be a property of how rarely a retry had happened before. See FAILURES.md, [2026-09-01: retrying against an LLM produced a worse answer than the one it replaced, and a real false match, for the first time](FAILURES.md#2026-09-01-retrying-against-an-llm-produced-a-worse-answer-than-the-one-it-replaced-and-a-real-false-match-for-the-first-time).
- **The forecaster's ceiling** turned out to be provable rather than tunable: given a short settlement lag, every unsettled order floors to the first horizon day by construction, which is why the backtest finds almost no signal past day one. See ARCHITECTURE.md's [Forward cash forecast](ARCHITECTURE.md#forward-cash-forecast-e14). This one produced no FAILURES.md entry; nothing broke, the honest number was simply smaller than hoped.
