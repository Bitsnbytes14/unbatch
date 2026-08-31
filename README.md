# unbatch

Settlement reconciliation agent — matches bank credit lines to expected settlement batches.

Built for the Razorpay AI Buildathon, Track 04 (AI Finance Controller).

## The problem

A payment gateway doesn't pay a merchant per transaction — it batches many captured payments into one lump credit, net of per-transaction fees, GST on those fees, refunds issued in the window, and chargebacks, settled under a single UTR. Finance sees one bank statement line and has to work backward: which orders does this cover, and why isn't it the amount they expected? Concretely: a merchant sees a bank credit of ₹4,18,332 and has to explain the gap against the ₹4,19,107 they expected from that day's captures.

## Quickstart

```bash
git clone https://github.com/Bitsnbytes14/unbatch.git
cd unbatch
uv sync
uv run unbatch generate --seed 42
uv run unbatch run --cached
uv run unbatch report
```

No API key needed to reproduce every number below — `data/` and `cache/` are both committed, and `--cached` replays the recorded LLM responses byte-for-byte instead of calling out. `tests/test_end_to_end.py` asserts exactly this: it unsets `OPENAI_API_KEY`, regenerates `data/` from the seed, runs the cascade, and checks the numbers below down to the fraction.

The command above reproduces the default (rules + LLM) arm. To reproduce the full three-arm ablation from the table below:

```bash
uv run unbatch run --seed 42 --no-llm            # arm A — rules only
uv run unbatch run --seed 42 --cached            # arm B — rules + LLM (default)
uv run unbatch run --seed 42 --llm-only --cached # arm C — LLM only
uv run unbatch metrics --seed 42 --arm no_llm
uv run unbatch metrics --seed 42 --arm with_llm
uv run unbatch metrics --seed 42 --arm llm_only
```

## The cascade

Cheapest-and-most-certain first. Each stage only ever sees what the previous stage left unresolved.

| Stage | What it does | Confidence |
|---|---|---|
| **L0** | UTR exact — the settlement UTR appears in the bank narration | 1.00 |
| **L1** | Amount + date exact — credit ties exactly to a computed batch net | 0.98 |
| **L2** | Batch composition — bounded subset-sum finds the payment subset that composes the credit | 0.90 |
| **L3** | Tolerance — credit checked directly against date-windowed batch nets within a fee-derived band | 0.75 |
| **L4** | LLM adjudication — classifies *why* the break happened and proposes a resolution | model-reported |

**Rules compute, the model explains.** The model never does arithmetic — every delta it reasons over is pre-computed by rules and handed to it as a fact. And it never sees the full batch: only the one unresolved credit and its handful of candidate compositions reach L4, never the other ~93 records the rules layer already closed.

## Measured results

Seed 42, 105 bank credits.

| Arm | Match rate | False-match rate | Break-reason accuracy | LLM calls | Cost |
|---|---|---|---|---|---|
| A — rules only (`--no-llm`) | 88.6% (93/105) | 0.0% | n/a (0 calls) | 0 | ₹0.00 |
| B — rules + LLM (default) | 88.6% (93/105) | 0.0% | 91.7% (11/12) | 12 | ₹0.12 |
| C — LLM only (`--llm-only`) | 13.3% (14/105) | 0.0% | 3.8% (4/105) | 105 | ₹1.05 |

Every number above comes straight out of `MetricsReport` in `src/unbatch/metrics.py` (`count_match_rate`, `false_match_rate`, `break_reason_accuracy`, `llm_call_count`, `llm_cost_paise`) — `tests/test_end_to_end.py` pins all of them.

## The ablation, honestly

The model adds **zero match rate** over rules alone — 88.6% either way. That's not a wash: the 12 credits reaching L4 are exactly the ones the rules layer correctly *declined* to guess on (7 have two equally valid exact-sum compositions, 4 sit outside the tolerance band, 1 is an unrelated credit), not ones it missed. A rules-only system has no way to say anything about *why* those 12 broke; it just leaves them as an exception list.

What the model adds is **91.7% break-reason classification accuracy (11 of 12) with cited evidence, for ₹0.12.**

The counter-experiment is arm C: skip straight to the LLM for all 105 credits instead of just the 12 rules can't close. It makes 8.8× as many calls (105 vs. 12) and scores 75.2 points worse on match rate (13.3% vs. 88.6%) and 87.9 points worse on break-reason accuracy (3.8% vs. 91.7%) — for ₹1.05 instead of ₹0.12. That comparison is measured against the same seed and the same model, not asserted: it's the argument for the cascade, not a claim about it.

## Total project spend

**118 paise** (~$0.13) across every LLM call made building and measuring this project, against a $4 budget: 1 paisa for the initial provider-verification call, 12 for the with-LLM arm, 105 for the LLM-only ablation arm.

## Limitations

- **Synthetic data.** All three input files and ground truth come from `generate.py` with a fixed seed. No real bank or gateway data was used or seen.
- **One reconciliation pair.** Bank credit lines against settlement-derived batches. No other document types (invoices, tax filings, bank fee schedules) are in scope.
- **Single-tenant, single-currency.** One merchant, INR only. No multi-merchant or multi-currency handling.
- **No live gateway integration.** Reads static CSVs, not a Razorpay (or any) API — there is no ingestion or streaming layer.
- **Subset composition is bounded by design, not by the data.** L2/L4's composition search caps the candidate pool at 48 lines and the subset size at 25 (`compose.py`); a real batch larger than that would need a different algorithm, not a bigger constant.
