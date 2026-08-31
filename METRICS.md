# METRICS.md

What we report and how each number is computed. Every figure in the README, the HTML report and the pitch video comes from here — no number gets quoted anywhere unless it is defined in this file and produced by `metrics.py`.

## Definitions

Let `C` = bank credit lines requiring reconciliation.

| metric | definition | why it's here |
|---|---|---|
| **count match rate** | resolved credits / total credits | the headline number the brief asks for |
| **value-weighted match rate** | ₹ resolved / ₹ total | a 90% count rate that misses the three largest credits is a failure |
| **false-match rate** | matched-but-wrong / matched, vs ground truth | the number that actually matters in finance |
| **exception rate** | unresolved / total | reported proudly, not hidden |
| **precision** | correct matches / all matches | |
| **recall** | correct matches / all resolvable | denominator excludes `unrelated_credit` and `orphan_settlement`, which are correctly unresolvable |
| **stage funnel** | resolved count per stage L0–L4 | shows the cascade earning its structure |
| **LLM call count** | calls / total credits | proves the model sees only the residue, not the batch |
| **LLM cost** | paise, from the audit log | honesty about unit economics |
| **cost per adjudicated credit** | LLM cost paise / LLM call count | the per-call unit cost, not just the total |
| **cost per exception** | LLM cost paise / exception count | what it costs, in LLM spend, per item a human ends up looking at — the framing a payments company actually thinks in |
| **break-reason accuracy** | correct `break_reason` / classified credits | see below — this is the model's real metric |
| **p50 / p95 latency** | per credit, per stage | |

## Break-reason accuracy — the model's real metric

Count and value-weighted match rate score **whether** a credit resolved and to what `payment_ids` — never **why**. By the time a credit reaches L4, `l4_llm.py` has already picked *some* expected batch to compare it against (the closest one, always — see ARCHITECTURE.md § L4), so match rate cannot see the adjudicator's actual job at all. That job is naming the right `break_reason`, and it is scored separately:

`decision.reason` (the model's `BreakReason`) against `GroundTruthCredit.break_type`, for every credit whose Decision carries an `llm_model` (it actually reached the adjudicator) and did not degrade to `adjudication_failed` (no classification was produced at all in that case — excluded from the denominator, counted only in `adjudication_failed_count`). `BreakReason` and `BreakType` deliberately share string values for every category that can reach L4, so this is a direct comparison, not a mapping to maintain.

Reported as:
- **break-reason accuracy**: correct / classified (a single number)
- **break-reason confusion**: `{ground_truth_break_type: {predicted_break_reason: count}}` — the full confusion table, not just the diagonal

This is the number the ablation's B−A delta (below) cannot capture on its own: a small match-rate delta can still hide a real, useful capability if the model is naming the right reason for the credits it does touch. Report both — a high break-reason accuracy alongside a small match-rate delta is a legitimate, specific claim ("the model correctly explains breaks even where it doesn't change the resolution"), not a consolation prize.

`unrelated_credit` and `orphan_settlement` are **correctly unresolved**. Counting them against recall would punish correct behaviour, so they are excluded from the recall denominator and reported separately as `correctly_rejected`.

## The ablation — this is the centrepiece

Run three arms on an identical seed and report all three side by side:

| arm | command | what it shows |
|---|---|---|
| **A — rules only** | `--no-llm` | the deterministic baseline |
| **B — rules + LLM** | default | the shipped system |
| **C — LLM only** | `--llm-only` | what naive "wrap a model around it" gets you |

Arm C is the one nobody else will run, and it is worth the extra hour. It is the evidence for the design claim rather than an assertion of it — the expected result is that C is both worse and dramatically more expensive than A, which is the whole argument for the cascade.

The claim we want to be able to make, with a chart behind it:

> Deterministic layers resolve X%. The adjudicator recovers a further Y%. Z% escalate to human review. An LLM-only pipeline scores worse than rules alone at ~N× the cost.

Report the delta B − A as **the value the AI actually added**. If that delta is small, say so. A submission that honestly reports "the model added 6 points and here's where" is stronger than one claiming an unverifiable 99%.

## Honest reporting rules

1. **Never report a match rate without the false-match rate beside it.** They trade off; showing one alone is misleading.
2. **The exception list is published in full**, with a reason per item. No truncation, no "and 12 others".
3. **No cherry-picking.** All metrics come from one seeded run, not a best-of. The brief says explicitly that one cherry-picked match proves nothing.
4. **Report failures of the model too** — malformed JSON count, retry count, `adjudication_failed` count. These are in the audit log already; surface them.
5. If a number in the README does not have a line in `metrics.py` producing it, delete the number.

## Target shape

Not targets to hit by tuning until they appear — targets that indicate the system is behaving sanely. Missing them honestly is fine; faking them is not.

- count match rate: 85–93%
- value-weighted: within ~3 points of count rate
- false-match rate: **< 1%**, ideally 0
- LLM calls: < 20% of credits
- exceptions: 7–15%, every one explained

**A 100% match rate means something is wrong.** The data contains deliberately unresolvable records. If the pipeline resolves everything, it is matching things it should not, and the false-match rate will show it.
