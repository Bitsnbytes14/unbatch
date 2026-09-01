# METRICS.md

This file defines every number the project reports and how `metrics.py` computes it. Read it if you need to know what a reported figure means or where it comes from.

Every figure in the README, the HTML report, and the pitch video comes from here. A number is not quoted anywhere unless it is defined in this file and produced by `metrics.py`.

## Definitions

Let `C` = bank credit lines requiring reconciliation.

| metric | definition | why it's here |
|---|---|---|
| **count match rate** | resolved credits / total credits | the headline number the brief asks for |
| **value-weighted match rate** | rupees resolved / rupees total | a 90% count rate that misses the three largest credits is a failure |
| **false-match rate** | matched-but-wrong / matched, vs ground truth | the false-positive rate; in reconciliation, a wrong match is more costly than an unresolved one |
| **exception rate** | unresolved / total | reported in full, not hidden |
| **precision** | correct matches / all matches | |
| **recall** | correct matches / all resolvable | denominator excludes `unrelated_credit` and `orphan_settlement`, which are correctly unresolvable |
| **stage funnel** | resolved count per stage L0 through L4 (the five stages of the matching cascade, cheapest and most certain first) | shows which stage resolved each credit |
| **LLM call count** | calls / total credits | shows how much of the dataset the model sees, versus the deterministic rules |
| **LLM cost** | paise (1/100 of a rupee), from the audit log | reports actual unit economics |
| **cost per adjudicated credit** | LLM cost paise / LLM call count | the per-call unit cost, not just the total |
| **cost per exception** | LLM cost paise / exception count | the LLM spend per item a human ends up reviewing |
| **break-reason accuracy** | correct `break_reason` / classified credits | see below, this is the model's own metric |
| **p50 / p95 latency** | per credit, per stage | |

## Break-reason accuracy: the model's own metric

Count and value-weighted match rate score **whether** a credit resolved and to what `payment_ids`, never **why**. By the time a credit reaches L4 (the LLM adjudication stage), `l4_llm.py` has already picked *some* expected batch to compare it against (the closest one, always; see ARCHITECTURE.md's L4 section), so match rate cannot see the adjudicator's actual job at all. That job is naming the right `break_reason`, and it is scored separately:

`decision.reason` (the model's `BreakReason`) against `GroundTruthCredit.break_type`, for every credit whose Decision carries an `llm_model` (meaning it actually reached the adjudicator, the module that calls the LLM) and did not degrade to `adjudication_failed`. No classification was produced at all in that case, so it is excluded from the denominator and counted only in `adjudication_failed_count`. `BreakReason` and `BreakType` deliberately share string values for every category that can reach L4, so this is a direct comparison, not a mapping to maintain.

Reported as:
- **break-reason accuracy**: correct / classified (a single number)
- **break-reason confusion**: `{ground_truth_break_type: {predicted_break_reason: count}}`, the full confusion table, not just the diagonal

The ablation's B-A delta (below) cannot capture this on its own: a small match-rate delta can still coexist with a model that names the right reason for the credits it does touch. Both are reported. A high break-reason accuracy alongside a small match-rate delta is a specific, separate claim: the model explains breaks correctly even where it doesn't change the resolution.

`unrelated_credit` and `orphan_settlement` are **correctly unresolved**. Counting them against recall would penalize correct behavior, so they are excluded from the recall denominator and reported separately as `correctly_rejected`.

## The ablation

Three arms are run on an identical seed and reported side by side:

| arm | command | what it shows |
|---|---|---|
| **A: rules only** | `--no-llm` | the deterministic baseline |
| **B: rules + LLM** | default | the shipped system |
| **C: LLM only** | `--llm-only` | what adjudicating every credit with the LLM alone, with no rules cascade in front of it, produces |

Arm C takes an extra hour to run and is not something every project measures. It is evidence for the cascade's design rather than an assertion of it: the expected result is that C is both worse and more expensive than A.

The shape of claim this section is meant to support, with a chart behind it:

> Deterministic layers resolve X%. The adjudicator recovers a further Y%. Z% escalate to human review. An LLM-only pipeline scores worse than rules alone at approximately N times the cost.

Report the delta B minus A as the value the LLM added on this measurement. If that delta is small, report it as small, with an explanation of where it came from, rather than substituting an unverifiable headline number.

## Reporting rules

1. **Never report a match rate without the false-match rate beside it.** The two trade off against each other, so showing one alone is misleading.
2. **The exception list is published in full**, with a reason per item. No truncation, no "and 12 others."
3. **No cherry-picking.** All metrics come from one seeded run, not a best-of. The brief states explicitly that one cherry-picked match proves nothing.
4. **Report failures of the model too**: malformed JSON count, retry count, `adjudication_failed` count. These are already in the audit log; they are surfaced, not omitted.
5. If a number in the README does not have a line in `metrics.py` producing it, delete the number.

## Target shape

These are not targets to hit by tuning until they appear. They are ranges that indicate the system is behaving sanely. Missing them honestly is an acceptable outcome; tuning to hit them is not.

- count match rate: 85-93%
- value-weighted: within about 3 points of count rate
- false-match rate: under 1%, ideally 0
- LLM calls: under 20% of credits
- exceptions: 7-15%, every one explained

**A 100% match rate means something is wrong.** The data contains deliberately unresolvable records. If the pipeline resolves everything, it is matching things it should not, and the false-match rate will show it.
