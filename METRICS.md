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
| **p50 / p95 latency** | per credit, per stage | |

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
