# METHODOLOGY.md

How this was built, as engineering method. AI use in an AI Builder Internship is the expected baseline, not a variable worth measuring — the content worth recording is how the work was directed and checked, not who typed which line.

## Docs before code

`CLAUDE.md`, `ARCHITECTURE.md`, `DATA_SPEC.md`, and `METRICS.md` existed before `src/unbatch` had a single module in it. The invariants, the cascade's stage order and confidence bands, the record shapes and break-type catalogue, and the exact definition of every reported metric were all fixed first.

The point isn't ceremony. A contract written before an implementation exists can't be quietly reshaped by whatever turned out to be convenient to build; a metric defined before any number exists can't be chosen because it happened to look good. METRICS.md's honest-reporting rules — never report a match rate without the false-match rate beside it, no cherry-picking across runs, delete any number with no line in `metrics.py` producing it — were fixed before a single arm had ever been run. That's what made the ablation's central finding (the model adds close to zero match rate, and that's reported as the correct outcome rather than explained away) a measurement instead of a narrative fitted around a result.

## The seven invariants, and why each one exists

1. **Money is `int` paise, never float, never `Decimal` in storage.** A float in a money path is a rounding-drift bug that passes tests right up until the one input that exposes it. Committing to `int` at the parsing boundary makes the entire bug class structurally impossible, not merely tested against.
2. **The LLM never does arithmetic.** A wrong sum from a model is indistinguishable from a right one until someone reconciles it by hand. Keeping every delta rules-computed means everything the model is shown is already a checkable fact, not a number it might silently get wrong.
3. **The LLM never sees the full batch.** Bounds the blast radius of a bad classification — it can misname a break, never leak or reconstruct data outside what it was handed — and ties call count to what the rules layer actually left unresolved, not to dataset size. If the call count approaches the record count, that's a signal the cascade itself has failed, not a cost line to accept.
4. **Bias to exception over wrong match.** A false match is a silent, compounding error discovered weeks later while reconciling something unrelated; an exception is a visible ten-minute task today. The two failure costs are not symmetric, so the metric that matters is false-match rate driven toward zero, not match rate driven toward 100.
5. **Every decision writes an audit row.** Makes the exception report and the HTML report *queries* over one source of truth, rather than a second, separately maintained list that can silently drift from what actually happened.
6. **Runs are reproducible.** A reviewer needs to verify every reported number without an API key and without depending on a live model answering the same way twice. `--cached` makes each number a fact about a committed file, not a claim about what one API call did on one occasion.
7. **Ground truth is never read by the pipeline.** The invariant the whole measurement effort stands on: any stage that can see the answer can be shaped, consciously or not, to match it, and a match rate that's possible to game silently is worthless as a measurement. Enforced by AST inspection (`tests/test_no_scoring_leak.py`) across the whole package, not left to convention.

## Metrics before the adjudicator

The rules-only baseline (arm A) was run, scored, and committed (`baseline_rules_only.json`) before `l4_llm.py` or `adjudicator.py` existed. That ordering is what makes the with-LLM arm's contribution a genuine before/after measurement against a number locked in first — not a number computed after the fact and framed to flatter whatever the LLM arm turned out to do. It's also what surfaced the ablation ceiling (below) before a single LLM call had been made.

## Review gates between milestones

Several of the project's real findings came from re-reading the cascade against its own docs at a milestone boundary, not from a failing test:

- **The L3 combinatorial failure.** L3 was originally a composition search modeled on L2's — "L2 but with slack." Reviewing it against ARCHITECTURE.md's own stated purpose for a tolerance band (absorb fee and rounding noise) found it was structurally the wrong operation, not a mistunable one: composing *within* a band turns "sums to X" into "sums to approximately X," and a modest candidate pool has thousands of near-misses to draw from. No test caught this first — reviewing what the stage was actually *for* did.
- **The L2-to-L4 candidate gap.** Reviewing `l4_llm.py` against the real shape of `ambiguous_composition`'s ground truth (a strict subset of a batch, never the whole batch) found that no candidate the code could construct was ever capable of being the right answer, for any model. This was invisible to the stubbed test suite, since none of those stubs ever populated `candidate_lines` — it surfaced only from reading the actual prompts being sent against real data.
- **The ablation ceiling.** Reading the committed rules-only baseline's own numbers against what the with-LLM arm could possibly add found that 11 of the 12 credits reaching L4 were either provably ambiguous or genuinely unrelated credits — meaning the ceiling on the LLM's own contribution to match rate was near one point, before a single live call was made. Caught by reading the baseline, not by running the with-LLM arm and being surprised by a small delta afterward.

## FAILURES.md as a live log

Entries are written at the moment something breaks, before the fix — never reconstructed afterward. A reconstructed log tends to smooth the wrong turns out of its own account, replacing the actual reasoning path with the reasoning that turned out to be right. Written live, the sequence itself is evidence: the entry recording gpt-5-nano's poor break-reason accuracy, and the *separate*, later entry finding the real cause was the candidates it was shown rather than the model itself, could only exist in that order if the diagnosis genuinely happened that way — not backfilled to look clean after the real cause was already known.

## Measurement overruling assumption

Three cases, each found by measuring something that could have made the design look worse rather than assuming it wouldn't:

1. **The ablation ceiling.** Found *before* the adjudicator existed, by measuring the rules-only baseline first: 11 of 12 credits reaching L4 were deterministically unresolvable by design, capping the LLM's possible contribution to match rate near one point. The response was to rebalance the data so the with-LLM arm would have a real ablation to report, not to quietly lower the bar for what "the model helped" would mean.
2. **The L3 false-accept diagnosis, proven wrong by its own follow-up measurement.** The hypothesised cause of a false-match floor found across six seeds was a small, deliberately-unclaimed line landing inside the tolerance band. A guard was built and verified for exactly that shape — then tracing the four *actual* false matches individually, batch by batch, found none of them matched the hypothesis at all; every one was a coincidental collision with a completely unrelated batch. The guard was kept, correctly, as real hardening for a real shape it does close. The false-match numbers were reported unchanged rather than described as fixed.
3. **The break-reason headline as a small-sample artifact.** 91.7% accuracy on twelve L4 credits from one seed was the number the project's AI-judgment argument leaned on. Pooling the same measurement across six seeds (69 classified credits, with the five new seeds' live calls made and their cache entries committed specifically so the number could be checked) produced 89.9% — close, not identical — with a per-seed range of 80.0% to 100.0%. No single seed, including the original one, was ever a stable estimate on its own.

The same instinct produced the LLM-only ablation arm in the first place: it was run specifically because it might have shown the whole cascade design to be unnecessary, not because it was expected to vindicate it.
