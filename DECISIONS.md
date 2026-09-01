# DECISIONS.md

An architecture decision record for what was deliberately **not** built, and why. Each entry is a considered rejection, not an apology — the reasoning should be evaluable on its own, independent of the deadline that also motivated it.

Format:

```
## <thing not built>
**Considered because:** the genuine case for it
**Rejected because:** the reasoning
**What we did instead:** the alternative that was built
**Would revisit if:** the condition that would change the answer
```

---

## A web frontend / dashboard

**Considered because:** interactive drill-down into exceptions, live filtering, and a more "product-like" surface for a demo.

**Rejected because:** a judge runs this from a fresh clone, possibly mid-presentation. A build step, a dev server, or a bound port is one more thing that can fail at the worst moment, for a dataset (~105 credits, a dozen headline numbers) that doesn't actually need pivoting or live filtering to be legible. FastAPI or Streamlit would both solve a serving problem this project doesn't have.

**What we did instead:** `report.py` renders one self-contained static HTML file via jinja2 — every byte, including CSS, inlined. No server, no JS framework, no build step. Opens directly in a browser from the repo.

**Would revisit if:** the exception list regularly ran into the thousands (where static tables stop being scannable), or this became a maintained internal tool with recurring users who need saved views — a one-shot audit artifact needs neither.

## Real Razorpay test-mode API integration

**Considered because:** would exercise a real network integration and look less like a toy.

**Rejected because:** test-mode only produces Razorpay's own synthetic test transactions — not a real merchant's settlement file, with real narration conventions, real fee-tier history, or real correlated timing. Integrating it would relabel the synthetic-data gap one layer down, not close it, while adding an API key requirement and account provisioning before the first fixture even generates.

**What we did instead:** `generate.py`'s seeded synthetic data, plus a narration-noise generator (E10) built specifically because narration realism — not transaction realism — was the actual open question, and it was addressed directly rather than by hoping a test-mode API's own narrations would happen to be messy in the right way.

**Would revisit if:** a real, anonymized merchant settlement file became available. That's the one thing synthetic data plus a noise model can't manufacture on its own.

## A second reconciliation pair (bank statement to invoices)

**Considered because:** a fuller finance-ops story, closer to a merchant's actual end-to-end workflow.

**Rejected because:** the brief asks to close one loop convincingly, not sketch two loosely. Splitting the same time budget across two pairs would halve the depth available for genuine ambiguity and variance analysis on either one — and a second pair built without the same rigor (multi-seed pooling, a noise curve, an adversarial worst case, break-reason accuracy) would be exactly the unmeasured, cherry-picked shape this project refuses to submit for the first pair.

**What we did instead:** all measurement effort on bank-credit-to-settlement-batch, pushed to a depth (six-seed pooling, adversarial construction, a noise sweep, a stated operating envelope) a second pair at the same budget could not have reached.

**Would revisit if:** the first loop were fully closed and de-risked — it isn't quite; see the operating envelope — and there were time earmarked specifically for breadth rather than depth.

## Multi-tenant support, auth, real-time ingestion

**Considered because:** a production system eventually needs all three.

**Rejected because:** each is its own subsystem with its own failure modes — session management, tenant isolation, streaming backpressure — none of which the brief scores. One loop closed properly beats three half-closed; the brief says a cherry-picked match proves nothing, and it does not ask for breadth either.

**What we did instead:** single-tenant, static CSV files as the input boundary. No auth layer, because there is no multi-user surface to gate.

**Would revisit if:** this were being scoped as a production service rather than a submission judged on one closed loop. Auth and multi-tenancy are usually the *first* thing a real deployment needs — just not the thing this brief measures.

## LangChain / LlamaIndex / any agent framework

**Considered because:** faster to scaffold an "LLM decides what happens next" loop, and demonstrates framework familiarity.

**Rejected because:** the cascade *is* the orchestration — a fixed, auditable, cheapest-first sequence of deterministic stages, with the model called exactly once per unresolved item for exactly one narrow classification. A framework's own agent loop (tool selection, multi-step planning) adds indirection with nothing left for it to decide: which stage handles what is precisely the logic that has to stay legible, not delegated to a framework's internal control flow. One plain SDK call site is also strictly faster to debug than tracing what a framework chose to do and why.

**What we did instead:** a single `openai` SDK call (`client.chat.completions.create`) with a strict JSON schema, called directly from `l4_llm.py`.

**Would revisit if:** a later milestone needed genuinely open-ended multi-step tool use — the model deciding on its own to go fetch more records, say. Nothing here ever needs the model to choose what to look at next, only to classify what it's already been handed.

## Embeddings / vector search for matching

**Considered because:** could match narrations or descriptions by semantic similarity instead of exact rules.

**Rejected because:** this problem is exact, at every stage — a UTR either appears in a narration or it doesn't (L0), an amount ties to a batch net exactly or within a fee-derived tolerance or it doesn't (L1/L3), a subset of real settlement lines sums to the credit exactly or it doesn't (L2). None of that is a similarity-search problem. Even rapidfuzz-level string similarity was considered and explicitly rejected for L0 (E10c) as more power than the measured problem needed; embeddings would add a whole model class, an index, and a threshold to tune on top of that, for answers that are already checkable.

**What we did instead:** substring/exact-equality checks and bounded exact subset-sum (`compose.py`).

**Would revisit if:** the matching problem itself changed shape — e.g. reconciling against free-text merchant descriptions with no structured UTR or amount field at all, where "probably the same thing" is genuinely the best signal available.

## Docker

**Considered because:** environment consistency, "works on my machine" insurance for a judge.

**Rejected because:** it adds a daemon dependency, an image pull, and a build step — failure modes that don't exist for a pure-Python project with no external services to containerize around. Docker solves dependency/OS drift and multi-service orchestration; this project has neither.

**What we did instead:** `uv sync` — a lockfile-pinned, single-command setup, faster to cold-start on a judge's machine than pulling an image.

**Would revisit if:** a real external service entered the stack (see Postgres, below) — same underlying reason either way.

## Postgres (instead of SQLite)

**Considered because:** SQL familiarity, looks more "production-grade," supports concurrent writers.

**Rejected because:** the audit log has exactly one writer — the cascade run itself — and needs to be inspectable with zero setup by anyone who clones the repo. A Postgres instance is a service to install, configure, and keep running, for an access pattern (single process, single run) that never needed a server in the first place.

**What we did instead:** stdlib `sqlite3`, one file per run, gitignored and regenerated fresh from `--cached` every time.

**Would revisit if:** the audit log needed concurrent writers, or needed to be queried by something other than this one codebase.

## Raising MAX_POOL to reduce pool_too_large exceptions

**Considered because:** `bench --scale 5000` showed `pool_too_large` firing on hundreds of credits — a higher cap would resolve more of them instead of exceptioning.

**Rejected because:** the same measurement found pools sitting *just under* the current cap already cost several seconds each (meet-in-the-middle's ~2^24-per-half worst case) — raising the cap relocates that expensive zone to a higher line count without removing it, and pushes the true worst case further into hang territory. "Refusing to solve is correct behaviour; hanging is not" is `compose.py`'s own stated principle. Raising the cap specifically to shrink an exception count is tuning the system to make a number look better, which METRICS.md's honesty rules exist to forbid.

**What we did instead:** `MAX_POOL` stays at 48; `pool_too_large` is reported as a real, visible exception with a real reason.

**Would revisit if:** a faster exact subset-sum algorithm replaced meet-in-the-middle. The 2^(n/2) growth rate is the actual constraint — raising the cap without addressing that would just move the problem, not solve it.

## Fixing L2's coincidental exact-sum collision, and the same collision shape at L3's tolerance check

**Considered because:** this is the entire remaining source of false matches — 0.0–2.11% across six organic seeds (bench_multiseed.json), rising to 5.21% on adversarial data (bench_adversarial.json). Closing it gets false-match rate to a true, general 0%.

**Rejected because:** both are two independently-generated quantities landing within a narrow numeric target of each other by chance — an unrelated batch's net equal to a credit's exact composition sum (L2), or inside its tolerance band (L3). Distinguishing coincidence from the real answer using only the numbers is impossible in principle, not merely difficult — both candidates pass every check the system has access to. The one shape that *was* a real logic gap (L3 mistaking a whole missing settlement line for tolerance noise) was found and fixed; tracing each of the remaining false matches individually confirmed none of them are that shape.

**What we did instead:** measured and published as a floor rather than left unmeasured or quietly tuned away — the multi-seed range, and the adversarial worst case, are both committed artifacts.

**Would revisit if:** an additional identifying signal existed that isn't currently in scope — e.g. a real settlement report's own reference-number field, distinguishing a genuinely duplicated UTR from a coincidentally-close-amount unrelated one, which synthetic data doesn't carry.

## Widening or narrowing L3's tolerance band to improve the numbers

**Considered because:** narrowing it would shrink the coincidence-collision window; widening it would catch more genuine fee-tier drift.

**Rejected because:** the band (`max(50 paise, 0.6% × credit)`) is derived once from the fee structure itself — the actual per-method fee rates and the largest fee-tier shift the generator models — not tuned against observed results. Adjusting it after seeing the false-match numbers would make the band a function of what makes a given run's metrics look best, exactly what METRICS.md's honesty rules exist to prevent.

**What we did instead:** the band stayed fixed at its fee-derived value across every measurement in this project — six seeds, the noise sweep, and the adversarial dataset. Its cost is reported, not hidden by moving the goalposts.

**Would revisit if:** the underlying fee structure changed — a new fee tier, a different GST treatment. The band should track the fees it exists to absorb, never the metrics it produces.

## Partial-UTR fuzzy matching (E10c)

**Considered because:** narration noise degrades L0 badly on its own — resolutions there fall from 79 to 21 as noise rises to maximum (bench_noise.json).

**Rejected because:** the same measurement showed L1 (amount + date exact, which never reads narration) catches every credit L0 loses to noise, at zero false-match cost, flat across seeds 42, 44, and 46. A new lower-confidence fuzzy-match path would be built to solve a problem the data shows the cascade doesn't have — while adding a new way to be confidently wrong.

**What we did instead:** nothing — L0 stays an exact substring check, and the redundancy it needed was already there in L1.

**Would revisit if:** a future noise curve on different data ever showed L1 genuinely failing to backstop L0's losses. This one doesn't.

## Fixing the duplicate_utr / ambiguous_composition confusion (E13)

**Considered because:** on the adversarial dataset every `duplicate_utr` credit that reached L4 was classified as `ambiguous_composition` instead — a consistent, 0%-accuracy confusion, not scattered noise.

**Rejected because:** from the model's vantage point at L4, the two are genuinely indistinguishable — both present as multiple plausible attributions for one credit, built from the same kind of candidate list. Telling them apart needs an upstream signal the data simply doesn't carry to the model (which specific structural cause produced the ambiguity), not a better-worded prompt asking it to guess harder.

**What we did instead:** reported the confusion plainly, on both the organic and adversarial confusion tables (bench_adjudication.json), rather than prompt-engineering around a single dataset's specific instances.

**Would revisit if:** the candidate-construction step (l4_llm.py) could attach a cause-specific signal to each candidate — e.g. explicitly flagging "this candidate arises from a duplicated UTR" versus "this candidate arises from a subset-sum tie" — giving the model something to condition on beyond the shape of the ambiguity.

## More than one retry on malformed LLM output

**Considered because:** could reduce `adjudication_failed_count`, which currently forces an item straight to exception instead of a real classification.

**Rejected because:** a second attempt, with the validation error appended as context, tests whether the first failure was a one-off flake. A third attempt after a second failure tests the same question against evidence that already answered it — either one correction is enough to get the model back onto schema, or it's hitting something a correction can't fix, and repeating the same intervention doesn't change which of those is true. Unbounded retries also unbound the cost of exactly the inputs the model already handles worst.

**What we did instead:** one retry, then degrade to `adjudication_failed` — a real, visible exception row, never a silent drop. Across every live call measured in this project (12 + 57 + 8 + 105 across the seed-42 baseline, seeds 43–47, the adversarial dataset, and the LLM-only arm), malformed JSON count is 0 — the strict schema enforcement at the API boundary made this retry path itself untested by real data, which is worth recording rather than assuming a number for.

**Would revisit if:** a real prompt/model combination showed a nonzero, non-negligible malformed rate where a second correction demonstrably helped. Nothing measured here gives that evidence yet.

## Batching audit-log commits to fix the scale bottleneck

**Considered because:** `bench --scale 5000` found `audit.record`'s per-row `commit()` — not the matching logic — is the dominant wall-clock cost once decision counts run into the hundreds. Batching commits would remove that cost outright.

**Rejected because:** `audit.record` is shared by every command that touches the audit log, and CLAUDE.md's own invariant is "every decision writes an audit row" — durable, one at a time, before a stage returns. Batching trades that guarantee for speed: a crash mid-batch would lose decisions already resolved in memory but never durably recorded, quietly breaking the exact property (a complete, trustworthy audit trail) the whole project's design argument rests on. Changing that mid-benchmark, to make one throughput number look better, is a cascade-behaviour change with a real cost, not a free optimization.

**What we did instead:** reported honestly as a measured, documented cost of the durability guarantee (FAILURES.md, ARCHITECTURE.md's Audit trail and Scale sections). At production scale (105 credits) it is invisible; it only matters at a synthetic scale this project doesn't operate at day to day.

**Would revisit if:** this ran as a genuinely high-throughput service rather than a per-run batch job. Durability-per-decision is the right default until volume makes the crash-window risk worth trading against.

**Revisited (2026-09-01):** the crash-window objection above assumed batching meant one commit for the whole run. `audit.record_many` batches one commit *per stage* instead, wrapped in try/rollback — a failure partway through a stage's writes rolls the whole stage back rather than leaving it half-written, so "every decision writes an audit row" still holds at the same granularity `run_cascade` already reasons in (one stage's decisions, all at once). That's a narrower, safer version of the batching this entry rejected, not the one it was rejecting. `bench --scale 5000`: 11.45s -> 3.47s.

## A layered domain/application/infrastructure restructure

**Considered because:** the module boundaries here (stages, compose, audit, adjudicator) aren't organized into the classic layered folders a longer-lived service would use, and a reviewer used to that shape might read the flat `src/unbatch/` layout as under-designed.

**Rejected because:** a full refactor of a 110-commit project days before submission would end the history with one enormous restructure commit, which reads worse than incremental discipline — exactly the history a reviewer is told to read (README.md's own pointer to the git log). The boundaries that matter already hold without the folder ceremony: the adjudicator isolates the LLM, stages are pure functions with an enforced signature, and metrics.py alone reads ground truth. Renaming that into `domain/`, `application/`, `infrastructure/` would move code without changing any of those actual guarantees.

**What we did instead:** kept the flat layout and let the existing module boundaries (documented in CLAUDE.md's Conventions section and enforced by the stage protocol) carry the separation a layered structure would otherwise be buying.

**Would revisit if:** the project grew past this scale into something with multiple bounded contexts genuinely competing for the same module — nothing here suggests that need yet.

## Separating the matcher from the decision policy

**Considered because:** L2/L3/L4's confidence thresholds and match/exception decisions are currently embedded in each stage's `run()`, rather than a matcher returning candidates for a separate policy layer to adjudicate.

**Rejected because:** same reasoning as the restructure above — this is a real architectural pattern, but not one this codebase is missing the benefit of. The confidence bands (ARCHITECTURE.md's § Confidence bands) already function as the policy layer: each stage's threshold for resolving vs. falling through *is* the policy, just expressed as a constant and a comparison rather than a separate object. Extracting that into its own layer this week would touch every stage for a separation the confidence-band design already provides.

**What we did instead:** left the policy where it's legible — next to the match logic it governs, documented in ARCHITECTURE.md rather than abstracted into its own module.

**Would revisit if:** a policy needed to vary independently of the matching logic (e.g., a per-merchant confidence threshold) — right now there is exactly one policy, so there is nothing yet for a separate layer to parameterize.

## A structured Evidence object replacing reason strings

**Considered because:** `Decision.reason` and `Decision.rationale` are free-form strings; a structured object (the specific fields that justified a match — which comparison passed, which line matched, by how much) would be more queryable and harder to leave vague.

**Rejected because:** this is a real improvement, wrong week — it touches every decision write in every stage, `audit.py`'s schema, `metrics.py`'s scoring, and `report.py`'s rendering all at once, for a benefit (structured querying over reasons) nothing in this submission's judging criteria currently needs. `unbatch exceptions` and the HTML report already read reason strings as the unit of explanation; nothing downstream is blocked on them being unstructured.

**What we did instead:** kept `reason` as a fixed enum-like string per stage (each stage has exactly one constant reason value, e.g. `l3_tolerance.REASON_MATCH`) and `rationale` as free text for L4's LLM explanation — structured enough for the report and exception query to group by, without a new schema migration.

**Would revisit if:** a downstream consumer needed to query on *why* structurally (e.g., "every decision where the matched line's date was more than N days off") rather than by the stage/reason-string pair already available.

## Dependency injection for the adjudicator

**Considered because:** `adjudicator.py` calls the OpenAI SDK directly; an injected client/provider interface would make swapping models or providers a constructor argument instead of an edit.

**Rejected because:** the boundary this would formalize already exists in practice, not just in principle — D0.5's provider swap from Anthropic to OpenAI (see `adjudicator.py`'s module docstring) touched exactly one module and nothing else, because every other stage only ever sees `adjudicator`'s pydantic-validated return type, never the provider SDK. Adding a formal injection seam now would be building an abstraction to prove a property the swap itself already demonstrated.

**What we did instead:** kept `adjudicator.py` as the single, already-isolated module every LLM call passes through — cache lookup, prompt construction, schema validation, and retry/degrade all live there, and nothing outside it imports the provider SDK.

**Would revisit if:** a second provider needed to run side by side (e.g., an A/B comparison), rather than one provider swapped for another — that is a genuinely different requirement DI would serve, which hasn't come up.
