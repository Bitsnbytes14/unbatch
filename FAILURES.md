# FAILURES.md

A live log of what broke and how it got fixed. Written at the moment of breakage, not reconstructed afterwards.

The Buildathon form asks *"what broke, and how you got out"* and says it is the first thing they read. This file is the raw material for that answer.

## Format

```
### <date> — <one-line symptom>
**Broke:** what actually happened, concretely
**Cause:** the real cause, not the first guess
**Fix:** what changed
**Kept:** what this changed about the design, if anything
```

Rules for keeping this useful:

- Write the entry **before** fixing, while the symptom is still confusing. The confusion is the interesting part.
- Record wrong first guesses. "I assumed X, it was actually Y" is the most valuable line in any entry.
- Include the near-misses — things that nearly shipped broken and got caught by a test or a sanity check.
- Do not clean it up later. Rough is credible.
- If `Kept:` is non-empty, that failure changed the architecture. Those are the entries worth talking about in the video.

## Log

### 2026-08-30 — generated CSVs had CRLF (and doubled CRs) on Windows
**Broke:** while building the money-handling round-trip tests for `generate.py`, a quick manual check of a naively-written CSV showed `b'a,b\r\r\n1,2\r\r\n'` — a doubled `\r` before every `\n`, not just a normal CRLF.
**Cause:** two independent things stacked. First, Python's default text-mode file writing on Windows translates every `\n` byte to `os.linesep` (`\r\n`) on write. Second, the stdlib `csv.writer` *also* defaults its own line terminator to `\r\n` regardless of platform. Writing `\r\n` through a text-mode handle that then translates `\n`→`\r\n` produces `\r\r\n`. I assumed "no newline handling in my code" meant "no newline surprises" — wrong; the default is never neutral on Windows.
**Fix:** open every generated file (CSV and JSON) with `newline=""` to disable the platform translation, and pass `lineterminator="\n"` explicitly to `csv.writer`. Verified byte-for-byte: `b'a,b\n1,2\n'`.
**Kept:** DATA_SPEC.md's "byte-identical on any machine" requirement is not automatically true just because the generator has no explicit newline logic — it has to be enforced at every `open()` call. Added a determinism test that inspects raw bytes for `\r` rather than trusting a round-trip-through-pandas comparison, since that would have hidden this exact bug (pandas would decode-and-recompare consistently within one OS and never notice the encoding was platform-dependent).

### 2026-08-30 — git itself would have re-broken the same newline fix on checkout
**Broke:** nothing shipped broken, but a `git show :data/order_ledger.csv | xxd` check while staging commit 10's fixtures confirmed the risk: `git config core.autocrlf` is `true` here (both locally and globally). That setting converts LF to CRLF on checkout for anything git guesses is text.
**Cause:** the staged blob was correctly LF-only (autocrlf only rewrites on the way *out*, and the working file was already LF), so nothing looked wrong in the diff or the warnings — `git add` printed "LF will be replaced by CRLF the next time Git touches it," which is easy to read as noise. It is not noise: it means a future `git clone` or `git checkout -- data/order_ledger.csv` on a machine with autocrlf on would silently hand back a CRLF file, while a fresh `unbatch generate --seed 42` run would still produce LF — the exact byte-identical guarantee this session already fixed once, reopened one layer up.
**Fix:** added `.gitattributes` forcing `text eol=lf` on `data/**` and `cache/**`, so git normalizes to LF on checkout regardless of the client's `core.autocrlf`.
**Kept:** a newline-determinism fix at the application layer isn't sufficient for files that are also committed as "evidence" — the version-control layer needs its own explicit line-ending policy, not an inherited platform default. Same lesson as the first entry, one layer up.

### 2026-08-30 — rebalancing to smaller batches made credits go negative
**Broke:** commit 12's rescale (18 batches of 4-9 lines down to 68 batches of 2-4 lines, to hit ~70 credits over ~200 payments) immediately hit `pydantic_core.ValidationError: credit_paise must be non-negative` while building the bank statement baseline — twice, from two different causes, back to back.
**Cause:** first guess was that only the settlement_split batches were at risk (splitting a batch's lines in half can put a refund-heavy half on its own). Fixing that wasn't enough — the very next run failed on `CHARGEBACK_BATCH_INDEX` instead: the flat `CHARGEBACK_FEE_PAISE` (Rs 500) penalty plus normal fees can exceed a 2-line batch's other line entirely, since the old 4-9-line batches had enough unrelated lines to absorb a fixed penalty that a 2-line batch can't. Smaller batches don't just scale down proportionally — a fixed cost that was noise against 8 lines is structural risk against 2.
**Fix:** two changes. (1) Organic (non-forced) refund/partial-refund rolls now check projected batch headroom before committing — `MIN_BATCH_NET_BUFFER_PAISE` — and silently revert the order to `captured` if a refund would push the running total negative, rather than trusting probability. (2) `CHARGEBACK_BATCH_INDEX` (and settlement_split, ambiguous_composition) are forced to a fixed 4 lines via `_FORCED_N_LINES` instead of the general 2-4 range, so there's always enough headroom for their fixed costs.
**Kept:** any batch that carries a fixed-cost break (a flat fee, a full refund) needs either a headroom check or a forced minimum size — "at least one of every break type" and "smaller batches" are in real tension, not just a scale change. Added `_SPECIAL_BATCH_INDICES` to suppress organic noise on every hand-built batch, so the two don't fight each other.

### 2026-08-30 — rounding_delta's guaranteed gap silently became zero
**Broke:** after fixing the negative-credit errors above, `generate()` ran clean, but a manual check of the rounding_delta credit showed `delta=0` — the one break type whose entire point is a nonzero paise-level gap had none.
**Cause:** the original 4-9-line batches had enough lines that *some* line's fee almost always landed on a rounding boundary by chance. At 2-4 lines, that's no longer reliable — this specific seed's `LARGE_VALUE_BATCH_INDEX` batch happened not to hit one. The "floor" case of L3's tolerance band had been relying on an unstated probabilistic assumption the whole time; shrinking the batch size just made it visible.
**Fix:** force two CARD-method lines in that batch to an exact gross (`ROUNDING_BOUNDARY_GROSS_PAISE = 500_025`) that lands on a half-paise fee boundary individually (500025 × 2% = 10000.5 → 10001 each) but sums to a whole-paise batch total (1000050 × 2% = 20001.0 exactly) — a deterministic 1-paise gap, not a hoped-for one.
**Kept:** every "at least one" guarantee in this generator now has to be true by construction, never by the odds looking favorable at a given batch size — this is the second time in one session a "probably enough lines" assumption broke silently (see chargeback, above).

<!--
Likely candidates, based on where the design is thin. Delete as they either happen or don't:
  - subset-sum blowup in L2 before pooling caps were added
  - float creeping into a money path and passing tests anyway
  - date window off-by-one dropping T+2 settlements
  - model returning prose around the JSON block
  - cache key not including model/prompt version, so stale responses replayed after a prompt change
  - GST rounding applied per-line vs per-batch giving different paise totals
  - ground_truth accidentally imported into a stage, inflating the score
-->
