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

### 2026-08-30 — seeded (delete this entry once the first real one lands)
**Broke:** nothing yet.
**Cause:** —
**Fix:** —
**Kept:** —

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
