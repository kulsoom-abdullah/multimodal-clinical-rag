# Gold label pre-registration — written before the index rebuild

Committed **before** the corpus is reingested, so the definition of a "hit" cannot be chosen
after seeing which definition produces a better number.

Baseline for comparison: `baseline_2026-08-03.txt` (100% Recall@3, 4/4, index of 5,997 chunks
built with the pre-fix ingestion).

---

## Why the definition needed settling

The rebuild changes what identity means. Ingestion previously overwrote every document's
`trial_id` with the name of the directory it was filed under; it now keeps the extracted value
and adds `protocol_number` / `amendment`. Two consequences for scoring:

1. A document's `trial_id` after the rebuild may legitimately differ from the one in the baseline
   index. The `205801` master protocol's sub-study SAP is registered separately from the master
   record, so `NCT03739710` and `NCT06926673` can both be correct for different documents of the
   same study.
2. A single study can carry more than one `protocol_id`, because the corpus mixes protocols with
   publications *about* those protocols. Recomputed from the current index:
   - `NCT02578680` → `189-12` (protocol) and `KEYNOTE-189` (the NEJM paper)
   - `NCT02423343` → `H9H-MC-JBEF` (protocol) and `<UNKNOWN>` (the BMC Cancer paper)

So neither `trial_id` nor `protocol_number` alone is a clean key. Gold is therefore defined as an
explicit **set of acceptable identity values per query**, enumerated below.

## Scoring rule (fixed now)

**Both keys are reported, every time, side by side.** Neither is dropped based on its result.

- **`trial_id` Recall@3** — kept because it is the only measure directly comparable to the
  committed baseline. This is the headline before/after comparison.
- **`protocol_number` Recall@3** — the measure that matches the corrected identity model. Reported
  alongside, and becomes the primary metric in later work.

A query counts as a hit if **any** of the top 3 reranked chunks carries an identity value in that
query's gold set for the key being scored. Rank within the top 3 does not matter. `n = 4`.

## Gold sets

| # | Query | Type | Gold `trial_id` | Gold `protocol_number` |
|---|---|---|---|---|
| 1 | What is the primary objective of study 3000-02-005? | protocol_lookup | `NCT05751629` | `3000-02-005` |
| 2 | What are the common side effects of PARP inhibitors? | semantic | `NCT05751629` | `3000-02-005` |
| 3 | What is the dosing schedule for Galunisertib? | semantic | `NCT02423343` | `H9H-MC-JBEF`, `<UNKNOWN>` |
| 4 | Show me Phase 3 trials for NSCLC | filter_phase | `NCT02578680` | ~~`189-12`, `KEYNOTE-189`~~ → **`189`, `KEYNOTE-189`** (corrected, see below) |

Queries 3 and 4 accept either identity value because the protocol document and the publication
reporting that same study are both legitimately responsive.

### Correction to Query 4's `protocol_number` gold — label error, not tuning

**Original gold:** `{189-12, KEYNOTE-189}`. **Corrected to:** `{189, KEYNOTE-189}`.

**Why it was wrong from the start.** `protocol_number` did not exist when this file was written —
it is introduced by the same change the rebuild implements. There were therefore no actual
`protocol_number` values to write gold against, and I used the **baseline index's `protocol_id`
values** as a stand-in. That was the error: gold for a new field, derived from a different field.

In the rebuilt index, `NCT02578680/Prot_SAP_001` carries `protocol_id=189-12` but
`protocol_number=189` with `amendment=Amendment 12` — the extractor read `189-12` as protocol 189,
amendment 12. Whether that reading is right is a separate question; what matters here is that the
original gold value `189-12` **never appears in the `protocol_number` field at all**, so the query
could not have scored a hit under any retrieval behaviour. It was unreachable, not failed.

**This is a label correction, not a threshold adjustment.** The distinction: retrieval returned the
correct document (`Prot_SAP_001`, which is why the same query scored HIT under `trial_id`); only
the label it was compared against was wrong. Nothing about the system was changed to make this
pass, and no other gold value was touched. The corrected value is read directly from the rebuilt
index and is recorded here before the re-scored eval is run.

Gold for the other three queries is unchanged, and `trial_id` gold is unchanged throughout.

**No eval query currently targets the `205801` family.** The question of whether a `205801` query
scores on `protocol_number` (one study) or on a single registration is therefore not exercised by
this eval set. Recorded here so the absence is a known gap rather than a silent one: it is part of
the deferred eval expansion, and the rule above (`protocol_number`, set-valued) is what will apply
when such a query is added.

## Known defects in this eval set, carried forward unchanged

Stated so the post-rebuild number is not read as more than it is:

- **n = 4.** One query flipping moves the figure by 25 points.
- **Query 1 is tautological.** Strict mode filters on `protocol_id = 3000-02-005`, and it is then
  scored on whether `NCT05751629` appears — the same 343 chunks. It cannot fail except by
  returning nothing.
- **Query 2's gold is contestable.** Three trials in the corpus mention PARP; niraparib in
  `3000-02-005` makes `NCT05751629` defensible, not exclusive.
- **Random floor is 25.0%** for the `trial_id` key on the baseline index, computed from per-trial
  chunk shares. It must be recomputed after the rebuild, since chunk shares change.

Replacing the tautological query and expanding beyond n=4 is deferred to a separate increment.

## Stop-and-report thresholds (fixed now)

The rebuild halts and is reported rather than accepted if any of these trip:

| Check | Threshold | Rationale |
|---|---|---|
| `trial_id` Recall@3 | **Any** query flips hit → miss | At n=4 a single flip is 25% of the set; too coarse to absorb quietly |
| Dangling parent refs | Must be exactly **0** | The join is the thing being fixed; nonzero means it is not fixed |
| Vectors per figure | Must be exactly **1** | Currently 225 vectors over 130 files; duplication means the rebuild did not replace |
| Failure text in image captions | Must be **0%** for all three shapes | 86% of retrievable image vectors are refusals today; any survivor means captioning still fails |
| Caption quality on redacted figures | Sample 10 redacted figures; stop if >2 lack substantive description | The vision benchmark used one clean figure; redacted figures are unproven |
| Total chunk count | Stop if it moves >10% from 5,997 | A large swing means chunking or traversal changed, not just captions |
| Spend | Stop at **$3** (estimate is ~$1.35) | 2× overrun means the cost model is wrong |
| Wall clock | Stop at **3 hours** | Added after the sequential build proved unbounded in time while nowhere near the spend limit |

A trip means stop and report with the evidence — not tune, and not re-run with different settings
to get a better figure.

### Gap in the original threshold set: time was never bounded

The first threshold set bounded **spend** but not **wall clock**. That was the wrong constraint to
pick. A sequential build made 5,772 strictly serial summarization calls whose throughput swung 20×
(25 chunks/min in one 4-minute window, 1.26 chunks/min over a 159-minute window), giving a
completion estimate somewhere between 4 and 73 hours — while total spend stayed at a few cents,
nowhere near the $3 limit. Cost was never going to stop a run that could not finish.

A 3-hour wall-clock threshold is therefore part of the pre-registered set. Recorded as a gap in the
original reasoning, not as a threshold discovered convenient after the fact.

### The four caption-failure shapes

The caption check scans for **all four**, each of which produces text that is indistinguishable
from a description once embedded:

1. Model refusal — `"I can't see the image at the path you gave..."` (the shape in the current index)
2. `Error analyzing image: <exception>` — a swallowed exception returned as the caption
3. `Image file not found: <name>` — a missing file returned as the caption
4. **`IGNORE_IMAGE` appearing as a substring of a longer caption** — the classification token
   narrated into the description instead of replacing it

Shape 4 is checked as: a stored caption may be *exactly* `IGNORE_IMAGE` (a legitimate decision that
this image carries no clinical information), or must not contain the token at all. Anything in
between fails.

**Why this was widened, and when.** Only shape 1 was known when this file was first written.
Shapes 2 and 3 were found while auditing timeout coverage in `ingest_data_advanced.py`: a bare
`except Exception` in the vision path returned the error text *as the caption*, and it sat inside
the `@backoff`-decorated function so vision failures never retried. Both were removed — the vision
path now raises.

This widening happened **before the rebuild produced any results**. Nothing was tuned to an
outcome; the check was extended because a second and third failure shape were discovered to exist,
at a point when there were no numbers to tune toward. Recorded here so the sequence is auditable.

**Shape 4 was added later, and the sequence there is different — stated plainly.** It was found in
output from an aborted run (78 captions written before the run died on exhausted API credits): 39
were exactly `IGNORE_IMAGE`, and the other 39 opened with `1. **CLASSIFY**: IGNORE_IMAGE` and then
described the figure anyway. So unlike shapes 2 and 3, this one *was* discovered by looking at
results. It is recorded here before any **scored** rebuild exists, and it tightens the check rather
than loosening it — but the honest description is "found in a failed run's output", not "predicted
in advance".

### Additional pre-registered check: trial identity provenance

Ingestion now records `trial_id_source` on every chunk — `extracted` when the document's own NCT ID
was recovered, `source_dir` when it fell back to the directory name because extraction returned a
placeholder. Report the split; it is descriptive, not pass/fail. The check that *is* pass/fail:
**no chunk may carry a placeholder as its `trial_id`.** An earlier build put 59.2% of chunks under
`<UNKNOWN>` or `To be determined` because a truthy-check treated those strings as real answers.

### Client configuration for the rebuild

One `summarizer_llm` previously served both the vision and text paths, so isolating vision
required a **third client** rather than raising a shared value:

| Client | Path | Timeout | Retries |
|---|---|---|---|
| `extractor_llm` (Claude Sonnet 4.5) | metadata extraction | 60s | 3 |
| `summarizer_llm` (gpt-4o-mini) | text / table chunks | 60s | 3 |
| `vision_llm` (gpt-4o-mini) | figure captioning | 120s | 3 |

120s for vision has headroom against the benchmark latencies measured on the genomic heatmap —
gpt-4o-mini 9.45s, gpt-4.1-mini 7.96s, Claude Sonnet 4.5 17.13s — while leaving room for larger
redacted pages. Timeouts are load-bearing: `@backoff` fires only on exceptions, so a socket that
stalls without erroring never triggers a retry.
