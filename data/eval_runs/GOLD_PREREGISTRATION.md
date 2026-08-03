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
| 4 | Show me Phase 3 trials for NSCLC | filter_phase | `NCT02578680` | `189-12`, `KEYNOTE-189` |

Queries 3 and 4 accept either identity value because the protocol document and the publication
reporting that same study are both legitimately responsive.

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
| Refusal text in image captions | Must be **0%** | 86% of retrievable image vectors are refusals today; any survivor means captioning still fails |
| Caption quality on redacted figures | Sample 10 redacted figures; stop if >2 lack substantive description | The vision benchmark used one clean figure; redacted figures are unproven |
| Total chunk count | Stop if it moves >10% from 5,997 | A large swing means chunking or traversal changed, not just captions |
| Spend | Stop at **$3** (estimate is ~$1.35) | 2× overrun means the cost model is wrong |

A trip means stop and report with the evidence — not tune, and not re-run with different settings
to get a better figure.
