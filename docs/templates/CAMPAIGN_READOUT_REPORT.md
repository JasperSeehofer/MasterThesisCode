# Template — Campaign Readout Report

**Status:** research-cycle amendment A7 (2026-08-13). **Owner:** the session that reads out a campaign.
**When:** after the mechanical readout and the adversarial adjudication have both landed, and *before*
the author's stage-5 decision. **Why:** the readout artifacts (`*_READOUT.md/.json`, the adjudication
JSON, the ledger row) are written for the record and for re-derivation. They are not written to be
*understood* by someone returning after a long campaign. This report is the one artifact whose job is
comprehension, and it is what the stage-5 decision is made against.

Origin: author signal 2026-08-13 — *"these kind of reports is exactly what I need after a research cycle
has run, because a lot of things happened and this clearly shows me the whole picture."* Worked example:
the venue-transfer readout (`results/venue_transfer_20260811/`, ledger row #99).

## Binding rules

1. **Never adjudicate.** The report presents the branch the registered tree fired and the decisions it
   puts in front of the author. It does not rule, recommend-by-implication, or soften a defect.
2. **Every number traceable.** Each figure comes from the raw per-seed records or the committed readout
   JSON — never from a summary the report itself computes without a script. Name the source file.
3. **Charts from raw data.** The per-trial distribution is drawn from the per-seed records, not from
   mean ± SE. A distribution against a truth marker communicates a coverage failure in one glance; a
   table of coverage fractions does not.
4. **Vocabulary is mandatory, not optional.** Every symbol in the scorecard gets a plain-language gloss
   *and* the value it took in the decision cell. Assume the reader has been away from the thread.
5. **The adjudicator's flags survive into the report.** Including the ones that do not change the branch,
   each labelled with its branch impact. A report that only carries the confirming half is not a readout.
6. **Portable.** Markdown is the source of record and must stand alone. A rendered artifact (HTML page,
   Pages entry) is a presentation layer over this file, never a substitute for it.

## Structure

### 1. Masthead
Eyebrow line: what kind of measurement, which thread, the scale (seeds / chunks / arrays), the date.
Then the question the campaign asked, phrased as a question a reader can answer yes or no.
Then a **verdict strip**: branch fired · one sentence of what that means · the words
"presented, not adjudicated".

### 2. The goal — what was actually being asked
Two blocks, side by side:
- **The prior finding** — what was already believed, in plain language, and how strongly.
- **The objection this campaign answers** — the specific reason the prior finding might not hold. This is
  the whole reason the campaign was run; if it cannot be stated in three sentences, the campaign was
  under-specified.

Close with the design in one line: what is varied, what is held fixed, and what was locked in advance.

### 3. The design — the arms and the control
One rung per arm, in the registered ladder order, each showing which realism/mechanism axes are ON and
the headline statistic it produced. State plainly what would have counted as an arm *breaking* the
effect (the killing-axis rule), so the reader can see the test could have failed.

Call out the **control** separately: the cell that proves the apparatus itself is unbiased. Its result
is the licence to believe everything else on the page.

### 4. The result — the money chart
The per-trial distribution of the recovered quantity, one row per cell, sharing one axis, with a truth
marker on each row. Then three headline numbers as cards, each with the expected value beside the
observed one:
- the direct decision statistic (e.g. coverage: observed vs the band that was locked),
- the effect size in the units the reader thinks in,
- the effect size relative to the estimator's own claimed uncertainty.

Follow with a **"read this before anything else"** callout naming the *shape* of the failure — not just
that it failed. Confidently-wrong, noisy-wide, and edge-railed are three different diseases with three
different repairs, and only prose distinguishes them.

### 5. The mechanism check
Whatever scaling or dose–response the pre-registration locked, plotted with its registered band drawn
in. This is the section that separates "the effect is real" from "we know what drives it" — and it is
usually the section that decides whether the next step is a repair or a mechanism study.

### 6. The scorecard
Every cell × channel against its locked band, in one scrollable table, decision row visually distinct.
Footnote the band values themselves and the N-row used for each. State that bands were fixed at
pre-registration and unchanged after readout.

### 7. The vocabulary
One card per symbol: the symbol, its plain-language name, two or three sentences of what it means and
why it is on the page, and its decision-cell value. Include the ones that came out clean — an absent
failure mode (e.g. rails at 0) is information.

### 8. Why the numbers stand
- **Validity:** the confound trigger set, evaluated, with the count that fired. The control's result.
  Pins, checksums, census, seed-plan disjointness, provenance chain.
- **Independent recompute:** the adjudicator's verdict, what it re-derived, from what rawness, with what
  agreement tolerance.

### 9. What the adjudicator flagged anyway
Ranked list. Each item: what it is, whether it changes the branch (state this explicitly), and what it
changes instead. New compliance deviations go first and are tagged as such — they enter the ratification
bundle. Interpretive caveats (an arm that proves less than it appears) go last and are tagged
`interpretive`, because they shape the *next* measurement rather than this one.

### 10. The decisions
Numbered, each with: what ratifying/choosing it authorizes, what it forecloses, and its consequence if
taken. Author-gated items say so on their face. This is the decision table the CLAUDE.md "Proposing
decisions" rule requires to live in a reviewable artifact rather than in chat.

### 11. Provenance footer
Arrays / partition / chunk and seed counts / completion state · pre-registration commit · instrument
commit · readout commit · the sentence "branch presented, not adjudicated; bands locked at
pre-registration and unchanged after readout".

## Delivery

Write the Markdown into the campaign's results directory as `CAMPAIGN_REPORT_<date>.md` and commit it
with the readout. Rendering it (Claude artifact today; a Pages entry or a book chapter later) is
encouraged for the decision conversation but is a presentation of this file, not a replacement — the
repository must carry the full report even if the rendered copy is never opened again.
