# Amendment Impact Tracker

**Established 2026-08-21 by author mandate** (verbatim, from the 2026-08-21 overnight-autonomy
instruction): *"I also want you to add a tracker to the existing amendements that get +1 each time
they contribute meaningfully. you can also implement new amendements."*

**Contract.** One row per amendment in `docs/RESEARCH_CYCLE.md`'s amendment ledger. The **count**
increments by exactly 1 each time the amendment *meaningfully contributes* — it changed a design
before a mistake shipped, caught a defect, forced a disclosure that mattered, or saved compute.
Every `+1` is backed by a dated evidence line appended under the row **in the same commit** that
claims it; a count with no matching evidence line is void. Routine compliance (the amendment was
merely followed without changing an outcome) is **not** a contribution. Counters start 2026-08-21
at 0; contributions before that date live in the amendment ledger's own evidence column and are
not retro-credited.

The Retrospective Ledger (`docs/RETROSPECTIVE_LEDGER.md`) field 2 is the primary feeder: at every
cycle end, the amendments that earned their keep are named there and credited here.

| amendment | one-line scope | count |
|---|---|---|
| A0 — establishment | the 7-stage cycle itself | 0 |
| A1 — free re-reads before compute | exhaust on-disk diagnostics before ANY new compute | 0 |
| A2 — paired read with class-summed comparisons | per-event read beside every aggregate | 0 |
| A3 — harness acceptance criteria | 2-channel, production-N, multi-candidate SBC | 0 |
| A4 — evidence correction on A1/A3 | corrected precedent numbers | 0 |
| A5 — Stage L external consult | literature rings R0–R4, `[LIT]` intake, gate item 6 | 0 |
| A6 — periodic assumption & performance audit | cadence/trigger re-validation ritual | 0 |
| A7 — campaign readout report | comprehension-first report before stage-5 decision | 0 |
| A8 — branch-referent / two-sidedness / execution-completeness / band-derivation | registration-time checks | 0 |
| A9 — sequential-escalation rule | PROPOSED, not adopted — tracked for completeness | 0 |
| A10 — invariance & blindness declaration | fixed invariants + blind-spot sentence per prereg | 0 |
| A11 — provenance freshness | {value, source, date, config} stamps; STALE may not be quoted | 0 |
| A12 — score-zero first diagnostic | E[∂_θ ln L]=0 at truth, class/covariate-resolved, free | 0 |
| A13 — engagement gate | instrument must demonstrably move the output | 0 |
| A14 — attribution ships with its falsifier | falsifier registered before banking | 0 |
| A15 — power-calibrated gates, can-fail controls | operating characteristics at actual N; no vacuous control | 0 |
| A16 — retrospective ledger + impact tracker | this instrument | 0 |

## Evidence lines

*(append `- YYYY-MM-DD A<k> +1 — <one clause> — <artifact/commit>` below, newest last)*
