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
| A1 — free re-reads before compute | exhaust on-disk diagnostics before ANY new compute | 2 |
| A2 — paired read with class-summed comparisons | per-event read beside every aggregate | 0 |
| A3 — harness acceptance criteria | 2-channel, production-N, multi-candidate SBC | 0 |
| A4 — evidence correction on A1/A3 | corrected precedent numbers | 0 |
| A5 — Stage L external consult | literature rings R0–R4, `[LIT]` intake, gate item 6 | 1 |
| A6 — periodic assumption & performance audit | cadence/trigger re-validation ritual | 0 |
| A7 — campaign readout report | comprehension-first report before stage-5 decision | 0 |
| A8 — branch-referent / two-sidedness / execution-completeness / band-derivation | registration-time checks | 0 |
| A9 — sequential-escalation rule | PROPOSED, not adopted — tracked for completeness | 0 |
| A10 — invariance & blindness declaration | fixed invariants + blind-spot sentence per prereg | 1 |
| A11 — provenance freshness | {value, source, date, config} stamps; STALE may not be quoted | 0 |
| A12 — score-zero first diagnostic | E[∂_θ ln L]=0 at truth, class/covariate-resolved, free | 1 |
| A13 — engagement gate | instrument must demonstrably move the output | 0 |
| A14 — attribution ships with its falsifier | falsifier registered before banking | 0 |
| A15 — power-calibrated gates, can-fail controls | operating characteristics at actual N; no vacuous control | 2 |
| A16 — retrospective ledger + impact tracker | this instrument | 0 |
| A17 — gate/band portability + realized-scatter re-check | operating characteristics survive transport | 0 |
| A18 — explicit bias reference | every scorer prints what each bias subtracts | 0 |
| A19 — symmetric falsifiers | every fireable branch carries a registered falsifier | 0 |
| A20 — clean-context adversarial verification | builder's context never verifies its own claim; typed narrative | 0 |

## Evidence lines

*(append `- YYYY-MM-DD A<k> +1 — <one clause> — <artifact/commit>` below, newest last)*

- 2026-08-21 A1 +1 — pre-check O2 (impostor-leg decomposition) found 73% of B-SEL's −0.1083 on banked CSVs at zero compute, firing C-SG's design-change trigger BEFORE its 51–69 CPU-h were spent — ledger row #149, `decompose_impostor_leg.py`
- 2026-08-21 A1 +1 — pre-check O3 (matched-channel read) overturned O2's "small residual" reading (−0.0291 was cancellation, the matched channel is −0.0846) on the same banked data, again pre-compute — ledger row #150, `decompose_matched_channel.py`
- 2026-08-21 A12 +1 — the score-zero frame (E[∂_h ln L_matched]=0 for the dark-conditional) is what defined the matched channel and localized the completion-leg violation; the per-event score-at-truth read (−0.28 → −0.06) carried the O2 readout — ledger rows #149–#150
- 2026-08-21 A15 +1 — forced the O2/O3 band design to state that a deterministic paired read has no sampling null, preventing a repeat of the A-7 `max(0.005, 2·SE)` mistake; materiality bands were derived from C-SG's resolution instead — prereg O2/O3 band registrations
- 2026-08-21 A5 +1 — independence-preserving Stage-L sweep found our venue sits BELOW Gray 2020's own validated completeness floor (G20-d) and that the O2/O3 per-sector decomposition has no literature precedent (documented field gap → candidate novel contribution) — docs/LITERATURE_WARNINGS.md, commit a80ce4b2
- 2026-08-21 A15 +1 — the pilot-first mandate + pre-committed STOP caught the orchestrator's own channel-ported GATE V (false-fails 5/12 known-informative reference seeds) before the 42-seed fleet ran — retrospective ledger entry 1, prereg "PILOT GATE V AMENDMENT"
- 2026-08-21 A10 +1 — the registered six-invariant blindness list converted the INTERNAL-DEFECT verdict into a named-conditional claim and forced the S̄_φ audit designation as the next step — ledger row #151 item 5
