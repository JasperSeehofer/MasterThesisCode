# DEDUP / CONFLICTS — end-of-fan-out verifier pass, items 1–11 (item 12 fragment truncated)

Filed 2026-08-30, as part of the REGISTERED END-OF-FAN-OUT VERIFIER PASS
(`REGISTRATION_END_VERIFIER_PASS_20260829.md`). Scope: dedup/cross-reference over the
19-item NODE payload supplied to this task, of which **items 1–11 arrived complete** and
**item 12 arrived truncated** (its JSON is cut off mid-`author_rule_items` string —
`"...Adoption returns to the author as a fresh [R`). Items 13–19 and item 20 (wave-3
blind readout, DEFERRED — cluster SSH down) were not present in the payload handed to
this file and are out of scope here; only items 1–11 (+ the item-12 fragment) are
analysed below. Nothing in this file re-adjudicates a verdict; it only cross-references
what the 11 independent verifier passes already found.

Counts: **(a) 6 duplicate/overlap clusters · (b) 1 flagged conflict-candidate (no hard
numeric contradiction found) · (c) 19 distinct author_rule_items reduced to 13 unique
questions, of which 7 map onto SYNTHESIS_DOCKET_2 §6 items 1–8 · (d) 6 governance-breach
entries from 4 items · (e) 1 item flagged for incomplete data (item 12, truncated
payload) — no item among 1–11 shows a vague/non-source method.**

---

## (a) Duplicated findings across items

1. **Shared-file line-number drift from concurrent wave-3/adoption commits, independently
   flagged as "expected churn, not an error" by four different verifiers.**
   - Item 1: ternary citation `bayesian_statistics.py:5187-5191` (write-time) drifted to
     `:5215-5219` (HEAD), +28 lines from `d4765539`/`0d0eb691`.
   - Item 6: full suite passed-count 1851 (record) vs 1896 (my re-run) — sibling nodes
     added tests between HEAD moves a794404c→b87ad2e6.
   - Item 9: site 2.3 citation `1692-1704` (record) vs `1691-1701` (pre-edit HEAD);
     full-suite 1851 vs 1896 passed, same root cause as item 6's.
   - Item 11: falsifier suite "52 passed" (record, HEAD dd63fe0c) vs 58 collected at
     current HEAD b87ad2e6 (+6 tests from other charter nodes).
   All four verifiers independently reached the same conclusion (non-material, shared
   mutable tree, not a defect in the node under review) — reported as four separate
   "discrepancies" rather than being cross-referenced to each other in the source
   records. No adjudication needed; flagged here only to avoid double-counting as four
   distinct defects.

2. **B0-A′ INSTRUMENT-DEFECT Z-scores independently re-derived twice from the same raw
   log/CSV set.** Item 2 computes this as its own primary decisive number (Z_b=-3.676431,
   Z_s=-7.078607, pooled N=461). Item 6 independently re-confirms the identical figures
   (`Z_b=-3.676430700268586`, `Z_s=-7.078606542881258`) by reading
   `hier_s0_registered_run/s0a_score.md` as background context for its own KW-Q1 read.
   Both computations agree to 6+ significant figures — a genuine independent
   cross-check, not a conflict, but the same underlying number was re-verified twice
   inside this same verifier pass without either item citing the other.

3. **The B7.3 `eff`/`mz_sel` adoption's "returns to author after wave-3" status is
   surfaced independently by three items** (6, 10, 11), all converging on the same
   fact set (falsifier (ii) unrun, ≈208–286 CPU-h at the old anchor; C4 IMMATERIAL-
   PREDICTED; suite 1896; independent verifier PASS) and all correctly deferring the
   ratification call to the author. This is docket §6 item 4 (see part (c) below) —
   three verifiers reaching the ledger row independently is corroboration, not new
   information, and should be read as one open item, not three.

4. **PA-HIER-31's two open smear-form contradictions (R1/R2′)** are surfaced by both
   item 1 (via the veto-invitation over `smear_global_selection` dispatch convention,
   and the b-grid-staleness return item) and item 2 (explicitly, "REVISION NOTE 2, R2′
   ... pending its own fresh author [RULE]"). Same underlying open item (docket §6
   item 2), reached independently by two verifiers auditing two different nodes (B1.1
   wave-1 record and Stage-0 record respectively) built on the same driver family.

5. **The F-ii mass-window design ruling** is the explicit subject of item 8's returns-
   to-author section (matches docket §6 item 1 verbatim) and is foreshadowed by item 7's
   own returns-to-author note ("wave-2 counterfactual arm ... proposed, NOT launched by
   this node" — the precursor to the same design question). Not a contradiction: item 7
   is the B5.1 gate-implementation record (the instrument), item 8 is the B5.2 C3
   counterfactual (the measurement) that the same open ruling actually turns on.

6. **The impostor-family open sub-convention question** appears with two different
   framings that are actually one open thread: item 5's row #167 D̃_φ sub-convention
   ("COMPLETED-SMALL vs COMPLETED-MATERIAL") and item 6's E21 divisor-gap caveat on
   KW-Q1's R (both trace to the same un-normalised D̃_φ/ρ(θ) algebra inside the s_imp
   bracket, per item 6's own citation of `B1_1_S0A_DEFECT_FORENSIC_20260829.md` E21).
   Neither item cites the other; both are pointing at the same unresolved algebra gap
   from two different vantage points (B4.1 claim-card decomposition vs B4.2 KW-Q1
   readout).

---

## (b) CONFLICTS

**No item's re-derived number or verdict directly contradicts another item's
re-derived number or verdict, or a record, within items 1–11.** One cross-item tension
is flagged for the adjudicator as a possible naming/quantity conflation, not a proven
contradiction:

- **Item 7 (B5.1) vs Item 8 (B5.2), "retention" figures.** Item 7 independently
  re-derives a fleet-level true-host retention of **0.9567→0.7890** (linear-k1.5 →
  log-k3) from a 24-arm jackknife (mean 0.7898±0.0455, SE 0.0093) and treats this as a
  confirmed, accurate production-relevant number (byte-identity + gate PASS). Item 8's
  R1 mechanism gate, reading the *actual* wave-2 C3 production arm against the baseline,
  finds host-recovery **identical at 66/76 (86.84%) in both arms** and explicitly
  frames this as **falsifying** "the mirror's 0.789±0.009 → [0.762,0.816] band" as a
  *retention-transfer prediction* for the specific production fleet, with a documented
  mechanism (100% of the 621 candidate-set changes are dark-class, 0/76 in-catalogue
  events change positivity). These are not the same statistic — item 7's number is an
  aggregate synthetic/injection-fleet retention rate; item 8's is a specific 76-event
  in-catalogue recovery count on one production fleet — and neither verifier disputes
  the other's arithmetic. But the two numbers are easy to conflate (both are "log-k3
  retention ≈0.79 vs a possible 66/76 read"), and item 8 itself uses the word
  "falsifies" against a figure item 7 independently confirmed as correct for its own
  registered measurement. **Flagged for the adjudicator to confirm these are indeed
  two distinct, non-conflicting statistics** (as both items' own text argues) before
  either number is used interchangeably in a synthesis document.

---

## (c) Union of author_rule_items — deduplicated, cross-referenced to docket 2 §6

`SYNTHESIS_DOCKET_2_20260829.md` §6 lists 8 numbered `[RULE]`s returning to the author.
Below, every `author_rule_items` entry across items 1–11 is folded into that numbering
where it matches; items with no §6 match are listed as **NOT IN DOCKET §6** — either
genuinely new since docket 2 was filed, or a narrower/process-level item docket 2 did
not itemize.

| # | Question (deduplicated) | Raised by | Docket 2 §6 match |
|---|---|---|---|
| 1 | Mass-window design: adopt log-k3 as documented design choice vs keep linear-k1.5 vs commission a k-scan; folds in WGEOM §9 F-ii | Items 7, 8 | **§6 item 1 (F-ii)** — exact match, item 8 quotes it near-verbatim |
| 2 | PA-HIER-31 open contradictions: (a) R1 unconditional `smear_sigma_z` pin vs `smear_global_selection=False` for CoR-P; (b) R2′ same pair for CoR-M/S0-A; b-grid staleness (driver still on ±0.02 vs PA-HIER-29's ±0.0661 re-anchor); veto invitation over the per-node smear dispatch convention | Items 1, 2 | **§6 item 2** — (a)/(b) match exactly; the b-grid-staleness and dispatch-convention sub-points are item 1's own gloss on the same open contradiction, not separately itemized in §6 |
| 3 | θ-consistent divisor fix (Σ^φ(θ) extension + sky-cone-radius flag) as a physics-change gate presentation for the next tree | Items 5 (p_Di hook ruling), 6 (E21 divisor-gap not folded into B4.2, non-physics-hook ruling for B4.3), 9 (judgment call on raw-z vs z̃ for sigma_z_pv — narrower, see row below) | **§6 item 3** — items 5 and 6's framing map onto it directly; item 9's raw-z-vs-z̃ judgment call is a *sibling* open question on the same commit family, not literally §6 item 3's sky-cone/Σ^φ(θ) scope — see "NOT IN DOCKET §6" note below |
| 4 | Ratify `mz_sel`/`eff` (B7.3) as production default, or revert to "off", pending falsifier (ii) and the wave-3 blind readout + `off` arm | Items 6 (row #253), 10 (falsifier (ii) outcome; wave-2 arm falsifier (iii) MATERIAL-DOWN review), 11 (row #220 falsifier (ii) unrun, PROVISIONAL cap) | **§6 item 4** — exact match, three independent citations (see dedup cluster (a)3 above) |
| 5 | G7 row 16 re-grade: "MEASURED, calibration-affecting" → "mock: zero by construction; real data: O(1) degeneracy" | Item 4 | **§6 item 5** — exact match |
| 6 | Launch S0-B now (REPORTED-ONLY, post-hoc ρ(θ) subtraction disclosed) vs launch-after-fix | Items 2 (explicit), 6 (B0-A′ forensic resolution as a precondition, same underlying gate) | **§6 item 6** — exact match |
| 7 | CMEM pooled-observation awareness: two independent fleets both read deficit-direction (row #219 p=0.0152; A1 p=0.0358); no pooled statistic computed or banked — leave unregistered or open a registration? | Item 3 (the item's own subject matter; not phrased as an open question in its `author_rule_items` list, but its "discrepancies" section explicitly confirms no pooled statistic exists anywhere) | **§6 item 7** — item 3 supplies the from-source confirmation this rule's premise rests on |
| 8 | R2c bank-vs-follow-up: bank-and-park vs a ≥90%-power follow-up registration | Item 3 (same relationship as row 7 above — item 3 is the A1 measurement this rule is about, not itself an open-question citation) | **§6 item 8** — same relationship as row 7 |
| 9 | **NOT IN DOCKET §6.** Whether a Stage P costing grant should proceed given the newly-measured 18.6× under-costing of θ-engaged (smeared) `evaluate()` calls | Item 1 | No match — a costing/scheduling question, not itemized in docket 2 §6's 8 rules; possibly folded into future costing of item 3/4's compute bands but not stated as such |
| 10 | **NOT IN DOCKET §6.** Whether S0-R (FALLBACK/DISARMED) should run on a future dedicated session | Item 1 | No match |
| 11 | **NOT IN DOCKET §6.** Row #167: whether the D̃_φ sub-convention for the impostor-weight-switch family also completes (COMPLETED-SMALL −0.00281±0.00047 vs COMPLETED-MATERIAL +0.0344) — bounds C1's [0,+0.123] remedy-family range | Item 5 | No match — open since row #167, predates docket 2, not re-surfaced there |
| 12 | **NOT IN DOCKET §6.** B6.1's judgment call (sigma_z_pv from raw host_z per prose vs from z̃ per the appended note's own formula literal) — flagged by its own builder for orchestrator/author confirmation, not yet ruled | Item 9 | No match — narrower than §6 item 3's Σ^φ(θ)/sky-cone scope; same commit family (θ-hook) but a distinct sub-question not itemized in docket 2 |
| 13 | **NOT IN DOCKET §6.** Whether `catalogue_numerator_survival_2d_center="auto"` (unimplemented) should be coded, and whether that code change itself requires a fresh physics-change gate presentation before being written | Item 10 | No match |

Process-only citations (ledger row numbers, standing rules 1/2, A8/A10/A11/A13/A14/A15/A20
tags, row #213/#246/#220 as evidentiary support) from items 1, 2, 3, 5, 6, 7, 9, 10, 11
are **not** separately listed above — they are authorization/provenance citations
backing the items' own verdicts, not open questions returning to the author, and are
recorded in each item's own `evidence_files`/`author_rule_items` list.

**Item 12 (truncated fragment):** the one visible line — "`BIAS_HISTORY_LEDGER.md` row
#253 ... 'Adoption returns to the author as a fresh [R...'" — is consistent with §6
item 4 (B7.3 ratification) but the payload cuts off before the item's own title,
verdict, or full content can be confirmed. Not counted in the table above; see (e).

---

## (d) Union of governance breaches

Items 1, 2, 3, 4, 11 report `governance_breaches: []` (none). The following 6 entries
come from 4 items:

1. **(Item 7)** Attribution-precision breach: the sentence "production changes inside
   the tree are covered too ... every gate goes to the end verifier" is twice labeled
   `(author, verbatim)` (runbook 37 §5; `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md`
   header) when ledger row #223's actual verbatim text differs materially in wording
   (though not in authorized substance). Violates CLAUDE.md's own attribution-precision
   convention.
2. **(Item 8)** A22 launch-stamp accuracy: "tree clean at both the local and cluster
   checkouts at launch time" (registration + C0 baseline gate, both) is contradicted by
   `tree_dirty_file_count=296` in every retrieved provenance JSON for both C0 and all
   four C3 tasks. C0 §13 offers an explanatory-but-unverified resolution (untracked-file
   count) that this pass could not independently confirm (SSH down).
3. **(Item 9)** Builder=runner conflation on this node's own regression/smoke-testing:
   no separate runner role specified in the charter for B6.1; disclosed in the record
   itself, and this verifier pass's independent re-execution supplies the missing
   rule-2 independence.
4. **(Item 9)** No refuter report reached the chair for this node before the chair's
   "consistent with sec1.2" finding was recorded — disclosed in the item's own framing;
   this verifier pass supplies the missing independent check.
5. **(Item 9)** The raw-z-vs-z̃ judgment call (see (c) row 12) is a live, undischarged
   returns-to-author item that item 9 itself files under `governance_breaches` rather
   than purely `author_rule_items` — listed here for completeness since the source item
   double-filed it.
6. **(Item 10)** B7.1's "panel clean after 0 rounds, two independent non-refuting
   reports" claim has **no separately-filed report artifact** anywhere under
   `fanout1_20260829/` — unlike item 9 (which discloses an analogous gap explicitly) or
   item 12 (which has an actual `B7_3_ADOPTION_VERIFIER_REPORT.md` file, per item 10's
   own comparison). The claim is only ever restated (record → docket → ledger row
   #231), never sourced to an independent file, and this evidentiary gap is **not**
   disclosed in the B7.1 record itself.

---

## (e) Items flagged for the adjudicator — method not clearly re-executed from source

**None of items 1–11 are flagged.** Every one of the 11 items names a from-scratch
verifier script under `verifier_pass/item{N}_rederive*.py` (or an equivalent direct
re-execution: item 1's pytest re-run + regex source match, item 3's import-and-rerun of
the sha1-pinned instrument, item 9's monkeypatched closed-form comparison), reads raw
CSV/JSON/source rather than the record's own restatement, and reports a numeric
match/mismatch against the record's claimed figures to stated precision. This satisfies
the falsification brief's re-execution requirement on its face for all 11.

**Item 12 is flagged** — its payload is truncated mid-JSON (cuts off inside
`author_rule_items` before title, verdict, decisive numbers, method, or evidence_files
appear). **Cannot assess whether item 12's verifier re-executed from source; the
adjudicator should request the complete item 12 payload before treating it as verified.**
No claim from item 12 is otherwise used or endorsed anywhere in this dedup file.

---

## Scope note (carried forward per task instruction)

Item 20 (the wave-3 blind readout) is **DEFERRED** in the parent verifier pass because
cluster SSH is down — disclosed here as instructed, though it does not bear on any of
the dedup/conflict findings above (all local reads). Items 13–19 were not included in
the NODE payload supplied to this dedup task and are therefore not covered by this file.
