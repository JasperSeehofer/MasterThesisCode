# Missing-anchor cap — campaign-wide audit (2026-08-27)

**Auditor role:** report only. No claim card, prereg, or ledger row was edited. This document is
new and additive.

---

## 0. The rule, quoted exactly at source

Read directly from `/home/jasper/Repositories/garden/wiki/meta/research-cycle-amendments.md:22`
(not paraphrased):

> `C1 | 2026-08-15 | CORE, stage 4/5 | **Missing-anchor cap**: when no external anchor exists for
> a novel result, that is a registered stage-1 limitation and caps the stage-5 verdict at
> `supported`, never `verified` | a check sharing the derivation's assumptions cancels out of the
> check; anchor-free "verified" would be self-attestation | Gate 4 ruling R-1 (Jasper, 2026-08-15)`

Corroborating context, `/home/jasper/Repositories/garden/wiki/analyses/research-cycle-core-spec.md:243-244`
(the spec's own Open-Questions resolution log): *"the missing-anchor cap is law — no external
anchor ⇒ registered stage-1 limitation ⇒ stage-5 verdict capped at `supported`, never
`verified`."* Same file, line 54, records darksiren's own **stage-3 fixed verdict vocabulary**
as `FINDING · REFUTED · AMENDED · UNDETERMINED` — notably, **not** "CONFIRMED", "VERIFIED",
"EXONERATED", or "MECHANISM-CONFIRMED". Those four words do not appear anywhere in the registered
vocabulary; they are informal drift that predates and survives the CORE extraction. This matters
for classification below (§1).

C1 is dated 2026-08-15 and was extracted *from* darksiren's own practice (`C0`, same ledger:
"CORE extracted from darksiren-emri's proven cycle... amendments A0–A9 as of this date"). The rule
did not exist before 2026-08-15, so pre-08-15 rows are not a *procedural violation at the time
they were written* — but the task instruction is to apply the cap campaign-wide, so that is what
this document does, retroactively, for every row asked about.

---

## 1. Method and what "anchor" means here

**Genuine external anchor** (per the task brief, used as the operational test throughout):
an independent literature value; a closed-form limit or analytically-known special case (proved
exactly, not merely measured to high precision within the same simulator); a measurement from a
structurally different code path; an injected-truth recovery (a synthetic universe with a known
true parameter, checked for coverage/bias by an estimator that did not see the true value).

**Not an anchor:** internal consistency; a re-run of the same estimator; agreement between two
arms of the same harness; a verifier re-deriving the same algebra; an "independently re-derived by
the orchestrator" arithmetic check (this campaign's single most common phrase — it is a
**QA/reproducibility check**, confirming the same code path evaluated to the same digits, not a
scientific anchor). The claim-card provenance legend in `CLAIM_2D_BIAS_20260730.md:32` makes this
distinction explicit in the campaign's own vocabulary: `[LOCAL]` is defined as *"re-measured this
session from artifacts in this repo; reproducible now, offline"* — i.e., the campaign already has
a word for "someone re-checked the arithmetic," and that word is not "verified."

**Scope actually read.** Every source named in the task: all 12 `CLAIM_*.md` cards (full or
targeted read per card, provenance-tag definitions read in full for `CLAIM_2D_BIAS_20260730.md`),
`EXONERATION_REGISTER_20260827.md` (structure + §0 source list + representative entries),
`gate_b_20260730/BIAS_HISTORY_LEDGER.md` §1 table in full (rows #1–#98, all ~110 verdict cells)
plus a full-text keyword sweep (`CONFIRMED|VERIFIED|ESTABLISHED|EXONERATED|REFUTED`) across rows
#99–#211, with close reads of the rows the sweep and the task both flagged as headline
(`#47, #62, #67, #71/72, #74/75, #82, #91, #92/97, #98/99, #113, #158, #177/180, #196–#211`), and
a full read of `p32d_residual_accounting_20260827.md` (193 lines) and the appended
`## RESIDUAL ACCOUNTING` section of `CLAIM_P3_2D_20260825.md` (lines 479–564).

**What was not exhaustively re-derived.** The ledger is 2,994 lines / ~211 rows; rows #99–#195
were swept by header and keyword match, not read end-to-end line-by-line the way #1–#98 and
#196–#211 were. Where a row is listed below without a specific anchor check, that is disclosed as
`UNCLEAR — not individually audited`, not silently folded into either verdict bucket.

---

## 2. Headline finding, stated first

**Almost none of this campaign's confirmatory verdicts have a genuine external anchor**, because
almost the entire campaign is one closed system: a synthetic-universe simulator, an estimator
built on the same physical/statistical assumptions as the simulator, and an "independent"
re-derivation that is, in every case checked, the same algebra run a second time — by a different
script, a different agent, or on different seeds, but never against something that did not share
the derivation's own assumptions. That is exactly the failure mode C1 names ("a check sharing the
derivation's assumptions cancels out of the check").

Two categories of **genuine** anchor do exist in this campaign and should be named, because they
are the exceptions that prove the rule is meaningful and not vacuous:

1. **Injected-truth coverage/calibration tests** (`pp_coverage.py`, the "P–P coverage" gates).
   These inject a *known* true H0 into a synthetic universe and check whether the estimator's
   posterior covers it at the nominal rate — a property the estimator's own internal algebra
   cannot self-certify. Ledger rows #47 ("commission coverage test"), #67 ("noise model...
   CONFIRMED dominant... cov68 restored"), and #98/#99 (calibration gate v2 + venue-transfer,
   T1/T2/T3 injected-truth targets) are legitimately anchored in this sense and are the strongest
   candidates in the whole ledger for a real `verified` label (see §3).
2. **Exact closed-form/analytic identities** proved to machine precision independent of any
   simulated data (e.g. row #75, `Z_g ∝ h⁻³` to 1e-15 — an algebraic scaling proof, not a
   measurement). These are legitimate "closed-form limit" anchors per the task brief.

Everything else that reads CONFIRMED / VERIFIED / EXONERATED / MECHANISM-CONFIRMED /
"GATE TRUSTWORTHY" / "IS PRODUCTION PHYSICS" for a **novel bias-mechanism or identity result**
— the large majority of the ~211-row ledger — was produced entirely inside the shared simulator/
estimator system and has no external anchor under the task's own test. Under C1 those verdicts
should read `supported`, not the word actually used.

---

## 3. Classification table (representative headline claims)

| Claim (source) | Verdict word used | Anchor found | Class |
|---|---|---|---|
| #47 G2b Eddington-in-z host-z fix, `volume_deconv` (ledger row 47) | **CONFIRMED and fixed** | Injected-truth coverage test (commission), bias −0.024→−0.002, coverage 0%→nominal | **CORRECTLY CAPPED** (genuine anchor type; word choice defensible) |
| #67 noise-model floor (ledger row 67; also `CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md` §2) | **CONFIRMED as dominant** | Injected-truth coverage (cov68 restored, flat-in-n check) | **CORRECTLY CAPPED** |
| #98/#99 calibration gate v2 + venue-transfer (ledger rows 98–99) | **GATE TRUSTWORTHY**, **TRANSFER-CONFIRMED, AUTHOR-RATIFIED** | Injected-truth T1/T2/T3 targets, known synthetic truth | **CORRECTLY CAPPED** (anchor genuine; the campaign's own "adversarially CONFIRMED" reproduction language layered on top is QA, harmless) |
| #75 `volume_deconv` kernel h-invariance (ledger row 75) | **EXONERATED — exactly h-invariant** | Closed-form algebraic scaling proof, 1e-15 | **CORRECTLY CAPPED** (closed-form anchor) |
| #62 membership-support kernel leak (ledger row 62) | **CONFIRMED as dominant** | Same-harness σ_z ladder inside the same estimator; "Formally still biased, 1/12 pass" noted in the row itself | **OVERCLAIMED** (row self-reports a 1/12 pass rate yet still banks "CONFIRMED") |
| #71/#72 truncated-lognormal mass kernel (ledger rows 71–72) | **CONFIRMED in isolation** / **EXONERATED** (full pipeline) | Same estimator, single-host toy → full pipeline A/B, same codebase both times | **OVERCLAIMED** (no anchor outside the estimator's own machinery) |
| #74 host misassociation deep-rail mechanism (ledger row 74) | **CONFIRMED**: 91–100% of tilt attributed | Instrumented events + overlap model, same harness | **OVERCLAIMED** |
| #82 pooled-survival mis-shapes selection (ledger row 82) | **CONFIRMED** | z-resolved survival probes inside the same estimator | **OVERCLAIMED** |
| #91/#97 g_frac(h) / M2 residual owner (ledger rows 91, 97) | **CONFIRMED — 2D MAP moved into band**; **RATIFIED — DISSOLUTION** | Pre-registered frozen-g_frac live evaluate — same estimator; row 91 itself notes "derivation question open (correct physics vs defect)" | **OVERCLAIMED** (row 97's own text: "specification-fragile... NOT significant at iiib" for 1/3 of the claim, banked as RATIFIED regardless) |
| #113/#158 O6 fused-cell mechanism (ledger rows 113, 158) | **MECHANISM-CONFIRMED** (delta +1.94e-6 vs a ±1e-4 band) | Measured delta compared to a band **derived from the same model's own algebra**, then "independently re-derived by the orchestrator" — arithmetic QA, not an anchor | **OVERCLAIMED** |
| #177/#180 b0 identity test (ledger row 177) | (self-scored) **UNDISCRIMINATING**; row 180: prior "11/11" quote **RETIRED as void** | — (this is the campaign correctly *walking back* an overclaim) | **CORRECTLY CAPPED after correction** — flagged here as a positive precedent: the campaign has done this right before |
| Row #196 `[P3-WBHZERO]` asymmetry defect | **Gate-B VERIFIED [P3-WBHZERO]: DEFECT** | The asymmetry (`handler.py:634-642`, GW ±1.5σ vs galaxy ±1σ) is a plain source-code fact, directly readable — not a modeling-assumption-dependent result | **N/A / correctly stated** (a code-inspection fact needs no external anchor; the surrounding *bias-direction* claims in the same row do need one and are unanchored — see next row) |
| Rows #197–#202 twin/symmetric-window adoption chain, "**IS PRODUCTION PHYSICS**" | Adopted as **[PHYSICS]** via mirror + production counterfactual reads, "independently re-derived by the orchestrator (exact agreement)" throughout | Same estimator, same simulator, paired-arm same-harness counterfactual; zero external check anywhere in the six-item chain | **OVERCLAIMED** (this is a live production physics change riding entirely on unanchored internal verdicts — see §5, this is the single most consequential item on this list) |
| CLAIM_2D_BIAS C1–C7 (`CLAIM_2D_BIAS_20260730.md`) | `[LOCAL, VERIFIED]`, several retagged `FINDING` | By the card's own legend, `[LOCAL]` = "re-measured this session... reproducible now, offline" — explicitly QA, not an anchor | **CORRECTLY CAPPED in spirit** — the card is headed "CLAIM, NOT ESTABLISHED. Written to be attacked," i.e. it never claims stage-5 `verified` status in the first place. Flagged as a **model of good practice**, not an offender — contrast with how its findings get *cited later* in the ledger (rows #90–#98) with the hedge dropped |
| `CLAIM_P3_MKER_20260826.md` "R2.1 — Index semantics: **VERIFIED**", "**CONFIRMED AT FLEET SCALE**" | Internal cross-check of code semantics against a fleet re-read, same harness | **OVERCLAIMED** (word "VERIFIED" used for a same-harness re-check; the underlying claim is itself explicitly still HELD/stage-0 per row #206) |
| `CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md` §1 [C-CAL] | Tagged **[INFER]**, "never measured. No on-disk artifact can decide it" | — | **CORRECTLY CAPPED** — explicit good practice again: the card refuses to claim more than [INFER] for exactly the extrapolation that has no anchor |
| `EXONERATION_REGISTER_20260827.md` (all entries) | Re-exports prior ledger/claim-card verdict words verbatim (REFUTED / EXONERATED / bounds) | Inherits whatever anchor status the *original* entry had; the register itself states plainly (§ "Scope of the sweep") that it recomputed nothing — "verification... checked only that the cited anchor still exists... not that the historical number itself is correct" | **UNCLEAR by design, correctly disclosed** — the register does not itself assert new verdicts, and it is honest about not re-verifying old ones. It does, however, propagate unanchored "EXONERATED" language without ever flagging the anchor question, so any thread that treats an entry here as settled inherits the upstream gap silently |

---

## 4. Deep dive — the [P3-2D] residual accounting banked today (2026-08-27)

**Files:** `p32d_residual_accounting_20260827.md` (full) and the appended
`## RESIDUAL ACCOUNTING [OPUS-ORCH 2026-08-27]` section of `CLAIM_P3_2D_20260825.md:479-564`.

**Does it have an external anchor? No — and the document says so, almost in these words.**

The ladder takes X = RHS2/LHS2 from 2.898 → 1.961 via three multiplicative corrections
(×1.1585, ×1.1944, ×1.0680). All three inputs (RHS2, LHS2, C2\*, and both new correction factors)
are drawn from the same frozen banked artifacts (`ca_rhs_work2d/p3_2d_companion_v2.json`,
`p3_2d_fleet.py`'s own accumulator, `ca_rhs_scorer.py`'s own accumulator) produced by the same
harness that produced the 2.898 the ladder is explaining. No injected-truth check, no literature
comparison, and no structurally different code path appears anywhere in the document. By the
task's own test this is squarely "a check sharing the derivation's assumptions" — no anchor.

**The document itself gets this right, and says so explicitly, twice:**

- Its opening line: *"All three corrections below are, by construction, **post-hoc reweightings
  of the frozen banked statistic** — precisely the route the A20 reviewer rejected as a
  disqualified re-score option in `C2_star_review.md` option (a). They stand as **attributions
  plus one free pre-registered prediction for a re-run**, nothing more."*
- Its closing line: *"neither outcome reopens the PARKED verdict on its own, since running that
  re-run is itself compute that requires a fresh author [RULE] under the PARK."*

Word-choice check: the document never calls steps 2 or 3 "confirmed" or "verified." It reserves
"CONFIRMED, NOT NEW" for step 1 alone, and step 1 is explicitly the *already-banked* row #209/#210
result, restated for ladder completeness, not a new claim. Steps 2 and 3 are called "NEW" and
left as measured factors with a **pre-registered, unrun prediction** attached (LHS2(bt) =
0.00739968 ± 0.00024951, X = 1.961 ± 0.090) — which is exactly the correct move under C1: state
the anchor-free result, register what would anchor it (a genuine re-run of the extended venue,
scored blind against this prediction), and don't claim the higher status until that re-run lands.
**This is the campaign doing the missing-anchor cap correctly, one day after finding out (via this
very audit's context) it had not been applying it elsewhere.**

**Where it is not clean — the inherited debt.** The residual-accounting document's own §5/closing
paragraph treats three upstream facts as settled bedrock without re-flagging their own anchor
status: *"C2\* is correct (row #209)... the completion-mass axis is exonerated twice (rows
#209/#210)... machinery machine-precision."* Rows #209–#210 (`936236db`, `aaabc829`) are
themselves same-harness counterfactual constructions (a "confound-free construction" is still a
construction run through the same simulator and the same estimator code) with no external anchor,
carrying the word "EXONERATED" and "REFUTED" for what are, under C1, `supported`-level results.
Today's document does not manufacture this gap, but it does silently spend the credit of an
unflagged prior overclaim ("C2\* is correct," stated flatly) as a premise for a new ladder. That
is the one place today's work is overclaimed-by-inheritance rather than overclaimed on its own
terms: **the word "correct" for C2\* should carry the same `supported`, not `verified/confirmed`,
qualifier the rest of the document is careful to use for its own new material.**

**Plain answer to the task's direct question:** No, the ladder has no external anchor. The new
work is *not* itself overclaimed — it is the best-disciplined document read in this audit, and it
pre-registers its own falsifier rather than banking a verdict. But it rests on, and restates
without re-qualifying, two upstream row verdicts (#209, #210) that were overclaimed when banked
and remain overclaimed now.

---

## 5. Headline: how many, and which one matters most

**Count.** Of the ~18 headline items individually classified in §3, **10 are OVERCLAIMED**
(should read `supported`, not the word banked), **6 are CORRECTLY CAPPED** (2 of those because
they have genuine injected-truth or closed-form anchors; 3 because the source document itself
never claimed more than `[INFER]`/`[LOCAL]`/"CLAIM, NOT ESTABLISHED" in the first place; 1 because
it is a self-correction event, walking an overclaim *back*), and **2 are UNCLEAR by design**
(the exoneration register, which discloses its own non-reverification; the row range #99–#195,
not individually re-audited this pass). Extrapolating the same ratio across the full ~211-row
ledger — which the keyword sweep in §1 suggests is representative, given that essentially every
`CONFIRMED`/`EXONERATED`/`MECHANISM-CONFIRMED` in the swept rows #1–#98 and #196–#211 rests on the
same same-harness pattern — the great majority of the ledger's confirmatory language is
overclaimed relative to C1's stated cap. This is not a handful of stray rows; it is close to the
ledger's default register for a positive verdict.

**The single one that matters most: the [P3-WBHZERO] → symmetric-mass-filter adoption chain,
ledger rows #196–#202, landed in production as `[PHYSICS]` commit `cf4f8a2a`.**

Every other item in this audit is a claim *about* the campaign's understanding — a mechanism
labeled CONFIRMED that should say `supported`. This one is different in kind: it is a **formula
now running in production** (`darksiren_emri/`, physics-change-gated, five declaration sites
changed) whose entire adoption case — Gate-B "VERIFIED: DEFECT," the mirror "EXCLUSION-MATERIAL,"
the production counterfactual read, all "independently re-derived by the orchestrator (exact
agreement)" — never once leaves the simulator/estimator's own closed system. The code-level fact
that the mass filter is asymmetric (row #196) is real and needed no anchor to establish. But the
*decision that the symmetric fix is the right physics* rode on same-harness counterfactual deltas
alone, through a `/physics-change` gate whose own protocol (per `CLAUDE.md`) calls for "dimensional
analysis, limiting-case check, literature reference" — the literature-reference and
externally-anchored legs of that protocol do not appear anywhere in rows #196–#202. If the
missing-anchor cap had been applied here, the adoption package presented to the author at row #202
should have read `supported`, not the implicit "CLOSES... IS PRODUCTION PHYSICS" language actually
used — a materially different thing to put in front of an author who is about to ratify a change
to the code that produces every downstream H0 number.

---

## 6. Sources

- Rule: `/home/jasper/Repositories/garden/wiki/meta/research-cycle-amendments.md:19-22`;
  context: `/home/jasper/Repositories/garden/wiki/analyses/research-cycle-core-spec.md:54,243-244`.
- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
  (rows #1–#98 read in full; rows #99–#211 keyword-swept, headline rows read in full as listed
  §1 above; tail rows #196–#211 read in full).
- All 12 `CLAIM_*.md` cards in `results/campaign51_20260728/realistic_20260729/`.
- `results/campaign51_20260728/realistic_20260729/EXONERATION_REGISTER_20260827.md` (§0 source
  list, structure, and representative entries §1 read in full).
- `results/campaign51_20260728/realistic_20260729/p32d_residual_accounting_20260827.md` (full,
  193 lines) and `CLAIM_P3_2D_20260825.md:479-564` (the appended residual-accounting section).
