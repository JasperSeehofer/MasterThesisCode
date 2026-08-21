# CLAIM INTAKE — [P3] pull-forward: does the catalogue-leg selection convention move the impostor drag?

**Opened:** 2026-08-22, per ledger row #159 D3 = B (author: pull [P3] forward from the row #110
paper-task deferral; "we first keep digging for the bias"). Research-cycle stage 0.
**Thread tag:** `[P3-IMP]`. Append-only below the committed line.

## 1. The claim (stage-0 statement)

**[INFER, from [DOC] inputs — not yet measured]** Replacing the production catalogue leg's
denominator-only selection convention (Gray 2020 Eq. (A.10) / MFG 2019, coded at
`bayesian_statistics.py:5889-5901` and `:6062-6072`: no per-host p_det in the numerator) with
the **paired selected-prior form** (per-host `w_pop·S̄_φ/α(h)` weighting with leave-one-out
impostor completion — the FULL-F arrangement of
`docs/PROPOSAL_GRAY_CONVENTION_PAPER_INTEGRATION_20260817.md`) **materially changes the impostor
catalogue leg's contribution to B-SEL's headline bias** (currently −0.079 of −0.108, 73%,
row #149 [LOCAL]).

**Why it is plausible enough to open** (all [DOC]):
- The FULL chain measured the coded convention's cost on the venue arm: coded-base tilt
  **+2644 ± 46.5 nats/h ≈ +0.0373 MAP bias**, driven to **zero-consistency by the paired form**
  (FULL-F **+30.6 ± 42.7**, 0.7σ) — an ~86× reduction (Gray-paper proposal §, row #116 chain).
- Sign structure is non-trivial: the coded-convention cost is **positive** (+0.037, H₀ high)
  while the impostor drag is **negative** (−0.079, H₀ low) — the convention fork and the
  impostor mechanism are not the same object, and their interaction is unmeasured. The claim is
  NOT "FULL-F cures the impostor drag"; it is "the decomposition of the headline bias is
  convention-dependent at a material level."
- Row #119 M-4 bounded only the **[P2]-induced mixture skew** (median +0.02–0.03 share, 161/1588
  events) — a different, narrower question than the full per-host convention fork.

**Refute by (cheapest decisive test, rule 3):** re-score the **banked** B-SEL 12-seed fleet's
catalogue leg under the FULL-F arrangement using the existing FULL-chain harness (the row #116
A-FULL machinery) on the banked diagnostics columns — zero or near-zero `evaluate()` — and
re-read the O2 impostor decomposition (`decompose_impostor_leg.py` pattern) under both
conventions. If the impostor-leg share and the headline decomposition move by less than a
pre-registered materiality band, the claim is refuted and [P3] returns to the row #110
paper-task deferral with that bound banked.

## 2. Exoneration check (rule 1 — both layers, checked 2026-08-22)

- **Layer 1 (`CLAIM_2D_BIAS_20260730.md`):** no entry names the catalogue-leg selection
  convention, host weighting, or the impostor mechanism. "p_det inside/outside" appears in the
  summary line but resolves to layer 2's items below. NOT exonerated here.
- **Layer 2 (ledger §2):**
  - Item 6 ⚠ *"Adding p_det inside the numerator ALONE — refuted and it breaks calibrated
    controls (#66). Only the joint model-σ + p_det-inside pair works (#67)."* — **adjacent, not
    covering**: [P3]-FULL-F is precisely the JOINT paired arrangement item 6's second sentence
    leaves open, and the FULL chain's zero-consistent result is the measured instance of that
    pair. This intake does NOT re-open the refuted "p_det alone" move; any stage-2 arm that
    degenerates to an unpaired numerator p_det is void by this exoneration.
  - Item 10 ⚠ *"B_num as a defective integral — exonerated (#80, #87)"* — completion-leg object;
    not this claim's target (the catalogue leg). Respected.
- **Conclusion: the [P3] paired-convention question is OPEN — no exoneration covers it.**

## 3. Stage-L R0 (mandatory lightweight sweep — satisfied by reference, not re-run)

An R0 sweep for the [C-SYM]/[P3] front ran 2026-08-18 (`docs/LITERATURE_WARNINGS.md` MFG-a row)
and the 2026-08-21 Stage-L rows stand:
- **G20-d** — Gray 2020's own MDC validates only 25–75% completeness; our venue is at 4.79%,
  outside the source paper's tested range (`VIOLATED`-adjacent). Any [P3] result is
  venue-scoped by this.
- **MFG-a** — the MFG consistency principle as used in this repo is a paraphrase, **UNCHECKED**
  verbatim; the data-deterministic/latent-thresholded fork is a repo-internal result with no
  literature statement either way. **Carried obligation:** verbatim-verify before the paper
  quotes it; a stage-2 prereg here may cite it only as a supported claim.
- **B25-a** — a Gray-style mixture estimator reported unbiased H₀ across completeness
  configurations (**UNCHECKED** counter-precedent) — a stage-5 interpretation input, not a
  design input.

## 4. Free re-reads available before any compute (rule 9 / [A1] — stage-1/2 inputs)

1. The 12 banked B-SEL seeds' `event_likelihoods.csv` diagnostics (per-event, per-host columns)
   — the substrate the O2 zeroing read used; candidate substrate for a FULL-F rescore.
2. `decompose_impostor_leg.py` + its output JSON (the committed O2 instrument) — the
   decomposition harness to run under both conventions.
3. The FULL-chain harness of row #116 (A-FULL arms; `AFULL2D_ARM_READOUT_20260817.md`,
   commit `bcd66529`) — locate and inventory at stage 2; if it reads banked columns, the
   Refute-by test is zero-`evaluate()`.
4. Row #119's `run_20260817_fusion_counterfactual/readout.json` — the M-4 skew per-event lists
   (which events move, by how much) for the paired per-event read rule 10/[A2] requires.

## 5. What stage 1 (information forecast) must answer before pre-registration

- Expected magnitude: what does the FULL-chain arithmetic predict for the 12-seed fleet's
  catalogue leg under FULL-F — is the plausible effect size ≳ the fleet SEM (0.019), i.e. can
  12 seeds resolve it at all? (Power first — the O4 lesson: compute axis leverage BEFORE bands.)
- Which of the impostor drag's −0.079 flows through terms the convention fork actually touches
  (per-host numerator weighting) vs terms it cannot (candidate membership, photo-z kernels) —
  a structural upper bound on the reachable effect, from banked columns.

**Stage 0 exit: intake complete** (claim + refute-by + both exoneration layers + R0). Next:
stage 1 information forecast, then stage-2 pre-registration (A10 invariants + structural
blindness + A21 identity discipline). No measurement runs before the prereg is committed.

---
*(committed line — append only below)*

## APPENDED 2026-08-22 — Stage 1: information forecast + Refute-by correction (inventory-driven)

**Inventory finding [DOC, agent-verified paths]:** the §1 Refute-by clause assumed a
zero-`evaluate()` banked-column rescore. The inventory refutes that premise: the banked B-SEL
CSVs (`arm_event_likelihoods/bsel_seed*/…/event_likelihoods.csv`, 12 banked seeds) are
per-(event, h) AGGREGATES — the per-host terms FULL-F needs (`w_pop`, `S̄_φ`, `1/imp_k` per
candidate) are never stored. `decompose_impostor_leg.py` is an algebraic subtraction on the
banked `L_cat_no_bh` column, not a rescorer. The FULL-chain harness
(`results/mechanism_study_20260813/l4_afull_premeasure.py`) computes per-host terms FRESH per
seed via `venue_transfer._draw_seed_realization` and can be pointed at a different seed fleet
without estimator changes.

**Corrected Refute-by (supersedes §1's, per A21 discipline — corrected at stage 1, before any
prereg):** point the FULL-chain harness machinery at the 12 banked B-SEL seeds; compute the
catalogue leg under BOTH conventions (coded denominator-only vs paired FULL-F selected-prior);
re-read the O2-style impostor decomposition under each. Fresh compute, cost to be measured by a
1-seed pilot with a costing line (A6/A17) BEFORE the fleet.

**Stage-1 forecast (what a perfect analysis could say):**
- **Structural bound [DOC/[INFER]]:** the fork re-weights the catalogue leg; its reachable
  effect on the headline bias is bounded by the leg's total removal effect — **[0, +0.079]**
  (row #149's O2 measurement is the limiting case "catalogue leg → 0"). The fork cannot touch
  candidate membership or photo-z kernels.
- **Power:** the 12-seed fleet SEM on the headline is ~0.019 ⇒ the fleet resolves the effect at
  ≥2σ only if the convention moves the bias by ≳0.04 — i.e. ≳50% of the removal effect. A
  smaller true effect lands in a REPORTED-BOUND branch, not a null claim. The stage-2 prereg
  MUST carry this as its axis-leverage statement (A17) and register the sub-resolvable branch
  as first-class.
- **Sign context [DOC]:** on the venue arm the coded convention's cost was POSITIVE
  (+0.0373 MAP) while the impostor drag is NEGATIVE (−0.079) — the prereg registers arms that
  can distinguish "fork reduces |bias|" from "fork increases |bias|" without preferring either.
- **Paired per-event read (rule 10/[A2]):** the M-4 per-event mover list is not persisted but is
  regenerable free from the on-disk `off_iiib/fused_iiib` CSVs — the prereg includes it as a
  secondary read.

**Stage-1 exit → stage 2 next:** pre-registration of the two-convention 12-seed measurement
(1-seed pilot + costing line first; A10 invariants; structural blindness: this design cannot
detect convention errors COMMON to both arms, e.g. an error inside `S̄_φ` itself — the same
blind spot disclosed in O6/O7).
