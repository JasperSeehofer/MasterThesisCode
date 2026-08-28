# PRE-REGISTRATION — [CMEM] the two zero-compute completion-membership reads (stage 2)

`[FABLE-ORCH 2026-08-28]` · thread opened by author ruling row #216 item 4; these reads
authorized by the author grant of 2026-08-28 ("please go ahead" on the docket's [CMEM]
stage-2 prereg [DO] — authorization stamp per runbook 36 §2). Claim card:
`CLAIM_COMPLETION_MEMBERSHIP_20260828.md` (intake-complete; exoneration layers checked).

**Class: structural/composition measurement. ZERO H₀-space reads. All verdicts capped
REPORTED-ONLY** (first-of-kind read; no calibrated control exists; mirrors the [HIER]
item-9 affordability cap).

## 1. Frozen inputs

| input | pin |
|---|---|
| estimator source | `bayesian_statistics.py` + `galaxy_catalogue/handler.py` at HEAD (post-θ-hook `d40fe5c8` line references; the traced structure predates the hook) |
| fleet | the 24 banked `p3_2d_fleet_20260825/bc_*_work` arms (the R-MKER-6 census basis): per-seed `simulations/diagnostics/event_likelihoods.csv` (h = 0.73 only) + `prepared_cramer_rao_bounds.csv` + posteriors JSONs |
| census convention | chord/radius per `handler.py` cone (chord form, §R2.6); **anchor gate: reproduce seed 900121 event 20 chord 1.674660e-03 (full-float 1.6746585172e-03) / radius 1.4956979546e-03 before any count is read** — STOP on mismatch |
| banked census comparand | 380/2261 outside, fraction 0.1681 (R-MKER-6 entry) — the flag recomputation must reproduce these exactly (gate C-G1) |

## 2. Read 1 — structural trace (free; decides the mechanism's shape)

Trace, with file:line citations at HEAD:
(a) the catalogue leg's numerator support: candidate list = the hard sky-cone ball
(`get_possible_hosts_from_ball_tree`), yes/no;
(b) the completion numerator `B_num`'s population factor over sky: does it integrate the
UN-catalogued population only (a `(1−f)`-class factor), or the full population, or a
cone-complement term;
(c) whether any term anywhere carries the hypothesis "true host in-catalogue but outside
the candidate ball".

**Registered outcomes (partition):**
- **S-SHARP** — (a) yes ∧ (b) un-catalogued-only ∧ (c) none: the catalogued-out-of-cone
  hypothesis weight appears in no numerator while the denominator covers it → the claim
  sharpens (the ~17 % class's in-catalogue weight is structurally dropped).
- **S-WEAK** — (b) full-population or (c) exists: partial/total rerouting exists; the claim
  weakens to the residual named by the trace.
- **S-OTHER** — the structure matches neither description; report it verbatim.

## 3. Read 2 — paired composition read (free; banked CSVs only)

Recompute the per-event outside/inside flag over the 24 bc arms (anchor-gated, §1), join to
`event_likelihoods.csv` (h = 0.73), and compare OUTSIDE vs INSIDE, per-seed stratified:

- **R2a** per-event catalogue share `c_i = 1 − B_num/(combined_no_bh · D_tilde_phi)`
  (exact from columns under the derived form B_num^φ = B_num), summarized by median + mean.
- **R2b** catalogue-collapse rate: fraction with `L_cat_no_bh == 0`.
- **R2c** median `combined_no_bh` ratio outside/inside (the likelihood-deficit-at-truth
  probe; fleet truth is h = 0.73).
- **Covariate check** (A2 discipline): `z_true` distributions of the two groups, to flag
  confounded pairing.

**Band (registered before any split is looked at):** significance by within-seed label
permutation (10 000 permutations, statistic = the outside−inside difference of medians,
two-sided). p < 0.01 ⇒ **DISPLACED** on that read; else **NOT-DISTINGUISHED**. Direction
expected under BOTH the defect and the as-designed model is lower catalogue share outside
(the true host is absent from the ball either way) — therefore R2a/R2b alone can NEVER
confirm the defect; they are composition evidence. **The defect-relevant probe is R2c**
(a truth-likelihood deficit is not predicted by the as-designed model in which the event's
weight is correctly re-routed). Structural blindness [A10]: single-h data cannot measure an
H₀-direction of any deficit; invariants — catalogue pin, fleet artifacts, h = 0.73, cone
convention (all frozen above; census anchor audited this cycle).

## 4. Verdict map (partition; REPORTED-ONLY cap on everything)

1. **C-DEFECT-CANDIDATE** — S-SHARP ∧ R2c DISPLACED (deficit direction): the mechanism is
   structurally real and expressed; next step is its own registered H₀-space measurement.
2. **C-STRUCTURAL-ONLY** — S-SHARP ∧ R2c NOT-DISTINGUISHED: structurally real, not
   expressed at truth-likelihood level in this venue at N = 2261; bound reported.
3. **C-REFUTED** — S-WEAK ∧ R2c NOT-DISTINGUISHED: rerouting exists and nothing is
   expressed — the intake claim is refuted at this venue.
4. **C-MIXED** — any other combination (incl. S-OTHER): report verbatim; the stage-L STUCK
   counter starts per the research-cycle rule.

**Falsifier of any banked attribution [A14]:** verdict 1's attribution is falsified if a
future registered arm that re-routes the dropped weight (e.g. adding the out-of-cone
in-catalogue term) fails to move the 380-event class's truth-likelihood by the deficit
measured here. Registered now; unrun ⇒ any attribution stays provisional.

## 5. Gates

- **C-G1**: recomputed flag reproduces the banked census (380/2261, 0.1681) exactly AND
  the §1 anchor full-float. FAIL ⇒ INSTRUMENT-DEFECT, stop.
- **C-G2**: catalogue-share identity sanity — `c_i ∈ [0, 1]` for ≥ 99.9 % of events (the
  derived-form identity check); violations quoted.

*Registered 2026-08-28 pre-execution; no split-dependent number appears above this line.
Instrument: `cmem_reads.py` (`⟨SUBMIT⟩` sha at launch). Both reads local, zero cluster.*
