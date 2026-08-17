# PHYSICS-CHANGE PROPOSAL — selection fusion in the completion legs (the L6 production transfer)

**Date:** 2026-08-17 · **Authorized to author:** ledger rows #115 item 3, #116 item 3 ·
**Governing derivations:** `results/mechanism_study_20260813/L6_DER2_CORRECT_FORM_2D_20260816.md`
(+ verifier addendum, `453d1b29`) and `L6_DER3_PRODUCTION_COMPLETION_LEG_20260816.md` (`e3eec5c0`) ·
**Evidence base:** the A-FULL-2D registered arm (rows #115–#116: DS-G1 −11.8 ± 0.61 in band,
2D bias +0.0006 ± 0.0013, coverage restored, 1D bit-untouched) ·
**Status: PRESENTED, NOT ADJUDICATED. No code changes accompany this document. Each item below
passes the full `/physics-change` gate (presentation → implementation → checks → `[PHYSICS]`
commit → gate-ledger row) individually if and when approved. A3 stands: the venue arm proves the
CHANNEL and the repair FORM; no production magnitude is claimed anywhere in this proposal.**

## 0. The claim being repaired

Under the pipeline's own detection model — latent-thresholded in the inference coordinates
(detection randomness supplied by marginalized extrinsics, independent of measurement noise;
L6-DER3 §3) — the correct per-event likelihood uses the SELECTED population prior, so the
detection survival's M-dependence must be integrated against the observed-mass likelihood in
ONE ∫dM. The coded `absolute_marginal` completion legs instead carry no survival factor in
either channel's numerator (the MFG/Gray denominator-only arrangement, exact only for
data-deterministic detection). The venue proved the 2D consequence (channel B) and validated
the fused repair on fresh seeds.

## 1. The items (each row of the decision table at the end)

### [P1] 2D completion leg: `g_i` → `g_sel,prod` (the fused survival)

- **Old** (`bayesian_statistics.py:4334-4363`, `completion_mass_factor_g` `:2012`):
  `B_num_wbh = ∫ (1−f_k) p_gw [dVc/dz/(1+z)] g_i(z;h) dz`,
  `g_i = ∫ dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M;z)`.
- **New:** same outer quadrature;
  `g_sel,prod(z;h) = ∫ dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M;z) · S_4D(d_L(z;h), x_M·M_z,det,i)`
  with `S_4D` queried exactly as `precompute_phi_marginal_survival` queries it (detector-frame
  mass, node d_L, isotropic, `_wbh_z_kwargs` rider), **non-adaptive pinned n_hermite = 64**
  (the Route-1 adaptive error bound does not cover the sharp S(x_M) factor — registered venue
  convention, carried over with its rationale).
- **Reference:** Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)–(7) selected-prior form;
  L6-DER2 §2–§3; venue arm rows #115–#116.
- **Dimensional analysis:** `S_4D` is dimensionless → `g_sel,prod` remains a density in `x_M`,
  the SAME measure as `mz_integral` — the 2D catalogue/completion addability (gate (i)) is
  preserved by construction.
- **Limiting cases:** `S→1` recovers `g_i` bit-exactly (the venue's S≡1 refactor gate, 0.0
  measured, becomes a unit test here); `σ_cond→0` gives `g_i·S(at the observed mass)` (the
  venue's sharp-likelihood limit, V4). Production exercises the broad-σ_cond regime — no venue
  magnitude intuition inherited.
- **Interaction:** the `:4352` φ-support warning must account for S-induced zeros; the
  `g_frac` diagnostics column and `freeze_g_frac_ref_h` counterfactual continue to work
  unchanged (they wrap `B_num_wbh/B_num`, whatever g is).

### [P2] 1D completion leg: S̄_φ numerator factor default-on (T3′ promotion)

- **Old:** `B_num = ∫ (1−f_k) p_gw [dVc/dz/(1+z)] dz` (selection-free);
  the S̄_φ-weighted form exists as the N-2 instrumentation branch
  (`completion_numerator_integrand_sel_1d`, default `"off"`).
- **New:** `B_num = ∫ (1−f_k) p_gw [dVc/dz/(1+z)] S̄_φ(z;h) dz` — the existing branch promoted
  to the `absolute_marginal` default (table READ, never rebuilt, exactly as coded).
- **Reference:** T3′ (`N2_SELECTION_NUMERATOR_DERIVATION_20260805`); L6-DER3 §4.
- **Pairing constraint (hard):** [P1] and [P2] ship together or not at all — fixing one channel
  manufactures a spurious selection term in the 2D−1D comparison (L6-DER3 §4). The venue arm's
  1D channel already carries the S̄_φ weight (A-FULL), so the arm is evidence for the PAIR.
- **Limiting case:** S̄_φ → 1 recovers the old B_num exactly; S̄_φ(z→z_max) → 0 correctly
  removes population support beyond the horizon (currently overweighted).

### [P3] Catalogue leg: the per-host selection weighting (the Gray-convention fork)

- **The fork:** under the latent model each catalogued candidate's numerator gains p_det
  weighting against its own mass marginal (`mz_integral` → single-∫dM with `S_4D`; 1D leg a
  per-host S̄-type factor) — contradicting the coded `:5204` convention ("a numerator p_det is
  the MFG most common mistake") which presumes data-deterministic detection. Mixture
  consistency is the forcing function: with [P2] on, an S̄-free catalogue leg is DOWN-weighted
  relative to completion wherever S̄_φ < 1 — a skew of the G/Ḡ mixture that does not exist in
  the venue (single-class candidates), so the arm is NOT evidence here.
- **This is the row #110 Gray-convention paper task's subject.** Options in the decision table;
  the derivation-coherent position is that [P2]+[P3] form one arrangement, but [P3] carries
  paper-facing convention weight the author may want to settle in the paper analysis first.
- **Reference:** Gray et al. (2020) arXiv:1908.06050 Eq. (A.10) vs MFG (2019); L6-DER3 §4.

### [P4] The measure prefactor (V2) + D-ii ratio-form GW factor (option C)

- **The question:** the M_z_obs-density-vs-ratio prefactor (`1/(σ_M·M(1+z))`, the D2 analogue)
  and the D-ii ratio-form distance factor are the same class of measure decision; production's
  broad σ_M makes V2 non-negligible (verifier amendment, `453d1b29`). One consistent measure
  ruling for both, with dimensional analysis across the catalogue/completion pair, belongs in
  whichever of [P1]/[P3] is approved — folded in, not separate code changes.
- **Constraint:** gate (i) measure invariance (legs addable in the same x_M density) is the
  acceptance criterion; any prefactor decision must show it explicitly.

### [P5] Verification plan (accompanies any approved subset)

1. Unit tests: S≡1 recovery of `g_i` (bit-exact); S̄_φ≡1 recovery of old `B_num`; byte-identity
   of every non-`absolute_marginal` path (the legacy/generator_marginal assembly untouched,
   gate (iii-a) style).
2. Regression: record old-vs-new `B_num`/`B_num_wbh`/`g_frac` on a pinned event set before the
   change lands ([PHYSICS] commit convention; PHYSICS-GATE-LEDGER row per gate run).
3. Counterfactual instrumentation FIRST, adoption SECOND: an N-2-style paired cell (old form vs
   fused form, same seeds/realizations) on the campaign venue measures the production-side
   magnitude BEFORE any campaign re-run is authorized — the production analogue of the venue's
   premeasure-then-arm discipline. Its bands are seeded from nothing (no prediction basis
   exists — A3); it is a measurement, not a test.
4. Campaign re-run scope + cost estimate returns with the counterfactual's result.

## 2. Decision table

| # | Item | Scope | Recommendation | Tag |
|---|---|---|---|---|
| 1 | [P1]+[P2] fused survival in BOTH completion legs (paired, `absolute_marginal` only) | production estimator | **Approve for implementation behind the full physics-change gate**, with [P5] items 1–2 in the same commit | [DO] |
| 2 | [P3] catalogue-leg selection weighting | production estimator + paper convention | **Defer to the Gray-convention paper task** (row #110) UNLESS the [P5-3] counterfactual shows the [P2]-induced mixture skew is material — then it returns here | [RULE] |
| 3 | [P4] measure ruling (V2 + D-ii option C) | folded into item 1's implementation | Settle inside item 1's presentation gate (old/new formula with prefactor explicit) | [DO] |
| 4 | [P5-3] production counterfactual cell before any campaign re-run | measurement | Approve as the next measurement after item 1 lands | [DO] |
| 5 | xhigh verifier on THIS proposal before item 1's implementation begins | discipline | Standard practice per L6 arc | [DO] |

**Binding default honored:** approving item 1 does NOT approve a campaign re-run (returns with
item 4's result); nothing here touches the paper's existing claims; the −11.7-class residual
and pool-vs-model mismatch remain open residuals of record.

*Append-only from its commit.*

---

## 3. Verifier amendments (2026-08-17, appended per append-only convention)

The row #117 item-5 adversarial verifier returned **GO-WITH-AMENDMENTS**
(`PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md`). Corrections of record to the
text above — the body is preserved unedited; where this section conflicts with it, this
section governs:

1. **[P1] regime statement (MAJOR-1):** "Production exercises the broad-σ_cond regime" is
   WRONG. Measured d_L-conditional σ_cond on the production CRB reference: p50 = 8.8e-8
   (p95 = 3.0e-7). The broad-σ_M figure belongs to the catalogue leg's host-mass Gaussian.
   Production's completion leg operates in the sharp-likelihood limit; expected action of the
   pair is 1D-dominated ([P2]), with [P1] correct-form but possibly near-inert. A3 unchanged.
2. **[P3] skew direction (MAJOR-3):** with [P2] on, the S̄-free catalogue leg is
   **OVER-weighted** relative to completion wherever S̄_φ < 1 — the body's "DOWN-weighted" is
   inverted. The defer-unless-material structure survives.
3. **[P1] quadrature rider (MAJOR-2):** the pinned n_hermite=64 choice is a substantive flip
   of the ratified Route-1 adaptive default on the hot path, not a rider — it returns to the
   author as its own line in the presentation gate.
4. **[P4] scope (MAJOR-4):** V2 is provably immaterial (≲1e-6) at completion-leg σ_cond and
   material only in the deferred catalogue leg — the ruling cannot be silently folded into
   item 1; the presentation gate carries it explicitly.
5. **[P5] additions:** S≡1 bit-exactness restated against `adaptive=False, n=64`; the [P5-3]
   counterfactual must decompose 1D and 2D contributions separately; freeze-ref-h,
   φ-support-warning, and external-caller byte-identity tests added (MINOR-1..3); DRAFT status
   of the N-2 production measurement and the #66/#67 calibration caveat carried (MINOR-4).
