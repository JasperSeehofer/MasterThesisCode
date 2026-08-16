# L6-DER3 — the production completion-leg derivation: does `completion_mass_factor_g` need the fused survival?

**Date:** 2026-08-16 · **Authorized:** ledger row #115 item 3 ("all approved") · **Standing
constraint:** A3 (venue magnitudes do NOT transfer; channel existence does) · **Status:
PRESENTED, NOT ADJUDICATED — this is the derivation that gives the reopened `/physics-change`
proposal its subject. No production code is changed here; no magnitude is claimed.**

## 1. The question (verifier item F, row #115)

The venue arc established (rows #113–#115): the coded 2D structure's S̄_φ×g factorization —
two ∫dM where the selected joint prior demands one ∫dM φ·p_det·N — owns channel B, and the
fused `g_sel` removes 91.4% of the venue's 2D−1D excess. Does the production
`absolute_marginal` completion leg commit an error of the same class, and what is its
correct form?

## 2. The production geometry (verified against code, commit `91c813df`)

- **2D completion numerator** (`bayesian_statistics.py:4334-4363`):
  `B_num_wbh = ∫ (1−f_k) p_gw dVc/(1+z) · g_i(z;h) dz`, with
  `g_i = ∫dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M;z)` — **no survival factor anywhere in the
  ∫dM, and no S̄_φ node weight either**. The window `[z_lower, z_upper]` moves with `h_eval`
  (`:4368-4384`), so the h-moving evaluation channel exists (A3).
- **1D completion numerator**: `(1−f_k) p_gw dVc/(1+z)` — selection-free. The S̄_φ-weighted
  form exists ONLY as the N-2 instrumentation branch
  (`completion_numerator_integrand_sel_1d`, default `"off"`; derivation T3′,
  `N2_SELECTION_NUMERATOR_DERIVATION_20260805`).
- **Denominator** `D̃^φ(h) = β_G^φ + β_Ḡ^φ` contracts S̄_φ against `f̄`/`(1−f̄)`·p_pop
  (`precompute_phi_selection_integrals`) — selection lives here, once, globally.
- **Catalogue leg** (`:5200-5250`): numerator = `gw_3d · mz_integral · z-prior`, with
  `mz_integral` the **per-galaxy** mass marginal (the host's own M_gal, not φ), and an
  explicit convention comment: *"p_det is applied solely in the denominator; a numerator
  p_det is the Mandel–Farr–Gair (2019) 'most common mistake' and biases H0 high"*
  (Gray et al. 2020, Eq. A.10).

So production is arranged in the **MFG/Gray denominator-only convention** — unlike the venue's
coded form (which carried S̄_φ×g in the numerator), production carries **neither** S̄_φ nor
any p_det M-coupling in its numerators.

## 3. The load-bearing fork: which detection model governs

The two arrangements answer to two different detection models:

- **Data-deterministic detection** (detection is a deterministic function of the *observed*
  data, e.g. a threshold on observed SNR): `p(data|θ, det) = p(data|θ)/P(det|θ)` on the
  detected support, the numerator p_det cancels, and the MFG/Gray denominator-only form is
  exact. A numerator p_det would then be the MFG "most common mistake."
- **Latent-thresholded detection** (detection is random given the inference coordinates,
  independent of the measurement noise): `p(data|θ, det) = p(data|θ)`, and the correct
  per-event likelihood is the **selected-prior form**
  `L_i ∝ ∫ p(data|z,M) · p_pop(z) φ(M) p_det(M,z,h) dz dM / α(h)` — p_det stays inside the
  numerator's ∫dM, coupled to the observed-mass likelihood. This is exactly the venue's
  correct joint prior (L6-DER2 §2) and the T3′ 1D derivation.

**Production is latent-thresholded in the inference coordinates.** The pipeline's detection
rule thresholds the SNR computed from the FULL parameter vector θ (waveform SNR, no noise
draw); `SimulationDetectionProbability` is the survival estimator
`P(SNR(θ) ≥ 20 | d_L, M_z)` with the randomness supplied by the marginalized extrinsic
parameters. Given only `(z, M)` — the coordinates the H0 likelihood integrates — detection is
genuinely random, and it is independent of the measurement noise on `(d_obs, M_z_obs)` given
θ. That is the latent-thresholded case: the selected-prior arrangement is the exact one, and
the denominator-only arrangement drops precisely the correlation between selection's
M-dependence and the observed mass. (The venue generator implements this same model by
construction — `closed_loop_gfrac` draws detection as a Bernoulli on the survival at the true
`(d_L, M_z)` — which is why its measured verdict bears on the production question at the
structural level: the channel is real when the model is latent-thresholded.)

The MFG-mistake warning at `:5204` is therefore **convention-conditional**: it is correct
under data-deterministic detection and wrong under the pipeline's own latent-thresholded
detection. This is the row #110 Gray-convention paper task's subject, now with a measured 2D
consequence behind it.

## 4. The correct-form production completion leg (the derivation)

Under the latent-thresholded model, the completion (out-of-catalogue) host term for event i
at Hubble value h is, before normalization by D̃^φ:

    B_num^{2D,correct} = ∫ dz (1−f_k(z)) [dVc/dz/(1+z)] p_gw(d_obs|z,h)
                          · ∫ dx_M N(x_M; μ_cond(z), σ_cond) φ_x(x_M;z)
                                 S_4D(d_L(z;h), x_M·M_z,det,i)          — ONE ∫dM

i.e. **`g_i` → `g_sel,prod`: insert the unmarginalized with-BH survival inside the existing
Gauss–Hermite contraction**, queried exactly as `precompute_phi_marginal_survival` queries it
(detector-frame mass `x_M·M_z,det,i`, the node's `d_L(z;h_eval)`, isotropic sky,
`_wbh_z_kwargs` rider). And the 1D completion numerator correspondingly gains the φ-marginal
factor:

    B_num^{1D,correct} = ∫ dz (1−f_k) [dVc/dz/(1+z)] p_gw(d_obs|z,h) · S̄_φ(z;h)

— which is **exactly the T3′/N-2 instrumentation branch promoted to default**. The
denominator D̃^φ is already the selected-population normalization (∫∫ w_pop φ p_det) and is
unchanged, mirroring the venue result (α untouched).

**The two channels stand or fall together.** Production's coded pair is (g_i | nothing): the
common S̄_φ-level factor is missing from BOTH numerators, and the M-coupling of selection is
missing from the 2D one. Relative to the correct pair (g_sel | S̄_φ), the coded **2D−1D
difference** is missing exactly the channel-B-class term — the fused ∫ φ p_det N versus the
factorized S̄_φ×(∫ φ N)… reduced here to its production arrangement, S̄_φ×(g_i/g_i-norm)
versus g_sel. Fixing only one channel manufactures a spurious selection term in the 2D−1D
comparison; the proposal must move both together. (The N-2 cell measured the 1D-level term in
production: central-difference tilt +30.9 nats/h at h=0.73, chord +24.6 in-band —
`results/run_20260805_n2sel1d/readout.json`, claim status DRAFT/unadjudicated. Cited as
context, not as evidence for the 2D term.)

**The catalogue leg is the same fork, per-galaxy.** Under the latent model each catalogued
candidate's numerator gains `p_det`-weighting against its own mass marginal
(`mz_integral` → ∫ N·p_gal·S_4D single-∫dM in the 2D leg; a per-host S̄-type factor in the
1D leg — with p_gal in place of φ). This contradicts the explicit `:5204` convention comment
and Gray et al. Eq. (A.10) as coded, so it is a **separate, author-gated decision** inside
the proposal (and the substance of the row #110 paper task) — NOT something to slip in as a
side effect of the completion-leg change. Note the measure-consistency gate (i): the
catalogue and completion 2D legs must stay addable densities in the same x_M measure,
whatever is decided.

## 5. Amendments that are LIVE here (V2/V4, verifier addendum `453d1b29`)

- **V2 (measure prefactor):** production host-mass errors are large (σ_M ~ 60–200%), so the
  M_z_obs-density-vs-ratio prefactor question (the D2-analogue `1/(σ_M·M(1+z))`) is NOT
  negligible here and must be settled in the proposal with dimensional analysis and a
  limiting case — jointly with the D-ii ratio-form GW factor (runbook item 4, option C folds
  into this proposal).
- **V4 (sharp-likelihood limit):** the venue's σ_cond ~ 1e-7 made g_sel ≈ g·S(at the observed
  mass); production's broad σ_cond exercises the genuinely non-separable regime, where the
  survival's mass-slope reweights the whole ∫dM. No venue intuition about the size or even
  the sign of the net production effect may be inherited (A3). The support-exit warning at
  `:4352` (g_i leaving the φ support) will interact with the added S factor and needs a
  regression test.

## 6. What this derivation does NOT establish

- Any production magnitude (A3): whether the campaign's observed 2D bias is owned by this
  term requires the production counterfactual instrumentation (the `freeze_g_frac_ref_h` /
  N-2-style cells are the template), which belongs to the proposal's verification plan.
- The catalogue-leg ruling (§4, separate decision).
- The −11.7-class residual's origin (open question of record).

## 7. Scope of the reopened `/physics-change` proposal (the subject, per row #112/#115)

The proposal (a reviewable artifact with the decision table inline, per the standing
convention) must cover, each with derivation, reference, dimensional analysis, limiting case,
and regression test:

1. **[P1]** 2D completion leg: `g_i` → `g_sel,prod` (fused survival inside the Hermite
   contraction), quadrature convention pinned (non-adaptive at the fused object, per the
   venue-registered convention and its rationale).
2. **[P2]** 1D completion leg: S̄_φ numerator factor default-on (promotion of the T3′
   branch), paired with [P1] — the channels move together or not at all.
3. **[P3]** Catalogue leg: latent-model p_det weighting (per-galaxy fused form) versus the
   coded Gray/MFG convention — presented as its own fork with the paper-task analysis;
   includes the measure-consistency gate (i) check.
4. **[P4]** The V2 measure prefactor + D-ii option C fold-in (one measure decision for both).
5. **[P5]** Verification plan: byte-identity of every untouched path; counterfactual
   instrumentation cells before/after; the venue arm (A-FULL-2D, `PREREGISTRATION_A_FULL_2D.md`)
   as the structural evidence base; campaign re-run scope and cost.

Denominators (D̃^φ, α-analogues, Σ^φ) are UNCHANGED in every item — the entire change is
numerator-side, exactly as in the venue repair.

*Append-only from its commit. The proposal itself is a separate document; nothing in
production changes until it passes the physics-change gate and returns from the author.*
