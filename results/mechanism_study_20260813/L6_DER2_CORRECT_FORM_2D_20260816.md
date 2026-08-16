# L6-DER2 — the correct-form 2D derivation: the factorization error and the A-FULL-2D candidate

**Date:** 2026-08-16 · **Authorized:** rows #112 (option A) / #114 (L6 ratified; runbook 15
item 2 is the standing next step) · **Status: PRESENTED, NOT ADJUDICATED — derivation only; the
mirror pre-measurement (§4) is the next mechanical step and the arbiter, per the Part-1/Part-2
lesson (prose predicts, the audit decides).**

## 1. The question (from L6-DER §3, ratified row #114)

Channel B — the φ-slope tracking of g's z-argument — is a real, measured +139 nats/h term of the
CODED 2D channel. Does the CORRECT joint (d_obs, M_z_obs)-density form cancel it, and if so
through what partner?

## 2. The derivation

**The generator's 2D data** (pinned event): M_z_obs = M_row(1+z_true)(1+σ_M(ρe1+√(1−ρ²)e2)),
with M_row pinned from the DETECTED (SNR-selected) pool — the M-side twin of the z-side pinning.

**The coded/A-FULL 2D structure factorizes selection and mass-likelihood:** the numerator carries
S̄_φ(z;h) = ∫dM φ(M)·p_det(M,z,h) as a z-weight (A-FULL ingredient 2; inside α for the coded
form), TIMES g = ∫dM φ(M)·N(M_z_obs-ratio | M, z, d-conditioning) as the mass factor — i.e.

    coded/A-FULL 2D weight  ∝  [∫ φ p_det dM] · [∫ φ N dM]                      (two ∫dM)

**The correct selected joint prior does not factorize.** For pinned events drawn from the
selected population, the host's (z, M) prior is the normalized selected density
w_pop(z)·φ(M)·p_det(M,z,h)/α(h), and the M-integral runs ONCE against the mass likelihood:

    correct 2D weight  ∝  ∫ dM  φ(M) · p_det(M,z,h) · N(M_z_obs ; M(1+z)·(…), σ_M·M(1+z))   (one ∫dM)

**This is a factorization error exactly analogous to D1+D4 being "one broken pairing":** p_det
and the M_z_obs-likelihood are both M-dependent and must share a single ∫dM; splitting them into
S̄_φ(z)×g double-marginalizes φ and decouples the selection's M-dependence from the observed
mass. Channel B's tracking term is the visible symptom: in the coded form, g tracks the RAW φ
slope down-mass as z*(h) rises; in the correct form the effective prior at the likelihood point
is φ·p_det, and p_det RISES with M (louder sources) — flattening the effective slope — while the
S̄_φ z-weight's own drift moves oppositely. For the pinned-detected population the correct form's
expected score should cancel channel B to the same quality the 1D pairing repair achieved (up to
the same pool-vs-model residual class). **This is the prediction, not the proof — §4 measures it.**

Also predicted, for free: the correct form needs **no d-side change beyond A-FULL** (the
conditional N(M_z_obs | d_obs, z) machinery — proj, σ_cond — is correct as coded; channel A
measured null), and the M_z_obs-density-vs-ratio question (the D2-analogue prefactor
1/(σ_M·M(1+z))) rides along inside the single ∫dM and should be included in the same candidate.

## 3. The A-FULL-2D candidate (code form, venue mirror first)

Replace, in the 2D channel only, the pair (S̄_φ(z;h) node-weight × `completion_mass_factor_g`)
by the single fused object

    g_sel(z, f; h) = ∫ dx_M  N(x_M ; mu_cond(f), sigma_cond) · φ_x(x_M; z) · S(x_M·M_z_obs·(…); z, h)

with S the UNmarginalized detection survival at the implied source mass (available from the
estimator's own detection machinery — the same S whose φ-marginal is the tabulated S̄_φ), the
existing Hermite quadrature reused, and the 1D channel keeping its A-FULL form untouched (its
S̄_φ(z) weight stays — for the 1D data the M-marginalization IS correct). Denominator/α
unchanged (it already integrates φ·p_det·w_pop).

## 4. The registered next steps (in order; nothing self-adjudicates)

1. **Mirror pre-measurement** (stage-4/5 method, c2 path, 15 MN0X seed replays): T2 of the
   fused-g_sel candidate vs the coded and A-FULL references. Prediction to be tested: the 2D−1D
   excess collapses (channel B cancelled to the ~few-nat level), 1D bit-untouched. Requires
   plumbing S(M,z,h) per Hermite node — the detection machinery exposes it (the S̄_φ tables'
   integrand); implementation is sonnet-class with the l6 mirror as the base.
2. xhigh verifier on this derivation + the pre-measurement together.
3. If confirmed: A-FULL-2D registered arm (A8-v2, fresh seeds) — author gate; and the
   production completion-leg counterpart derivation (the same fusion question in
   `completion_numerator_integrand_with_bh_mass`, `absolute_marginal` path) — the reopened
   `/physics-change` proposal's subject.

*Append-only from its commit. No repair installed; the venue mirror measures first.*
