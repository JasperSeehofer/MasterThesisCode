# C₂ re-derivation — the 2D bounded-identity constant, from the committed definitions

**Blind-deriver task, 2026-08-26.** Sources consumed: `darksiren_emri/bayesian_inference/bayesian_statistics.py`
(`path_a_mixture_objects` :2407–2465, the φ-leg tables :2081–2097, the p_i assembly :5556–5609,
the L_cat divisor branch :5094–5112, the with-BH numerator integrand :6600–6660, `_mz_sel_2d_expectation`
:5918–5990, `B_num_wbh` :5282–5340), `darksiren_emri/validation/correspondence_1d.py` (the
`catalogue_selected`/`catalogue_selected_2d`/`population_selected` draw laws, `kernel_smeared_survival`
:1242–1345, `_draw_kernel_survival_redshifts` :1440–1500, `_draw_2d_accepted_latents` :1610–1700),
`results/campaign51_20260728/realistic_20260729/p3_b0_identity_test.py` (`c_star`, `mass_companion` — the
1D C\* analog), `p3_2d_companion.py` (`compute_sigma_tilde_4d`, `_segmented_integral_batch`),
`PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md` (the 1D identity's registered form), and
`PREREGISTRATION_P3_2D_20260825.md` §1 + PA-2D-1 F4 **read only after the derivation below was fixed**,
as the candidate to check against. Not read: BIAS_HISTORY_LEDGER.md, PA-2D-8/PA-2D-9, `p3_2d_rhs2_20260826/`.

---

## 0. Objects, exactly as committed

Per-event, at h = H_TRUE = 0.73, Σ^φ slot, fused cell, absolute_marginal:

- **w₂ = A₂/(A₂+B₂)** with A₂ = α_G_φ·L_cat_with_bh, B₂ = B_num_wbh (read from `event_likelihoods.csv`).
- **α_G_φ = Σ^4D/n̂_w^φ = β_G_φ·Σ^4D/Σ^φ** (`path_a_mixture_objects`: `n_hat_w_phi = sigma_phi/beta_G_phi`,
  `alpha_G_phi = sigma_4d/n_hat_w_phi`); equivalently α_G_φ = β_G_φ·r_Malm with r_Malm = Σ^4D/Σ^φ.
- **L_cat_with_bh = (Σ_ball w_g N_g^{2D})/Σ^4D** — under absolute_marginal the with-BH catalogue leg
  divides by `global_denom_with_bh = Σ^4D` (:5022, :5106; the [P3-RPHI] "phi" slot swaps only the
  no-BH divisor to Σ^φ and "deliberately" leaves the with-BH divisor at Σ^4D, :5013–5022).
- **N_g^{2D}(d)** (per-candidate with-BH numerator, twin/mz_sel arrangement) =
  ∫ k̄_g(z) · p_gw(d̂|z) · [ mz_integral·E[S_4D] ](z) dz, where k̄_g(z) ∝ N(z; z_g, σ_g,eff)·w_pop(z)·f_k(z)
  window-normalized (`galaxy_redshift_prior_pdf`, the volume_deconv kernel) and, by the Gaussian-product
  identity (`_mz_sel_2d_expectation` docstring), mz_integral·E[S_4D] =
  ∫ p_gal(M|g)·p(M̂_z|M(1+z))·S_4D(d_L(z;h), M(1+z)) dx_M, with p_gal = N(M_eff_g, σ_Mg) (Eddington-shifted
  mean, the F2-resolved "eff" centering). **No S̄_φ(z) factor appears in the with-BH numerator** — the
  adopted 1D twin factor (`catalogue_numerator_survival="phi"`) multiplies the without-BH integrand only
  (:6362–6368; ":the WITHOUT-BH numerator is untouched" / mz_sel "with-BH channel ONLY").
- **B_num_wbh(d)** (fused) = ∫ (1−f̄(z))·w_pop(z)·p_gw(d̂|z)·g_sel(z; d) dz with
  g_sel(z;d) = ∫ φ_pop(M)·p(M̂_z|M(1+z))·S_4D(d_L(z), M(1+z)) dx_M (:5282–5340, `completion_mass_factor_g_sel`).
- **β_G_φ = ∫ f̄·S̄_φ·p_pop dz, β̄_Ḡ_φ = ∫ (1−f̄)·S̄_φ·p_pop dz** (:2087–2088), p_pop = dV_c/dz/(1+z).
- **Σ^φ = Σ_g w_g·S̄_φ(z_g)** (point-evaluated, mass-blind catalogue sum); **Σ^4D** the mass-aware sum;
  w_g = R_eff_per_mbh(M_g)/(1+z_g).
- **Σ̃^4D = Σ_g w_g·S̃_4D,g**, S̃_4D,g = ∫ k̄_g(z)·S̄_4D,g(z) dz, S̄_4D,g(z) = E_{M~N(M_eff_g,σ_Mg)}[S_4D(d_L(z;h), M(1+z))]
  (`p3_2d_companion.py`: `numerator/z_norm` in `_segmented_integral_batch` — kernel × mass-marginal survival,
  **no S̄_φ(z) inside the z-integrand**).

Draw laws (`correspondence_1d.py`):

- **Class-G venue (b0i2d, `catalogue_selected_2d`)**: host g ∝ w_g·S̃_φ,g; z_true|g ∝ k̄_g(z)·S̄_φ(z)
  (`_draw_kernel_survival_redshifts` — the 1D law, unchanged); latent M ~ p_gal(·|g); whole-triple
  Bernoulli acceptance with probability S_4D(d_L(z_true), M(1+z_true)); (d̂, M̂_z) drawn jointly from the
  donor Fisher 2×2 block centered at the host's own latent truth.
- **Completion class (`population_selected`)**: z ∝ w_pop(z)·(1−f̄(z;h_true))·S̄_φ(z;h_true)
  (`selected_population_z_weights`), normalization = β̄_Ḡ_φ (the same trapezoid objects as :2088).
- **F-0 acceptance**: 1_acc(d) = {σ_dL/d̂ < 0.10 ∧ SNR ≥ 20}, a deterministic function of the event row,
  applied per draw on BOTH sides; both sides are **drawn-count normalized** (÷200 per seed on the LHS,
  ÷N_syn on the RHS — the PA-CA-1 convention carried over).

## 1. The general two-density lemma (the acceptance measure)

Let g₂(d) be the class-G data density and ḡ₂(d) the completion-class data density (both normalized over
the *unfiltered* event space — the drawn-count normalization makes each banked mean a Monte-Carlo estimate
of E[·, with 1_acc inside], not a conditional mean). Suppose there exist constants a₂, b₂ with

    A₂(d) = a₂·g₂(d)   and   B₂(d) = b₂·ḡ₂(d)   pointwise in d.                     (L)

Then, since w₂ = A₂/(A₂+B₂) and 1−w₂ = B₂/(A₂+B₂):

    E_G[(1−w₂)·1_acc] = ∫ g₂·(B₂/(A₂+B₂))·1_acc dd = (1/a₂)·∫ (A₂B₂/(A₂+B₂))·1_acc dd
    E_Ḡ[ w₂·1_acc ]   = ∫ ḡ₂·(A₂/(A₂+B₂))·1_acc dd = (1/b₂)·∫ (A₂B₂/(A₂+B₂))·1_acc dd

The symmetric overlap integral ∫ A₂B₂/(A₂+B₂)·1_acc dd is common to both sides and cancels — this is the
cancellation step that makes the constant a pure ratio of normalizations, and it is why the F-0 filter,
being a common multiplicative indicator, never enters the constant:

    **C₂ = a₂/b₂**,   so that   C₂·E_G[(1−w₂)·1_acc] = E_Ḡ[w₂·1_acc].               (I)

This is exactly the structure of the 1D C-A identity (C\*·E_G[1−w] = E_Ḡ[w], PREREGISTRATION_CA §1),
where a = β_G_φ·ρ, b = β̄_Ḡ_φ, ρ = Σ̃^φ/Σ^φ (`p3_b0_identity_test.c_star`).

## 2. The completion side: b₂ = β̄_Ḡ_φ exactly (the tower identity)

Integrate B_num_wbh over the observables. ∫ p_gw(d̂|z) dd̂ = 1, and by the tower identity
∫∫ φ_pop(M)·p(M̂_z|M(1+z))·S_4D dM dM̂ = ∫ φ_pop(M)·S_4D(d_L(z), M(1+z)) dM = S̄_φ(z;h) — the same
φ-marginal survival table `precompute_phi_marginal_survival` provides. Hence

    ∫ B_num_wbh(d) dd = ∫ (1−f̄)·w_pop·S̄_φ dz = β̄_Ḡ_φ.

Moreover the completion-class predictive of the mixture generates z ∝ w_pop(1−f̄)S̄_φ and then
M̂|z from g_sel(·)/S̄_φ(z) — the S̄_φ cancels between the z-draw weight and the mass-conditional
normalizer — so ḡ₂(d) = B_num_wbh(d)/β̄_Ḡ_φ **pointwise**, i.e. **b₂ = β̄_Ḡ_φ with no ρ-type
correction and no mass factor** (this is the "completion-class 2D mass = β̄_Ḡ_φ exactly" leg).

## 3. The catalogue side: the Σ^4D cancellation, then a₂ = β_G_φ·Σ̃^4D/Σ^φ

First the divisor bookkeeping, purely from the committed estimator:

    A₂ = α_G_φ·L_cat_with_bh
       = [β_G_φ·Σ^4D/Σ^φ] · [Σ_ball w_g N_g^{2D}(d) / Σ^4D]
       = (β_G_φ/Σ^φ) · Σ_g w_g N_g^{2D}(d).                                          (Σ^4D cancels)

**This is why Σ^φ — and not Σ^4D — appears in C₂'s denominator**: α_G_φ carries Σ^4D/Σ^φ while the
with-BH catalogue leg is divided by Σ^4D; the Σ^4D's cancel identically (h-node by h-node), leaving the
mass-blind Σ^φ as the only surviving catalogue-sum divisor on the class-G side.

Now the class-G data-law mass. Under the mixture's OWN class-G predictive — hosts ∝ w_g, event data from
the with-BH numerator kernel itself (the same pairing convention the 1D derivation used, where the venue
kernel k̄_g·S̄_φ equals the 1D twin numerator kernel exactly) — the unnormalized class-G density is
U₂(d) = Σ_g w_g N_g^{2D}(d), and its total mass is

    ∫ U₂ dd = Σ_g w_g ∫ k̄_g(z) [∫∫ p_gal·p(M̂|·)·S_4D dM dM̂] [∫ p_gw dd̂] dz
            = Σ_g w_g ∫ k̄_g(z)·S̄_4D,g(z) dz = Σ_g w_g·S̃_4D,g = **Σ̃^4D**,

*exactly* the companion's object (same window-normalized volume_deconv kernel k̄_g, same
Eddington-shifted Gaussian p_gal = N(M_eff_g, σ_Mg), same S_4D interpolant). Hence g₂ = U₂/Σ̃^4D and

    A₂(d) = (β_G_φ/Σ^φ)·Σ̃^4D·g₂(d)   ⇒   **a₂ = β_G_φ·Σ̃^4D/Σ^φ**.

## 4. The constant

    ┌──────────────────────────────────────────────────────────────┐
    │   C₂ = a₂/b₂ = β_G_φ · Σ̃^4D / ( Σ^φ · β̄_Ḡ_φ )              │
    └──────────────────────────────────────────────────────────────┘

Equivalent committed-object forms: C₂ = (β_G_φ/β̄_Ḡ_φ)·r_Malm·ρ₂ with ρ₂ = Σ̃^4D/Σ^4D
(the 2D analog of the 1D ρ = Σ̃^φ/Σ^φ), and C₂ = α_G_φ·ρ₂/β̄_Ḡ_φ. The sums appear as:
**Σ̃^4D in the numerator** (the class-G draw-law contraction — total data-mass of the with-BH numerator
law), **Σ^φ in the denominator** (the surviving catalogue divisor after the α_G_φ ⊗ 1/Σ^4D cancellation),
**β_G_φ in the numerator / β̄_Ḡ_φ in the denominator** (the class-odds pair, as in 1D). Σ^4D appears
nowhere — it cancels completely.

## 5. Numerical evaluation (committed values)

With β_G_φ = 153322777.12146157, β̄_Ḡ_φ = 888403790.0, Σ^φ = 980867125.6740596,
Σ̃^4D = 348078892.5018141, r_Malm(0.73) = 1/2.6124925:

    C₂ = 153322777.12146157 × 348078892.5018141 / (980867125.6740596 × 888403790.0)
       = **0.06124403326364123**  (≈ 0.0612440333)

Cross-checks: Σ^4D = Σ^φ·r_Malm = 375452609.21; ρ₂ = Σ̃^4D/Σ^4D = 0.92709142 (a sub-unity smearing
ratio, the plausible 2D analog of ρ ≈ 1 with the mass-marginal broadening pulling it below 1);
(β_G_φ/β̄_Ḡ_φ)·r_Malm·ρ₂ = 0.061244033263641216 — agrees to 1.6e-16 relative (float round-trip).

## 6. Comparison to PA-2D-1 F4 — verdict

PA-2D-1 F4 registers: **C₂\* ≡ β_G_φ·Σ̃^4D/(Σ^φ·β̄_Ḡ_φ) — "Σ^4D cancels completely; Σ̃^4D (the venue
draw-law contraction) is the ONE new number."**

**VERDICT: AGREE.** My independent derivation reproduces F4's formula factor-for-factor — the same
Σ^4D cancellation, the same Σ^φ-in-denominator placement, Σ̃^4D as the class-G contraction, β̄_Ḡ_φ exact
on the completion side with no extra mass factor — and the same number, 0.06124403326364123. Neither of
the stage-0 bracket readings (ρ₂ = ρ or ρ₂ = 1) is the exact one; the exact ρ₂ is Σ̃^4D/Σ^4D, which is
what F4's collapsed form encodes.

**One qualification, flagged for the record (not a disagreement with F4's algebra):** F4's label for
Σ̃^4D — "the venue draw-law contraction" — is exact for the mixture's own class-G predictive (the pairing
under which the constant is derived, and the one the 1D C\* derivation itself used), but the
*implemented* b0i2d class-G draw (`_draw_2d_accepted_latents`) retains the 1D S̄_φ(z) weight in the
z-true draw (`_draw_kernel_survival_redshifts`: density ∝ k̄_g·S̄_φ) and layers the Bernoulli(S_4D)
acceptance on top — its own docstring discloses the equivalence to an S̃_4D,g-reweighting holds only
"up to the (unchanged) z-marginal's own existing survival weighting". The implemented venue's exact
class-G contraction is therefore Σ_g w_g·∫k̄_g·S̄_φ·S̄_4D,g dz (a both-survivals object ≠ Σ̃^4D, since
the mz_sel with-BH numerator — unlike the 1D twin's without-BH numerator — carries no per-candidate
S̄_φ(z) to match it), and against that venue no *pure* constant closes identity (I) exactly: the
pointwise ratio A₂/g₂ inherits an event-level S̄_φ(z_ev) tilt. In 1D this factor was matched on both
sides (venue kernel = twin kernel = k̄_g·S̄_φ) and hence invisible; in 2D it is unmatched. C₂ as
registered is exact for the model-side pairing; against the banked b0i2d fleet it is exact only modulo
that disclosed venue-drift term.
